from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(CURRENT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from patchcore.patchcore import PatchCoreModel
from patchcore.preprocess import list_images


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def save_panel(
    original_image,
    processed_image,
    mask: np.ndarray,
    score: float,
    threshold: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis in axes:
        axis.axis("off")

    axes[0].imshow(np.asarray(original_image))
    axes[0].set_title("Original")

    axes[1].imshow(np.asarray(processed_image))
    axes[1].set_title("Processed")

    axes[2].imshow(np.asarray(processed_image))
    overlay = axes[2].imshow(mask.astype(np.float32), cmap="jet", alpha=0.55)
    axes[2].set_title("Heatmap")
    figure.colorbar(overlay, ax=axes[2], fraction=0.046, pad=0.04)

    status = "ANOMALY" if score >= threshold else "OK"
    color = "salmon" if status == "ANOMALY" else "lightgreen"
    figure.text(
        0.5,
        0.02,
        f"score={score:.6f} threshold={threshold:.6f} status={status}",
        ha="center",
        fontsize=12,
        bbox={"boxstyle": "round", "facecolor": color, "alpha": 0.85},
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal PatchCore inference.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True, help="Single image or folder of images.")
    parser.add_argument("--output-dir", default="infer_results")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    LOGGER.info("Loading model from %s on %s", args.model_dir, device)
    load_start = time.perf_counter()
    model = PatchCoreModel.from_model_dir(args.model_dir, device=device)
    load_elapsed = time.perf_counter() - load_start
    LOGGER.info(
        "Model loaded in %.3fs | backbone=%s layers=%s sampler=%s ratio=%.3f",
        load_elapsed,
        model.config.backbone_name,
        ",".join(model.config.feature_layers),
        model.config.sampler_name,
        model.config.sample_ratio,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    above_dir = output_dir / "above_threshold"
    below_dir = output_dir / "below_threshold"
    above_dir.mkdir(exist_ok=True)
    below_dir.mkdir(exist_ok=True)

    rows = []
    image_paths = list_images(args.input)
    LOGGER.info("Found %d image(s) in %s", len(image_paths), args.input)

    total_start = time.perf_counter()
    for index, image_path in enumerate(image_paths, start=1):
        image_start = time.perf_counter()
        preprocess_start = time.perf_counter()
        tensor, original_image, processed_image, _ = model.preprocessor(image_path)
        preprocess_elapsed = time.perf_counter() - preprocess_start

        infer_start = time.perf_counter()
        result = model.infer_tensor(tensor=tensor, image_path=image_path)
        infer_elapsed = time.perf_counter() - infer_start

        destination = above_dir if result.image_score >= args.threshold else below_dir
        panel_path = destination / f"{image_path.stem}_panel.png"
        render_start = time.perf_counter()
        save_panel(
            original_image=original_image,
            processed_image=processed_image,
            mask=result.mask,
            score=result.image_score,
            threshold=args.threshold,
            output_path=panel_path,
        )
        render_elapsed = time.perf_counter() - render_start
        image_elapsed = time.perf_counter() - image_start
        status = "ANOMALY" if result.image_score >= args.threshold else "OK"

        rows.append(
            {
                "filename": str(image_path),
                "score": f"{result.image_score:.6f}",
                "threshold": f"{args.threshold:.6f}",
                "is_anomaly": int(result.image_score >= args.threshold),
                "panel_path": str(panel_path),
            }
        )
        LOGGER.info(
            "[%d/%d] %s | score=%.6f threshold=%.6f status=%s | preprocess=%.3fs infer=%.3fs render=%.3fs total=%.3fs",
            index,
            len(image_paths),
            image_path.name,
            result.image_score,
            args.threshold,
            status,
            preprocess_elapsed,
            infer_elapsed,
            render_elapsed,
            image_elapsed,
        )

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "score", "threshold", "is_anomaly", "panel_path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    total_elapsed = time.perf_counter() - total_start
    LOGGER.info("Saved inference results to %s", output_dir)
    LOGGER.info(
        "Processed %d image(s) in %.3fs | avg=%.3fs/image",
        len(image_paths),
        total_elapsed,
        total_elapsed / max(1, len(image_paths)),
    )


if __name__ == "__main__":
    main()


" python infer.py --model-dir artifacts/dinov2_bank --input dataset/corona_images/test/broken --output-dir infer_out/dinov2 --threshold 1.0"