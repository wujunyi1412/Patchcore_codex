from __future__ import annotations

import argparse
import os
import sys

import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(CURRENT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from patchcore.onnx import export_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PatchCore embedder to ONNX.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default="onnx_model")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    output_path = export_onnx(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=device,
        opset=args.opset,
    )
    print(f"Saved ONNX model to {output_path}")


if __name__ == "__main__":
    main()
