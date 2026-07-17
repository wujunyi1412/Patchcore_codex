from __future__ import annotations

import argparse
import os
import sys

import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(CURRENT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from patchcore import PatchCoreModel
from patchcore import default_config
from patchcore.backbones import list_backbones


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal PatchCore memory bank.")
    parser.add_argument("--train-dir", required=True, help="Folder containing normal training images.")
    parser.add_argument("--output-dir", required=True, help="Directory to store the trained memory bank.")
    parser.add_argument(
        "--backbone",
        default="wide_resnet50_2",
        choices=list_backbones(),
        help="Backbone used for feature extraction.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="Feature layers to extract. Default depends on the backbone.",
    )
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument(
        "--preprocess",
        default="none",
        choices=["none", "gaussian_blur", "median", "sharpen"],
    )
    parser.add_argument("--pretrain-embed-dim", type=int, default=1024)
    parser.add_argument("--target-embed-dim", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=3)
    parser.add_argument("--patch-stride", type=int, default=1)
    parser.add_argument("--num-neighbors", type=int, default=1)
    parser.add_argument(
        "--sampler",
        default="identity",
        choices=["identity", "random", "greedy_coreset", "approx_greedy_coreset"],
        help="Sampling strategy used before building the memory bank.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Keep only a ratio of patch embeddings before indexing.",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    config = default_config(
        backbone_name=args.backbone,
        feature_layers=args.layers,
        image_size=args.image_size,
        preprocess=args.preprocess,
        pretrain_embed_dim=args.pretrain_embed_dim,
        target_embed_dim=args.target_embed_dim,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        num_neighbors=args.num_neighbors,
        sampler_name=args.sampler,
        sample_ratio=args.sample_ratio,
    )
    model = PatchCoreModel(config=config, device=device)
    stats = model.fit(train_path=args.train_dir)
    model.save(args.output_dir, stats)
    print(f"Saved model to {args.output_dir}")
    print(
        f"train_images={stats.train_image_count} embeddings={stats.embedding_count} "
        f"embedding_dim={stats.embedding_dim} sampler={args.sampler} "
        f"sample_ratio={args.sample_ratio}"
    )


if __name__ == "__main__":
    main()
