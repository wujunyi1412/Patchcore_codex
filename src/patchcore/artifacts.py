from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from patchcore.common import FaissIndex


@dataclass
class ModelConfig:
    backbone_name: str
    feature_layers: list[str]
    image_size: int
    preprocess: str
    pretrain_embed_dim: int
    target_embed_dim: int
    patch_size: int
    patch_stride: int
    num_neighbors: int
    sampler_name: str
    sample_ratio: float


@dataclass
class BankStats:
    train_image_count: int
    embedding_count: int
    embedding_dim: int


def save_metadata(output_dir: str | Path, config: ModelConfig, stats: BankStats) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = {"config": asdict(config), "stats": asdict(stats)}
    (target / "model_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def load_metadata(model_dir: str | Path) -> tuple[ModelConfig, BankStats]:
    payload = json.loads(Path(model_dir, "model_config.json").read_text(encoding="utf-8"))
    config = ModelConfig(**payload["config"])
    stats = BankStats(**payload["stats"])
    return config, stats


def save_index(output_dir: str | Path, index: FaissIndex) -> None:
    index.save(str(Path(output_dir, "memory_bank.faiss")))


def load_index(model_dir: str | Path, on_gpu: bool = False) -> FaissIndex:
    index = FaissIndex(on_gpu=on_gpu)
    index.load(str(Path(model_dir, "memory_bank.faiss")))
    return index
