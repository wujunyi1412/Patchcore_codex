from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from patchcore import backbones
from patchcore.artifacts import BankStats
from patchcore.artifacts import ModelConfig
from patchcore.artifacts import load_index
from patchcore.artifacts import load_metadata
from patchcore.artifacts import save_index
from patchcore.artifacts import save_metadata
from patchcore.common import FaissIndex
from patchcore.common import RescaleSegmentor
from patchcore.embedding import PatchcoreEmbedder
from patchcore.preprocess import ImagePreprocessor
from patchcore.preprocess import list_images
from patchcore.sampler import create_sampler


@dataclass
class InferenceResult:
    image_path: str
    image_score: float
    patch_scores: np.ndarray
    mask: np.ndarray
    patch_shape: tuple[int, int]


class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, feature_layers: list[str]) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_layers = feature_layers
        self._handles = []
        self._outputs: dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _resolve_module(self, dotted_name: str) -> torch.nn.Module:
        module: torch.nn.Module = self.backbone
        for part in dotted_name.split("."):
            if part.isdigit():
                module = module[int(part)]  # type: ignore[index]
            else:
                module = getattr(module, part)
        return module

    def _register_hooks(self) -> None:
        for layer_name in self.feature_layers:
            module = self._resolve_module(layer_name)
            self._handles.append(module.register_forward_hook(self._capture(layer_name)))

    def _capture(self, layer_name: str):
        def hook(_module, _inputs, output):
            self._outputs[layer_name] = output

        return hook

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        self._outputs.clear()
        _ = self.backbone(images)
        return [self._outputs[name] for name in self.feature_layers]


class PatchCoreModel:
    def __init__(self, config: ModelConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.preprocessor = ImagePreprocessor(
            image_size=config.image_size,
            preprocess=config.preprocess,
        )
        backbone = backbones.load(config.backbone_name).to(device)
        self.extractor = FeatureExtractor(backbone, config.feature_layers).to(device)
        self.extractor.eval()
        self.embedder = PatchcoreEmbedder(
            pretrain_embed_dim=config.pretrain_embed_dim,
            target_embed_dim=config.target_embed_dim,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
        ).to(device)
        self.embedder.eval()
        self.index = FaissIndex(on_gpu=False)
        self.segmentor = RescaleSegmentor(target_size=(config.image_size, config.image_size))

    @classmethod
    def from_model_dir(cls, model_dir: str | Path, device: torch.device) -> "PatchCoreModel":
        config, _ = load_metadata(model_dir)
        model = cls(config=config, device=device)
        model.index = load_index(model_dir, on_gpu=False)
        return model

    def _embed_tensor(self, tensor: torch.Tensor):
        with torch.no_grad():
            feature_maps = self.extractor(tensor.to(self.device, dtype=torch.float32))
            output = self.embedder(feature_maps)
        return output.embeddings.detach().cpu().numpy(), output.patch_shape

    def embed_paths(self, image_paths: list[str | Path]) -> np.ndarray:
        embeddings = []
        for image_path in image_paths:
            tensor, _, _, _ = self.preprocessor(image_path)
            image_embeddings, _ = self._embed_tensor(tensor)
            embeddings.append(image_embeddings)
        if not embeddings:
            raise ValueError("No training images found.")
        return np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)

    def fit(self, train_path: str | Path) -> BankStats:
        image_paths = list_images(train_path)
        features = self.embed_paths(image_paths)
        sampler = create_sampler(
            name=self.config.sampler_name,
            ratio=self.config.sample_ratio,
            device=str(self.device),
        )
        features = sampler.sample(features)

        self.index.fit(features)
        return BankStats(
            train_image_count=len(image_paths),
            embedding_count=int(features.shape[0]),
            embedding_dim=int(features.shape[1]),
        )

    def save(self, output_dir: str | Path, stats: BankStats) -> None:
        save_metadata(output_dir, self.config, stats)
        save_index(output_dir, self.index)

    def infer_path(self, image_path: str | Path) -> InferenceResult:
        tensor, _, _, _ = self.preprocessor(image_path)
        return self.infer_tensor(tensor=tensor, image_path=image_path)

    def infer_tensor(
        self,
        tensor: torch.Tensor,
        image_path: str | Path = "<memory>",
    ) -> InferenceResult:
        embeddings, patch_shape = self._embed_tensor(tensor)
        search = self.index.search(embeddings, self.config.num_neighbors)
        patch_scores = np.mean(search.distances, axis=1)
        image_score = float(np.max(patch_scores)) if len(patch_scores) else 0.0
        patch_map = patch_scores.reshape(1, patch_shape[0], patch_shape[1]).astype(
            np.float32,
            copy=False,
        )
        mask = self.segmentor(patch_map)[0]
        return InferenceResult(
            image_path=str(image_path),
            image_score=image_score,
            patch_scores=patch_scores,
            mask=mask,
            patch_shape=patch_shape,
        )


def default_config(
    backbone_name: str = "wide_resnet50_2",
    feature_layers: list[str] | None = None,
    image_size: int = 320,
    preprocess: str = "none",
    pretrain_embed_dim: int = 1024,
    target_embed_dim: int = 1024,
    patch_size: int = 3,
    patch_stride: int = 1,
    num_neighbors: int = 1,
    sampler_name: str = "identity",
    sample_ratio: float = 1.0,
) -> ModelConfig:
    layers = list(feature_layers or backbones.default_layers(backbone_name))
    return ModelConfig(
        backbone_name=backbone_name,
        feature_layers=layers,
        image_size=int(image_size),
        preprocess=preprocess,
        pretrain_embed_dim=int(pretrain_embed_dim),
        target_embed_dim=int(target_embed_dim),
        patch_size=int(patch_size),
        patch_stride=int(patch_stride),
        num_neighbors=int(num_neighbors),
        sampler_name=sampler_name,
        sample_ratio=float(sample_ratio),
    )
