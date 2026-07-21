from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Iterable

import torch
from torchvision import models


@dataclass(frozen=True)
class BackboneDefinition:
    name: str
    builder: Callable[[], torch.nn.Module]
    default_layers: tuple[str, ...]


class DinoV2FeatureMapBackbone(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()

    @staticmethod
    def _block_index(layer_name: str) -> int:
        prefix = "blocks."
        if not layer_name.startswith(prefix):
            raise ValueError(
                f"DINOv2 layer names must look like 'blocks.N', got '{layer_name}'."
            )
        return int(layer_name[len(prefix) :])

    def extract_feature_maps(
        self,
        images: torch.Tensor,
        feature_layers: list[str],
    ) -> list[torch.Tensor]:
        block_indices = [self._block_index(layer_name) for layer_name in feature_layers]
        outputs = self.model.get_intermediate_layers(
            images,
            n=block_indices,
            reshape=True,
            return_class_token=False,
            norm=True,
        )
        return list(outputs)


def _build_resnet50() -> torch.nn.Module:
    try:
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    except AttributeError:
        return models.resnet50(pretrained=True)


def _build_resnet101() -> torch.nn.Module:
    try:
        return models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
    except AttributeError:
        return models.resnet101(pretrained=True)


def _build_wide_resnet50_2() -> torch.nn.Module:
    try:
        return models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2
        )
    except AttributeError:
        return models.wide_resnet50_2(pretrained=True)


def _build_dinov2_vits14() -> torch.nn.Module:
    return DinoV2FeatureMapBackbone("dinov2_vits14")


def _build_dinov2_vitb14() -> torch.nn.Module:
    return DinoV2FeatureMapBackbone("dinov2_vitb14")


def _build_dinov2_vitl14() -> torch.nn.Module:
    return DinoV2FeatureMapBackbone("dinov2_vitl14")


def _build_dinov2_vitg14() -> torch.nn.Module:
    return DinoV2FeatureMapBackbone("dinov2_vitg14")


_BACKBONES: dict[str, BackboneDefinition] = {
    "dinov2_vitb14": BackboneDefinition(
        name="dinov2_vitb14",
        builder=_build_dinov2_vitb14,
        default_layers=("blocks.5", "blocks.11"),
    ),
    "dinov2_vitg14": BackboneDefinition(
        name="dinov2_vitg14",
        builder=_build_dinov2_vitg14,
        default_layers=("blocks.19", "blocks.39"),
    ),
    "dinov2_vitl14": BackboneDefinition(
        name="dinov2_vitl14",
        builder=_build_dinov2_vitl14,
        default_layers=("blocks.11", "blocks.23"),
    ),
    "dinov2_vits14": BackboneDefinition(
        name="dinov2_vits14",
        builder=_build_dinov2_vits14,
        default_layers=("blocks.5", "blocks.11"),
    ),
    "resnet50": BackboneDefinition(
        name="resnet50",
        builder=_build_resnet50,
        default_layers=("layer2", "layer3"),
    ),
    "resnet101": BackboneDefinition(
        name="resnet101",
        builder=_build_resnet101,
        default_layers=("layer2", "layer3"),
    ),
    "wide_resnet50_2": BackboneDefinition(
        name="wide_resnet50_2",
        builder=_build_wide_resnet50_2,
        default_layers=("layer2", "layer3"),
    ),
}


def list_backbones() -> list[str]:
    return sorted(_BACKBONES.keys())


def get_definition(name: str) -> BackboneDefinition:
    if name not in _BACKBONES:
        available = ", ".join(list_backbones())
        raise KeyError(f"Unsupported backbone '{name}'. Available: {available}")
    return _BACKBONES[name]


def load(name: str) -> torch.nn.Module:
    model = get_definition(name).builder()
    model.eval()
    return model


def default_layers(name: str) -> tuple[str, ...]:
    return get_definition(name).default_layers


def validate_layers(layers: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(layer.strip() for layer in layers if layer and layer.strip())
    if not normalized:
        raise ValueError("At least one feature layer is required.")
    return normalized
