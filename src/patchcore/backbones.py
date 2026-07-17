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


_BACKBONES: dict[str, BackboneDefinition] = {
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
