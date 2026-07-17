from __future__ import annotations

import json
from pathlib import Path

import torch

from patchcore.patchcore import PatchCoreModel


class OnnxEmbeddingModel(torch.nn.Module):
    def __init__(self, model: PatchCoreModel) -> None:
        super().__init__()
        self.extractor = model.extractor
        self.embedder = model.embedder
        self.feature_layers = tuple(model.config.feature_layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_maps = self.extractor(images)
        output = self.embedder(feature_maps)
        return output.embeddings


def export_onnx(
    model_dir: str,
    output_dir: str,
    device: torch.device,
    opset: int = 17,
) -> Path:
    model = PatchCoreModel.from_model_dir(model_dir=model_dir, device=device)
    wrapper = OnnxEmbeddingModel(model).to(device)
    wrapper.eval()

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    onnx_path = target / "patchcore_embedder.onnx"
    dummy = torch.randn(1, 3, model.config.image_size, model.config.image_size, device=device)
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embeddings"],
        dynamic_axes={
            "input": {0: "batch"},
            "embeddings": {0: "patches"},
        },
    )

    metadata = {
        "image_size": model.config.image_size,
        "feature_layers": model.config.feature_layers,
        "embedding_dim": model.config.target_embed_dim,
        "patch_size": model.config.patch_size,
        "patch_stride": model.config.patch_stride,
        "num_neighbors": model.config.num_neighbors,
    }
    (target / "onnx_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return onnx_path
