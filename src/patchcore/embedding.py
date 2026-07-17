from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EmbeddingOutput:
    embeddings: torch.Tensor
    patch_shape: tuple[int, int]


class PatchMaker:
    def __init__(self, patch_size: int, patch_stride: int = 1) -> None:
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)

    def patchify(self, feature_map: torch.Tensor):
        padding = (self.patch_size - 1) // 2
        unfold = torch.nn.Unfold(
            kernel_size=self.patch_size,
            stride=self.patch_stride,
            padding=padding,
        )
        unfolded = unfold(feature_map)
        patch_h = int(
            (feature_map.shape[-2] + 2 * padding - (self.patch_size - 1) - 1)
            / self.patch_stride
            + 1
        )
        patch_w = int(
            (feature_map.shape[-1] + 2 * padding - (self.patch_size - 1) - 1)
            / self.patch_stride
            + 1
        )
        unfolded = unfolded.reshape(
            feature_map.shape[0],
            feature_map.shape[1],
            self.patch_size,
            self.patch_size,
            -1,
        )
        unfolded = unfolded.permute(0, 4, 1, 2, 3)
        return unfolded, (patch_h, patch_w)


class PatchcoreEmbedder(torch.nn.Module):
    def __init__(
        self,
        pretrain_embed_dim: int,
        target_embed_dim: int,
        patch_size: int,
        patch_stride: int = 1,
    ) -> None:
        super().__init__()
        self.pretrain_embed_dim = int(pretrain_embed_dim)
        self.target_embed_dim = int(target_embed_dim)
        self.patch_maker = PatchMaker(patch_size=patch_size, patch_stride=patch_stride)

    def _align_patch_grid(
        self,
        patches: torch.Tensor,
        patch_shape: tuple[int, int],
        ref_shape: tuple[int, int],
    ) -> torch.Tensor:
        if patch_shape == ref_shape:
            return patches

        reshaped = patches.reshape(
            patches.shape[0],
            patch_shape[0],
            patch_shape[1],
            *patches.shape[2:],
        )
        reshaped = reshaped.permute(0, 3, 4, 5, 1, 2)
        base_shape = reshaped.shape
        reshaped = reshaped.reshape(-1, *reshaped.shape[-2:])
        reshaped = F.interpolate(
            reshaped.unsqueeze(1),
            size=ref_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        reshaped = reshaped.reshape(*base_shape[:-2], *ref_shape)
        reshaped = reshaped.permute(0, 4, 5, 1, 2, 3)
        return reshaped.reshape(len(reshaped), -1, *reshaped.shape[-3:])

    def _pool_layer(self, patches: torch.Tensor) -> torch.Tensor:
        patches = patches.reshape(len(patches), 1, -1)
        return F.adaptive_avg_pool1d(patches, self.pretrain_embed_dim).squeeze(1)

    def forward(self, ordered_feature_maps: list[torch.Tensor]) -> EmbeddingOutput:
        patch_sets = [self.patch_maker.patchify(feature_map) for feature_map in ordered_feature_maps]
        ref_shape = patch_sets[0][1]

        aligned = []
        for patches, patch_shape in patch_sets:
            aligned.append(self._align_patch_grid(patches, patch_shape, ref_shape))

        flattened = [patches.reshape(-1, *patches.shape[-3:]) for patches in aligned]
        pooled = [self._pool_layer(patches) for patches in flattened]
        merged = torch.stack(pooled, dim=1)
        merged = merged.reshape(len(merged), 1, -1)
        embeddings = F.adaptive_avg_pool1d(merged, self.target_embed_dim).reshape(
            len(merged), -1
        )
        return EmbeddingOutput(embeddings=embeddings, patch_shape=ref_shape)
