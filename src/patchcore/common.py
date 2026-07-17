from __future__ import annotations

import os
from dataclasses import dataclass

import faiss
import numpy as np
import scipy.ndimage as ndimage


@dataclass
class SearchResult:
    distances: np.ndarray
    indices: np.ndarray


class FaissIndex:
    def __init__(self, on_gpu: bool = False, num_workers: int = 4) -> None:
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = bool(on_gpu)
        self.search_index = None

    def _create_index(self, dimension: int):
        if self.on_gpu:
            return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(),
                dimension,
                faiss.GpuIndexFlatConfig(),
            )
        return faiss.IndexFlatL2(dimension)

    def _index_to_cpu(self, index):
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _index_to_gpu(self, index):
        if self.on_gpu:
            return faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        return index

    def fit(self, features: np.ndarray) -> None:
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape={features.shape}")
        self.search_index = self._create_index(int(features.shape[1]))
        self.search_index.add(features.astype(np.float32, copy=False))

    def search(self, query_features: np.ndarray, k: int) -> SearchResult:
        if self.search_index is None:
            raise RuntimeError("Index is empty. Fit or load it before search.")
        distances, indices = self.search_index.search(
            query_features.astype(np.float32, copy=False),
            int(k),
        )
        return SearchResult(distances=distances, indices=indices)

    def save(self, path: str) -> None:
        if self.search_index is None:
            raise RuntimeError("Index is empty. Nothing to save.")
        faiss.write_index(self._index_to_cpu(self.search_index), path)

    def load(self, path: str) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(path))


class RescaleSegmentor:
    def __init__(self, target_size: int | tuple[int, int], smoothing: float = 4.0):
        self.target_size = target_size
        self.smoothing = float(smoothing)

    def __call__(self, patch_scores: np.ndarray) -> list[np.ndarray]:
        import torch
        import torch.nn.functional as F

        scores = torch.from_numpy(patch_scores.astype(np.float32, copy=False)).unsqueeze(1)
        scores = F.interpolate(
            scores,
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        arrays = scores.cpu().numpy()
        return [ndimage.gaussian_filter(item, sigma=self.smoothing) for item in arrays]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
