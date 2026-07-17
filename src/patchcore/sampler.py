from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SamplerConfig:
    name: str
    ratio: float


class BaseSampler:
    def __init__(self, ratio: float) -> None:
        ratio = float(ratio)
        if ratio <= 0 or ratio > 1:
            raise ValueError("sample_ratio must be in (0, 1].")
        self.ratio = ratio

    def sample(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _target_count(self, features: np.ndarray) -> int:
        return max(1, int(len(features) * self.ratio))


class IdentitySampler(BaseSampler):
    def sample(self, features: np.ndarray) -> np.ndarray:
        return features


class RandomSampler(BaseSampler):
    def sample(self, features: np.ndarray) -> np.ndarray:
        if self.ratio >= 1.0:
            return features
        target_count = self._target_count(features)
        indices = np.random.choice(len(features), size=target_count, replace=False)
        indices.sort()
        return features[indices]


class GreedyCoresetSampler(BaseSampler):
    def __init__(self, ratio: float, device: str = "cpu") -> None:
        super().__init__(ratio)
        self.device = torch.device(device)

    def sample(self, features: np.ndarray) -> np.ndarray:
        if self.ratio >= 1.0:
            return features

        target_count = self._target_count(features)
        if target_count >= len(features):
            return features

        feature_tensor = torch.from_numpy(features.astype(np.float32, copy=False)).to(self.device)
        selected_indices = [0]
        min_distances = torch.cdist(feature_tensor[:1], feature_tensor, p=2).squeeze(0)

        while len(selected_indices) < target_count:
            next_index = int(torch.argmax(min_distances).item())
            selected_indices.append(next_index)
            distances = torch.cdist(
                feature_tensor[next_index : next_index + 1],
                feature_tensor,
                p=2,
            ).squeeze(0)
            min_distances = torch.minimum(min_distances, distances)

        return features[np.array(selected_indices, dtype=np.int64)]


class ApproximateGreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        ratio: float,
        device: str = "cpu",
        projection_dim: int = 128,
        start_points: int = 10,
    ) -> None:
        super().__init__(ratio)
        self.device = torch.device(device)
        self.projection_dim = int(projection_dim)
        self.start_points = int(start_points)

    def _project(self, features: np.ndarray) -> torch.Tensor:
        feature_tensor = torch.from_numpy(features.astype(np.float32, copy=False)).to(self.device)
        if feature_tensor.shape[1] <= self.projection_dim:
            return feature_tensor

        generator = torch.Generator(device=self.device)
        generator.manual_seed(0)
        projection = torch.randn(
            feature_tensor.shape[1],
            self.projection_dim,
            generator=generator,
            device=self.device,
            dtype=feature_tensor.dtype,
        )
        projection = projection / torch.sqrt(torch.tensor(self.projection_dim, device=self.device))
        return feature_tensor @ projection

    def sample(self, features: np.ndarray) -> np.ndarray:
        if self.ratio >= 1.0:
            return features

        target_count = self._target_count(features)
        if target_count >= len(features):
            return features

        projected = self._project(features)
        start_count = min(self.start_points, len(features))
        initial_indices = torch.linspace(
            0,
            len(features) - 1,
            steps=start_count,
            device=self.device,
        ).long()
        min_distances = torch.cdist(projected[initial_indices], projected, p=2).min(dim=0).values
        selected_indices = initial_indices.cpu().numpy().tolist()

        while len(selected_indices) < target_count:
            next_index = int(torch.argmax(min_distances).item())
            selected_indices.append(next_index)
            distances = torch.cdist(
                projected[next_index : next_index + 1],
                projected,
                p=2,
            ).squeeze(0)
            min_distances = torch.minimum(min_distances, distances)

        unique_indices = np.unique(np.array(selected_indices, dtype=np.int64))
        if len(unique_indices) > target_count:
            unique_indices = unique_indices[:target_count]
        return features[unique_indices]


def create_sampler(name: str, ratio: float, device: str = "cpu") -> BaseSampler:
    normalized = name.lower()
    if normalized == "identity":
        return IdentitySampler(ratio=1.0)
    if normalized == "random":
        return RandomSampler(ratio=ratio)
    if normalized == "greedy_coreset":
        return GreedyCoresetSampler(ratio=ratio, device=device)
    if normalized == "approx_greedy_coreset":
        return ApproximateGreedyCoresetSampler(ratio=ratio, device=device)
    raise ValueError(
        "Unsupported sampler '{}'. Available: identity, random, greedy_coreset, "
        "approx_greedy_coreset".format(name)
    )
