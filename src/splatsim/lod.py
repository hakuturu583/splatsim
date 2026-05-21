"""Level-of-Detail (LOD) system for Gaussian Splatting.

Pre-computes importance-based LOD tiers at load time and provides
zero-cost tensor slicing at render time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from splatsim._conversions import GaussianTensors
from splatsim.dataclass.lod_config import LodConfig


@dataclass
class LodIndex:
    """Pre-computed LOD metadata for a single GaussianTensors instance.

    Gaussians are assumed to be sorted by importance (descending).
    Each tier is defined by a count, and rendering tier *k* means
    using the first ``tier_counts[k]`` Gaussians — a simple slice.
    """

    tier_counts: list[int]
    tier_max_distances: list[float]
    centroid: Tensor  # [3]


class LodManager:
    """Manages LOD pre-computation and per-frame tier selection."""

    def __init__(self, config: LodConfig) -> None:
        self._config = config

    @property
    def config(self) -> LodConfig:
        return self._config

    def precompute(self, tensors: GaussianTensors) -> tuple[GaussianTensors, LodIndex]:
        """Sort Gaussians by importance and compute tier boundaries.

        Importance is defined as ``max_scale * opacity``.  Larger, more
        opaque Gaussians contribute most to the rendered image and are
        therefore ranked highest.

        Returns the re-ordered tensors and a :class:`LodIndex` that
        stores tier counts and the centroid for distance computation.
        """
        n = tensors.means.shape[0]

        # importance = max scale axis * opacity
        max_scale = tensors.scales.max(dim=1).values  # [N]
        importance = max_scale * tensors.opacities  # [N]

        sorted_idx = importance.argsort(descending=True)

        sorted_tensors = GaussianTensors(
            means=tensors.means[sorted_idx],
            quats=tensors.quats[sorted_idx],
            scales=tensors.scales[sorted_idx],
            opacities=tensors.opacities[sorted_idx],
            colors=tensors.colors[sorted_idx],
            sh_degree=tensors.sh_degree,
        )

        # Compute tier counts (sorted by max_distance ascending)
        tiers = sorted(self._config.tiers, key=lambda t: t.max_distance)
        tier_counts: list[int] = []
        tier_max_distances: list[float] = []
        for tier in tiers:
            count = max(1, int(n * tier.fraction))
            tier_counts.append(count)
            tier_max_distances.append(tier.max_distance)

        centroid = sorted_tensors.means.mean(dim=0)

        lod_index = LodIndex(
            tier_counts=tier_counts,
            tier_max_distances=tier_max_distances,
            centroid=centroid,
        )
        return sorted_tensors, lod_index

    def select_tier(self, lod_index: LodIndex, camera_position: Tensor) -> int:
        """Choose the LOD tier based on camera-to-centroid distance."""
        dist = torch.linalg.norm(camera_position - lod_index.centroid).item()
        for i, max_d in enumerate(lod_index.tier_max_distances):
            if dist <= max_d:
                return i
        # Beyond all thresholds — use the coarsest (last) tier
        return len(lod_index.tier_max_distances) - 1

    def apply(
        self,
        tensors: GaussianTensors,
        lod_index: LodIndex,
        tier: int,
    ) -> GaussianTensors:
        """Return a sliced view of *tensors* for the given LOD tier."""
        n = lod_index.tier_counts[tier]
        if n >= tensors.means.shape[0]:
            return tensors
        return GaussianTensors(
            means=tensors.means[:n],
            quats=tensors.quats[:n],
            scales=tensors.scales[:n],
            opacities=tensors.opacities[:n],
            colors=tensors.colors[:n],
            sh_degree=tensors.sh_degree,
        )
