"""Level-of-Detail (LOD) system for Gaussian Splatting.

Pre-computes importance-based LOD tiers at load time and provides
zero-cost tensor slicing at render time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    centroid: tuple[float, float, float]


class LodManager:
    """Manages LOD pre-computation and per-frame tier selection."""

    def __init__(self, config: LodConfig) -> None:
        self._tiers = sorted(config.tiers, key=lambda t: t.max_distance)

    def precompute(self, tensors: GaussianTensors) -> tuple[GaussianTensors, LodIndex]:
        """Sort Gaussians by importance and compute tier boundaries.

        Importance is defined as ``max_scale * opacity``.  Larger, more
        opaque Gaussians contribute most to the rendered image and are
        therefore ranked highest.

        Returns the re-ordered tensors and a :class:`LodIndex` that
        stores tier counts and the centroid for distance computation.
        """
        n = tensors.means.shape[0]

        max_scale = tensors.scales.max(dim=1).values  # [N]
        importance = max_scale * tensors.opacities  # [N]
        sorted_idx = importance.argsort(descending=True)

        sorted_tensors = tensors[sorted_idx]

        tier_counts: list[int] = []
        tier_max_distances: list[float] = []
        for tier in self._tiers:
            tier_counts.append(max(1, int(n * tier.fraction)))
            tier_max_distances.append(tier.max_distance)

        # Store centroid as plain floats for CPU-side distance computation.
        c = sorted_tensors.means.mean(dim=0)
        centroid = (c[0].item(), c[1].item(), c[2].item())

        lod_index = LodIndex(
            tier_counts=tier_counts,
            tier_max_distances=tier_max_distances,
            centroid=centroid,
        )
        return sorted_tensors, lod_index

    def filter(
        self,
        tensors: GaussianTensors,
        lod_index: LodIndex,
        camera_position: tuple[float, float, float],
    ) -> GaussianTensors:
        """Select the appropriate LOD tier and return sliced tensors.

        Combines tier selection (by camera-to-centroid distance) and
        tensor slicing into a single call.
        """
        cx, cy, cz = lod_index.centroid
        px, py, pz = camera_position
        dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)

        tier = len(lod_index.tier_max_distances) - 1
        for i, max_d in enumerate(lod_index.tier_max_distances):
            if dist <= max_d:
                tier = i
                break

        n = lod_index.tier_counts[tier]
        if n >= tensors.means.shape[0]:
            return tensors
        return tensors[:n]
