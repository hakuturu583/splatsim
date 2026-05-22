"""Level-of-Detail (LOD) system for Gaussian Splatting.

Pre-computes importance-based LOD tiers at load time and provides
zero-cost tensor slicing at render time.  Supports two modes:

* **Centroid mode** (default): single centroid per source, fast but
  breaks when camera moves far from the centroid.
* **Octree mode** (``max_gaussians_per_cell`` set): density-adaptive
  spatial partitioning via recursive octree subdivision.  Each leaf
  cell applies LOD independently based on camera-to-cell distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from splatsim._conversions import GaussianTensors
from splatsim.dataclass.lod_config import LodConfig


@dataclass
class LodIndex:
    """Pre-computed LOD metadata for a single GaussianTensors instance."""

    tier_counts: list[int]
    tier_max_distances: list[float]
    tier_max_distances_t: Tensor = field(repr=False)
    """[T] float32 GPU tensor of tier max distances (cached for filter)."""
    centroid: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # --- octree / grid mode (all None in centroid mode) ---
    cell_centers: Tensor | None = field(default=None, repr=False)
    """[C, 3] center of each leaf cell."""
    cell_ranges: Tensor | None = field(default=None, repr=False)
    """[C, 2] int64 (start, end) into the sorted tensors."""
    cell_tier_counts: Tensor | None = field(default=None, repr=False)
    """[C, T] int64 per-cell Gaussian count for each tier."""


class LodManager:
    """Manages LOD pre-computation and per-frame tier selection."""

    def __init__(self, config: LodConfig) -> None:
        self._tiers = sorted(config.tiers, key=lambda t: t.max_distance)
        self._max_gpc = config.max_gaussians_per_cell

    # ------------------------------------------------------------------
    # Precompute
    # ------------------------------------------------------------------

    def precompute(self, tensors: GaussianTensors) -> tuple[GaussianTensors, LodIndex]:
        """Sort Gaussians by importance and compute LOD metadata.

        When ``max_gaussians_per_cell`` is set, performs octree-based
        spatial partitioning; otherwise falls back to centroid mode.
        """
        n = tensors.means.shape[0]
        max_scale = tensors.scales.max(dim=1).values  # [N]
        importance = max_scale * tensors.opacities  # [N]

        if self._max_gpc is not None:
            return self._precompute_octree(tensors, importance, n)
        return self._precompute_centroid(tensors, importance, n)

    def _build_global_stats(
        self, sorted_tensors: GaussianTensors, n: int
    ) -> tuple[list[int], list[float], tuple[float, float, float]]:
        """Compute tier counts, max distances, and centroid."""
        tier_counts = [max(1, int(n * t.fraction)) for t in self._tiers]
        tier_max_distances = [t.max_distance for t in self._tiers]
        c = sorted_tensors.means.mean(dim=0)
        centroid = (c[0].item(), c[1].item(), c[2].item())
        return tier_counts, tier_max_distances, centroid

    def _precompute_centroid(
        self,
        tensors: GaussianTensors,
        importance: Tensor,
        n: int,
    ) -> tuple[GaussianTensors, LodIndex]:
        sorted_idx = importance.argsort(descending=True)
        sorted_tensors = tensors[sorted_idx]
        tier_counts, tier_max_distances, centroid = self._build_global_stats(
            sorted_tensors, n
        )

        return sorted_tensors, LodIndex(
            tier_counts=tier_counts,
            tier_max_distances=tier_max_distances,
            tier_max_distances_t=torch.tensor(
                tier_max_distances, device=tensors.means.device, dtype=torch.float32
            ),
            centroid=centroid,
        )

    def _precompute_octree(
        self,
        tensors: GaussianTensors,
        importance: Tensor,
        n: int,
    ) -> tuple[GaussianTensors, LodIndex]:
        device = tensors.means.device

        # --- octree subdivision (CPU) ---
        means_cpu = tensors.means.detach().cpu()
        aabb_min = means_cpu.min(dim=0).values
        aabb_max = means_cpu.max(dim=0).values
        all_indices = torch.arange(n)

        leaves: list[tuple[Tensor, Tensor, Tensor]] = []  # (indices, lo, hi)
        self._subdivide(means_cpu, all_indices, aabb_min, aabb_max, leaves, depth=0)

        # --- sort within each leaf by importance (descending) ---
        importance_cpu = importance.detach().cpu()
        ordered_indices: list[Tensor] = []
        cell_centers_list: list[Tensor] = []
        cell_ranges_list: list[tuple[int, int]] = []
        offset = 0

        for leaf_indices, lo, hi in leaves:
            leaf_imp = importance_cpu[leaf_indices]
            local_order = leaf_imp.argsort(descending=True)
            ordered = leaf_indices[local_order]
            ordered_indices.append(ordered)

            count = len(ordered)
            cell_ranges_list.append((offset, offset + count))
            cell_centers_list.append((lo + hi) / 2.0)
            offset += count

        final_order = torch.cat(ordered_indices).to(device)
        sorted_tensors = tensors[final_order]

        # --- build cell metadata ---
        num_cells = len(leaves)
        num_tiers = len(self._tiers)

        cell_centers = torch.stack(cell_centers_list).to(
            device=device, dtype=torch.float32
        )
        cell_ranges = torch.tensor(cell_ranges_list, device=device, dtype=torch.int64)
        cell_counts = cell_ranges[:, 1] - cell_ranges[:, 0]

        cell_tier_counts = torch.zeros(
            num_cells, num_tiers, device=device, dtype=torch.int64
        )
        for j, tier in enumerate(self._tiers):
            cell_tier_counts[:, j] = (
                (cell_counts.float() * tier.fraction).clamp(min=1).long()
            )

        tier_counts, tier_max_distances, centroid = self._build_global_stats(
            sorted_tensors, n
        )

        return sorted_tensors, LodIndex(
            tier_counts=tier_counts,
            tier_max_distances=tier_max_distances,
            tier_max_distances_t=torch.tensor(
                tier_max_distances, device=device, dtype=torch.float32
            ),
            centroid=centroid,
            cell_centers=cell_centers,
            cell_ranges=cell_ranges,
            cell_tier_counts=cell_tier_counts,
        )

    def _subdivide(
        self,
        means: Tensor,
        indices: Tensor,
        lo: Tensor,
        hi: Tensor,
        leaves: list[tuple[Tensor, Tensor, Tensor]],
        depth: int,
        max_depth: int = 10,
    ) -> None:
        """Recursively subdivide a cell into octants."""
        if self._max_gpc is None or len(indices) <= self._max_gpc or depth >= max_depth:
            if len(indices) > 0:
                leaves.append((indices, lo.clone(), hi.clone()))
            return

        mid = (lo + hi) / 2.0
        pts = means[indices]

        for octant in range(8):
            child_lo = lo.clone()
            child_hi = hi.clone()
            mask = torch.ones(len(indices), dtype=torch.bool)
            for axis in range(3):
                if octant & (1 << axis):
                    child_lo[axis] = mid[axis]
                    mask &= pts[:, axis] >= mid[axis]
                else:
                    child_hi[axis] = mid[axis]
                    mask &= pts[:, axis] < mid[axis]

            child_indices = indices[mask]
            if len(child_indices) > 0:
                self._subdivide(
                    means,
                    child_indices,
                    child_lo,
                    child_hi,
                    leaves,
                    depth + 1,
                    max_depth,
                )

    # ------------------------------------------------------------------
    # Per-frame filter
    # ------------------------------------------------------------------

    def filter(
        self,
        tensors: GaussianTensors,
        lod_index: LodIndex,
        camera_position: Tensor,
    ) -> GaussianTensors:
        """Select the appropriate LOD tier and return filtered tensors.

        Args:
            camera_position: [3] float32 tensor on the same device.
        """
        if lod_index.cell_centers is not None:
            return self._filter_octree(tensors, lod_index, camera_position)
        return self._filter_centroid(tensors, lod_index, camera_position)

    def _filter_centroid(
        self,
        tensors: GaussianTensors,
        lod_index: LodIndex,
        camera_position: Tensor,
    ) -> GaussianTensors:
        cx, cy, cz = lod_index.centroid
        px, py, pz = (
            camera_position[0].item(),
            camera_position[1].item(),
            camera_position[2].item(),
        )
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

    def _filter_octree(
        self,
        tensors: GaussianTensors,
        lod_index: LodIndex,
        camera_position: Tensor,
    ) -> GaussianTensors:
        assert lod_index.cell_centers is not None  # noqa: S101
        assert lod_index.cell_ranges is not None  # noqa: S101
        assert lod_index.cell_tier_counts is not None  # noqa: S101

        device = lod_index.cell_centers.device
        num_cells = lod_index.cell_centers.shape[0]

        # 1. Camera-to-cell distances (camera_position is already a GPU tensor)
        dists = torch.norm(
            lod_index.cell_centers - camera_position.unsqueeze(0), dim=1
        )  # [C]

        # 2. Tier selection per cell (tier_max_distances_t is pre-cached)
        max_d = lod_index.tier_max_distances_t  # [T]
        tier_idx = (
            (dists.unsqueeze(1) <= max_d.unsqueeze(0)).to(torch.int64).argmax(dim=1)
        )  # [C]
        exceeds_all = dists > max_d[-1]
        tier_idx[exceeds_all] = len(lod_index.tier_max_distances) - 1

        # 3. Per-cell selected counts
        selected_counts = lod_index.cell_tier_counts[
            torch.arange(num_cells, device=device), tier_idx
        ]  # [C]

        # 4. Build flat index tensor
        starts = lod_index.cell_ranges[:, 0]  # [C]
        max_count = selected_counts.max()  # scalar tensor, stays on GPU

        offsets = (
            torch.arange(max_count.item(), device=device)
            .unsqueeze(0)
            .expand(num_cells, -1)
        )
        abs_indices = starts.unsqueeze(1) + offsets  # [C, max_count]
        mask = offsets < selected_counts.unsqueeze(1)  # [C, max_count]
        selected_indices = abs_indices[mask]  # [total_selected]

        if selected_indices.shape[0] >= tensors.means.shape[0]:
            return tensors
        return tensors[selected_indices]
