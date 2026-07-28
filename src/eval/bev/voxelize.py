"""Pure-torch hard voxelisation feeding the BEV encoder ONNX.

Turns a raw point cloud into the ``(voxels, num_points_per_voxel, coors)`` triple
the ONNX consumes -- the "hard voxelization" of mmdet3d / BEVFusion: points are
gridded, at most ``max_num_points`` are kept per occupied voxel, and each voxel's
feature is the raw stack of its points (the ONNX runs the mean-VFE internally).

Runs on CPU or CUDA (no custom op), so it is unit-testable without TensorRT.
"""

from __future__ import annotations

import torch

from .config import BEVConfig


def hard_voxelize(
    points: torch.Tensor, cfg: BEVConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Voxelise ``points`` (N, F) into ``(voxels, num_points, coors)``.

    Args:
        points: (N, F) tensor with ``F >= 3`` and the first three columns
            ``(x, y, z)``. ``F`` should equal ``cfg.num_point_features``.
        cfg: geometry / capacity configuration.

    Returns:
        ``voxels`` (M, max_num_points, F) float, zero-padded; ``num_points`` (M,)
        int32 = points per voxel (capped at ``max_num_points``); ``coors`` (M, 3)
        int32 voxel indices in ``cfg.coors_order`` (default ``(z, y, x)``).
        ``M`` is the occupied-voxel count, capped at ``cfg.max_voxels``.
    """
    K = cfg.max_num_points
    F = points.shape[1]
    gx, gy, gz = cfg.grid_size
    lo = points.new_tensor(cfg.point_cloud_range[:3])
    vsz = points.new_tensor(cfg.voxel_size)

    idx = torch.floor((points[:, :3] - lo) / vsz).long()  # (N, 3) x,y,z
    in_range = (
        (idx[:, 0] >= 0)
        & (idx[:, 0] < gx)
        & (idx[:, 1] >= 0)
        & (idx[:, 1] < gy)
        & (idx[:, 2] >= 0)
        & (idx[:, 2] < gz)
    )
    points = points[in_range]
    idx = idx[in_range]
    if points.shape[0] == 0:
        return (
            points.new_zeros(0, K, F),
            points.new_zeros(0, dtype=torch.int32),
            points.new_zeros(0, 3, dtype=torch.int32),
        )

    # Linear voxel id (x-major), then group points by occupied voxel.
    lin = (idx[:, 0] * gy + idx[:, 1]) * gz + idx[:, 2]
    uniq, inverse, counts = torch.unique(
        lin, return_inverse=True, return_counts=True, sorted=True
    )
    M = uniq.shape[0]

    # Rank of each point within its voxel (0-based), via a stable sort by voxel.
    order = torch.argsort(inverse, stable=True)
    group_start = torch.zeros(M, dtype=torch.long, device=points.device)
    group_start[1:] = torch.cumsum(counts, 0)[:-1]
    within = torch.empty_like(inverse)
    within[order] = (
        torch.arange(inverse.shape[0], device=points.device)
        - group_start[inverse[order]]
    )

    # Keep the first K points per voxel, and cap the number of voxels.
    keep = (within < K) & (inverse < cfg.max_voxels)
    vox_id = inverse[keep]
    slot = within[keep]
    M = min(M, cfg.max_voxels)

    voxels = points.new_zeros(M, K, F)
    voxels[vox_id, slot] = points[keep]
    num_points = counts[:M].clamp(max=K).to(torch.int32)

    uniq = uniq[:M]
    ux = uniq // (gy * gz)
    rem = uniq % (gy * gz)
    uy = rem // gz
    uz = rem % gz
    axes = {"xyz": (ux, uy, uz), "zyx": (uz, uy, ux)}
    if cfg.coors_order not in axes:
        raise ValueError(f"Unsupported coors_order {cfg.coors_order!r}")
    coors = torch.stack(axes[cfg.coors_order], dim=1).to(torch.int32)
    return voxels, num_points, coors
