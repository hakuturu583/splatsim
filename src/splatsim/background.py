from __future__ import annotations

import importlib as _importlib
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from splatsim._conversions import GaussianTensors, cloud_to_tensors, quat_multiply

_3dgs_io = _importlib.import_module("3dgs_io")
_load_tileset = _3dgs_io.load_tileset
_merge_tileset = _3dgs_io.merge_tileset


class Background:
    """Loads a Cesium 3D Tileset and stores as GPU-ready Gaussian tensors."""

    def __init__(
        self,
        tileset_path: str | Path,
        *,
        device: torch.device = torch.device("cuda"),
        use_sh: bool = False,
        max_tiles: int | None = None,
    ) -> None:
        tiles = _load_tileset(str(tileset_path), max_tiles=max_tiles)

        # Extract root tile rotation (tile-local -> ECEF).
        # 3D Tiles transform is column-major flat [16], so reshape and transpose.
        root_transform = np.array(tiles[0].transform, dtype=np.float64).reshape(4, 4).T
        ecef_rotation = root_transform[:3, :3]  # tile-local -> ECEF

        cloud = _merge_tileset(tiles)
        tensors = cloud_to_tensors(cloud, device, use_sh=use_sh)

        # Re-center: subtract ECEF centroid for numerical stability.
        self._origin = tensors.means.mean(dim=0).clone()
        tensors.means = tensors.means - self._origin

        # Undo ECEF rotation so that the tile-local frame is restored
        # (Y=up, Z=back in RUB). Use SVD to get a clean rotation matrix.
        u, _, vt = np.linalg.svd(ecef_rotation)
        r_clean = u @ vt
        if np.linalg.det(r_clean) < 0:
            u[:, -1] *= -1
            r_clean = u @ vt
        # R maps tile-local -> ECEF, so R^T maps ECEF -> tile-local.
        r_inv = torch.tensor(r_clean.T, device=device, dtype=torch.float32)

        # Rotate positions back to tile-local frame
        tensors.means = tensors.means @ r_inv.T

        # Rotate quaternions: convert R^T to quaternion, compose with existing
        r_inv_quat = _rotation_matrix_to_quat(r_inv)
        tensors.quats = quat_multiply(r_inv_quat, tensors.quats)

        self._tensors = tensors

    @property
    def origin(self) -> Tensor:
        """The ECEF centroid that was subtracted (GeoReference origin)."""
        return self._origin

    @property
    def tensors(self) -> GaussianTensors:
        return self._tensors

    @property
    def num_gaussians(self) -> int:
        return self._tensors.means.shape[0]


def _rotation_matrix_to_quat(r: Tensor) -> Tensor:
    """Convert a 3x3 rotation matrix to a (w,x,y,z) quaternion."""
    # Shepperd's method
    trace = r[0, 0] + r[1, 1] + r[2, 2]

    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (r[2, 1] - r[1, 2]) * s
        y = (r[0, 2] - r[2, 0]) * s
        z = (r[1, 0] - r[0, 1]) * s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s

    return torch.stack([w, x, y, z])
