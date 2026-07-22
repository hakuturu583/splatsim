from __future__ import annotations

import importlib as _importlib
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from splatsim._conversions import GaussianTensors, cloud_to_tensors, quat_multiply
from splatsim.lod import LodIndex, LodManager

_3dgs_io = _importlib.import_module("3dgs_io")
_load_tileset = _3dgs_io.load_tileset
_merge_tileset = _3dgs_io.merge_tileset


class Background:
    """Loads a 3D Tileset or a v2 scene USDZ as GPU-ready Gaussian tensors."""

    def __init__(
        self,
        source_path: str | Path,
        *,
        device: torch.device = torch.device("cuda"),
        use_sh: bool = False,
        max_tiles: int | None = None,
        lod_manager: LodManager | None = None,
    ) -> None:
        path = Path(source_path)
        if path.suffix.lower() == ".usdz":
            # 3dgs_io scene USDZ bundles scene.json + chunks/*.spz; load each
            # SPZ chunk (already baked in the ENU world frame) as a tensor
            # chunk. The scene's ecef_anchor is the ENU world→ECEF transform.
            from splatsim._usdz import load_spz_scene

            tensors, anchor = load_spz_scene(path, device, use_sh=use_sh)
            self._ecef_rotation = anchor[:3, :3].copy()
            self._ecef_translation = anchor[:3, 3].copy()
        else:
            tiles = _load_tileset(str(path), max_tiles=max_tiles)

            # Extract the root tile's ECEF transform for GeoReference.
            # 3D Tiles stores column-major; reshape then transpose to row-major.
            root_tf = np.array(tiles[0].transform, dtype=np.float64).reshape(4, 4).T
            self._ecef_rotation = root_tf[:3, :3].copy()
            self._ecef_translation = root_tf[:3, 3].copy()

            if len(tiles) == 1:
                # Single tile: use raw tile-local cloud directly (already RUB, Y=up).
                # Avoids the ECEF rotation that merge_tileset would apply.
                cloud = tiles[0].cloud
            else:
                # Multi-tile: merge into ECEF, then undo the root rotation
                # so that the result stays in the tile-local orientation.
                cloud = _merge_tileset(tiles)
                self._undo_ecef_rotation(cloud, device)
            tensors = cloud_to_tensors(cloud, device, use_sh=use_sh)

        # Re-center to the cloud's tile-local centroid for numerical
        # stability. Despite living *near* the GeoReference origin in some
        # tilesets, this offset is the gaussians' own centroid in the
        # tile-local frame, not the ECEF origin (see ``ecef_translation``).
        self._tile_local_centroid = tensors.means.mean(dim=0).clone()
        tensors.means = tensors.means - self._tile_local_centroid

        # LOD: sort by importance and compute tier boundaries.
        self._lod_index: LodIndex | None = None
        if lod_manager is not None:
            tensors, self._lod_index = lod_manager.precompute(tensors)

        self._tensors = tensors

    def _undo_ecef_rotation(self, cloud: object, device: torch.device) -> None:
        """Undo the ECEF rotation in-place on a merged cloud.

        Converts from ECEF orientation back to tile-local RUB frame.
        """
        n = cloud.num_points  # ty: ignore[unresolved-attribute]
        positions = np.array(
            cloud.positions,  # ty: ignore[unresolved-attribute]
            dtype=np.float32,
        ).reshape(n, 3)

        # Clean rotation via SVD
        u, _, vt = np.linalg.svd(self._ecef_rotation)
        r_clean = u @ vt
        if np.linalg.det(r_clean) < 0:
            u[:, -1] *= -1
            r_clean = u @ vt
        r_inv = r_clean.T  # ECEF -> tile-local

        # Subtract ECEF centroid, rotate back, write positions
        centroid = positions.mean(axis=0)
        positions = (positions - centroid) @ r_inv.T
        cloud.positions = positions.reshape(-1)  # ty: ignore[unresolved-attribute]

        # Rotate quaternions: spz stores (x,y,z,w)
        quats_xyzw = np.array(
            cloud.rotations,  # ty: ignore[unresolved-attribute]
            dtype=np.float32,
        ).reshape(n, 4)
        quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]]
        quats_t = torch.from_numpy(quats_wxyz).to(device)

        r_inv_t = torch.tensor(r_inv, device=device, dtype=torch.float32)
        r_inv_quat = _rotation_matrix_to_quat(r_inv_t)
        rotated = quat_multiply(r_inv_quat, quats_t)

        # Back to spz (x,y,z,w) order
        rotated_np = rotated.cpu().numpy()[:, [1, 2, 3, 0]]
        cloud.rotations = rotated_np.reshape(-1)  # ty: ignore[unresolved-attribute]

    @property
    def lod_index(self) -> LodIndex | None:
        return self._lod_index

    @property
    def tile_local_centroid(self) -> Tensor:
        """Tile-local centroid subtracted from gaussian means for numerical stability.

        This is ``means.mean(dim=0)`` in the coordinate frame of the loaded
        gaussians: tile-local RUB for a 3D Tiles source, or Z-up ENU world for
        a v2 USDZ. It is **not** an ECEF translation. Renderers that consume
        world-frame poses must add it back to the translation column of their
        world-to-camera matrices to align with the re-centered cloud.

        For the ECEF translation of the root tile, see :attr:`ecef_translation`.
        """
        return self._tile_local_centroid

    @property
    def ecef_translation(self) -> np.ndarray:
        """ECEF translation of the root tile, in meters (3,) ``float64``.

        Read from the 3D Tiles root transform or the USDZ scene's
        ``world.ecef_anchor``; never applied to the gaussians.
        """
        return self._ecef_translation

    @property
    def ecef_rotation(self) -> np.ndarray:
        """ECEF rotation of the root tile, 3x3 ``float64``.

        Read from the 3D Tiles root transform or the USDZ scene's
        ``world.ecef_anchor``; never applied to USDZ gaussians. For multi-tile
        tilesets, the inverse is applied to the merged cloud before centering.
        """
        return self._ecef_rotation

    @property
    def tensors(self) -> GaussianTensors:
        return self._tensors

    @property
    def num_gaussians(self) -> int:
        return self._tensors.means.shape[0]


def _rotation_matrix_to_quat(r: Tensor) -> Tensor:
    """Convert a 3x3 rotation matrix to a (w,x,y,z) quaternion."""
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
