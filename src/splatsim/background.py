from __future__ import annotations

import importlib as _importlib
from pathlib import Path

import torch
from torch import Tensor

from splatsim._conversions import GaussianTensors, cloud_to_tensors

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
        cloud = _merge_tileset(tiles)

        tensors = cloud_to_tensors(cloud, device, use_sh=use_sh)

        # ECEF coordinates are huge (~millions of meters).
        # Re-center to local origin for numerical stability.
        self._origin = tensors.means.mean(dim=0).clone()
        tensors.means = tensors.means - self._origin

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
