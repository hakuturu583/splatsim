from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    import spz

from splatsim._conversions import (
    GaussianTensors,
    apply_rigid_transform,
    cloud_to_tensors,
)


class RigidBody:
    """A rigid Gaussian object that can be positioned and rotated in the scene."""

    def __init__(
        self,
        source: str | Path,
        *,
        device: torch.device = torch.device("cuda"),
        use_sh: bool = False,
    ) -> None:
        cloud = _load_cloud(source)
        self._base_tensors = cloud_to_tensors(cloud, device, use_sh=use_sh)
        self._device = device
        self.position = torch.zeros(3, device=device, dtype=torch.float32)
        self.rotation = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32
        )  # identity quaternion (wxyz)

    def set_pose(
        self,
        position: tuple[float, float, float] | Tensor,
        rotation: tuple[float, float, float, float] | Tensor | None = None,
    ) -> None:
        """Set the world-space pose of this rigid body.

        Args:
            position: (x, y, z) translation.
            rotation: (w, x, y, z) quaternion. If None, keeps current rotation.
        """
        if isinstance(position, tuple):
            self.position = torch.tensor(
                position, device=self._device, dtype=torch.float32
            )
        else:
            self.position = position.to(device=self._device, dtype=torch.float32)

        if rotation is not None:
            if isinstance(rotation, tuple):
                self.rotation = torch.tensor(
                    rotation, device=self._device, dtype=torch.float32
                )
            else:
                self.rotation = rotation.to(device=self._device, dtype=torch.float32)

    @property
    def tensors(self) -> GaussianTensors:
        """Return transformed tensors with current pose applied."""
        return apply_rigid_transform(self._base_tensors, self.position, self.rotation)

    @property
    def num_gaussians(self) -> int:
        return self._base_tensors.means.shape[0]


def _load_cloud(
    source: str | Path,
) -> spz.GaussianCloud:  # ty: ignore[unresolved-attribute]
    """Load a GaussianCloud from file, auto-detecting format by extension."""
    import importlib

    _3dgs_io = importlib.import_module("3dgs_io")
    load_gltf = _3dgs_io.load_gltf
    load_ply = _3dgs_io.load_ply
    load_spz = _3dgs_io.load_spz

    path = Path(source)
    suffix = path.suffix.lower()

    if suffix == ".spz":
        return load_spz(str(path))
    elif suffix in (".glb", ".gltf"):
        return load_gltf(str(path))
    elif suffix == ".ply":
        return load_ply(str(path))
    else:
        msg = f"Unsupported file format: {suffix}. Use .spz, .glb, .gltf, or .ply."
        raise ValueError(msg)
