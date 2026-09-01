from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    import spz

from splatsim._conversions import (
    MAX_NON_YAW_RAD,
    GaussianTensors,
    apply_rigid_transform,
    cloud_to_tensors,
    yaw_from_quat,
)
from splatsim.lod import LodIndex, LodManager


class RigidBody:
    """A rigid Gaussian object that can be positioned and rotated in the scene."""

    def __init__(
        self,
        source: str | Path,
        *,
        device: torch.device = torch.device("cuda"),
        use_sh: bool = False,
        lod_manager: LodManager | None = None,
        rotate_sh: bool = False,
    ) -> None:
        self._init_from_tensors(
            cloud_to_tensors(_load_cloud(source), device, use_sh=use_sh),
            device=device,
            lod_manager=lod_manager,
            rotate_sh=rotate_sh,
        )

    @classmethod
    def from_tensors(
        cls,
        base_tensors: GaussianTensors,
        *,
        device: torch.device,
        lod_manager: LodManager | None = None,
        rotate_sh: bool = False,
    ) -> RigidBody:
        """Build a rigid body from tensors that are already on the device.

        Used by :class:`~splatsim.actor_assets.ActorAssetLibrary`, whose
        Gaussians arrive from a scene bundle's asset bank rather than a
        standalone file — and whose base tensors are shared between every
        instance spawned from the same asset.
        """
        body = cls.__new__(cls)
        body._init_from_tensors(
            base_tensors, device=device, lod_manager=lod_manager, rotate_sh=rotate_sh
        )
        return body

    def _init_from_tensors(
        self,
        base_tensors: GaussianTensors,
        *,
        device: torch.device,
        lod_manager: LodManager | None,
        rotate_sh: bool,
    ) -> None:
        # LOD: sort base tensors by importance and store tier boundaries.
        self._lod_index: LodIndex | None = None
        if lod_manager is not None:
            base_tensors, self._lod_index = lod_manager.precompute(base_tensors)

        self._base_tensors = base_tensors
        self._device = device
        self._rotate_sh = rotate_sh
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
    def base_tensors(self) -> GaussianTensors:
        """Return the untransformed (local-space) base tensors."""
        return self._base_tensors

    @property
    def rotate_sh(self) -> bool:
        """Whether posing this body also re-expresses its colour SH in world.

        See :func:`splatsim._conversions.apply_rigid_transform`.
        """
        return self._rotate_sh

    @property
    def sh_rotation_tilt(self) -> float:
        """Radians this body's pose departs from a pure yaw.

        Colour SH is re-expressed by rotating about ``+Z`` only, which is exact
        for the yaw-only poses road vehicles have. A caller that poses a body
        far off the ground plane can read this to know the view-dependent
        colour is approximate; ``0.0`` when :attr:`rotate_sh` is off, since
        nothing is being approximated.
        """
        if not self._rotate_sh:
            return 0.0
        _yaw, tilt = yaw_from_quat(self.rotation)
        return float(tilt)

    @property
    def sh_rotation_is_exact(self) -> bool:
        """True while the current pose is within :data:`MAX_NON_YAW_RAD` of a yaw."""
        return self.sh_rotation_tilt <= MAX_NON_YAW_RAD

    @property
    def tensors(self) -> GaussianTensors:
        """Return transformed tensors with current pose applied."""
        return apply_rigid_transform(
            self._base_tensors,
            self.position,
            self.rotation,
            rotate_sh=self._rotate_sh,
        )

    @property
    def lod_index(self) -> LodIndex | None:
        return self._lod_index

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
