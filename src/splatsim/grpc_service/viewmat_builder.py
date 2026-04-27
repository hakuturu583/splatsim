"""Build gsplat-compatible view / intrinsic matrices from camera pose."""

from __future__ import annotations

import torch
from torch import Tensor

from splatsim._conversions import quat_to_rotation_matrix


def build_viewmat_from_pose(
    position: tuple[float, float, float],
    rotation_wxyz: tuple[float, float, float, float],
    device: torch.device,
) -> Tensor:
    """Build a 4x4 world-to-camera view matrix from a tile-local pose.

    The pose is in re-centered tile-local coordinates where:

    - *position* is ``(x, y, z)``
    - *rotation_wxyz* is a ``(w, x, y, z)`` quaternion representing
      camera orientation.  The corresponding rotation matrix's columns
      are ``[forward(X), right(Y), up(Z)]`` in tile-local world
      coordinates — the same convention as
      ``SplatSimCameraSensor._compute_viewmat``.

    The function remaps to gsplat RDF convention and inverts to produce
    a world-to-camera view matrix.
    """
    q = torch.tensor(rotation_wxyz, device=device, dtype=torch.float32)
    R_c2w = quat_to_rotation_matrix(q)  # [3, 3], columns = [fwd, right, up]

    # Remap: camera (X=fwd, Y=right, Z=up) → gsplat (X=right, Y=down, Z=fwd)
    R_rdf = torch.stack([R_c2w[:, 1], -R_c2w[:, 2], R_c2w[:, 0]], dim=1)

    # Invert camera-to-world → world-to-camera
    R_w2c = R_rdf.T
    pos = torch.tensor(position, device=device, dtype=torch.float32)
    t_w2c = -(R_w2c @ pos)

    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    viewmat[:3, :3] = R_w2c
    viewmat[:3, 3] = t_w2c
    return viewmat


def build_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: torch.device,
) -> Tensor:
    """Build a 3x3 intrinsic matrix K from pinhole camera parameters."""
    return torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
