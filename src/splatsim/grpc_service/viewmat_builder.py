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
      the camera-to-world rotation **already in gsplat RDF convention**
      (X=right, Y=down, Z=forward).  The client is responsible for
      applying the RDF remapping before sending.

    The function inverts this to produce a world-to-camera view matrix.
    """
    q = torch.tensor(rotation_wxyz, device=device, dtype=torch.float32)
    R_c2w = quat_to_rotation_matrix(q)  # [3, 3], already in RDF convention

    # Invert camera-to-world → world-to-camera
    R_w2c = R_c2w.T
    pos = torch.tensor(position, device=device, dtype=torch.float32)
    t_w2c = -(R_w2c @ pos)

    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    viewmat[:3, :3] = R_w2c
    viewmat[:3, 3] = t_w2c
    return viewmat


def build_base_to_world_from_pose(
    position: tuple[float, float, float],
    rotation_wxyz: tuple[float, float, float, float],
    device: torch.device,
) -> Tensor:
    """Build a 4x4 base_link→world transform from a tile-local base pose.

    Unlike :func:`build_viewmat_from_pose` (camera path, which *inverts* the
    pose into a world-to-camera view matrix), the LiDAR renderer consumes the
    base pose directly: it composes ``sensor_to_world = base_to_world @ s2b``
    internally. So no inversion happens here.

    - *position* is the base_link origin ``(x, y, z)`` in tile-local coords.
    - *rotation_wxyz* is a ``(w, x, y, z)`` quaternion for the base_link→world
      rotation. base_link follows the ROS convention (X=forward, Y=left,
      Z=up), matching the frame the sensor→base extrinsic is expressed in.
    """
    q = torch.tensor(rotation_wxyz, device=device, dtype=torch.float32)
    R_b2w = quat_to_rotation_matrix(q)  # [3, 3]

    base_to_world = torch.eye(4, device=device, dtype=torch.float32)
    base_to_world[:3, :3] = R_b2w
    base_to_world[:3, 3] = torch.tensor(position, device=device, dtype=torch.float32)
    return base_to_world


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
