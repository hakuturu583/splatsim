from __future__ import annotations

import math

import torch

from splatsim.grpc_service.viewmat_builder import (
    build_base_to_world_from_pose,
    build_viewmat_from_pose,
)


def test_base_to_world_is_not_inverted() -> None:
    """The LiDAR base pose is used directly (base→world), not inverted.

    Unlike :func:`build_viewmat_from_pose` (which produces world-to-camera),
    the LiDAR renderer composes ``sensor_to_world = base_to_world @ s2b``, so
    the translation column must hold the base position verbatim.
    """
    device = torch.device("cpu")
    pos = (1.0, 2.0, 3.0)
    quat = (1.0, 0.0, 0.0, 0.0)  # identity

    m = build_base_to_world_from_pose(pos, quat, device)

    assert torch.allclose(m[:3, :3], torch.eye(3))
    assert m[:3, 3].tolist() == [1.0, 2.0, 3.0]
    assert m[3].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_base_to_world_applies_rotation() -> None:
    """A 90° yaw rotates the base axes into world without touching position."""
    device = torch.device("cpu")
    half = math.radians(90.0) / 2.0
    quat = (math.cos(half), 0.0, 0.0, math.sin(half))  # +90° about world z

    m = build_base_to_world_from_pose((5.0, 0.0, 0.0), quat, device)

    # base +x maps to world +y under a +90° yaw.
    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(m[:3, :3], expected, atol=1e-6)
    assert m[:3, 3].tolist() == [5.0, 0.0, 0.0]


def test_base_to_world_differs_from_camera_viewmat() -> None:
    """base→world (LiDAR) is the inverse relationship of world→camera."""
    device = torch.device("cpu")
    pos = (2.0, -1.0, 4.0)
    quat = (1.0, 0.0, 0.0, 0.0)

    b2w = build_base_to_world_from_pose(pos, quat, device)
    viewmat = build_viewmat_from_pose(pos, quat, device)

    # Camera viewmat translation is -(R_w2c @ pos); base_to_world keeps pos.
    assert b2w[:3, 3].tolist() == list(pos)
    assert viewmat[:3, 3].tolist() == [-2.0, 1.0, -4.0]
