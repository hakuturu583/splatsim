"""CARLA → splatsim frames of the carla_canvas backend (numpy only)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from splatsim.carla_canvas.frames import (
    SceneFrame,
    carla_rays_to_panorama,
    column_of,
    default_n_columns,
    panorama_columns,
    quaternion_wxyz,
)

# CARLA (x east, y south, z up) -> a right-handed scene sharing its origin (y north)
FLIP = np.diag([1.0, -1.0, 1.0, 1.0])


def carla_pose(location=(0.0, 0.0, 0.0), yaw_deg=0.0) -> np.ndarray:
    """A 4x4 CARLA pose with only a yaw (CARLA: right positive)."""
    y = math.radians(yaw_deg)
    m = np.eye(4)
    m[:3, 0] = (math.cos(y), math.sin(y), 0.0)  # forward
    m[:3, 1] = (-math.sin(y), math.cos(y), 0.0)  # right
    m[:3, 3] = location
    return m


def test_camera_viewmat_is_rdf() -> None:
    frame = SceneFrame.from_matrix(FLIP)
    view = frame.viewmat(carla_pose((1.0, 2.0, 3.0)))
    ahead = view @ np.array([6.0, -2.0, 3.0, 1.0])  # 5 m forward (CARLA +x)
    np.testing.assert_allclose(ahead[:3], [0.0, 0.0, 5.0], atol=1e-12)
    right = view @ np.array([1.0, -3.0, 3.0, 1.0])  # CARLA y+1 (right) = scene y-1
    np.testing.assert_allclose(right[:3], [1.0, 0.0, 0.0], atol=1e-12)
    above = view @ np.array([1.0, -2.0, 4.0, 1.0])
    np.testing.assert_allclose(
        above[:3], [0.0, -1.0, 0.0], atol=1e-12
    )  # up is -y (RDF)


def test_flu_pose_is_a_proper_rotation() -> None:
    frame = SceneFrame.from_matrix(FLIP)
    m = frame.flu_pose(carla_pose((1.0, 2.0, 0.0), yaw_deg=90.0))
    r = m[:3, :3]
    assert np.isclose(np.linalg.det(r), 1.0)
    # CARLA yaw +90 faces south (CARLA +y), i.e. the scene's -y
    np.testing.assert_allclose(r[:, 0], [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(m[:3, 3], [1.0, -2.0, 0.0])
    # FLU's left (+y) is CARLA's -right: east after a right turn to the south
    np.testing.assert_allclose(r[:, 1], [1.0, 0.0, 0.0], atol=1e-12)


def test_geo_frame_uses_the_rotation_linearly() -> None:
    class FakeGeo:
        r_enu_to_tile = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        def carla_rotation_to_tile_local(self, r):
            return self.r_enu_to_tile @ np.diag([1.0, -1.0, 1.0]) @ r

        def carla_position_to_tile_local(self, x, y, z):
            return np.array([x + 100.0, y, z])

    frame = SceneFrame.from_geo(FakeGeo())
    pose = carla_pose((1.0, 2.0, 3.0), yaw_deg=30.0)
    position, axes = frame.pose(pose)
    np.testing.assert_allclose(position, [101.0, 2.0, 3.0])
    np.testing.assert_allclose(
        axes, FakeGeo().carla_rotation_to_tile_local(pose[:3, :3])
    )


@pytest.mark.parametrize("yaw", [0.0, 45.0, 135.0, -100.0, 179.0])
def test_quaternion_round_trip(yaw: float) -> None:
    r = SceneFrame.from_matrix(FLIP).flu_pose(carla_pose(yaw_deg=yaw))[:3, :3]
    w, x, y, z = quaternion_wxyz(r)
    back = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    np.testing.assert_allclose(back, r, atol=1e-12)
    assert w >= 0


def test_panorama_columns_and_carla_azimuths() -> None:
    n = 1024
    cols = panorama_columns(n)
    assert cols[0] > 3.14 and cols[-1] < -3.14  # +π -> -π, as splatsim
    np.testing.assert_array_equal(column_of(cols, n), np.arange(n))
    # CARLA yaw +90 (right) is FLU azimuth -90: three quarters along the sweep
    rows, c = carla_rays_to_panorama(
        np.array([10.0, 0.0, -10.0]),
        np.array([0, 2, 1]),
        np.array([90.0, 0.0, -180.0]),
        n,
    )
    assert rows.tolist() == [0, 2, 1]
    assert abs(cols[c[0]] - (-math.pi / 2)) < 2 * math.pi / n
    assert abs(cols[c[1]]) < 2 * math.pi / n
    assert c[2] in (0, n - 1)  # ±180 wraps
    with pytest.raises(ValueError):
        carla_rays_to_panorama(np.array([0.0, 5.0]), np.array([0]), np.array([0.0]), n)


def test_default_columns() -> None:
    assert default_n_columns(56000, 32, 10) == 360  # 175 per turn: at least 360
    assert default_n_columns(1_310_720, 64, 10) == 2048
    assert default_n_columns(1e9, 1, 1) == 8192
