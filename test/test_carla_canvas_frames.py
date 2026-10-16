"""CARLA → splatsim frames of the carla_canvas backend (numpy only)."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("carla_canvas")

from carla_canvas.geometry import to_frame  # noqa: E402

from splatsim.carla_canvas.frames import (
    SceneFrame,
    column_of,
    default_n_columns,
    panorama_columns,
    quaternion_wxyz,
    rays_to_panorama,
)

# CARLA (x east, y south, z up) -> a right-handed scene sharing its origin (y north)
FLIP = np.diag([1.0, -1.0, 1.0, 1.0])


def carla_pose(location=(0.0, 0.0, 0.0), yaw_deg=0.0) -> np.ndarray:
    """A CARLA pose with only a yaw (CARLA: right positive), as carla_canvas
    hands it to a right_handed renderer."""
    return to_frame(_carla_pose(location, yaw_deg), "right_handed")


def _carla_pose(location=(0.0, 0.0, 0.0), yaw_deg=0.0) -> np.ndarray:
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


def test_geo_frame_turns_poses_with_the_positions() -> None:
    """CARLA x is the MGRS grid's east, off true east by the grid convergence:
    orientations follow the position map, not an ENU rotation."""
    convergence, k = math.radians(0.72), 1.00017  # Tokyo; the projection's scale

    class FakeGeo:
        r_enu_to_tile = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        c, s = math.cos(convergence), math.sin(convergence)
        grid_to_enu = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        def carla_position_to_tile_local(self, x, y, z):
            grid = np.array([x, -y, z])
            offset = np.array([100.0, 0.0, 0.0])
            return self.r_enu_to_tile @ (k * self.grid_to_enu @ grid) + offset

    geo = FakeGeo()
    frame = SceneFrame.from_geo(geo)
    carla = _carla_pose((1.0, 2.0, 3.0), yaw_deg=30.0)
    m = frame.flu_pose(to_frame(carla, "right_handed"))
    position, forward = m[:3, 3], m[:3, 0]
    np.testing.assert_allclose(
        position, geo.carla_position_to_tile_local(1.0, 2.0, 3.0)
    )
    # a point 10 m ahead of the pose (in CARLA) lies along its forward axis
    ahead = geo.carla_position_to_tile_local(*(carla[:3, 3] + 10.0 * carla[:3, 0]))
    direction = (ahead - position) / np.linalg.norm(ahead - position)
    np.testing.assert_allclose(forward, direction, atol=1e-9)
    np.testing.assert_allclose(
        frame.rotation, geo.r_enu_to_tile @ geo.grid_to_enu, atol=1e-9
    )
    assert np.isclose(np.linalg.det(m[:3, :3]), 1.0)


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
    # FLU azimuth -90 (CARLA yaw +90, right): three quarters along the sweep
    rows, c = rays_to_panorama(
        np.array([10.0, 0.0, -10.0]),
        np.array([0, 2, 1]),
        np.array([-90.0, 0.0, 180.0]),
        n,
    )
    assert rows.tolist() == [0, 2, 1]
    assert abs(cols[c[0]] - (-math.pi / 2)) < 2 * math.pi / n
    assert abs(cols[c[1]]) < 2 * math.pi / n
    assert c[2] in (0, n - 1)  # ±180 wraps
    with pytest.raises(ValueError):
        rays_to_panorama(np.array([0.0, 5.0]), np.array([0]), np.array([0.0]), n)


def test_default_columns() -> None:
    assert default_n_columns(56000, 32, 10) == 360  # 175 per turn: at least 360
    assert default_n_columns(1_310_720, 64, 10) == 2048
    assert default_n_columns(1e9, 1, 1) == 8192
    # 90° sweeps: a quarter turn's rays, so 4x the columns over 360°
    assert default_n_columns(1_310_720, 64, 10, 90.0) == 8192
    assert default_n_columns(327_680, 64, 10, 90.0) == 2048


def test_a_scaled_matrix_scales_points_not_orientations() -> None:
    scaled = SceneFrame.from_matrix(
        2.0 * np.asarray(FLIP, dtype=np.float64) + np.diag([0, 0, 0, -1.0])
    )
    plain = SceneFrame.from_matrix(FLIP)
    pose = carla_pose(yaw_deg=30.0)
    np.testing.assert_allclose(
        scaled.flu_pose(pose)[:3, :3], plain.flu_pose(pose)[:3, :3]
    )
    position = scaled.flu_pose(carla_pose(location=(1.0, 2.0, 3.0)))[:3, 3]
    np.testing.assert_allclose(position, [2.0, -4.0, 6.0])
    assert scaled.scale == pytest.approx(2.0) and plain.scale == 1.0
    with pytest.raises(ValueError):
        SceneFrame.from_matrix(np.diag([1.0, 2.0, 1.0, 1.0]))
