"""CARLA world → splatsim scene (tile-local) frames for the carla_canvas backend.

The renderer works in carla_canvas's ``right_handed`` coordinate frame
(CARLA's with y negated, bodies x forward / y left / z up): carla_canvas
converts CARLA's poses before handing them over. splatsim renders in a
right-handed tile-local frame. A :class:`SceneFrame` maps one onto the other, either through the map's
geo-reference (:mod:`splatsim.geo_transform`) or
through an explicit matrix, and derives what splatsim's renderers take:

* cameras: a world-to-camera view matrix in gsplat's RDF convention;
* LiDARs: a sensor-to-world pose in FLU (gsplat's LiDAR frame);
* actors: a position and a ``wxyz`` quaternion of the body in FLU.

numpy only: no torch, no CUDA. The CARLA-side conventions (sensor axes,
OpenCV camera axes, quaternions) are carla_canvas's
(:mod:`carla_canvas.geometry`); only the CARLA → tile-local map is splatsim's.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from carla_canvas.geometry import FLIP_Y, quaternion_xyzw, view_matrix_cv
from numpy.typing import NDArray

Array = NDArray[np.float64]


class SceneFrame:
    """carla_canvas's ``right_handed`` frame (CARLA's with y negated: the
    OpenDRIVE frame; bodies x forward, y left, z up) → tile-local.

    The renderer declares ``coordinate_frame = "right_handed"``, so poses come
    from carla_canvas already right-handed and this map is a proper one:
    ``position`` maps a point, ``rotation`` (determinant +1) directions.
    """

    def __init__(
        self, position: Callable[[Array], Array], rotation: Array, scale: float = 1.0
    ) -> None:
        self._position = position
        self.rotation = np.asarray(rotation, dtype=np.float64)
        #: scene units per CARLA metre (a ``matrix`` with a uniform scale)
        self.scale = float(scale)

    @staticmethod
    def from_matrix(matrix: Array) -> SceneFrame:
        """An explicit 4x4 **CARLA-world** → tile-local matrix, as in the
        configuration (rotation, a uniform scale and the y flip a right-handed
        scene needs). Points get the scale; orientations only the rotation."""
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (4, 4):
            raise ValueError("the CARLA → scene matrix must be 4x4")
        # right_handed → CARLA (y negated) → tile-local
        linear, offset = m[:3, :3] @ FLIP_Y, m[:3, 3]
        scale = abs(float(np.linalg.det(linear))) ** (1.0 / 3.0)
        rotation = linear / scale if scale > 0.0 else linear
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise ValueError(
                "the CARLA → scene matrix must be a rotation (and flip) times a "
                "uniform scale"
            )
        if np.linalg.det(rotation) < 0:
            raise ValueError(
                "the CARLA → scene matrix must flip handedness (CARLA is "
                "left-handed, the scene right-handed)"
            )
        return SceneFrame(lambda p: linear @ p + offset, rotation, scale)

    @staticmethod
    def from_geo(geo_transform, at: Array | None = None) -> SceneFrame:
        """Through a :class:`splatsim.geo_transform.GeoTransform` (the map's
        OpenDRIVE geo-reference and the scene's ECEF anchor).

        Orientations follow the position map: its Jacobian at ``at`` (an
        OpenDRIVE point, default the map origin), orthonormalized. The
        OpenDRIVE x is the MGRS grid's east, which is not true east (the grid
        convergence, up to ~1.7 degrees at a UTM zone's edge), so an ENU-based
        rotation would turn poses against positions."""

        def position(p: Array) -> Array:
            x, y, z = p
            return np.asarray(
                geo_transform.carla_position_to_tile_local(x, -y, z), dtype=np.float64
            )

        return SceneFrame(position, local_rotation(position, at))

    # --- poses -------------------------------------------------------------------

    def flu_pose(self, pose: Array) -> Array:
        """4x4 body-to-world in tile-local of a ``right_handed`` (FLU) pose:
        a LiDAR's sensor-to-world, an actor's pose. A proper rotation."""
        m = np.asarray(pose, dtype=np.float64)
        out = np.eye(4)
        out[:3, :3] = self.rotation @ m[:3, :3]
        out[:3, 3] = self._position(m[:3, 3])
        return out

    def viewmat(self, pose: Array) -> Array:
        """4x4 world-to-camera (gsplat RDF: x right, y down, z forward) of a
        ``right_handed`` camera pose (looking along its x)."""
        return view_matrix_cv(self.flu_pose(pose), "right_handed")


def local_rotation(
    position: Callable[[Array], Array], at: Array | None = None, step: float = 1.0
) -> Array:
    """The rotation closest to the Jacobian of a position map at ``at``: central differences of ``step``
    metres, orthonormalized by the polar decomposition."""
    p0 = np.zeros(3) if at is None else np.asarray(at, dtype=np.float64)
    jacobian = np.column_stack(
        [
            (position(p0 + step * e) - position(p0 - step * e)) / (2.0 * step)
            for e in np.eye(3)
        ]
    )
    u, _, vt = np.linalg.svd(jacobian)
    return u @ vt


def quaternion_wxyz(rotation: Array) -> tuple[float, float, float, float]:
    """``(w, x, y, z)`` of a proper rotation matrix (splatsim's order)."""
    x, y, z, w = quaternion_xyzw(rotation)
    return w, x, y, z


# --- LiDAR panorama ----------------------------------------------------------------


def panorama_columns(n_columns: int) -> Array:
    """The azimuths (radians, FLU: atan2(y, x)) of splatsim's panorama columns:
    +π → -π, as :func:`splatsim.lidar_renderer._sensor_column_azimuths`."""
    return np.linspace(
        math.pi - 1e-4, -math.pi + 1e-4, int(n_columns), dtype=np.float64
    )


def column_of(azimuth_flu: Array, n_columns: int) -> NDArray[np.int64]:
    """The nearest panorama column of FLU azimuths (radians), wrapping around."""
    first, step = math.pi - 1e-4, (2.0 * math.pi - 2e-4) / (int(n_columns) - 1)
    col = np.rint((first - np.asarray(azimuth_flu, dtype=np.float64)) / step).astype(
        np.int64
    )
    return np.mod(col, int(n_columns))


def rays_to_panorama(
    elevations_deg: Array, channel: Array, azimuths_deg: Array, n_columns: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """(row, column) of each LiDAR ray in a panorama whose rows are the lasers'
    elevations in CARLA's order (``elevations_deg``, descending: row i is laser
    i), from the rays' FLU azimuths (degrees, left positive:
    ``LidarView.body_azimuths()`` in the ``right_handed`` frame)."""
    rows = np.asarray(channel, dtype=np.int64)
    if len(elevations_deg) and np.any(np.diff(np.asarray(elevations_deg)) >= 0):
        raise ValueError("the lasers' elevations must be strictly descending")
    return rows, column_of(np.radians(azimuths_deg), n_columns)


def default_n_columns(
    points_per_second: float,
    channels: int,
    rotation_frequency: float,
    horizontal_fov: float = 360.0,
) -> int:
    """Panorama columns over 360°: CARLA's ray spacing per laser (a revolution's
    points over horizontal_fov; at least 360, at most 8192), so each ray has
    its own column."""
    per_turn = points_per_second / max(channels, 1) / max(rotation_frequency, 1e-6)
    per_turn *= 360.0 / min(max(horizontal_fov, 1e-6), 360.0)
    return int(min(max(round(per_turn), 360), 8192))
