"""CARLA world → splatsim scene (tile-local) frames for the carla_canvas backend.

carla_canvas hands the renderer CARLA poses (Unreal's left-handed frame: x
forward, y right, z up; columns of a pose's rotation are the body's forward,
right and up axes). splatsim renders in a right-handed tile-local frame. A
:class:`SceneFrame` maps one onto the other, either through the map's
geo-reference (:mod:`splatsim.geo_transform`) or
through an explicit matrix, and derives what splatsim's renderers take:

* cameras: a world-to-camera view matrix in gsplat's RDF convention;
* LiDARs: a sensor-to-world pose in FLU (gsplat's LiDAR frame);
* actors: a position and a ``wxyz`` quaternion of the body in FLU.

numpy only: no torch, no CUDA.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class SceneFrame:
    """CARLA world → tile-local.

    ``position`` maps a CARLA world point; ``rotation`` maps CARLA world
    directions (it carries the handedness flip, so its determinant is -1).
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
        """An explicit 4x4 CARLA-world → tile-local matrix (rotation, a uniform
        scale and the y flip a right-handed scene needs). Points get the scale;
        orientations only the rotation."""
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (4, 4):
            raise ValueError("the CARLA → scene matrix must be 4x4")
        linear, offset = m[:3, :3], m[:3, 3]
        scale = abs(float(np.linalg.det(linear))) ** (1.0 / 3.0)
        rotation = linear / scale if scale > 0.0 else linear
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise ValueError(
                "the CARLA → scene matrix must be a rotation (and flip) times a "
                "uniform scale"
            )
        return SceneFrame(lambda p: linear @ p + offset, rotation, scale)

    @staticmethod
    def from_geo(geo_transform, at: Array | None = None) -> SceneFrame:
        """Through a :class:`splatsim.geo_transform.GeoTransform` (the map's
        OpenDRIVE geo-reference and the scene's ECEF anchor).

        Orientations follow the position map: its Jacobian at ``at`` (a CARLA
        point, default the map origin), orthonormalized. CARLA's x is the
        MGRS grid's east, which is not true east (the grid convergence, up to
        ~1.7 degrees at a UTM zone's edge), so ENU-based
        ``carla_rotation_to_tile_local`` would turn poses against positions."""

        def position(p: Array) -> Array:
            return np.asarray(
                geo_transform.carla_position_to_tile_local(*p), dtype=np.float64
            )

        return SceneFrame(position, local_rotation(position, at))

    # --- poses -------------------------------------------------------------------

    def pose(self, carla_pose: Array) -> tuple[Array, Array]:
        """(position, axes) of a 4x4 CARLA pose in tile-local: axes' columns are
        the body's forward, right and up directions."""
        m = np.asarray(carla_pose, dtype=np.float64)
        return np.asarray(
            self._position(m[:3, 3]), dtype=np.float64
        ), self.rotation @ m[:3, :3]

    def viewmat(self, carla_pose: Array) -> Array:
        """4x4 world-to-camera (gsplat RDF: x right, y down, z forward) of a
        CARLA camera pose (looking along its x)."""
        position, axes = self.pose(carla_pose)
        rdf = np.column_stack([axes[:, 1], -axes[:, 2], axes[:, 0]])
        view = np.eye(4)
        view[:3, :3] = rdf.T
        view[:3, 3] = -(rdf.T @ position)
        return view

    def flu_pose(self, carla_pose: Array) -> Array:
        """4x4 body-to-world in FLU (x forward, y left, z up): a LiDAR's
        sensor-to-world, an actor's pose. A proper rotation."""
        position, axes = self.pose(carla_pose)
        out = np.eye(4)
        out[:3, :3] = np.column_stack([axes[:, 0], -axes[:, 1], axes[:, 2]])
        out[:3, 3] = position
        return out


def local_rotation(
    position: Callable[[Array], Array], at: Array | None = None, step: float = 1.0
) -> Array:
    """The rotation (with its reflection: CARLA is left-handed) closest to the
    Jacobian of a position map at ``at``: central differences of ``step``
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
    """``(w, x, y, z)`` of a proper rotation matrix."""
    r = np.asarray(rotation, dtype=np.float64)
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = (
            0.25 * s,
            (r[2, 1] - r[1, 2]) / s,
            (r[0, 2] - r[2, 0]) / s,
            (r[1, 0] - r[0, 1]) / s,
        )
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = (
            (r[2, 1] - r[1, 2]) / s,
            0.25 * s,
            (r[0, 1] + r[1, 0]) / s,
            (r[0, 2] + r[2, 0]) / s,
        )
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = (
            (r[0, 2] - r[2, 0]) / s,
            (r[0, 1] + r[1, 0]) / s,
            0.25 * s,
            (r[1, 2] + r[2, 1]) / s,
        )
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = (
            (r[1, 0] - r[0, 1]) / s,
            (r[0, 2] + r[2, 0]) / s,
            (r[1, 2] + r[2, 1]) / s,
            0.25 * s,
        )
    w, x, y, z = q
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
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


def carla_rays_to_panorama(
    elevations_deg: Array, channel: Array, azimuths_deg: Array, n_columns: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """(row, column) of each CARLA LiDAR ray in a panorama whose rows are the
    lasers' elevations in CARLA's order (``elevations_deg``, descending: row i is
    laser i). CARLA's azimuth is a yaw (right positive); FLU's is to the left."""
    rows = np.asarray(channel, dtype=np.int64)
    if len(elevations_deg) and np.any(np.diff(np.asarray(elevations_deg)) >= 0):
        raise ValueError("the lasers' elevations must be strictly descending")
    return rows, column_of(-np.radians(azimuths_deg), n_columns)


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
