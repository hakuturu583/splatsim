"""Shared geometry primitives: quaternion / rotation / pose-interpolation math.

Before this module the same handful of transforms was re-implemented across
``_usdz``, ``lidar_renderer``, ``grpc_service.pose_buffer``,
``carla_integration`` and ``eval/eval_lidar`` — each with its own quaternion
order and its own SLERP/LERP copy. They now all delegate here.

Quaternion order convention
---------------------------
Unless a function's ``order`` argument says otherwise, quaternions are
**(w, x, y, z)** ("wxyz") — the gsplat / ROS / ``spz`` convention and the
majority order across splatsim. The common exception is
``3dgs_io.parse_rig_trajectories``, which exposes rotations as **(x, y, z, w)**
("xyzw", the scipy / Eigen / glTF order); pass ``order="xyzw"`` there.

Every function operates on plain array-likes and returns ``float64`` numpy
arrays. The differentiable, torch-based rotation used inside the Gaussian
render path is deliberately kept separate as
:func:`splatsim._conversions.quat_to_rotation_matrix` (same wxyz convention).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

QuatOrder = str  # "wxyz" | "xyzw"
# Anything convertible to a small fixed-length float vector (tuple/list/ndarray).
Vec = Sequence[float] | np.ndarray


def _as_wxyz(q: Vec, order: QuatOrder) -> tuple[float, float, float, float]:
    """Return ``(w, x, y, z)`` from a quaternion given in ``order``."""
    a, b, c, d = (float(v) for v in q)
    if order == "wxyz":
        return a, b, c, d
    if order == "xyzw":
        return d, a, b, c
    raise ValueError(f"quaternion order must be 'wxyz' or 'xyzw', got {order!r}")


def quat_to_matrix(q: Vec, *, order: QuatOrder = "wxyz") -> np.ndarray:
    """Convert a quaternion to a 3x3 rotation matrix (float64).

    The quaternion is normalized first, so callers need not pre-normalize.
    ``order`` selects the input component order (see the module docstring);
    the default is ``"wxyz"``.
    """
    w, x, y, z = _as_wxyz(q, order)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("cannot build a rotation from a zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def mat4(r: np.ndarray, t: Vec) -> np.ndarray:
    """Assemble a 4x4 rigid transform from a 3x3 rotation and a translation."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = np.asarray(r, dtype=np.float64)
    m[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return m


def rpy_deg_to_matrix(rpy_deg: Vec) -> np.ndarray:
    """3x3 rotation from intrinsic roll/pitch/yaw in **degrees** (Z·Y·X order)."""
    roll, pitch, yaw = (math.radians(float(v)) for v in rpy_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def slerp(q0: Vec, q1: Vec, t: float) -> np.ndarray:
    """Shortest-path spherical linear interpolation between two unit quaternions.

    Both inputs are normalized first and the result is a unit quaternion in the
    **same component order** as the inputs (the SLERP math is order-agnostic, so
    this works for both wxyz and xyzw as long as ``q0`` and ``q1`` share one).
    Near-antipodal inputs fall back to a normalized LERP for numerical stability;
    ``t`` outside ``[0, 1]`` extrapolates.
    """
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:  # take the shorter arc
        b = -b
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        result = a + t * (b - a)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    s0 = math.sin((1.0 - t) * theta) / sin_theta
    s1 = math.sin(t * theta) / sin_theta
    return s0 * a + s1 * b


def lerp(p0: Vec, p1: Vec, t: float) -> np.ndarray:
    """Linear interpolation ``(1 - t) * p0 + t * p1`` (float64 array)."""
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    return a + t * (b - a)
