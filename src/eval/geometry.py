"""Small numpy geometry helpers shared across the LiDAR evaluation.

These are deliberately dependency-light (numpy + the shared
:mod:`splatsim._geometry` primitives) so the metric modules can import them
without pulling in torch, t4-devkit or rerun.
"""

from __future__ import annotations

import numpy as np

from splatsim._geometry import mat4, quat_to_matrix, slerp


def pose_to_matrix(translation, rotation) -> np.ndarray:
    """4x4 rigid transform from a T4 record's translation + pyquaternion rotation.

    ``rotation`` is a ``pyquaternion.Quaternion`` (as carried by ``EgoPose`` /
    ``CalibratedSensor``), exposing a 3x3 ``rotation_matrix``.
    """
    return mat4(np.asarray(rotation.rotation_matrix, dtype=np.float64), translation)


def transform(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform ``t`` to an (N, 3) point array."""
    return pts @ t[:3, :3].T + t[:3, 3]


def interp_ego_map(
    ts_us: np.ndarray, trans: np.ndarray, quats: list, t_us: float
) -> np.ndarray:
    """Interpolated ego(base)->map 4x4 pose at unix-microsecond time ``t_us``.

    Translation is linearly interpolated; rotation is SLERP'd (via the shared
    :func:`splatsim._geometry.slerp`) between the two bracketing ``ego_pose``
    records. ``quats`` are the records' ``pyquaternion.Quaternion`` rotations,
    read here as ``(w, x, y, z)`` via ``.elements``. Queries outside the
    recorded span clamp to the nearest endpoint. Used to reconstruct the
    sweep-end pose that drives the rolling-shutter render.
    """
    if t_us <= ts_us[0]:
        i0 = i1 = 0
        a = 0.0
    elif t_us >= ts_us[-1]:
        i0 = i1 = ts_us.shape[0] - 1
        a = 0.0
    else:
        i1 = int(np.searchsorted(ts_us, t_us))
        i0 = i1 - 1
        a = float((t_us - ts_us[i0]) / (ts_us[i1] - ts_us[i0]))
    pos = trans[i0] * (1.0 - a) + trans[i1] * a
    q = slerp(quats[i0].elements, quats[i1].elements, a)  # wxyz
    return mat4(quat_to_matrix(q, order="wxyz"), pos)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Rigid (no-scale) transform ``T`` minimizing ``|T*src - dst|`` (Kabsch).

    Args:
        src: (N, 3) source points.
        dst: (N, 3) target points, paired row-wise with ``src``.

    Returns:
        ``(T, rmse)`` where ``T`` is a 4x4 transform mapping ``src`` onto
        ``dst`` and ``rmse`` is the residual RMS distance after alignment.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    s_c = src - mu_s
    d_c = dst - mu_d
    cov = d_c.T @ s_c / src.shape[0]
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    s = np.diag([1.0, 1.0, d])
    r = u @ s @ vt
    t = mu_d - r @ mu_s
    out = mat4(r, t)
    aligned = transform(out, src)
    rmse = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))
    return out, rmse


def subsample(xyz: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly subsample rows of ``xyz`` down to ``max_points`` (no-op if fewer)."""
    n = xyz.shape[0]
    if max_points <= 0 or n <= max_points:
        return xyz
    idx = rng.choice(n, size=max_points, replace=False)
    return xyz[idx]
