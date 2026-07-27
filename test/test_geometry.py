"""Unit tests for the shared geometry primitives in ``splatsim._geometry``.

These pin the round-trip identities (quat↔matrix), the SLERP endpoints /
shortest-path / clamping behaviour, and the wxyz/xyzw order convention, so the
consolidation from PR #71's follow-up (issue #72) cannot silently drift.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from splatsim._geometry import lerp, mat4, quat_to_matrix, rpy_deg_to_matrix, slerp


def _rand_unit_quat(rng: np.random.Generator) -> np.ndarray:
    """Random unit quaternion in wxyz order."""
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def test_identity_quaternion_is_identity_matrix() -> None:
    np.testing.assert_allclose(quat_to_matrix((1.0, 0.0, 0.0, 0.0)), np.eye(3))


def test_quat_to_matrix_is_orthonormal_and_proper() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = quat_to_matrix(_rand_unit_quat(rng))
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(r), 1.0)


def test_quat_to_matrix_normalizes_input() -> None:
    # A non-unit quaternion must give the same rotation as its normalized form.
    q = np.array([2.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(quat_to_matrix(q), np.eye(3), atol=1e-12)


def test_quat_to_matrix_zero_norm_raises() -> None:
    with pytest.raises(ValueError):
        quat_to_matrix((0.0, 0.0, 0.0, 0.0))


def test_quat_order_wxyz_vs_xyzw_agree() -> None:
    # The same physical rotation expressed in the two component orders must
    # produce the same matrix.
    rng = np.random.default_rng(1)
    for _ in range(50):
        w, x, y, z = _rand_unit_quat(rng)
        m_wxyz = quat_to_matrix((w, x, y, z), order="wxyz")
        m_xyzw = quat_to_matrix((x, y, z, w), order="xyzw")
        np.testing.assert_allclose(m_wxyz, m_xyzw, atol=1e-12)


def test_quat_to_matrix_matches_90deg_z_rotation() -> None:
    # 90° about +z (wxyz): w = cos(45°), z = sin(45°).
    s = math.sqrt(0.5)
    r = quat_to_matrix((s, 0.0, 0.0, s))
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(r, expected, atol=1e-12)


def test_invalid_order_raises() -> None:
    with pytest.raises(ValueError):
        quat_to_matrix((1.0, 0.0, 0.0, 0.0), order="xzyw")


def test_mat4_assembles_rotation_and_translation() -> None:
    r = quat_to_matrix((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))
    t = (1.0, 2.0, 3.0)
    m = mat4(r, t)
    assert m.shape == (4, 4)
    np.testing.assert_allclose(m[:3, :3], r)
    np.testing.assert_allclose(m[:3, 3], t)
    np.testing.assert_allclose(m[3], (0.0, 0.0, 0.0, 1.0))


def test_rpy_zero_is_identity() -> None:
    np.testing.assert_allclose(
        rpy_deg_to_matrix((0.0, 0.0, 0.0)), np.eye(3), atol=1e-12
    )


def test_rpy_yaw_90_matches_quaternion() -> None:
    # Yaw 90° about +z must match the equivalent quaternion rotation.
    s = math.sqrt(0.5)
    np.testing.assert_allclose(
        rpy_deg_to_matrix((0.0, 0.0, 90.0)),
        quat_to_matrix((s, 0.0, 0.0, s)),
        atol=1e-12,
    )


def test_slerp_endpoints() -> None:
    rng = np.random.default_rng(2)
    q0 = _rand_unit_quat(rng)
    q1 = _rand_unit_quat(rng)
    if np.dot(q0, q1) < 0.0:  # slerp takes the shorter arc (sign-flipped)
        q1 = -q1
    np.testing.assert_allclose(slerp(q0, q1, 0.0), q0, atol=1e-12)
    np.testing.assert_allclose(slerp(q0, q1, 1.0), q1, atol=1e-12)


def test_slerp_result_is_unit() -> None:
    rng = np.random.default_rng(3)
    for _ in range(50):
        q = slerp(_rand_unit_quat(rng), _rand_unit_quat(rng), rng.uniform(0, 1))
        assert np.isclose(np.linalg.norm(q), 1.0)


def test_slerp_takes_shortest_path() -> None:
    # q and -q are the same rotation; slerp must not swing the long way round.
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = -q0
    mid = slerp(q0, q1, 0.5)
    # Halfway between q0 and its antipode stays at q0 (shortest arc has 0 angle).
    np.testing.assert_allclose(np.abs(mid), np.abs(q0), atol=1e-9)


def test_slerp_near_identical_is_stable() -> None:
    # Near-coincident inputs must stay unit and interpolate toward q1.
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([1.0, 1e-5, 0.0, 0.0])
    q1 = q1 / np.linalg.norm(q1)
    mid = slerp(q0, q1, 0.5)
    assert np.isclose(np.linalg.norm(mid), 1.0)
    assert mid[1] > 0.0  # interpolated toward q1


def test_slerp_xyzw_order_matches_wxyz() -> None:
    # The same interpolation done in xyzw must equal the wxyz result reordered.
    rng = np.random.default_rng(4)
    q0 = _rand_unit_quat(rng)
    q1 = _rand_unit_quat(rng)
    t = 0.37
    wxyz = slerp(q0, q1, t, order="wxyz")
    xyzw = slerp(q0[[1, 2, 3, 0]], q1[[1, 2, 3, 0]], t, order="xyzw")
    np.testing.assert_allclose(xyzw, wxyz[[1, 2, 3, 0]], atol=1e-12)


def test_slerp_matches_matrix_halfway_rotation() -> None:
    # SLERP halfway of a 90° z-rotation is a 45° z-rotation.
    s = math.sqrt(0.5)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([s, 0.0, 0.0, s])
    mid = slerp(q0, q1, 0.5)
    ang = math.radians(45.0)
    expected = np.array([math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)])
    np.testing.assert_allclose(mid, expected, atol=1e-9)


def test_lerp_endpoints_and_midpoint() -> None:
    p0 = (0.0, 0.0, 0.0)
    p1 = (2.0, 4.0, 6.0)
    np.testing.assert_allclose(lerp(p0, p1, 0.0), p0)
    np.testing.assert_allclose(lerp(p0, p1, 1.0), p1)
    np.testing.assert_allclose(lerp(p0, p1, 0.5), (1.0, 2.0, 3.0))
