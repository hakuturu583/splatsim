"""Pose accumulation and interpolation for frame-rate rendering."""

from __future__ import annotations

import bisect
import math
import threading
from dataclasses import dataclass


@dataclass
class TimestampedPose:
    """A camera pose with an associated timestamp in nanoseconds."""

    time_ns: int
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]  # wxyz quaternion


class PoseBuffer:
    """Accumulates timestamped poses and interpolates at arbitrary times.

    Thread-safe via a :class:`threading.Lock`.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._max_size = max_size
        self._times: list[int] = []
        self._poses: list[TimestampedPose] = []
        self._lock = threading.Lock()

    def append(self, pose: TimestampedPose) -> None:
        """Add a new pose.  Poses must arrive in non-decreasing time order."""
        with self._lock:
            self._times.append(pose.time_ns)
            self._poses.append(pose)
            if len(self._times) > self._max_size:
                self._times.pop(0)
                self._poses.pop(0)

    def interpolate(self, time_ns: int) -> TimestampedPose | None:
        """Interpolate pose at *time_ns*.

        Returns ``None`` if *time_ns* is outside the buffered range.
        """
        with self._lock:
            if not self._times:
                return None

            idx = bisect.bisect_right(self._times, time_ns)

            # Exact match on the last element or beyond range
            if idx == 0:
                return None  # before all poses
            if idx >= len(self._times):
                # After last pose — only return if exactly at the last pose
                if self._times[-1] == time_ns:
                    return self._poses[-1]
                return None

            p0 = self._poses[idx - 1]
            p1 = self._poses[idx]

            if p0.time_ns == p1.time_ns:
                return p0

            t = (time_ns - p0.time_ns) / (p1.time_ns - p0.time_ns)
            pos = _lerp_position(p0.position, p1.position, t)
            rot = _slerp(p0.rotation, p1.rotation, t)
            return TimestampedPose(time_ns=time_ns, position=pos, rotation=rot)

    def trim_before(self, time_ns: int) -> None:
        """Remove poses older than *time_ns*, keeping one for interpolation."""
        with self._lock:
            idx = bisect.bisect_left(self._times, time_ns)
            # Keep at least one pose before time_ns
            keep_from = max(0, idx - 1)
            if keep_from > 0:
                del self._times[:keep_from]
                del self._poses[:keep_from]

    @property
    def latest_time_ns(self) -> int | None:
        with self._lock:
            return self._times[-1] if self._times else None

    @property
    def earliest_time_ns(self) -> int | None:
        with self._lock:
            return self._times[0] if self._times else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._times)


def _slerp(
    q0: tuple[float, float, float, float],
    q1: tuple[float, float, float, float],
    t: float,
) -> tuple[float, float, float, float]:
    """Spherical linear interpolation (wxyz convention, shortest path)."""
    dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]

    # Ensure shortest path
    if dot < 0.0:
        q1 = (-q1[0], -q1[1], -q1[2], -q1[3])
        dot = -dot

    # Clamp for numerical safety
    dot = min(dot, 1.0)

    if dot > 0.9995:
        # Very close — use linear interpolation and normalise
        w = q0[0] + t * (q1[0] - q0[0])
        x = q0[1] + t * (q1[1] - q0[1])
        y = q0[2] + t * (q1[2] - q0[2])
        z = q0[3] + t * (q1[3] - q0[3])
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        s0 = math.sin((1.0 - t) * theta) / sin_theta
        s1 = math.sin(t * theta) / sin_theta
        w = s0 * q0[0] + s1 * q1[0]
        x = s0 * q0[1] + s1 * q1[1]
        y = s0 * q0[2] + s1 * q1[2]
        z = s0 * q0[3] + s1 * q1[3]

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / norm, x / norm, y / norm, z / norm)


def _lerp_position(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    """Linear interpolation between two 3D positions."""
    return (
        p0[0] + t * (p1[0] - p0[0]),
        p0[1] + t * (p1[1] - p0[1]),
        p0[2] + t * (p1[2] - p0[2]),
    )
