"""Frame-rate-based render scheduling."""

from __future__ import annotations


class FrameScheduler:
    """Determines the next rendering timestamp based on frame rate.

    Given a clock initial value and frame rate, computes a sequence of
    render times: ``t0``, ``t0 + 1/fps``, ``t0 + 2/fps``, ...
    """

    def __init__(self, clock_initial_ns: int, frame_rate: float) -> None:
        self._frame_period_ns = int(1_000_000_000 / frame_rate)
        self._next_render_ns = clock_initial_ns
        self._frame_count = 0

    @property
    def next_render_time_ns(self) -> int:
        """The timestamp at which the next frame should be rendered."""
        return self._next_render_ns

    def should_render(self, current_time_ns: int) -> bool:
        """Return ``True`` if *current_time_ns* >= next render time."""
        return current_time_ns >= self._next_render_ns

    def advance(self) -> int:
        """Advance to the next frame and return the consumed render time."""
        consumed = self._next_render_ns
        self._next_render_ns += self._frame_period_ns
        self._frame_count += 1
        return consumed

    @property
    def frame_count(self) -> int:
        """Number of frames rendered so far."""
        return self._frame_count
