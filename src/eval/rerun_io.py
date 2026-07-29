"""Rerun import guard + tiny logging helpers shared by the metrics.

Kept separate so metric modules can log scalars / images without each
re-implementing the optional-dependency guard.
"""

from __future__ import annotations

import numpy as np

# Shared colour vocabulary so the 3D geometry view and the metric legends agree:
# a viewer reads "green = GT, orange = rendered" across both. Referenced by the
# runner's geometry logging and the metrics' SeriesLines styling.
GT_COLOR = (80, 200, 120)  # GT, in-range static
RENDER_COLOR = (255, 130, 40)  # rendered static
DYNAMIC_COLOR = (230, 60, 60)  # GT dynamic / dynamic boxes
RENDER_DYNAMIC_COLOR = (150, 40, 40)  # rendered points inside a dynamic box
OUT_OF_RANGE_COLOR = (120, 120, 120)  # GT outside the sim envelope
TRAJ_COLOR = (120, 160, 255)  # ego trajectory


def require_rerun():
    """Import rerun-sdk, or exit with an actionable message."""
    try:
        import rerun as rr
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SystemExit(
            "rerun-sdk is required for LiDAR evaluation but is not installed.\n"
            "Install the optional 'eval' extra:  uv sync --extra eval"
        ) from exc
    return rr


def log_scalar(rr, path: str, value: float) -> None:
    """Log ``value`` to Rerun at ``path`` when it is finite (skip NaN/inf)."""
    if np.isfinite(value):
        rr.log(path, rr.Scalars(value))
