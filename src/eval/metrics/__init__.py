"""Evaluation metrics, one item per module.

Each metric implements :class:`~eval.metrics.base.LidarEvalMetric`: it scores the
shared per-frame :class:`~eval.frame.FrameData` (rendered vs GT clouds at the same
pose), logs its own signals to Rerun, and prints a final summary. New evaluation
items are added by dropping a new module here and registering it in
:func:`build_metrics`.

Metrics are intentionally decoupled from the geometry view: the runner logs the
shared 3D scene (point clouds, ego, boxes) once per frame; each metric logs only
its derived quantities (scalars, feature images).
"""

from __future__ import annotations

from ..context import EvalContext
from .base import LidarEvalMetric
from .bev_encoder import BEVEncoderMetric
from .chamfer import ChamferMetric

__all__ = [
    "LidarEvalMetric",
    "ChamferMetric",
    "BEVEncoderMetric",
    "build_metrics",
]


def build_metrics(args, ctx: EvalContext) -> list[LidarEvalMetric]:
    """Instantiate the metrics selected on the command line.

    ``--metrics`` is a comma-separated list drawn from the registry keys below;
    the default runs every registered metric. A metric that cannot initialise
    (e.g. the BEV encoder without its TensorRT backend) raises from its own
    constructor, so selection failures surface with an actionable message.
    """
    registry = {
        "chamfer": ChamferMetric,
        "bev": BEVEncoderMetric,
    }
    requested = [m.strip() for m in (args.metrics or "").split(",") if m.strip()]
    if not requested:
        requested = list(registry)
    unknown = [m for m in requested if m not in registry]
    if unknown:
        raise SystemExit(
            f"Unknown --metrics {unknown}; choose from {sorted(registry)}."
        )
    return [registry[name](args, ctx) for name in requested]
