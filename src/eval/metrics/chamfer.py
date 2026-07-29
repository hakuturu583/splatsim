"""Geometric agreement: symmetric Chamfer distance (raw + range-aware).

Two variants are computed per frame over the *static* (non-dynamic) subset:

* **raw** -- over all static GT points.
* **ranged** -- over only the GT points the LiDAR sim could physically return
  (inside some sensor's range shell + vertical FOV; see
  :func:`eval.sensors.coverage_mask`), so the metric isn't penalised for returns
  the sensor model cannot make.

Each variant reports ``(symmetric, render->gt, gt->render)`` in metres:
``render->gt`` is the completeness of the render, ``gt->render`` the coverage of
GT by the render.
"""

from __future__ import annotations

import numpy as np
import torch

from ..context import EvalContext
from ..frame import FrameData
from ..geometry import subsample
from ..rerun_io import log_scalar
from .base import LidarEvalMetric, summarize_series

# (rerun entity path, legend label, RGB colour). The per-frame value order must
# match this tuple order.
CHAMFER_SERIES: tuple[tuple[str, str, tuple[int, int, int]], ...] = (
    ("metrics/chamfer/raw/symmetric", "raw symmetric", (255, 200, 40)),
    ("metrics/chamfer/raw/render_to_gt", "raw render->gt", (255, 130, 40)),
    ("metrics/chamfer/raw/gt_to_render", "raw gt->render", (80, 200, 120)),
    ("metrics/chamfer/ranged/symmetric", "ranged symmetric", (180, 120, 255)),
    ("metrics/chamfer/ranged/render_to_gt", "ranged render->gt", (120, 160, 255)),
    ("metrics/chamfer/ranged/gt_to_render", "ranged gt->render", (40, 190, 190)),
)


def chamfer_distance(
    a: torch.Tensor, b: torch.Tensor, *, chunk: int = 4096
) -> tuple[float, float, float]:
    """Symmetric mean Chamfer distance (in metres) between two point sets.

    Args:
        a: (N, 3) point cloud on some device.
        b: (M, 3) point cloud on the same device.
        chunk: rows of ``a`` / ``b`` per ``cdist`` tile, to bound memory.

    Returns:
        ``(symmetric, a_to_b, b_to_a)`` where ``a_to_b`` is the mean nearest-
        neighbour distance from ``a`` into ``b`` (completeness of the render vs
        GT) and ``b_to_a`` the reverse (coverage of GT by the render).
    """

    def _nn_mean(src: torch.Tensor, dst: torch.Tensor) -> float:
        if src.numel() == 0 or dst.numel() == 0:
            return float("nan")
        # Accumulate on-device; a single .item() at the end avoids a GPU->CPU
        # sync per chunk.
        total = src.new_zeros(())
        for i in range(0, src.shape[0], chunk):
            dists = torch.cdist(src[i : i + chunk], dst)  # (chunk, M)
            total += dists.min(dim=1).values.sum()
        return (total / src.shape[0]).item()

    a_to_b = _nn_mean(a, b)
    b_to_a = _nn_mean(b, a)
    symmetric = float(np.nanmean([a_to_b, b_to_a]))
    return symmetric, a_to_b, b_to_a


class ChamferMetric(LidarEvalMetric):
    """Symmetric Chamfer distance between the rendered and GT point clouds."""

    name = "chamfer"

    def __init__(self, args, ctx: EvalContext) -> None:
        self._raw_sym: list[float] = []
        self._ranged_sym: list[float] = []

    def setup_rerun(self, rr, ctx: EvalContext) -> None:
        for path, name, color in CHAMFER_SERIES:
            rr.log(path, rr.SeriesLines(names=[name], colors=[color]), static=True)

    def update(self, rr, frame: FrameData, ctx: EvalContext) -> dict[str, float]:
        args, device, rng = ctx.args, ctx.device, ctx.rng

        # Chamfer over the *static* subset. "raw" uses all static GT; "ranged"
        # further restricts GT to the sim's range + FOV envelope so the metric
        # isn't penalised for GT returns the sensor model physically can't make.
        rd_dev = torch.from_numpy(
            subsample(frame.rd_static_xyz, args.max_points, rng)
        ).to(device)
        gt_dev = torch.from_numpy(
            subsample(frame.gt_static_xyz, args.max_points, rng)
        ).to(device)
        raw = chamfer_distance(rd_dev, gt_dev)

        gtr_dev = torch.from_numpy(
            subsample(frame.gt_ranged_xyz, args.max_points, rng)
        ).to(device)
        ranged = chamfer_distance(rd_dev, gtr_dev)

        for (path, _, _), value in zip(CHAMFER_SERIES, (*raw, *ranged)):
            log_scalar(rr, path, value)

        self._raw_sym.append(raw[0])
        self._ranged_sym.append(ranged[0])
        return {"raw": raw[0], "ranged": ranged[0]}

    def summarize(self) -> None:
        summarize_series("chamfer raw   ", self._raw_sym, unit=" m")
        summarize_series("chamfer ranged", self._ranged_sym, unit=" m")
