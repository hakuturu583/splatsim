"""The metric interface every evaluation item implements."""

from __future__ import annotations

import abc

import numpy as np

from ..context import EvalContext
from ..frame import FrameData


def summarize_series(name: str, values: list[float], unit: str = "") -> None:
    """Print mean/median/min/max of ``values`` (finite only) as a summary line.

    Shared by every metric so the distribution-stats formatting lives in one
    place. ``name`` is the full label (e.g. ``"chamfer raw"``); ``unit`` is an
    optional suffix appended to each statistic (e.g. ``" m"``).
    """
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size:
        print(
            f"[summary] {name} over {arr.size} frames: "
            f"mean={arr.mean():.4f}{unit}  median={np.median(arr):.4f}{unit}  "
            f"min={arr.min():.4f}{unit}  max={arr.max():.4f}{unit}"
        )
    else:
        print(f"[summary] {name}: no finite values computed.")


class LidarEvalMetric(abc.ABC):
    """One evaluation item scoring a rendered scan against GT at the same pose.

    A metric owns its Rerun entity paths and its running statistics. The runner
    drives it with three hooks:

    * :meth:`setup_rerun` -- once, after the recording is opened, to declare
      static entities (e.g. ``SeriesLines`` styling).
    * :meth:`update` -- once per frame, to score + log the frame and return the
      headline scalar(s) the console progress line should show.
    * :meth:`summarize` -- once at the end, to print aggregate statistics.

    Subclasses define their own ``__init__(args, ctx)`` and may raise there to
    signal the metric is unavailable (e.g. a missing optional backend);
    :func:`eval.metrics.build_metrics` lets that surface to the user.
    """

    #: short registry key / Rerun namespace for this metric.
    name: str = "metric"

    def setup_rerun(self, rr, ctx: EvalContext) -> None:  # noqa: B027 - optional
        """Declare static Rerun entities. Default: nothing."""

    @abc.abstractmethod
    def update(self, rr, frame: FrameData, ctx: EvalContext) -> dict[str, float]:
        """Score ``frame``, log this metric's signals, and return headline scalars.

        The returned ``{label: value}`` dict is appended to the per-frame console
        line; keep it to one or two numbers.
        """

    def summarize(self) -> None:  # noqa: B027 - optional
        """Print aggregate statistics over all frames. Default: nothing."""
