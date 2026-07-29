"""Per-frame driver: render once, log the shared 3D scene, run every metric.

The runner owns the shared geometry view (point clouds, ego, boxes, trajectory)
and the Rerun timeline; each metric owns only its derived signals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .context import build_context
from .frame import FrameData
from .geometry import transform
from .metrics import build_metrics
from .rerun_io import (
    DYNAMIC_COLOR,
    GT_COLOR,
    OCCLUDED_COLOR,
    OUT_OF_RANGE_COLOR,
    RENDER_COLOR,
    RENDER_DYNAMIC_COLOR,
    TRAJ_COLOR,
    require_rerun,
)


def _log_geometry(rr, frame: FrameData, radius: float) -> None:
    """Log one frame's clouds, dynamic boxes, and ego transform to Rerun."""
    b2w = frame.base_to_world
    gt_world = transform(b2w, frame.gt_xyz)
    rd_world = transform(b2w, frame.rd_xyz)
    gt_keep = frame.gt_keep
    cover = frame.gt_cover
    # Occlusion-shadow points shown on their own (not double-counted as dynamic).
    gt_shadow = frame.gt_occluded & ~frame.gt_dynamic
    rd_shadow = frame.rd_occluded & ~frame.rd_dynamic

    # GT: dynamic (red, masked out), static in-range (green, scored by ranged),
    # static out-of-range (grey, only the raw metric counts it), shadow (purple).
    rr.log(
        "world/gt_dynamic",
        rr.Points3D(gt_world[frame.gt_dynamic], colors=DYNAMIC_COLOR, radii=radius),
    )
    rr.log(
        "world/gt_occluded",
        rr.Points3D(gt_world[gt_shadow], colors=OCCLUDED_COLOR, radii=radius),
    )
    rr.log(
        "world/gt_lidar",
        rr.Points3D(gt_world[cover & gt_keep], colors=GT_COLOR, radii=radius),
    )
    rr.log(
        "world/gt_out_of_range",
        rr.Points3D(
            gt_world[~cover & gt_keep], colors=OUT_OF_RANGE_COLOR, radii=radius
        ),
    )
    # Rendered: static (orange, scored), fell-in-a-dynamic-box (dark red), or in a
    # dynamic object's occlusion shadow (purple, dropped as GT-invisible).
    rr.log(
        "world/rendered_lidar",
        rr.Points3D(rd_world[frame.rd_keep], colors=RENDER_COLOR, radii=radius),
    )
    rr.log(
        "world/rendered_dynamic",
        rr.Points3D(
            rd_world[frame.rd_dynamic], colors=RENDER_DYNAMIC_COLOR, radii=radius
        ),
    )
    rr.log(
        "world/rendered_occluded",
        rr.Points3D(rd_world[rd_shadow], colors=OCCLUDED_COLOR, radii=radius),
    )
    centers, half_sizes, quats = frame.boxes
    rr.log(
        "world/dynamic_boxes",
        rr.Boxes3D(
            centers=centers,
            half_sizes=half_sizes,
            quaternions=[rr.Quaternion(xyzw=q) for q in quats],
            colors=DYNAMIC_COLOR,
        ),
    )
    rr.log("world/ego", rr.Transform3D(translation=b2w[:3, 3], mat3x3=b2w[:3, :3]))


def run(args) -> None:
    from .frame import eval_frame

    rr = require_rerun()
    device = torch.device(args.device)
    ctx = build_context(args, device)
    metrics = build_metrics(args, ctx)
    print(f"[eval] metrics: {', '.join(m.name for m in metrics)}")

    # --- rerun recording -----------------------------------------------------
    rr.init("splatsim_lidar_eval", spawn=False)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(output))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for metric in metrics:
        metric.setup_rerun(rr, ctx)

    # --- per-frame loop ------------------------------------------------------
    samples = list(ctx.t4.sample)
    stride = max(1, args.stride)
    selected = samples[::stride]
    if args.max_frames:
        selected = selected[: args.max_frames]

    traj_world: list[np.ndarray] = []
    print(f"[eval] rendering {len(selected)} frames")

    for i, sample in enumerate(selected):
        frame = eval_frame(ctx, sample, i)
        rr.set_time("frame", sequence=i)
        rr.set_time("stamp", duration=frame.seconds)
        _log_geometry(rr, frame, args.point_radius)

        headline: dict[str, float] = {}
        for metric in metrics:
            headline.update(metric.update(rr, frame, ctx))
        traj_world.append(frame.base_to_world[:3, 3].copy())

        metric_str = " ".join(f"{k}={v:.4f}" for k, v in headline.items())
        print(
            f"  [{i + 1}/{len(selected)}] t={frame.seconds:.2f}s "
            f"gt={frame.gt_xyz.shape[0]:>7d} "
            f"({int((frame.gt_cover & frame.gt_keep).sum())} in-range, "
            f"{int(frame.gt_dynamic.sum())} dynamic, "
            f"{int((frame.gt_occluded & ~frame.gt_dynamic).sum())} shadow) "
            f"render={frame.rd_xyz.shape[0]:>7d} "
            f"({int(frame.rd_dynamic.sum())} dynamic, "
            f"{int((frame.rd_occluded & ~frame.rd_dynamic).sum())} shadow) | {metric_str}"
        )

    # The full ego path is logged once (static) rather than re-serialising the
    # growing polyline every frame (which is O(N^2) over the recording).
    if len(traj_world) >= 2:
        rr.log(
            "world/trajectory",
            rr.LineStrips3D(
                [np.asarray(traj_world, dtype=np.float32)], colors=TRAJ_COLOR
            ),
            static=True,
        )

    # --- summary -------------------------------------------------------------
    print()
    for metric in metrics:
        metric.summarize()
    print(f"[done] wrote Rerun recording to {output}")
