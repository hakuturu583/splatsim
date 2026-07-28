"""Per-frame render + masking, producing the shared :class:`FrameData`.

Each GT sample is rendered once (all render LiDARs unioned to mirror the GT
``LIDAR_CONCAT``), mapped into the common base_link frame, and annotated with
the dynamic-object and sim-coverage masks. The resulting :class:`FrameData` is
handed to every metric so the expensive render happens exactly once per frame.
"""

from __future__ import annotations

import dataclasses
from functools import cached_property

import numpy as np
import torch

from .context import EvalContext
from .geometry import interp_ego_map, pose_to_matrix, transform
from .sensors import coverage_mask


@dataclasses.dataclass
class FrameData:
    """One frame's rendered + GT clouds and masks, all in base_link.

    Coordinates are base_link (x-forward, y-left, z-up). Intensity is carried
    alongside xyz because the BEV encoder consumes it as a point feature. The
    three boolean masks (``gt_dynamic`` / ``rd_dynamic`` over each cloud, and
    ``gt_cover`` = inside the sim's range+FOV envelope) let each metric pick the
    subset it should score. ``base_to_world`` places the clouds in the 3D view;
    ``boxes`` = ``(centers, half_sizes, quats_xyzw)`` are the dynamic-object
    boxes already expressed in the world frame.
    """

    index: int
    seconds: float
    gt_xyz: np.ndarray
    gt_intensity: np.ndarray
    rd_xyz: np.ndarray
    rd_intensity: np.ndarray
    gt_dynamic: np.ndarray
    rd_dynamic: np.ndarray
    gt_cover: np.ndarray
    base_to_world: np.ndarray
    boxes: tuple[np.ndarray, np.ndarray, np.ndarray]

    # Masked subsets every metric shares, gathered once (cached on first access).
    @cached_property
    def gt_static_xyz(self) -> np.ndarray:
        """GT points with dynamic-object returns removed."""
        return self.gt_xyz[~self.gt_dynamic]

    @cached_property
    def rd_static_xyz(self) -> np.ndarray:
        """Rendered points with dynamic-box returns removed."""
        return self.rd_xyz[~self.rd_dynamic]

    @cached_property
    def gt_ranged_xyz(self) -> np.ndarray:
        """Static GT points inside the sim's range + FOV envelope."""
        return self.gt_xyz[self.gt_cover & ~self.gt_dynamic]


def _dynamic_box_mask(pts_map: np.ndarray, boxes: list, margin: float) -> np.ndarray:
    """Boolean mask of ``pts_map`` (T4 map frame) lying inside any annotated box.

    T4 ``sample_annotation`` boxes are the dataset's movable/dynamic objects
    (vehicles, pedestrians, ...). The splat scene is a *static* reconstruction, so
    GT returns off those objects -- and any Gaussians the scene may have frozen in
    their place -- corrupt the metrics. Masking both clouds by these boxes
    isolates static-reconstruction quality from dynamic content.

    Each box carries a map-frame ``position``/``rotation`` and a
    ``shape.size = (width, length, height)`` (extents along local y/x/z). A point
    is inside when, expressed in the box's local frame, it falls within the
    half-extents (optionally grown by ``margin`` metres on every side).
    """
    inside = np.zeros(pts_map.shape[0], dtype=bool)
    if pts_map.shape[0] == 0 or not boxes:
        return inside
    for box in boxes:
        center = np.asarray(box.position, dtype=np.float64)
        rot = np.asarray(box.rotation.rotation_matrix, dtype=np.float64)  # local->map
        width, length, height = (float(v) for v in box.shape.size)
        half = np.array([length, width, height], dtype=np.float64) / 2.0 + margin
        local = (pts_map - center) @ rot  # R^T (p - c): map -> box-local
        inside |= np.all(np.abs(local) <= half, axis=1)
    return inside


def _world_boxes(
    ctx: EvalContext, boxes: list
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centers, half_sizes, quats_xyzw) of map-frame boxes in the world frame."""
    if not boxes:
        z3 = np.empty((0, 3), dtype=np.float32)
        return z3, z3, np.empty((0, 4), dtype=np.float32)
    r_mw = ctx.map_to_world[:3, :3]
    centers, halfs, quats = [], [], []
    for box in boxes:
        centers.append(transform(ctx.map_to_world, np.asarray(box.position)[None])[0])
        width, length, height = (float(v) for v in box.shape.size)
        halfs.append([length / 2.0, width / 2.0, height / 2.0])
        # Compose the box orientation into the world frame, then reorder
        # pyquaternion's (w, x, y, z) to the (x, y, z, w) Rerun expects.
        rot = type(box.rotation)(matrix=r_mw @ np.asarray(box.rotation.rotation_matrix))
        w, x, y, z = rot.elements
        quats.append([x, y, z, w])
    return (
        np.asarray(centers, dtype=np.float32),
        np.asarray(halfs, dtype=np.float32),
        np.asarray(quats, dtype=np.float32),
    )


def eval_frame(ctx: EvalContext, sample, index: int) -> FrameData:
    """Render + mask a single GT sample in the common base_link frame."""
    args = ctx.args
    sd = ctx.t4.get("sample_data", sample.data[ctx.gt_channel])
    ego = ctx.t4.get("ego_pose", sd.ego_pose_token)
    seconds = float(sd.timestamp) * 1e-6

    def _to_world(m_np: np.ndarray) -> torch.Tensor:
        # ego(base)->map, then map->world(align), then re-center to Gaussians.
        m = torch.from_numpy(m_np.astype(np.float32)).to(ctx.device)
        m = ctx.align @ m
        m[:3, 3] = m[:3, 3] - ctx.centroid
        return m

    ego_in_map = pose_to_matrix(ego.translation, ego.rotation)
    base_to_world = _to_world(ego_in_map)

    # Rolling shutter: the spinning sweep finishes ~sweep_period after the frame
    # timestamp while the ego keeps moving. Reconstruct that sweep-end base pose
    # by interpolating the ego trajectory and feed both ends to the renderer.
    base_to_world_end = None
    if args.rolling_shutter:
        ego_end_map = interp_ego_map(
            ctx.ego_ts_us,
            ctx.ego_trans,
            ctx.ego_quat,
            float(sd.timestamp) + args.sweep_period_s * 1e6,
        )
        base_to_world_end = _to_world(ego_end_map)

    # GT scan: (>=4, N) -> (N, 3) + intensity in its calibrated_sensor frame,
    # then base_link. LidarPointCloud rows are (x, y, z, intensity[, ring]).
    gt_path = ctx.t4.get_sample_data_path(sd.token)
    gt_pc = ctx.LidarPointCloud.from_file(gt_path)
    gt_pts = gt_pc.points
    gt_xyz = np.ascontiguousarray(gt_pts[:3].T, dtype=np.float32)
    gt_intensity = (
        np.ascontiguousarray(gt_pts[3], dtype=np.float32)
        if gt_pts.shape[0] > 3
        else np.zeros(gt_xyz.shape[0], dtype=np.float32)
    )
    gt_base = transform(ctx.gt_s2b, gt_xyz)

    # Rendered scan: render every LiDAR at its USDZ mount (with the sweep-end
    # pose when rolling shutter is on), map each into base_link, and union them
    # to mirror the GT LIDAR_CONCAT.
    rd_parts, rd_int_parts = [], []
    for ld in ctx.lidars:
        panorama = ld.renderer.render(
            base_to_world, scene=ctx.scene, base_to_world_end=base_to_world_end
        )
        rendered = ld.renderer.panorama_to_point_cloud(
            panorama,
            drop_threshold=args.drop_threshold,
            alpha_threshold=args.alpha_threshold,
        )
        rd_parts.append(transform(ld.s2b, rendered["xyz"]))
        rd_int_parts.append(rendered["intensity"])
    if rd_parts:
        rd_base = np.concatenate(rd_parts, axis=0)
        rd_intensity = np.concatenate(rd_int_parts, axis=0).astype(np.float32)
    else:
        rd_base = np.empty((0, 3), np.float32)
        rd_intensity = np.empty((0,), np.float32)

    # Dynamic-object masking: drop GT and rendered points inside the frame's
    # annotated 3D boxes (a static scene cannot reproduce moving objects). Boxes
    # come back in the T4 map frame, so test the clouds there.
    boxes = ctx.t4.get_box3ds(sd.token) if args.mask_dynamic else []
    if boxes:
        gt_dynamic = _dynamic_box_mask(
            transform(ego_in_map, gt_base), boxes, args.dynamic_margin
        )
        rd_dynamic = _dynamic_box_mask(
            transform(ego_in_map, rd_base), boxes, args.dynamic_margin
        )
    else:
        gt_dynamic = np.zeros(gt_base.shape[0], dtype=bool)
        rd_dynamic = np.zeros(rd_base.shape[0], dtype=bool)

    gt_cover = coverage_mask(gt_base, ctx.lidars)

    return FrameData(
        index=index,
        seconds=seconds,
        gt_xyz=gt_base,
        gt_intensity=gt_intensity,
        rd_xyz=rd_base,
        rd_intensity=rd_intensity,
        gt_dynamic=gt_dynamic,
        rd_dynamic=rd_dynamic,
        gt_cover=gt_cover,
        base_to_world=base_to_world.cpu().numpy(),
        boxes=_world_boxes(ctx, boxes),
    )
