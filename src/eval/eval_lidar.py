#!/usr/bin/env python3
"""Evaluate splatsim LiDAR rendering against a WebAuto / T4 dataset.

The splatsim scene (a reconstructed 3DGS ``.usdz``) carries **no** ground-truth
LiDAR. This script pulls the ground truth from the matching WebAuto / T4 dataset
instead: it walks the dataset's GT ego trajectory, renders a LiDAR panorama from
the splat scene at every GT pose, and compares each rendered scan to the recorded
GT LiDAR scan with a symmetric Chamfer distance. Two variants are computed per
frame: a **raw** Chamfer over all GT points, and a **range-aware** ("ranged")
Chamfer that first drops GT points outside the sim's range + FOV envelope (see
:func:`_coverage_mask`), so the metric isn't penalised for returns the sensor
model physically cannot make. Everything — the two point clouds and both Chamfer
series — is logged to a `Rerun` recording (``.rrd``) so the geometry and the
metrics can be inspected together on a shared timeline.

Two corrections keep the comparison fair against a *static* reconstruction:

* **Rolling shutter.** A spinning LiDAR paints its panorama over a finite sweep
  (~100 ms) while the ego is moving, so the scan is not a single-instant
  snapshot. The render mirrors this: the sweep-end ego pose is reconstructed by
  interpolating the T4 ``ego_pose`` trajectory ``--sweep-period-s`` after the
  frame's timestamp, and both start/end poses drive the renderer's
  motion-during-sweep path (``--no-rolling-shutter`` to disable).
* **Dynamic-object masking.** The GT LiDAR sees moving vehicles/pedestrians that
  a static scene cannot reproduce. Before scoring, GT and rendered points that
  fall inside the frame's annotated 3D boxes are dropped so the Chamfer distance
  reflects static geometry only (``--no-mask-dynamic`` to disable).

Both heavy dependencies (``t4-devkit`` for the dataset, ``rerun-sdk`` for the
recording) are **optional** and live behind the ``eval`` extra::

    uv sync --extra eval

Example
-------
::

    uv run python src/eval/eval_lidar.py \
        --scene /path/to/scene.usdz \
        --data-root ~/.webauto/datasets \
        --dataset-id 0123abcd-... \
        --output outputs/eval_lidar.rrd

Coordinate frames
-----------------
* The T4 dataset ``ego_pose`` is the base_link pose in the dataset *map* frame
  (ENU, z-up, ROS base_link = x-forward/y-left/z-up).
* The splat scene lives in an ENU world that is re-centered to the background's
  ``tile_local_centroid`` for numerical stability.
* ``--align`` bridges the two. ``auto`` (default) fits a rigid transform between
  the T4 ego trajectory and the scene's own recorded rig trajectory (both from
  the same drive) via Umeyama; ``identity`` assumes the two ENU frames already
  share an origin; ``file`` loads a 4x4 ``.npy``. The chosen transform maps map
  → uncentered-ENU-world; the centroid is then subtracted to reach the frame the
  Gaussians live in.

The render sensor's mount (height, orientation, beam table) comes from the scene
USDZ's own rig LiDAR calibration, not from the T4 ``LIDAR_CONCAT`` extrinsic —
that concat frame sits at base_link (ground), so rendering there would bury the
rays in the road. The rendered scan (render-sensor frame) and the GT scan (its
calibrated_sensor frame) are both mapped into base_link, where the Chamfer
distance is computed; both are additionally transformed into the scene world
frame for the Rerun 3D view.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from splatsim._usdz import _mat4, _rig_in_world, load_rig_trajectories
from splatsim.dataclass import SceneConfig
from splatsim.lidar_renderer import (
    LidarRenderer,
    build_lidar_sensors_from_config,
)
from splatsim.scene import Scene, print_progress


# ── optional dependency guards ──────────────────────────────────────────────


def _require_t4():
    """Import t4-devkit, or exit with an actionable message."""
    try:
        from t4_devkit import T4Devkit
        from t4_devkit.dataclass import LidarPointCloud
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SystemExit(
            "t4-devkit is required for LiDAR evaluation but is not installed.\n"
            "Install the optional 'eval' extra:  uv sync --extra eval"
        ) from exc
    return T4Devkit, LidarPointCloud


def _require_rerun():
    """Import rerun-sdk, or exit with an actionable message."""
    try:
        import rerun as rr
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SystemExit(
            "rerun-sdk is required for LiDAR evaluation but is not installed.\n"
            "Install the optional 'eval' extra:  uv sync --extra eval"
        ) from exc
    return rr


# ── geometry helpers ────────────────────────────────────────────────────────


def _pose_to_matrix(translation, rotation) -> np.ndarray:
    """4x4 rigid transform from a T4 record's translation + pyquaternion rotation.

    ``rotation`` is a ``pyquaternion.Quaternion`` (as carried by ``EgoPose`` /
    ``CalibratedSensor``), exposing a 3x3 ``rotation_matrix``.
    """
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = np.asarray(rotation.rotation_matrix, dtype=np.float64)
    m[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return m


def _transform(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform ``t`` to an (N, 3) point array."""
    return pts @ t[:3, :3].T + t[:3, 3]


def _interp_ego_map(
    ts_us: np.ndarray, trans: np.ndarray, quats: list, t_us: float
) -> np.ndarray:
    """Interpolated ego(base)→map 4x4 pose at unix-microsecond time ``t_us``.

    Translation is linearly interpolated; rotation is SLERP'd (via the same
    ``pyquaternion.Quaternion`` the T4 records already carry) between the two
    bracketing ``ego_pose`` records. Queries outside the recorded span clamp to
    the nearest endpoint. Used to reconstruct the sweep-end pose that drives the
    rolling-shutter render.
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
    q0 = quats[i0]
    quat = q0 if i0 == i1 else type(q0).slerp(q0, quats[i1], a)
    return _mat4(np.asarray(quat.rotation_matrix, dtype=np.float64), pos)


def _dynamic_box_mask(pts_map: np.ndarray, boxes: list, margin: float) -> np.ndarray:
    """Boolean mask of ``pts_map`` (T4 map frame) lying inside any annotated box.

    T4 ``sample_annotation`` boxes are the dataset's movable/dynamic objects
    (vehicles, pedestrians, …). The splat scene is a *static* reconstruction, so
    GT returns off those objects — and any Gaussians the scene may have frozen in
    their place — corrupt the Chamfer distance. Masking both clouds by these
    boxes isolates static-reconstruction quality from dynamic content.

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
        rot = np.asarray(box.rotation.rotation_matrix, dtype=np.float64)  # local→map
        width, length, height = (float(v) for v in box.shape.size)
        half = np.array([length, width, height], dtype=np.float64) / 2.0 + margin
        local = (pts_map - center) @ rot  # R^T (p - c): map → box-local
        inside |= np.all(np.abs(local) <= half, axis=1)
    return inside


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Rigid (no-scale) transform ``T`` minimizing ``|T·src - dst|`` (Kabsch).

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
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    out[:3, 3] = t
    aligned = (r @ src.T).T + t
    rmse = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))
    return out, rmse


def _subsample(
    xyz: np.ndarray, max_points: int, rng: np.random.Generator
) -> np.ndarray:
    """Randomly subsample rows of ``xyz`` down to ``max_points`` (no-op if fewer)."""
    n = xyz.shape[0]
    if max_points <= 0 or n <= max_points:
        return xyz
    idx = rng.choice(n, size=max_points, replace=False)
    return xyz[idx]


def _coverage_mask(gt_base: np.ndarray, lidars: list["_Lidar"]) -> np.ndarray:
    """Boolean mask of GT points the LiDAR simulation could actually return.

    A GT point (in base_link) is *coverable* if, expressed in some render
    sensor's frame, it lies within that sensor's range shell ``[min, max]`` and
    its vertical FOV ``[el_min, el_max]`` (azimuth is a full 360° spin, so it
    imposes no constraint). Points outside every sensor's envelope — e.g. beyond
    the sim's ``max_range`` or above/below the beam fan — can never be rendered,
    so excluding them isolates reconstruction quality from sensor-model limits.
    """
    if gt_base.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    mask = np.zeros(gt_base.shape[0], dtype=bool)
    for ld in lidars:
        # p_sensor = R^T (p - t), with s2b = [R | t] (sensor -> base).
        p = (gt_base - ld.s2b[:3, 3]) @ ld.s2b[:3, :3]
        rng = np.linalg.norm(p, axis=1)
        el = np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1]))
        mask |= (
            (rng >= ld.min_range)
            & (rng <= ld.max_range)
            & (el >= ld.el_min)
            & (el <= ld.el_max)
        )
    return mask


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
        # Accumulate on-device; a single .item() at the end avoids a GPU→CPU
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


# ── rerun logging ───────────────────────────────────────────────────────────

# Chamfer time-series to log: (rerun entity path, legend label, RGB colour).
# Two variants: "raw" over all GT points, and "ranged" over only the GT points
# the LiDAR sim could actually return (range + FOV envelope; see _coverage_mask).
# The per-frame value order in the loop must match this tuple order.
CHAMFER_SERIES: tuple[tuple[str, str, tuple[int, int, int]], ...] = (
    ("metrics/chamfer/raw/symmetric", "raw symmetric", (255, 200, 40)),
    ("metrics/chamfer/raw/render_to_gt", "raw render→gt", (255, 130, 40)),
    ("metrics/chamfer/raw/gt_to_render", "raw gt→render", (80, 200, 120)),
    ("metrics/chamfer/ranged/symmetric", "ranged symmetric", (180, 120, 255)),
    ("metrics/chamfer/ranged/render_to_gt", "ranged render→gt", (120, 160, 255)),
    ("metrics/chamfer/ranged/gt_to_render", "ranged gt→render", (40, 190, 190)),
)


def _log_scalar(rr, path: str, value: float) -> None:
    if np.isfinite(value):
        rr.log(path, rr.Scalars(value))


# ── dataset access ──────────────────────────────────────────────────────────


def _resolve_dataset_dir(args) -> str:
    """Locate the T4 dataset directory from ``--dataset-dir`` or root + id."""
    if args.dataset_dir:
        return str(args.dataset_dir)
    if not (args.data_root and args.dataset_id):
        raise SystemExit(
            "Provide either --dataset-dir, or both --data-root and --dataset-id."
        )
    path = Path(args.data_root).expanduser() / args.dataset_id
    if not path.exists():
        raise SystemExit(f"Dataset directory not found: {path}")
    return str(path)


def _pick_lidar_channel(sample, requested: str) -> str:
    """Return the LiDAR channel to read for a sample.

    Honours ``requested`` when present, otherwise falls back to the first channel
    whose name contains ``LIDAR`` (e.g. ``LIDAR_CONCAT`` / ``LIDAR_TOP``).
    """
    if requested in sample.data:
        return requested
    for channel in sample.data:
        if "LIDAR" in channel.upper():
            return channel
    raise SystemExit(
        f"No LiDAR channel found in sample. Available channels: {list(sample.data)}"
    )


def _t4_ego_trajectory(t4) -> tuple[np.ndarray, np.ndarray]:
    """(positions, timestamps_us) of every ego_pose record, sorted by time."""
    poses = sorted(t4.ego_pose, key=lambda p: p.timestamp)
    xyz = np.asarray([np.asarray(p.translation, dtype=np.float64) for p in poses])
    ts = np.asarray([float(p.timestamp) for p in poses], dtype=np.float64)
    return xyz, ts


def _ego_pose_table(t4) -> tuple[np.ndarray, np.ndarray, list]:
    """(timestamps_us, translations, rotations) of every ego_pose, time-sorted.

    ``rotations`` are the records' own ``pyquaternion.Quaternion`` objects, fed
    as-is to :func:`_interp_ego_map` for the rolling-shutter sweep-end pose.
    """
    poses = sorted(t4.ego_pose, key=lambda p: p.timestamp)
    ts = np.asarray([float(p.timestamp) for p in poses], dtype=np.float64)
    trans = np.asarray(
        [np.asarray(p.translation, dtype=np.float64) for p in poses], dtype=np.float64
    )
    quats = [p.rotation for p in poses]
    return ts, trans, quats


def _usdz_rig_trajectory(usdz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """(positions, timestamps_us) of the scene's recorded rig trajectory."""
    rigs = load_rig_trajectories(usdz_path)
    for rig in rigs:
        poses = getattr(rig, "poses", None)
        if poses:
            xyz = np.asarray([_rig_in_world(p)[:3, 3] for p in poses])
            ts = np.asarray([float(p.timestamp_us) for p in poses], dtype=np.float64)
            return xyz, ts
    return np.empty((0, 3)), np.empty((0,))


def _compute_alignment(args, t4, usdz_path: str | None) -> np.ndarray:
    """Resolve the map → uncentered-ENU-world 4x4 transform per ``--align``."""
    if args.align == "identity":
        print("[align] identity (assuming shared ENU origin)")
        return np.eye(4, dtype=np.float64)

    if args.align == "file":
        if not args.align_file:
            raise SystemExit("--align file requires --align-file PATH.npy")
        mat = np.load(args.align_file).astype(np.float64)
        if mat.shape != (4, 4):
            raise SystemExit(f"--align-file must hold a 4x4 matrix, got {mat.shape}")
        print(f"[align] loaded 4x4 from {args.align_file}")
        return mat

    # auto: rigid-fit the T4 ego trajectory onto the scene's rig trajectory.
    if not usdz_path:
        print("[align] auto requested but scene has no USDZ rig trajectory; identity")
        return np.eye(4, dtype=np.float64)
    rig_xyz, rig_ts = _usdz_rig_trajectory(usdz_path)
    ego_xyz, ego_ts = _t4_ego_trajectory(t4)
    if rig_xyz.shape[0] < 3 or ego_xyz.shape[0] < 3:
        print("[align] not enough poses to fit; falling back to identity")
        return np.eye(4, dtype=np.float64)

    # Correspond each rig pose to the nearest-in-time ego pose (both are unix
    # microseconds), keeping only matches within the max time gap.
    order = np.argsort(ego_ts)
    ego_ts_sorted = ego_ts[order]
    ego_xyz_sorted = ego_xyz[order]
    max_dt_us = args.align_max_dt_s * 1e6
    hi = np.clip(np.searchsorted(ego_ts_sorted, rig_ts), 1, ego_ts_sorted.shape[0] - 1)
    lo = hi - 1
    pick = np.where(
        np.abs(ego_ts_sorted[lo] - rig_ts) <= np.abs(ego_ts_sorted[hi] - rig_ts), lo, hi
    )
    keep = np.abs(ego_ts_sorted[pick] - rig_ts) <= max_dt_us
    src = ego_xyz_sorted[pick[keep]]
    dst = rig_xyz[keep]
    if src.shape[0] < 3:
        print(
            "[align] auto: <3 timestamp matches between T4 ego and USDZ rig "
            f"(within {args.align_max_dt_s}s); falling back to identity. "
            "Pass --align identity or --align-file if the frames differ."
        )
        return np.eye(4, dtype=np.float64)
    transform, rmse = umeyama(src, dst)
    print(
        f"[align] auto: fitted rigid map→world from {src.shape[0]} matched poses, "
        f"RMSE={rmse:.3f} m"
    )
    if rmse > args.align_rmse_warn:
        print(
            f"[align] WARNING: alignment RMSE {rmse:.3f} m exceeds "
            f"{args.align_rmse_warn} m — trajectories may not correspond. "
            "Inspect the .rrd and consider --align-file."
        )
    return transform


# ── sensor spec ─────────────────────────────────────────────────────────────


class _Lidar:
    """A render sensor: its renderer, sensor→base mount, and coverage envelope.

    ``min_range``/``max_range`` and ``el_min``/``el_max`` (radians) describe the
    range shell + vertical FOV the sim can return, used by :func:`_coverage_mask`
    to build the range-aware Chamfer metric.
    """

    __slots__ = (
        "name",
        "renderer",
        "s2b",
        "min_range",
        "max_range",
        "el_min",
        "el_max",
    )

    def __init__(
        self,
        name: str,
        renderer: LidarRenderer,
        s2b: np.ndarray,
        el_min: float,
        el_max: float,
    ) -> None:
        self.name = name
        self.renderer = renderer
        self.s2b = s2b
        self.min_range = renderer.min_range_m
        self.max_range = (
            renderer.max_range_m if renderer.max_range_m is not None else float("inf")
        )
        self.el_min = el_min
        self.el_max = el_max


def _spec_el_bounds(spec) -> tuple[float, float]:
    """(min, max) beam elevation in radians for a spec.

    The explicit per-beam table wins over the uniform-span fallback — the same
    precedence :mod:`splatsim.lidar_renderer` uses to build the beam pattern.
    """
    if spec.row_elevations_rad:
        return float(min(spec.row_elevations_rad)), float(max(spec.row_elevations_rad))
    return float(spec.el_lo_rad), float(spec.el_hi_rad)


def _estimate_gt_azimuth_columns(
    gt_base: np.ndarray, s2b: np.ndarray, el_min_rad: float, el_max_rad: float
) -> int | None:
    """Estimate a spinning LiDAR's azimuth column count from GT point density.

    GT points (base_link) are expressed in the sensor frame and restricted to its
    vertical FOV; within thin elevation rings the fundamental azimuth step is the
    20th-percentile gap between consecutive returns (robust to occlusion gaps and
    the rare dual return). ``round(360° / step)`` is the samples per revolution.
    Returns None when the GT is too sparse in this FOV to estimate (e.g. a
    narrow-FOV auxiliary LiDAR that the concat barely populates).
    """
    p = (gt_base - s2b[:3, 3]) @ s2b[:3, :3]  # base -> sensor (R orthonormal)
    el = np.degrees(np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1])))
    lo, hi = np.degrees(el_min_rad), np.degrees(el_max_rad)
    in_fov = (el >= lo) & (el <= hi)
    el = el[in_fov]  # work on the in-FOV subset only
    az = np.degrees(np.arctan2(p[in_fov, 1], p[in_fov, 0]))
    ring_steps: list[float] = []
    for center in np.linspace(lo + 2.0, hi - 2.0, 40):
        ring = np.abs(el - center) < 0.04
        if ring.sum() < 300:
            continue
        gaps = np.diff(np.sort(az[ring]))
        # Exclude dual-return duplicates (~0) and occlusion gaps (>1°); what
        # remains is the firing interval within the ring.
        gaps = gaps[(gaps > 0.01) & (gaps < 1.0)]
        if gaps.size > 50:
            ring_steps.append(float(np.median(gaps)))
    if not ring_steps:
        return None
    # The true grid step is the finest consistent ring (occlusion only widens
    # gaps, never narrows them); the 10th percentile is a robust "finest".
    step = float(np.percentile(ring_steps, 10))
    return int(round(360.0 / step))


def _resolve_n_columns(args, specs, gt_base) -> int | None:
    """Resolve one azimuth column count applied to every render sensor (or None).

    ``--n-columns``: an integer overrides directly; ``usdz`` keeps each sensor's
    stored value (``None``); ``auto`` (default) derives one resolution from the
    GT density. Because ``LIDAR_CONCAT`` carries no per-sensor labels, a single
    value measured from the densest (highest-beam) LiDAR is applied to all —
    per-sensor estimates from the merged cloud are unreliable. Logs its choice.
    """
    keep_msg = "[n_columns] keeping per-sensor USDZ value"
    mode = args.n_columns
    if mode == "usdz":
        print(f"{keep_msg} (usdz)")
        return None
    if mode != "auto":
        print(f"[n_columns] {int(mode)} (override)")
        return int(mode)
    if gt_base is None:
        print(f"{keep_msg} (no GT)")
        return None
    ref = max(specs, key=lambda s: len(s.row_elevations_rad) or s.n_rows_uniform)
    est = _estimate_gt_azimuth_columns(gt_base, ref.s2b, *_spec_el_bounds(ref))
    if est is None:
        print(f"{keep_msg} (GT too sparse)")
        return None
    print(f"[n_columns] {est} (gt-density from {ref.name})")
    return est


def _build_lidar_renderers(config, args, device, gt_base=None) -> list[_Lidar]:
    """Build one renderer per USDZ rig LiDAR, to be aggregated at eval time.

    The T4 GT is ``LIDAR_CONCAT`` — the point cloud of *all* physical LiDARs
    merged. To compare like-for-like the render side must likewise enable every
    LiDAR the scene knows about and union their scans. Each sensor's mount
    (height / orientation) and per-beam table come from the scene USDZ's own rig
    calibration (``config.lidar_sensors``, via the production
    ``build_lidar_sensors_from_config`` path); the T4 ``LIDAR_CONCAT``
    calibrated_sensor sits at base_link (ground) and must NOT be used as a mount.

    Azimuth resolution (``n_columns``) is resolved per :func:`_resolve_n_columns`
    — by default measured from the GT density, since the scene's stored value is
    typically a library default (e.g. 2048) rather than the real sensor's.

    ``--lidar-name`` (comma-separated) restricts to a subset; the default is all.

    Returns a list of :class:`_Lidar`.
    """
    sensors = list(config.lidar_sensors or [])
    if not sensors:
        raise SystemExit(
            "Scene USDZ carries no LiDAR calibration (config.lidar_sensors is "
            "empty); cannot determine the sensor mounts. Use a scene exported "
            "with rig LiDAR extrinsics."
        )
    if args.lidar_name:
        wanted = {n.strip() for n in args.lidar_name.split(",") if n.strip()}
        sensors = [c for c in sensors if c.name in wanted]
        if not sensors:
            avail = [c.name for c in (config.lidar_sensors or [])]
            raise SystemExit(
                f"--lidar-name {args.lidar_name!r} matched none of {avail}"
            )

    specs = build_lidar_sensors_from_config(sensors)
    n_columns = _resolve_n_columns(args, specs, gt_base)

    out: list[_Lidar] = []
    for cfg_sensor, spec in zip(sensors, specs):
        el_min, el_max = _spec_el_bounds(spec)
        if n_columns is not None:
            spec = dataclasses.replace(spec, n_columns=n_columns)

        min_range = (
            args.min_range if args.min_range is not None else cfg_sensor.min_range_m
        )
        max_range = (
            args.max_range if args.max_range is not None else cfg_sensor.max_range_m
        )
        renderer = LidarRenderer(
            spec,
            device=device,
            min_range_m=float(min_range),
            max_range_m=float(max_range),
        )
        out.append(
            _Lidar(spec.name, renderer, spec.s2b.astype(np.float32), el_min, el_max)
        )
    return out


# ── evaluation context ──────────────────────────────────────────────────────


@dataclasses.dataclass
class _EvalContext:
    """Run-invariant state shared by every per-frame evaluation.

    Built once by :func:`_build_context`; consumed by :func:`_eval_frame` and
    :func:`_log_frame`. ``align`` (map→world) and ``centroid`` are on-device
    tensors; ``gt_s2b`` (calibrated_sensor→base_link) is a numpy 4x4.
    """

    args: argparse.Namespace
    device: torch.device
    rng: np.random.Generator
    t4: Any
    LidarPointCloud: Any
    scene: Scene
    lidars: list[_Lidar]
    gt_channel: str
    gt_s2b: np.ndarray
    align: torch.Tensor
    centroid: torch.Tensor
    # map(T4)→world(Gaussian) 4x4 (numpy): align with the centroid folded into
    # the translation. Places map-frame annotation boxes into the 3D view.
    map_to_world: np.ndarray
    # ego(base)→map trajectory for rolling-shutter sweep-end interpolation.
    # ``ego_quat`` holds the records' pyquaternion.Quaternion rotations.
    ego_ts_us: np.ndarray
    ego_trans: np.ndarray
    ego_quat: list


def _load_scene(args, device) -> tuple[SceneConfig, Scene]:
    """Load the splat scene; return its config and the built Scene."""
    print(f"[scene] loading {args.scene}")
    config = SceneConfig.from_source(args.scene)
    scene = Scene.from_config(config, device=device, progress=print_progress)
    if scene.background is None:
        raise SystemExit("Scene has no background; cannot evaluate LiDAR against it.")
    return config, scene


def _build_context(args, device) -> _EvalContext:
    """Load scene + dataset, resolve alignment and render sensors into a context."""
    T4Devkit, LidarPointCloud = _require_t4()
    rng = np.random.default_rng(args.seed)

    # --- scene ---------------------------------------------------------------
    config, scene = _load_scene(args, device)
    usdz_path = config.background_usdz
    assert scene.background is not None  # guaranteed by _load_scene
    centroid = (
        scene.background.tile_local_centroid.detach().cpu().numpy().astype(np.float64)
    )

    # --- dataset -------------------------------------------------------------
    dataset_dir = _resolve_dataset_dir(args)
    print(f"[dataset] loading T4 dataset from {dataset_dir}")
    t4 = T4Devkit(dataset_dir, revision=args.revision, verbose=args.verbose)

    align = _compute_alignment(args, t4, usdz_path)
    align_t = torch.from_numpy(align.astype(np.float32)).to(device)
    centroid_t = torch.from_numpy(centroid.astype(np.float32)).to(device)
    # map→world = align, then re-center to the Gaussians (subtract the centroid
    # from the translation). Constant across frames.
    map_to_world = align.copy()
    map_to_world[:3, 3] = map_to_world[:3, 3] - centroid
    ego_ts_us, ego_trans, ego_quat = _ego_pose_table(t4)

    # --- sensor + renderer ---------------------------------------------------
    samples = list(t4.sample)
    if not samples:
        raise SystemExit("Dataset has no samples.")
    gt_channel = _pick_lidar_channel(samples[0], args.lidar_channel)
    first_sd = t4.get("sample_data", samples[0].data[gt_channel])
    gt_cs = t4.get("calibrated_sensor", first_sd.calibrated_sensor_token)
    # GT scans live in their calibrated_sensor frame; this maps them to base_link
    # (for LIDAR_CONCAT that frame is base_link itself, i.e. an identity mount).
    gt_s2b = _pose_to_matrix(gt_cs.translation, gt_cs.rotation).astype(np.float32)
    # First GT scan (in base_link) — only needed to derive azimuth resolution,
    # so skip the extra file read unless --n-columns auto will use it.
    gt0_base = None
    if args.n_columns == "auto":
        gt0 = LidarPointCloud.from_file(t4.get_sample_data_path(first_sd.token))
        gt0_base = _transform(
            gt_s2b, np.ascontiguousarray(gt0.points[:3].T, np.float32)
        )

    # Render sensors come from the scene USDZ's own rig LiDAR calibration (mount
    # height + beam table), NOT the GT concat frame. All are rendered and unioned
    # to match the GT LIDAR_CONCAT — see _build_lidar_renderers.
    lidars = _build_lidar_renderers(config, args, device, gt_base=gt0_base)
    print(f"[sensor] {len(lidars)} render LiDAR(s), GT channel={gt_channel}")
    for ld in lidars:
        print(
            f"  - {ld.name}: mount={ld.s2b[:3, 3].round(3).tolist()} "
            f"rows={ld.renderer.n_rows} cols={ld.renderer.n_columns} "
            f"range=[{ld.renderer.min_range_m}, {ld.renderer.max_range_m}] m"
        )

    return _EvalContext(
        args=args,
        device=device,
        rng=rng,
        t4=t4,
        LidarPointCloud=LidarPointCloud,
        scene=scene,
        lidars=lidars,
        gt_channel=gt_channel,
        gt_s2b=gt_s2b,
        align=align_t,
        centroid=centroid_t,
        map_to_world=map_to_world,
        ego_ts_us=ego_ts_us,
        ego_trans=ego_trans,
        ego_quat=ego_quat,
    )


# ── per-frame evaluation ─────────────────────────────────────────────────────


@dataclasses.dataclass
class _FrameResult:
    """Everything one frame produces: geometry (base_link) + both Chamfer tuples.

    ``raw``/``ranged`` are each ``(symmetric, render→gt, gt→render)`` in metres;
    ``b2w`` is the base_link→world 4x4 used to place the clouds in the 3D view.
    ``cover`` / ``dyn_gt`` are boolean masks over ``gt_base`` (in the sim's range
    + FOV envelope; inside a dynamic-object box) and ``dyn_rd`` over ``rd_base``;
    all three are kept so the 3D view can colour points by category even though
    the Chamfer metrics score only the static, in-envelope subset. ``boxes`` is
    ``(centers, half_sizes, quats_xyzw)`` in the world frame for the box overlay.
    """

    seconds: float
    gt_base: np.ndarray
    rd_base: np.ndarray
    cover: np.ndarray
    dyn_gt: np.ndarray
    dyn_rd: np.ndarray
    boxes: tuple[np.ndarray, np.ndarray, np.ndarray]
    b2w: np.ndarray
    raw: tuple[float, float, float]
    ranged: tuple[float, float, float]


def _world_boxes(
    ctx: _EvalContext, boxes: list
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centers, half_sizes, quats_xyzw) of map-frame boxes in the world frame."""
    if not boxes:
        z3 = np.empty((0, 3), dtype=np.float32)
        return z3, z3, np.empty((0, 4), dtype=np.float32)
    r_mw = ctx.map_to_world[:3, :3]
    centers, halfs, quats = [], [], []
    for box in boxes:
        centers.append(_transform(ctx.map_to_world, np.asarray(box.position)[None])[0])
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


def _eval_frame(ctx: _EvalContext, sample) -> _FrameResult:
    """Render + score a single GT sample in the common base_link frame."""
    args = ctx.args
    sd = ctx.t4.get("sample_data", sample.data[ctx.gt_channel])
    ego = ctx.t4.get("ego_pose", sd.ego_pose_token)
    seconds = float(sd.timestamp) * 1e-6

    # ego(base)→map, then map→world(align), then re-center to Gaussians.
    ego_in_map = _pose_to_matrix(ego.translation, ego.rotation)
    base_to_world = torch.from_numpy(ego_in_map.astype(np.float32)).to(ctx.device)
    base_to_world = ctx.align @ base_to_world
    base_to_world[:3, 3] = base_to_world[:3, 3] - ctx.centroid

    # Rolling shutter: the spinning sweep finishes ~sweep_period after the frame
    # timestamp while the ego keeps moving. Reconstruct that sweep-end base pose
    # by interpolating the ego trajectory and feed both ends to the renderer.
    base_to_world_end = None
    if args.rolling_shutter:
        ego_end_map = _interp_ego_map(
            ctx.ego_ts_us,
            ctx.ego_trans,
            ctx.ego_quat,
            float(sd.timestamp) + args.sweep_period_s * 1e6,
        )
        b2w_end = torch.from_numpy(ego_end_map.astype(np.float32)).to(ctx.device)
        b2w_end = ctx.align @ b2w_end
        b2w_end[:3, 3] = b2w_end[:3, 3] - ctx.centroid
        base_to_world_end = b2w_end

    # GT scan: (4, N) -> (N, 3) in its calibrated_sensor frame -> base_link.
    gt_path = ctx.t4.get_sample_data_path(sd.token)
    gt_pc = ctx.LidarPointCloud.from_file(gt_path)
    gt_xyz = np.ascontiguousarray(gt_pc.points[:3].T, dtype=np.float32)
    gt_base = _transform(ctx.gt_s2b, gt_xyz)

    # Rendered scan: render every LiDAR at its USDZ mount (with the sweep-end
    # pose when rolling shutter is on), map each into base_link, and union them
    # to mirror the GT LIDAR_CONCAT.
    rd_parts = []
    for ld in ctx.lidars:
        panorama = ld.renderer.render(
            base_to_world, scene=ctx.scene, base_to_world_end=base_to_world_end
        )
        rendered = ld.renderer.panorama_to_point_cloud(
            panorama,
            drop_threshold=args.drop_threshold,
            alpha_threshold=args.alpha_threshold,
        )
        rd_parts.append(_transform(ld.s2b, rendered["xyz"]))
    rd_base = (
        np.concatenate(rd_parts, axis=0) if rd_parts else np.empty((0, 3), np.float32)
    )

    # Dynamic-object masking: drop GT and rendered points inside the frame's
    # annotated 3D boxes (a static scene cannot reproduce moving objects). Boxes
    # come back in the T4 map frame, so test the clouds there.
    boxes = ctx.t4.get_box3ds(sd.token) if args.mask_dynamic else []
    if boxes:
        dyn_gt = _dynamic_box_mask(
            _transform(ego_in_map, gt_base), boxes, args.dynamic_margin
        )
        dyn_rd = _dynamic_box_mask(
            _transform(ego_in_map, rd_base), boxes, args.dynamic_margin
        )
    else:
        dyn_gt = np.zeros(gt_base.shape[0], dtype=bool)
        dyn_rd = np.zeros(rd_base.shape[0], dtype=bool)

    # Chamfer in base_link over the *static* subset. "raw" uses all static GT;
    # "ranged" further restricts GT to the sim's range + FOV envelope so the
    # metric isn't penalised for GT returns the sensor model physically can't make.
    rd_static = rd_base[~dyn_rd]
    rd_dev = torch.from_numpy(_subsample(rd_static, args.max_points, ctx.rng)).to(
        ctx.device
    )
    gt_dev = torch.from_numpy(
        _subsample(gt_base[~dyn_gt], args.max_points, ctx.rng)
    ).to(ctx.device)
    raw = chamfer_distance(rd_dev, gt_dev)

    cover = _coverage_mask(gt_base, ctx.lidars)
    gt_ranged = gt_base[cover & ~dyn_gt]
    gtr_dev = torch.from_numpy(_subsample(gt_ranged, args.max_points, ctx.rng)).to(
        ctx.device
    )
    ranged = chamfer_distance(rd_dev, gtr_dev)

    return _FrameResult(
        seconds=seconds,
        gt_base=gt_base,
        rd_base=rd_base,
        cover=cover,
        dyn_gt=dyn_gt,
        dyn_rd=dyn_rd,
        boxes=_world_boxes(ctx, boxes),
        b2w=base_to_world.cpu().numpy(),
        raw=raw,
        ranged=ranged,
    )


def _log_frame(rr, ctx: _EvalContext, result: _FrameResult, i: int) -> None:
    """Log one frame's clouds, ego transform, and Chamfer scalars to Rerun."""
    radius = ctx.args.point_radius
    b2w, cover, dyn_gt, dyn_rd = result.b2w, result.cover, result.dyn_gt, result.dyn_rd
    # Transform both (base_link) clouds into the scene world frame for the
    # 3D view. The ego/base_link pose drives the sensor transform + track.
    gt_world = _transform(b2w, result.gt_base)
    rd_world = _transform(b2w, result.rd_base)

    rr.set_time("frame", sequence=i)
    rr.set_time("stamp", duration=result.seconds)
    # GT split into three categories: dynamic (red, masked out of the metric),
    # static in-range (green, used by the ranged metric), and static
    # out-of-range (grey, only counted by the raw metric).
    static_gt = ~dyn_gt
    rr.log(
        "world/gt_dynamic",
        rr.Points3D(gt_world[dyn_gt], colors=(230, 60, 60), radii=radius),
    )
    rr.log(
        "world/gt_lidar",
        rr.Points3D(gt_world[cover & static_gt], colors=(80, 200, 120), radii=radius),
    )
    rr.log(
        "world/gt_out_of_range",
        rr.Points3D(gt_world[~cover & static_gt], colors=(120, 120, 120), radii=radius),
    )
    # Rendered: static (orange, scored) vs points that fell in a dynamic box
    # (dark red, masked out — e.g. Gaussians frozen where an object once was).
    rr.log(
        "world/rendered_lidar",
        rr.Points3D(rd_world[~dyn_rd], colors=(255, 130, 40), radii=radius),
    )
    rr.log(
        "world/rendered_dynamic",
        rr.Points3D(rd_world[dyn_rd], colors=(150, 40, 40), radii=radius),
    )
    centers, half_sizes, quats = result.boxes
    rr.log(
        "world/dynamic_boxes",
        rr.Boxes3D(
            centers=centers,
            half_sizes=half_sizes,
            quaternions=[rr.Quaternion(xyzw=q) for q in quats],
            colors=(230, 60, 60),
        ),
    )
    rr.log(
        "world/ego",
        rr.Transform3D(translation=b2w[:3, 3], mat3x3=b2w[:3, :3]),
    )
    for (path, _, _), value in zip(CHAMFER_SERIES, (*result.raw, *result.ranged)):
        _log_scalar(rr, path, value)


# ── summary ──────────────────────────────────────────────────────────────────


def _summarize(raw_syms: list[float], ranged_syms: list[float]) -> None:
    """Print the final raw + ranged symmetric-Chamfer statistics."""

    def _stats(label: str, values: list[float]) -> None:
        finite = np.asarray([c for c in values if np.isfinite(c)], dtype=np.float64)
        if finite.size:
            print(
                f"[summary] {label} symmetric Chamfer over {finite.size} frames: "
                f"mean={finite.mean():.4f} m  median={np.median(finite):.4f} m  "
                f"min={finite.min():.4f} m  max={finite.max():.4f} m"
            )
        else:
            print(f"[summary] {label}: no finite Chamfer values computed.")

    print()
    _stats("raw   ", raw_syms)
    _stats("ranged", ranged_syms)


# ── main pipeline ───────────────────────────────────────────────────────────


def run(args) -> None:
    rr = _require_rerun()
    device = torch.device(args.device)
    ctx = _build_context(args, device)

    # --- rerun recording -----------------------------------------------------
    rr.init("splatsim_lidar_eval", spawn=False)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(output))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for path, name, color in CHAMFER_SERIES:
        rr.log(path, rr.SeriesLines(names=[name], colors=[color]), static=True)

    # --- per-frame loop ------------------------------------------------------
    samples = list(ctx.t4.sample)
    stride = max(1, args.stride)
    selected = samples[::stride]
    if args.max_frames:
        selected = selected[: args.max_frames]

    traj_world: list[np.ndarray] = []
    raw_sym_values: list[float] = []
    ranged_sym_values: list[float] = []
    print(f"[eval] rendering {len(selected)} frames")

    for i, sample in enumerate(selected):
        result = _eval_frame(ctx, sample)
        _log_frame(rr, ctx, result, i)

        raw_sym_values.append(result.raw[0])
        ranged_sym_values.append(result.ranged[0])
        traj_world.append(result.b2w[:3, 3].copy())

        print(
            f"  [{i + 1}/{len(selected)}] t={result.seconds:.2f}s "
            f"gt={result.gt_base.shape[0]:>7d} "
            f"({int((result.cover & ~result.dyn_gt).sum())} in-range, "
            f"{int(result.dyn_gt.sum())} dynamic) "
            f"render={result.rd_base.shape[0]:>7d} ({int(result.dyn_rd.sum())} dynamic) "
            f"| chamfer raw={result.raw[0]:.4f}m ranged={result.ranged[0]:.4f}m"
        )

    # The full ego path is logged once (static) rather than re-serialising the
    # growing polyline every frame (which is O(N²) over the recording).
    if len(traj_world) >= 2:
        rr.log(
            "world/trajectory",
            rr.LineStrips3D(
                [np.asarray(traj_world, dtype=np.float32)], colors=(120, 160, 255)
            ),
            static=True,
        )

    # --- summary -------------------------------------------------------------
    _summarize(raw_sym_values, ranged_sym_values)
    print(f"[done] wrote Rerun recording to {output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_argument_group("scene / dataset")
    src.add_argument(
        "--scene", required=True, help="Scene USDZ path or SceneConfig source."
    )
    src.add_argument(
        "--data-root",
        help="Base directory holding WebAuto datasets as <data-root>/<dataset-id>.",
    )
    src.add_argument("--dataset-id", help="WebAuto / T4 dataset ID (folder name).")
    src.add_argument(
        "--dataset-dir",
        help="Direct path to the T4 dataset dir (overrides --data-root/--dataset-id).",
    )
    src.add_argument(
        "--revision", default=None, help="Dataset version (default: latest)."
    )
    src.add_argument(
        "--lidar-channel",
        default="LIDAR_CONCAT",
        help="GT LiDAR channel to read from the T4 dataset (default: LIDAR_CONCAT).",
    )

    sensor = p.add_argument_group("sensor / rendering")
    sensor.add_argument(
        "--lidar-name",
        default=None,
        help="Comma-separated USDZ rig LiDAR names to render (default: all, "
        "unioned to match the GT LIDAR_CONCAT).",
    )
    sensor.add_argument(
        "--n-columns",
        default="auto",
        help="Azimuth columns per render LiDAR: 'auto' (derive from GT density, "
        "default), an integer, or 'usdz' (keep the scene's stored value).",
    )
    sensor.add_argument(
        "--min-range", type=float, default=None, help="Override min range (m)."
    )
    sensor.add_argument(
        "--max-range", type=float, default=None, help="Override max range (m)."
    )
    sensor.add_argument(
        "--rolling-shutter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Model motion-during-sweep: render with the interpolated sweep-end "
        "ego pose (default: on; --no-rolling-shutter for a single-instant scan).",
    )
    sensor.add_argument(
        "--sweep-period-s",
        type=float,
        default=0.1,
        help="LiDAR sweep duration in seconds, i.e. the start→end time span used "
        "for rolling shutter (default: 0.1 = a 10 Hz spinning LiDAR).",
    )
    sensor.add_argument(
        "--drop-threshold",
        type=float,
        default=0.5,
        help="Drop a rendered sample when its ray-drop probability exceeds this "
        "(default: 0.5).",
    )
    sensor.add_argument(
        "--alpha-threshold",
        type=float,
        default=0.1,
        help="Drop a rendered sample when its accumulated alpha is below this "
        "(default: 0.1).",
    )

    align = p.add_argument_group("alignment (T4 map → splat world)")
    align.add_argument(
        "--align",
        choices=("auto", "identity", "file"),
        default="auto",
        help="How to map T4 map poses into the splat world (default: auto).",
    )
    align.add_argument("--align-file", help="4x4 .npy transform for --align file.")
    align.add_argument(
        "--align-max-dt-s",
        type=float,
        default=0.1,
        help="Max timestamp gap when matching poses for auto alignment.",
    )
    align.add_argument(
        "--align-rmse-warn",
        type=float,
        default=1.0,
        help="Warn if auto-alignment RMSE exceeds this (m).",
    )

    ev = p.add_argument_group("evaluation / output")
    ev.add_argument(
        "--output", default="outputs/eval_lidar.rrd", help="Output .rrd path."
    )
    ev.add_argument(
        "--mask-dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop GT and rendered points inside the frame's annotated 3D boxes "
        "before scoring, so the static reconstruction isn't penalised for moving "
        "objects (default: on; --no-mask-dynamic to score raw clouds).",
    )
    ev.add_argument(
        "--dynamic-margin",
        type=float,
        default=0.25,
        help="Grow each dynamic box by this many metres per side when masking, to "
        "catch returns just outside the annotated extent (default: 0.25).",
    )
    ev.add_argument("--stride", type=int, default=1, help="Use every Nth sample.")
    ev.add_argument(
        "--max-frames", type=int, default=0, help="Cap number of frames (0=all)."
    )
    ev.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Subsample each cloud to this many points for Chamfer (0=off).",
    )
    ev.add_argument(
        "--point-radius", type=float, default=0.03, help="Rerun point radius (m)."
    )
    ev.add_argument("--device", default="cuda", help="Torch device (default: cuda).")
    ev.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling.")
    ev.add_argument("--verbose", action="store_true", help="Verbose T4 table loading.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
