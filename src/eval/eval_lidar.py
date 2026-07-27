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
import sys
from pathlib import Path

import numpy as np
import torch

from splatsim._usdz import _rig_in_world, load_rig_trajectories
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


def _build_lidar_renderers(config, args, device) -> list[_Lidar]:
    """Build one renderer per USDZ rig LiDAR, to be aggregated at eval time.

    The T4 GT is ``LIDAR_CONCAT`` — the point cloud of *all* physical LiDARs
    merged. To compare like-for-like the render side must likewise enable every
    LiDAR the scene knows about and union their scans. Each sensor's mount
    (height / orientation) and per-beam table come from the scene USDZ's own rig
    calibration (``config.lidar_sensors``, via the production
    ``build_lidar_sensors_from_config`` path); the T4 ``LIDAR_CONCAT``
    calibrated_sensor sits at base_link (ground) and must NOT be used as a mount.

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
    out: list[_Lidar] = []
    for cfg_sensor, spec in zip(sensors, specs):
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
        # Vertical FOV bounds: explicit beam table if present, else uniform span.
        if spec.row_elevations_rad:
            el_min = float(min(spec.row_elevations_rad))
            el_max = float(max(spec.row_elevations_rad))
        else:
            el_min, el_max = float(spec.el_lo_rad), float(spec.el_hi_rad)
        out.append(
            _Lidar(spec.name, renderer, spec.s2b.astype(np.float32), el_min, el_max)
        )
    return out


# ── main pipeline ───────────────────────────────────────────────────────────


def run(args) -> None:
    T4Devkit, LidarPointCloud = _require_t4()
    rr = _require_rerun()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # --- scene ---------------------------------------------------------------
    print(f"[scene] loading {args.scene}")
    config = SceneConfig.from_source(args.scene)
    usdz_path = config.background_usdz
    scene = Scene.from_config(config, device=device, progress=print_progress)
    if scene.background is None:
        raise SystemExit("Scene has no background; cannot evaluate LiDAR against it.")
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

    # Render sensors come from the scene USDZ's own rig LiDAR calibration (mount
    # height + beam table), NOT the GT concat frame. All are rendered and unioned
    # to match the GT LIDAR_CONCAT — see _build_lidar_renderers.
    lidars = _build_lidar_renderers(config, args, device)
    print(f"[sensor] {len(lidars)} render LiDAR(s), GT channel={gt_channel}")
    for ld in lidars:
        print(
            f"  - {ld.name}: mount={ld.s2b[:3, 3].round(3).tolist()} "
            f"rows={ld.renderer.n_rows} cols={ld.renderer.n_columns} "
            f"range=[{ld.renderer.min_range_m}, {ld.renderer.max_range_m}] m"
        )

    # --- rerun recording -----------------------------------------------------
    rr.init("splatsim_lidar_eval", spawn=False)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(output))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for path, name, color in CHAMFER_SERIES:
        rr.log(path, rr.SeriesLines(names=[name], colors=[color]), static=True)

    # --- per-frame loop ------------------------------------------------------
    stride = max(1, args.stride)
    selected = samples[::stride]
    if args.max_frames:
        selected = selected[: args.max_frames]

    traj_world: list[np.ndarray] = []
    raw_sym_values: list[float] = []
    ranged_sym_values: list[float] = []
    print(f"[eval] rendering {len(selected)} frames")

    for i, sample in enumerate(selected):
        sd = t4.get("sample_data", sample.data[gt_channel])
        ego = t4.get("ego_pose", sd.ego_pose_token)
        seconds = float(sd.timestamp) * 1e-6

        # ego(base)→map, then map→world(align), then re-center to Gaussians.
        ego_in_map = torch.from_numpy(
            _pose_to_matrix(ego.translation, ego.rotation).astype(np.float32)
        ).to(device)
        base_to_world = align_t @ ego_in_map
        base_to_world[:3, 3] = base_to_world[:3, 3] - centroid_t

        # GT scan: (4, N) -> (N, 3) in its calibrated_sensor frame -> base_link.
        gt_path = t4.get_sample_data_path(sd.token)
        gt_pc = LidarPointCloud.from_file(gt_path)
        gt_xyz = np.ascontiguousarray(gt_pc.points[:3].T, dtype=np.float32)
        gt_base = _transform(gt_s2b, gt_xyz)

        # Rendered scan: render every LiDAR at its USDZ mount, map each into
        # base_link, and union them to mirror the GT LIDAR_CONCAT.
        rd_parts = []
        for ld in lidars:
            panorama = ld.renderer.render(base_to_world, scene=scene)
            rendered = ld.renderer.panorama_to_point_cloud(
                panorama,
                drop_threshold=args.drop_threshold,
                alpha_threshold=args.alpha_threshold,
            )
            rd_parts.append(_transform(ld.s2b, rendered["xyz"]))
        rd_base = (
            np.concatenate(rd_parts, axis=0)
            if rd_parts
            else np.empty((0, 3), np.float32)
        )

        # Chamfer distance in the common base_link frame. "raw" uses all GT;
        # "ranged" restricts GT to the sim's range + FOV envelope so the metric
        # isn't penalised for GT returns the sensor model physically can't make.
        rd_dev = torch.from_numpy(_subsample(rd_base, args.max_points, rng)).to(device)
        gt_dev = torch.from_numpy(_subsample(gt_base, args.max_points, rng)).to(device)
        raw = chamfer_distance(rd_dev, gt_dev)

        cover = _coverage_mask(gt_base, lidars)
        gt_ranged = gt_base[cover]
        gtr_dev = torch.from_numpy(_subsample(gt_ranged, args.max_points, rng)).to(
            device
        )
        ranged = chamfer_distance(rd_dev, gtr_dev)

        raw_sym_values.append(raw[0])
        ranged_sym_values.append(ranged[0])

        # Transform both (base_link) clouds into the scene world frame for the
        # 3D view. The ego/base_link pose drives the sensor transform + track.
        b2w = base_to_world.cpu().numpy()
        gt_world = _transform(b2w, gt_base)
        rd_world = _transform(b2w, rd_base)
        traj_world.append(b2w[:3, 3].copy())

        rr.set_time("frame", sequence=i)
        rr.set_time("stamp", duration=seconds)
        # GT split by coverage: in-range (green, used by the ranged metric) vs
        # out-of-range (grey, only counted by the raw metric).
        rr.log(
            "world/gt_lidar",
            rr.Points3D(
                gt_world[cover], colors=(80, 200, 120), radii=args.point_radius
            ),
        )
        rr.log(
            "world/gt_out_of_range",
            rr.Points3D(
                gt_world[~cover], colors=(120, 120, 120), radii=args.point_radius
            ),
        )
        rr.log(
            "world/rendered_lidar",
            rr.Points3D(rd_world, colors=(255, 130, 40), radii=args.point_radius),
        )
        rr.log(
            "world/ego",
            rr.Transform3D(translation=b2w[:3, 3], mat3x3=b2w[:3, :3]),
        )
        for (path, _, _), value in zip(CHAMFER_SERIES, (*raw, *ranged)):
            _log_scalar(rr, path, value)

        print(
            f"  [{i + 1}/{len(selected)}] t={seconds:.2f}s "
            f"gt={gt_base.shape[0]:>7d} ({int(cover.sum())} in-range) "
            f"render={rd_base.shape[0]:>7d} | chamfer raw={raw[0]:.4f}m "
            f"ranged={ranged[0]:.4f}m"
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
    _stats("raw   ", raw_sym_values)
    _stats("ranged", ranged_sym_values)
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
        "--min-range", type=float, default=None, help="Override min range (m)."
    )
    sensor.add_argument(
        "--max-range", type=float, default=None, help="Override max range (m)."
    )
    sensor.add_argument("--drop-threshold", type=float, default=0.5)
    sensor.add_argument("--alpha-threshold", type=float, default=0.1)

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
