#!/usr/bin/env python3
"""Evaluate splatsim LiDAR rendering against a WebAuto / T4 dataset.

The splatsim scene (a reconstructed 3DGS ``.usdz``) carries **no** ground-truth
LiDAR. This script pulls the ground truth from the matching WebAuto / T4 dataset
instead: it walks the dataset's GT ego trajectory, renders a LiDAR panorama from
the splat scene at every GT pose, and compares each rendered scan to the recorded
GT LiDAR scan with a symmetric Chamfer distance. Everything — the two point
clouds and the per-frame Chamfer distance — is logged to a `Rerun` recording
(``.rrd``) so the geometry and the metric can be inspected together on a shared
timeline.

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

Because the renderer is given the GT sensor's own extrinsic as its mount, the
rendered point cloud and the GT point cloud share the sensor frame, so the
Chamfer distance is computed there directly (no cross-frame resampling). The
point clouds are additionally transformed into the scene world frame for the
Rerun 3D view.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from splatsim._usdz import _rig_in_world, load_rig_trajectories
from splatsim.dataclass import SceneConfig
from splatsim.dataclass.lidar_config import LidarConfig
from splatsim.lidar_renderer import LidarRenderer, LidarSensorSpec, is_known_sensor
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
CHAMFER_SERIES: tuple[tuple[str, str, tuple[int, int, int]], ...] = (
    ("metrics/chamfer/symmetric", "symmetric", (255, 200, 40)),
    ("metrics/chamfer/render_to_gt", "render→gt", (255, 130, 40)),
    ("metrics/chamfer/gt_to_render", "gt→render", (80, 200, 120)),
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


def _build_sensor_spec(args, cs_record) -> tuple[LidarSensorSpec, float, float]:
    """Build a :class:`LidarSensorSpec` mounted at the GT sensor's extrinsic.

    Returns ``(spec, min_range_m, max_range_m)`` — ranges come from the model
    preset unless overridden on the CLI.
    """
    s2b = _pose_to_matrix(cs_record.translation, cs_record.rotation)

    preset = LidarConfig.for_sensor(args.sensor_type)
    n_columns = args.n_columns if args.n_columns else preset.n_columns
    fps = preset.fps
    min_range = args.min_range if args.min_range is not None else preset.min_range_m
    max_range = args.max_range if args.max_range is not None else preset.max_range_m

    spec = LidarSensorSpec(
        name=args.lidar_channel,
        sensor_type=args.sensor_type if is_known_sensor(args.sensor_type) else "",
        s2b=s2b,
        n_columns=int(n_columns),
        spinning_frequency_hz=float(fps),
        n_rows_uniform=int(args.n_rows),
        el_hi_rad=np.radians(args.el_hi_deg),
        el_lo_rad=np.radians(args.el_lo_deg),
    )
    return spec, float(min_range), float(max_range)


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
    first_channel = _pick_lidar_channel(samples[0], args.lidar_channel)
    first_sd = t4.get("sample_data", samples[0].data[first_channel])
    cs_record = t4.get("calibrated_sensor", first_sd.calibrated_sensor_token)
    spec, min_range, max_range = _build_sensor_spec(args, cs_record)
    renderer = LidarRenderer(
        spec, device=device, min_range_m=min_range, max_range_m=max_range
    )
    s2b_t = torch.from_numpy(spec.s2b.astype(np.float32)).to(device)
    print(
        f"[sensor] channel={first_channel} type={args.sensor_type or 'uniform'} "
        f"rows={renderer.n_rows} cols={renderer.n_columns} "
        f"range=[{min_range}, {max_range}] m"
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
    cd_values: list[float] = []
    print(f"[eval] rendering {len(selected)} frames")

    for i, sample in enumerate(selected):
        sd = t4.get("sample_data", sample.data[first_channel])
        ego = t4.get("ego_pose", sd.ego_pose_token)
        seconds = float(sd.timestamp) * 1e-6

        # ego(base)→map, then map→world(align), then re-center to Gaussians.
        ego_in_map = torch.from_numpy(
            _pose_to_matrix(ego.translation, ego.rotation).astype(np.float32)
        ).to(device)
        base_to_world = align_t @ ego_in_map
        base_to_world[:3, 3] = base_to_world[:3, 3] - centroid_t

        # GT scan (sensor frame): (4, N) -> (N, 3).
        gt_path = t4.get_sample_data_path(sd.token)
        gt_pc = LidarPointCloud.from_file(gt_path)
        gt_xyz = np.ascontiguousarray(gt_pc.points[:3].T, dtype=np.float32)

        # Rendered scan (same sensor frame, since s2b == GT extrinsic).
        panorama = renderer.render(base_to_world, scene=scene)
        rendered = renderer.panorama_to_point_cloud(
            panorama,
            drop_threshold=args.drop_threshold,
            alpha_threshold=args.alpha_threshold,
        )
        rendered_xyz = rendered["xyz"]

        # Chamfer distance in the shared sensor frame.
        gt_sub = _subsample(gt_xyz, args.max_points, rng)
        rd_sub = _subsample(rendered_xyz, args.max_points, rng)
        gt_dev = torch.from_numpy(gt_sub).to(device)
        rd_dev = torch.from_numpy(rd_sub).to(device)
        cd_sym, cd_r2g, cd_g2r = chamfer_distance(rd_dev, gt_dev)
        cd_values.append(cd_sym)

        # Transform both clouds into the scene world frame for the 3D view.
        sensor_to_world = (base_to_world @ s2b_t).cpu().numpy()
        gt_world = (gt_xyz @ sensor_to_world[:3, :3].T) + sensor_to_world[:3, 3]
        rd_world = (rendered_xyz @ sensor_to_world[:3, :3].T) + sensor_to_world[:3, 3]
        traj_world.append(sensor_to_world[:3, 3].copy())

        rr.set_time("frame", sequence=i)
        rr.set_time("stamp", duration=seconds)
        rr.log(
            "world/gt_lidar",
            rr.Points3D(gt_world, colors=(80, 200, 120), radii=args.point_radius),
        )
        rr.log(
            "world/rendered_lidar",
            rr.Points3D(rd_world, colors=(255, 130, 40), radii=args.point_radius),
        )
        rr.log(
            "world/sensor",
            rr.Transform3D(
                translation=sensor_to_world[:3, 3],
                mat3x3=sensor_to_world[:3, :3],
            ),
        )
        for path, value in zip(
            (p for p, _, _ in CHAMFER_SERIES), (cd_sym, cd_r2g, cd_g2r)
        ):
            _log_scalar(rr, path, value)

        print(
            f"  [{i + 1}/{len(selected)}] t={seconds:.2f}s "
            f"gt={gt_xyz.shape[0]:>7d} render={rendered_xyz.shape[0]:>7d} "
            f"chamfer={cd_sym:.4f}m (r→g {cd_r2g:.4f} / g→r {cd_g2r:.4f})"
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
    finite = np.asarray([c for c in cd_values if np.isfinite(c)], dtype=np.float64)
    if finite.size:
        print(
            "\n[summary] Chamfer distance over "
            f"{finite.size} frames: "
            f"mean={finite.mean():.4f} m  median={np.median(finite):.4f} m  "
            f"min={finite.min():.4f} m  max={finite.max():.4f} m"
        )
    else:
        print("\n[summary] no finite Chamfer values computed.")
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
        help="LiDAR channel to evaluate (default: LIDAR_CONCAT).",
    )

    sensor = p.add_argument_group("sensor / rendering")
    sensor.add_argument(
        "--sensor-type",
        default="OT128",
        help="Scan pattern for rendering: OT128 / XT32 / HDL64E, or '' for uniform.",
    )
    sensor.add_argument(
        "--n-columns", type=int, default=0, help="Override azimuth bins."
    )
    sensor.add_argument(
        "--n-rows", type=int, default=128, help="Beam count for the uniform fallback."
    )
    sensor.add_argument(
        "--el-hi-deg", type=float, default=15.0, help="Uniform-fallback top elevation."
    )
    sensor.add_argument(
        "--el-lo-deg",
        type=float,
        default=-25.0,
        help="Uniform-fallback bottom elevation.",
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
