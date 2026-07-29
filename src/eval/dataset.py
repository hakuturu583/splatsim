"""T4 / WebAuto dataset access + T4-map -> splat-world alignment.

Everything that reaches into ``t4-devkit`` lives here: the optional-dependency
guard, dataset-directory resolution, LiDAR-channel selection, ego-pose tables,
and the Umeyama trajectory alignment that bridges the dataset map frame and the
Gaussian world frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import umeyama


def require_t4():
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


def resolve_dataset_dir(args) -> str:
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


def pick_lidar_channel(sample, requested: str) -> str:
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


def ego_pose_table(t4) -> tuple[np.ndarray, np.ndarray, list]:
    """(timestamps_us, translations, rotations) of every ego_pose, time-sorted.

    ``rotations`` are the records' own ``pyquaternion.Quaternion`` objects, fed
    as-is to :func:`eval.geometry.interp_ego_map` for the rolling-shutter
    sweep-end pose.
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
    from splatsim._usdz import _rig_in_world, load_rig_trajectories

    rigs = load_rig_trajectories(usdz_path)
    for rig in rigs:
        poses = getattr(rig, "poses", None)
        if poses:
            xyz = np.asarray([_rig_in_world(p)[:3, 3] for p in poses])
            ts = np.asarray([float(p.timestamp_us) for p in poses], dtype=np.float64)
            return xyz, ts
    return np.empty((0, 3)), np.empty((0,))


def compute_alignment(args, t4, usdz_path: str | None) -> np.ndarray:
    """Resolve the map -> uncentered-ENU-world 4x4 transform per ``--align``."""
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
    ego_ts, ego_xyz, _ = ego_pose_table(t4)
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
        f"[align] auto: fitted rigid map->world from {src.shape[0]} matched poses, "
        f"RMSE={rmse:.3f} m"
    )
    if rmse > args.align_rmse_warn:
        print(
            f"[align] WARNING: alignment RMSE {rmse:.3f} m exceeds "
            f"{args.align_rmse_warn} m -- trajectories may not correspond. "
            "Inspect the .rrd and consider --align-file."
        )
    return transform
