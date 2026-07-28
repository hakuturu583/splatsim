"""Run-invariant evaluation context.

:class:`EvalContext` bundles everything built once at start-up (scene, dataset
handle, render sensors, alignment, ego-pose tables) and shared, read-only, by
every per-frame evaluation and every metric.
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import Any

import numpy as np
import torch

from splatsim.dataclass import SceneConfig
from splatsim.scene import Scene, print_progress

from .dataset import (
    compute_alignment,
    ego_pose_table,
    pick_lidar_channel,
    require_t4,
    resolve_dataset_dir,
)
from .geometry import pose_to_matrix, transform
from .sensors import Lidar, build_lidar_renderers


@dataclasses.dataclass
class EvalContext:
    """Run-invariant state shared by every per-frame evaluation and metric.

    Built once by :func:`build_context`; consumed by :mod:`eval.frame` and the
    metrics. ``align`` (map->world) and ``centroid`` are on-device tensors;
    ``gt_s2b`` (calibrated_sensor->base_link) is a numpy 4x4.
    """

    args: argparse.Namespace
    device: torch.device
    rng: np.random.Generator
    t4: Any
    LidarPointCloud: Any
    scene: Scene
    lidars: list[Lidar]
    gt_channel: str
    gt_s2b: np.ndarray
    align: torch.Tensor
    centroid: torch.Tensor
    # map(T4)->world(Gaussian) 4x4 (numpy): align with the centroid folded into
    # the translation. Places map-frame annotation boxes into the 3D view.
    map_to_world: np.ndarray
    # ego(base)->map trajectory for rolling-shutter sweep-end interpolation.
    # ``ego_quat`` holds the records' pyquaternion.Quaternion rotations.
    ego_ts_us: np.ndarray
    ego_trans: np.ndarray
    ego_quat: list


def load_scene(args, device) -> tuple[SceneConfig, Scene]:
    """Load the splat scene; return its config and the built Scene."""
    print(f"[scene] loading {args.scene}")
    config = SceneConfig.from_source(args.scene)
    scene = Scene.from_config(config, device=device, progress=print_progress)
    if scene.background is None:
        raise SystemExit("Scene has no background; cannot evaluate LiDAR against it.")
    return config, scene


def build_context(args, device) -> EvalContext:
    """Load scene + dataset, resolve alignment and render sensors into a context."""
    T4Devkit, LidarPointCloud = require_t4()
    rng = np.random.default_rng(args.seed)

    # --- scene ---------------------------------------------------------------
    config, scene = load_scene(args, device)
    usdz_path = config.background_usdz
    assert scene.background is not None  # guaranteed by load_scene
    centroid = (
        scene.background.tile_local_centroid.detach().cpu().numpy().astype(np.float64)
    )

    # --- dataset -------------------------------------------------------------
    dataset_dir = resolve_dataset_dir(args)
    print(f"[dataset] loading T4 dataset from {dataset_dir}")
    t4 = T4Devkit(dataset_dir, revision=args.revision, verbose=args.verbose)

    align = compute_alignment(args, t4, usdz_path)
    align_t = torch.from_numpy(align.astype(np.float32)).to(device)
    centroid_t = torch.from_numpy(centroid.astype(np.float32)).to(device)
    # map->world = align, then re-center to the Gaussians (subtract the centroid
    # from the translation). Constant across frames.
    map_to_world = align.copy()
    map_to_world[:3, 3] = map_to_world[:3, 3] - centroid
    ego_ts_us, ego_trans, ego_quat = ego_pose_table(t4)

    # --- sensor + renderer ---------------------------------------------------
    samples = list(t4.sample)
    if not samples:
        raise SystemExit("Dataset has no samples.")
    gt_channel = pick_lidar_channel(samples[0], args.lidar_channel)
    first_sd = t4.get("sample_data", samples[0].data[gt_channel])
    gt_cs = t4.get("calibrated_sensor", first_sd.calibrated_sensor_token)
    # GT scans live in their calibrated_sensor frame; this maps them to base_link
    # (for LIDAR_CONCAT that frame is base_link itself, i.e. an identity mount).
    gt_s2b = pose_to_matrix(gt_cs.translation, gt_cs.rotation).astype(np.float32)
    # First GT scan (in base_link) -- only needed to derive azimuth resolution,
    # so skip the extra file read unless --n-columns auto will use it.
    gt0_base = None
    if args.n_columns == "auto":
        gt0 = LidarPointCloud.from_file(t4.get_sample_data_path(first_sd.token))
        gt0_base = transform(gt_s2b, np.ascontiguousarray(gt0.points[:3].T, np.float32))

    # Render sensors come from the scene USDZ's own rig LiDAR calibration (mount
    # height + beam table), NOT the GT concat frame. All are rendered and unioned
    # to match the GT LIDAR_CONCAT -- see build_lidar_renderers.
    lidars = build_lidar_renderers(config, args, device, gt_base=gt0_base)
    print(f"[sensor] {len(lidars)} render LiDAR(s), GT channel={gt_channel}")
    for ld in lidars:
        print(
            f"  - {ld.name}: mount={ld.s2b[:3, 3].round(3).tolist()} "
            f"rows={ld.renderer.n_rows} cols={ld.renderer.n_columns} "
            f"range=[{ld.renderer.min_range_m}, {ld.renderer.max_range_m}] m"
        )

    return EvalContext(
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
