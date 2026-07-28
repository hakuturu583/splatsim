#!/usr/bin/env python3
"""Evaluate splatsim LiDAR rendering against a WebAuto / T4 dataset.

The splatsim scene (a reconstructed 3DGS ``.usdz``) carries **no** ground-truth
LiDAR. This script pulls the ground truth from the matching WebAuto / T4 dataset
instead: it walks the dataset's GT ego trajectory, renders a LiDAR panorama from
the splat scene at every GT pose, and scores the rendered scan against the
recorded GT scan with a set of pluggable **metrics** (see :mod:`eval.metrics`).
Everything -- the two point clouds and every metric's signals -- is logged to a
single `Rerun` recording (``.rrd``) so geometry and metrics share a timeline.

Metrics (``--metrics``, default: all):

* ``chamfer`` -- symmetric Chamfer distance (raw + range-aware), in metres.
* ``bev`` -- OnePlanner BEV-encoder feature similarity: both clouds are pushed
  through the BEV encoder and the ``[512, 180, 180]`` feature maps are compared
  (cosine / relative-L2), with PCA-RGB and cosine-heatmap views logged as images.

Two corrections keep the comparison fair against a *static* reconstruction:

* **Rolling shutter.** A spinning LiDAR paints its panorama over a finite sweep
  (~100 ms) while the ego is moving. The render mirrors this by interpolating the
  sweep-end ego pose (``--no-rolling-shutter`` to disable).
* **Dynamic-object masking.** GT/rendered points inside the frame's annotated 3D
  boxes are dropped before scoring so the metrics reflect static geometry only
  (``--no-mask-dynamic`` to disable).

Dependencies are optional and layered behind extras::

    uv sync --extra eval   # t4-devkit + rerun-sdk  (chamfer metric)
    uv sync --extra bev    # + tensorrt             (bev metric)

The BEV metric additionally needs the OnePlanner encoder ONNX and the
``autoware_tensorrt_plugins`` shared library; point at them with ``--bev-onnx`` /
``--bev-plugins`` or the ``$ONEPLANNER_BEV_ONNX`` / ``$ONEPLANNER_TRT_PLUGINS``
environment variables.

Example
-------
::

    uv run python -m eval.eval_lidar \
        --scene /path/to/scene.usdz \
        --data-root ~/.webauto/datasets \
        --dataset-id 0123abcd-... \
        --bev-onnx ~/models/oneplanner_bev_encoder.onnx \
        --bev-plugins .../libautoware_tensorrt_plugins.so \
        --output outputs/eval_lidar.rrd

Coordinate frames
-----------------
* The T4 dataset ``ego_pose`` is the base_link pose in the dataset *map* frame
  (ENU, z-up, ROS base_link = x-forward/y-left/z-up).
* The splat scene lives in an ENU world re-centered to the background's
  ``tile_local_centroid`` for numerical stability.
* ``--align`` bridges the two (see :func:`eval.dataset.compute_alignment`).

The render sensor's mount + beam table come from the scene USDZ's own rig LiDAR
calibration, not the T4 ``LIDAR_CONCAT`` extrinsic. Rendered and GT scans are
both mapped into base_link, where the metrics are computed, and into the scene
world frame for the Rerun 3D view.
"""

from __future__ import annotations

import argparse
import sys

from .runner import run


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
        help="LiDAR sweep duration in seconds for rolling shutter (default: 0.1).",
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

    align = p.add_argument_group("alignment (T4 map -> splat world)")
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

    bev = p.add_argument_group("bev encoder metric")
    bev.add_argument(
        "--bev-backend",
        default="spconv",
        choices=("spconv", "tensorrt"),
        help="BEV encoder backend: 'spconv' (default, onnx2torch + spconv, "
        "pure-pip) or 'tensorrt' (needs --bev-plugins).",
    )
    bev.add_argument(
        "--bev-onnx",
        default=None,
        help="Path to oneplanner_bev_encoder.onnx (or $ONEPLANNER_BEV_ONNX).",
    )
    bev.add_argument(
        "--bev-plugins",
        default=None,
        help="Path to libautoware_tensorrt_plugins.so (or $ONEPLANNER_TRT_PLUGINS).",
    )
    bev.add_argument(
        "--bev-engine-cache",
        default=None,
        help="Path to cache the built TensorRT engine (default: next to the ONNX).",
    )
    bev.add_argument(
        "--bev-fp16",
        action="store_true",
        help="Build the BEV TensorRT engine with FP16 (faster, slightly lossy).",
    )
    bev.add_argument(
        "--bev-no-intensity",
        action="store_true",
        help="Zero the intensity channel fed to the BEV encoder.",
    )

    ev = p.add_argument_group("evaluation / output")
    ev.add_argument(
        "--metrics",
        default="",
        help="Comma-separated metrics to run: chamfer, bev (default: all).",
    )
    ev.add_argument(
        "--output", default="outputs/eval_lidar.rrd", help="Output .rrd path."
    )
    ev.add_argument(
        "--mask-dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop GT and rendered points inside the frame's annotated 3D boxes "
        "before scoring (default: on; --no-mask-dynamic to score raw clouds).",
    )
    ev.add_argument(
        "--dynamic-margin",
        type=float,
        default=0.25,
        help="Grow each dynamic box by this many metres per side when masking "
        "(default: 0.25).",
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
