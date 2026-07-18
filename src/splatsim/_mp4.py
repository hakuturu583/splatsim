"""Offline MP4 rendering along a scene USDZ's rig camera trajectory."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from splatsim._usdz import (
    camera_intrinsics_K,
    first_camera,
    iter_world_to_camera_interpolated,
    read_rig_trajectories,
    read_scene_json,
)

if TYPE_CHECKING:
    from splatsim.renderer import Renderer
    from splatsim.scene import Scene


def render_trajectory_mp4(
    scene: Scene,
    renderer: Renderer,
    usdz_path: str | Path,
    *,
    output_path: str | Path,
    camera_name: str | None = None,
    fps: int = 30,
) -> int:
    """Render ``scene`` along the rig camera's GT trajectory to an MP4 file.

    The trajectory and intrinsics come from the USDZ's ``rig_trajectories``
    sidecar. ``camera_name`` selects which rig camera (defaults to the first
    one); it must match the camera used to seed ``renderer.width``/``height``
    so that resolutions agree.

    Returns the number of frames written.
    """
    usdz_path = Path(usdz_path)
    output_path = Path(output_path)

    meta = read_scene_json(usdz_path)
    rig_uri = meta.get("extras", {}).get("rig_trajectories")
    if not rig_uri:
        raise ValueError(
            f"{usdz_path}: scene.json has no rig_trajectories sidecar; "
            "--mp4 requires a USDZ exported with rig trajectories"
        )
    rigs = read_rig_trajectories(usdz_path, rig_uri)

    cam = first_camera(rigs, name=camera_name)
    if cam is None:
        raise ValueError(f"{usdz_path}: no camera found in rig_trajectories")

    K_np, width, height = camera_intrinsics_K(cam)
    if renderer.width != width or renderer.height != height:
        raise ValueError(
            f"renderer size ({renderer.width}x{renderer.height}) does not "
            f"match camera {cam.name!r} resolution ({width}x{height})"
        )

    K = torch.tensor(K_np, device=renderer.device, dtype=torch.float32)

    if scene.background is None:
        raise ValueError("scene.background must be loaded before rendering MP4")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty: ignore[unresolved-attribute]
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {output_path}")

    frame_count = 0
    try:
        for _ts, w2c in iter_world_to_camera_interpolated(
            rigs,
            background=scene.background,
            fps=float(fps),
            name=camera_name,
        ):
            viewmat = torch.tensor(w2c, device=renderer.device, dtype=torch.float32)
            with torch.no_grad():
                rgb = renderer.render(viewmat, K, scene=scene, camera_name=cam.name)
            rgb_u8 = (rgb.clamp(0.0, 1.0) * 255).byte().cpu().numpy()
            bgr_u8 = np.ascontiguousarray(rgb_u8[:, :, ::-1])
            writer.write(bgr_u8)
            frame_count += 1
            if frame_count % 30 == 0:
                sys.stderr.write(f"\r  Rendering MP4: frame {frame_count}")
                sys.stderr.flush()
    finally:
        writer.release()
        if frame_count >= 30:
            sys.stderr.write("\n")
            sys.stderr.flush()

    if frame_count == 0:
        raise ValueError(
            f"{usdz_path}: rig_trajectories has no poses for camera "
            f"{cam.name!r}; no frames were rendered"
        )

    return frame_count
