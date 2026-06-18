"""Reader helpers for the scene USDZ format emitted by ``3dgs_io.save_scene_usdz``.

A scene USDZ archive bundles:

* ``default.usda`` — USD stage referencing ``tileset.json`` and ``scene.json``.
* ``scene.json`` — splatsim.scene/v1 metadata (world transform, render defaults, ...).
* ``tileset.json`` — Cesium 3D Tiles document declaring ``EXT_3dgs_spz``.
* ``chunks/chunk_NNNNNN.spz`` — Niantic SPZ binaries, one per tile.

3dgs_io only ships a writer (``save_scene_usdz``); this module is splatsim's
reader.
"""

from __future__ import annotations

import importlib as _importlib
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from splatsim._conversions import GaussianTensors, cloud_to_tensors

_3dgs_io = _importlib.import_module("3dgs_io")
_load_spz = _3dgs_io.load_spz
_parse_rig_trajectories = _3dgs_io.parse_rig_trajectories


def read_scene_json(usdz_path: str | Path) -> dict[str, Any]:
    """Read ``scene.json`` out of a scene USDZ without extracting the whole archive."""
    with zipfile.ZipFile(usdz_path) as zf:
        if "scene.json" not in zf.namelist():
            raise ValueError(
                f"{usdz_path}: missing scene.json (not a 3dgs_io scene USDZ)"
            )
        return json.loads(zf.read("scene.json"))


def extract_scene_usdz(usdz_path: str | Path) -> Path:
    """Extract a scene USDZ to a fresh temp directory and return its path."""
    out_dir = Path(tempfile.mkdtemp(prefix="splatsim_usdz_"))
    with zipfile.ZipFile(usdz_path) as zf:
        zf.extractall(out_dir)
    return out_dir


def load_spz_tileset(
    tileset_path: str | Path,
    device: torch.device,
    *,
    use_sh: bool = False,
) -> tuple[GaussianTensors, np.ndarray]:
    """Load a Cesium 3D Tiles document whose children are SPZ files.

    Returns the concatenated :class:`GaussianTensors` together with the
    root tile's transform (row-major 4x4, ECEF→tile-local convention).
    """
    tileset_path = Path(tileset_path)
    base_dir = tileset_path.parent
    with tileset_path.open() as f:
        tileset = json.load(f)

    root = tileset["root"]
    # 3D Tiles stores transforms column-major; flip to row-major.
    root_transform = (
        np.asarray(
            root.get("transform", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
            dtype=np.float64,
        )
        .reshape(4, 4)
        .T
    )

    tensor_list: list[GaussianTensors] = []
    for child in root.get("children", []):
        chunk_uri = child["content"]["uri"]
        cloud = _load_spz(str(base_dir / chunk_uri))
        if cloud.num_points == 0:
            continue
        tensor_list.append(cloud_to_tensors(cloud, device, use_sh=use_sh))

    if not tensor_list:
        raise ValueError(f"{tileset_path}: no SPZ chunks found")

    merged = _concat_tensors(tensor_list)
    return merged, root_transform


def first_camera(rigs: list[Any]) -> Any | None:
    """Return the first camera nested in the first rig that has one."""
    for rig in rigs:
        if rig.cameras:
            return rig.cameras[0]
    return None


def camera_to_viewer_intrinsics(
    camera: Any,
) -> tuple[int | None, int | None, float | None]:
    """Approximate ``(width, height, fov_y_deg)`` from a ``3dgs_io.Camera``.

    splatsim only supports a pinhole camera, so non-pinhole models
    (``ftheta`` and similar fisheye lenses) are mapped to a best-effort
    vertical FOV derived from the model's ``max_angle`` parameter.
    """
    params = camera.camera_model.parameters or {}
    resolution = params.get("resolution")
    width: int | None = int(resolution[0]) if resolution else None
    height: int | None = int(resolution[1]) if resolution else None

    model_type = (camera.camera_model.type or "").lower()
    fov_y_deg: float | None = None
    if model_type == "pinhole":
        fy = params.get("fy")
        if fy and height:
            fov_y_deg = float(np.degrees(2 * np.arctan(height / (2 * float(fy)))))
    elif model_type == "ftheta":
        max_angle = params.get("max_angle")
        if max_angle is not None and width and height:
            # max_angle is the half-cone angle (radian). Approximate the
            # vertical FOV by scaling proportionally to the height of the
            # image diagonal.
            diag = float(np.hypot(width, height))
            fov_y_deg = float(np.degrees(2 * float(max_angle) * height / diag))

    return width, height, fov_y_deg


def read_rig_trajectories(usdz_path: str | Path, rig_uri: str) -> list[Any]:
    """Read ``rig_trajectories.json`` out of a scene USDZ and parse it.

    Returns ``list[3dgs_io.RigTrajectory]``; cameras live inside each rig
    under :attr:`RigTrajectory.cameras`.
    """
    with zipfile.ZipFile(usdz_path) as zf:
        if rig_uri not in zf.namelist():
            raise ValueError(f"{usdz_path}: missing {rig_uri}")
        doc = json.loads(zf.read(rig_uri))
    return _parse_rig_trajectories(doc)


def initial_camera_pose_from_rig_trajectories(
    rigs: list[Any],
) -> tuple[tuple[float, float, float], float] | None:
    """Compose the first rig pose with its first camera's extrinsics.

    Returns ``(world_position, yaw_deg)`` where yaw is the rotation of the
    composed sensor-in-world transform around the +Z (up) axis; the world
    position is in root-local frame (the same frame as the gaussians).
    Returns ``None`` if there is no rig or no camera.
    """
    for rig in rigs:
        if not rig.poses or not rig.cameras:
            continue
        rig_pose = rig.poses[0]
        cam = rig.cameras[0]

        r_rig = _quat_to_matrix(rig_pose.rotation)
        t_rig = np.asarray(rig_pose.translation, dtype=np.float64)
        r_sensor = _quat_to_matrix(cam.extrinsics.rotation)
        t_sensor = np.asarray(cam.extrinsics.translation, dtype=np.float64)

        # T_sensor_world = T_rig_world @ T_sensor_rig
        t_sensor_world = t_rig + r_rig @ t_sensor
        r_sensor_world = r_rig @ r_sensor

        # splatsim yaw rotates around +Z; at yaw=0 the camera looks along
        # world -Y. ``CameraExtrinsics`` follows the OpenGL/glTF convention,
        # so the camera looks along its own -Z. World forward direction:
        fwd_world = r_sensor_world @ np.array([0.0, 0.0, -1.0])
        # Project to the horizontal plane for the yaw. Falls back to 0
        # for cameras pointing nearly straight up/down (degenerate yaw).
        horiz = float(np.hypot(fwd_world[0], fwd_world[1]))
        if horiz < 1e-6:
            yaw_deg = 0.0
        else:
            yaw_rad = float(np.arctan2(fwd_world[0], -fwd_world[1]))
            yaw_deg = float(np.degrees(yaw_rad))

        position = (
            float(t_sensor_world[0]),
            float(t_sensor_world[1]),
            float(t_sensor_world[2]),
        )
        return position, yaw_deg
    return None


def _quat_to_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _concat_tensors(tensors: list[GaussianTensors]) -> GaussianTensors:
    sh_degrees = {t.sh_degree for t in tensors}
    if len(sh_degrees) != 1:
        raise ValueError(f"Mixed SH degrees across chunks: {sh_degrees}")
    sh_degree = sh_degrees.pop()
    return GaussianTensors(
        means=torch.cat([t.means for t in tensors], dim=0),
        quats=torch.cat([t.quats for t in tensors], dim=0),
        scales=torch.cat([t.scales for t in tensors], dim=0),
        opacities=torch.cat([t.opacities for t in tensors], dim=0),
        colors=torch.cat([t.colors for t in tensors], dim=0),
        sh_degree=sh_degree,
    )
