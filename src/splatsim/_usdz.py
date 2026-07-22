"""Reader helpers for the scene USDZ format emitted by ``3dgs_io.save_scene_usdz``.

A scene USDZ archive bundles:

* ``default.usda`` — USD stage referencing ``scene.json`` and the SPZ chunks.
* ``scene.json`` — splatsim.scene/v2 metadata (``world.ecef_anchor``, render
  defaults, ``extras`` sidecar references, ...).
* ``chunks/chunk_NNNNNN.spz`` — Niantic SPZ binaries whose numeric axes are
  already baked in the scene's Z-up ENU world frame.

3dgs_io only ships a writer (``save_scene_usdz``); this module is splatsim's
reader.
"""

from __future__ import annotations

import importlib as _importlib
import json
import tempfile
import typing
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from splatsim._conversions import GaussianTensors, cloud_to_tensors

if typing.TYPE_CHECKING:
    from splatsim.background import Background

_3dgs_io = _importlib.import_module("3dgs_io")
_frame_convention = _importlib.import_module("3dgs_io.frame_convention")
_spz_io = _importlib.import_module("3dgs_io.spz_io")
_decode_lidar_sidecar = _3dgs_io.decode_lidar_sidecar
_load_spz = _spz_io.load_spz_world
_parse_rig_trajectories = _3dgs_io.parse_rig_trajectories
_validate_frame_convention = _frame_convention.validate_frame_convention
_validate_rigid_transform = _frame_convention.validate_rigid_transform


def read_scene_json(usdz_path: str | Path) -> dict[str, Any]:
    """Read and validate a frame-explicit v2 ``scene.json``."""
    with zipfile.ZipFile(usdz_path) as zf:
        if "scene.json" not in zf.namelist():
            raise ValueError(
                f"{usdz_path}: missing scene.json (not a 3dgs_io scene USDZ)"
            )
        meta = json.loads(zf.read("scene.json"))
    if not isinstance(meta, dict):
        raise ValueError(f"{usdz_path}: scene.json must be a JSON object")
    if meta.get("schema") != "splatsim.scene/v2":
        raise ValueError(f"{usdz_path}: scene.json must use splatsim.scene/v2")
    world = meta.get("world")
    if not isinstance(world, dict):
        raise ValueError(f"{usdz_path}: scene.json is missing world metadata")
    _validate_frame_convention(world.get("frame_convention"))
    _validate_rigid_transform(world.get("ecef_anchor"), where="scene world.ecef_anchor")
    gaussians = meta.get("gaussians")
    if not isinstance(gaussians, dict) or gaussians.get("frame") != "world":
        raise ValueError(f"{usdz_path}: scene gaussians must use the world frame")
    return meta


def load_spz_scene(
    usdz_path: str | Path,
    device: torch.device,
    *,
    use_sh: bool = False,
) -> tuple[GaussianTensors, np.ndarray]:
    """Load the SPZ gaussian chunks bundled in a scene USDZ.

    Returns the concatenated :class:`GaussianTensors` — baked in the scene's
    Z-up ENU world frame, so no coordinate transform is applied — together
    with the scene's ``ecef_anchor`` (row-major 4x4, ENU world→ECEF) read
    from ``scene.json`` (``world.ecef_anchor``).

    The embedded ``tileset.json`` is deliberately not interpreted. In the v2
    format every SPZ chunk already contains world-frame coordinates, so the
    chunks can be loaded directly without Cesium tile transforms.
    """
    usdz_path = Path(usdz_path)
    meta = read_scene_json(usdz_path)
    ecef_anchor = _validate_rigid_transform(
        meta["world"]["ecef_anchor"], where="scene world.ecef_anchor"
    )
    ext_meta = meta["gaussians"].get("ext_attributes")
    sidecar_suffix: str | None = None
    if ext_meta is not None:
        if not isinstance(ext_meta, dict):
            raise ValueError(f"{usdz_path}: gaussians.ext_attributes must be an object")
        if ext_meta.get("extension") != "EXT_gaussian_lidar":
            raise ValueError(
                f"{usdz_path}: unsupported gaussian extension "
                f"{ext_meta.get('extension')!r}"
            )
        sidecar_suffix = ext_meta.get("sidecar_suffix")
        if not isinstance(sidecar_suffix, str) or not sidecar_suffix.startswith("."):
            raise ValueError(f"{usdz_path}: invalid gaussian extension sidecar_suffix")

    tensor_list: list[GaussianTensors] = []
    with zipfile.ZipFile(usdz_path) as zf:
        chunk_names = sorted(
            n for n in zf.namelist() if n.startswith("chunks/") and n.endswith(".spz")
        )
        if not chunk_names:
            raise ValueError(f"{usdz_path}: no SPZ chunks found under chunks/")
        for name in chunk_names:
            with tempfile.NamedTemporaryFile(suffix=".spz") as tmp:
                tmp.write(zf.read(name))
                tmp.flush()
                cloud = _load_spz(tmp.name)
            if cloud.num_points == 0:
                continue
            tensors = cloud_to_tensors(cloud, device, use_sh=use_sh)
            if sidecar_suffix is not None:
                sidecar_name = str(Path(name).with_suffix(sidecar_suffix))
                if sidecar_name not in zf.namelist():
                    raise ValueError(
                        f"{usdz_path}: missing LiDAR sidecar {sidecar_name}"
                    )
                attrs = _decode_lidar_sidecar(zf.read(sidecar_name))
                intensity = attrs.get("lidar_intensity_raw")
                raydrop = attrs.get("lidar_raydrop_logit")
                if intensity is None or raydrop is None:
                    raise ValueError(
                        f"{usdz_path}: incomplete LiDAR attributes in {sidecar_name}"
                    )
                if (
                    len(intensity) != cloud.num_points
                    or len(raydrop) != cloud.num_points
                ):
                    raise ValueError(
                        f"{usdz_path}: LiDAR sidecar count does not match {name}"
                    )
                tensors.intensity_raw = torch.from_numpy(intensity).to(device)
                tensors.raydrop_logit = torch.from_numpy(raydrop).to(device)
            tensor_list.append(tensors)

    if not tensor_list:
        raise ValueError(f"{usdz_path}: SPZ chunks contain no gaussians")

    return _concat_tensors(tensor_list), ecef_anchor


def first_camera(rigs: list[Any], name: str | None = None) -> Any | None:
    """Return a camera from the rigs.

    If ``name`` is ``None``, picks the camera whose OpenCV +Z (forward) is
    most aligned with the vehicle's forward axis (rig +X). This avoids
    initializing the viewer with a back- or side-facing camera just because
    it sorts first in ``rig_trajectories.json``. Ties (and fully unaligned
    rigs) fall back to the first camera in the first rig that has one.

    If ``name`` is given, returns the camera whose ``name`` attribute
    matches; raises ``ValueError`` if no such camera exists, including the
    available names in the message.
    """
    if name is None:
        best: tuple[float, Any] | None = None
        fallback: Any | None = None
        for rig in rigs:
            for cam in rig.cameras or []:
                if fallback is None:
                    fallback = cam
                # extrinsics.to_matrix() is sensor-in-rig; the camera's
                # optical axis (sensor +Z) expressed in rig coords is its
                # third column. Vehicle forward is rig +X, so the X
                # component of that column measures forwardness.
                try:
                    forwardness = float(cam.extrinsics.to_matrix()[0, 2])
                except Exception:
                    continue
                if best is None or forwardness > best[0]:
                    best = (forwardness, cam)
        if best is not None and best[0] > 0.5:
            return best[1]
        return fallback

    available: list[str] = []
    for rig in rigs:
        for cam in rig.cameras or []:
            if cam.name == name:
                return cam
            available.append(cam.name)
    raise ValueError(
        f"camera {name!r} not found in rig_trajectories; available: {available}"
    )


def camera_to_viewer_intrinsics(
    camera: Any,
) -> tuple[int | None, int | None, float | None]:
    """Approximate ``(width, height, fov_y_deg)`` from a ``3dgs_io.Camera``.

    splatsim only supports a pinhole projection, so ``fov_y_deg`` is derived
    from ``fy`` for ``pinhole`` and ``opencv`` models. Other models (e.g.
    ``ftheta`` fisheye) degrade gracefully to ``fov_y_deg=None``.
    """
    params = camera.camera_model.parameters or {}
    resolution = params.get("resolution")
    width: int | None = int(resolution[0]) if resolution else None
    height: int | None = int(resolution[1]) if resolution else None

    model_type = (camera.camera_model.type or "").lower()
    fov_y_deg: float | None = None
    if model_type in ("pinhole", "opencv"):
        fy = params.get("fy")
        if fy and height:
            fov_y_deg = float(np.degrees(2 * np.arctan(height / (2 * float(fy)))))

    return width, height, fov_y_deg


def camera_intrinsics_K(camera: Any) -> tuple[np.ndarray, int, int]:
    """Return ``(K, width, height)`` for a pinhole/opencv rig camera.

    ``K`` is a 3x3 OpenCV intrinsic matrix in pixel units; ``width``/``height``
    are the image resolution. Raises ``ValueError`` if the camera is not a
    ``pinhole``/``opencv`` model or lacks the required parameters.
    """
    params = camera.camera_model.parameters or {}
    model_type = (camera.camera_model.type or "").lower()
    if model_type not in ("pinhole", "opencv"):
        raise ValueError(
            f"camera {camera.name!r}: K-matrix render requires a pinhole "
            f"or opencv camera, got {model_type!r}"
        )
    resolution = params.get("resolution")
    if not resolution or len(resolution) < 2:
        raise ValueError(f"camera {camera.name!r}: missing resolution")
    width = int(resolution[0])
    height = int(resolution[1])
    fx_raw = params.get("fx")
    fy_raw = params.get("fy")
    if fx_raw is None:
        fx_raw = fy_raw
    if fy_raw is None:
        fy_raw = fx_raw
    if fx_raw is None or fy_raw is None:
        raise ValueError(f"camera {camera.name!r}: missing fx/fy")
    fx = float(fx_raw)
    fy = float(fy_raw)
    cx = float(params.get("cx", width / 2.0))
    cy = float(params.get("cy", height / 2.0))
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K, width, height


def iter_world_to_camera_uncentered(
    rigs: list[Any], *, name: str | None = None
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(timestamp, world_to_camera)`` per rig pose for one camera.

    ``world_to_camera`` is a 4x4 OpenCV extrinsic (``+Z`` forward) that maps
    points in the ENU world frame to the camera frame. This is the frame the
    gaussians live in *before* :class:`Background` re-centers them, so this
    iterator is rarely what callers want directly — prefer
    :func:`iter_world_to_camera`, which compensates for the re-centering.

    The camera pose in world is ``rig_in_world @ sensor_in_rig`` and
    ``world_to_camera`` is its inverse.
    """
    rig, cam = _find_rig_with_camera(rigs, name=name)
    if rig is None or cam is None:
        return
    for pose in rig.poses:
        yield float(pose.timestamp_us), np.linalg.inv(_camera_in_world(pose, cam))


def _apply_tile_local_centroid(w2c: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Shift a world-frame OpenCV w2c so it lines up with a re-centered scene."""
    w2c = w2c.copy()
    w2c[:3, 3] = w2c[:3, 3] + w2c[:3, :3] @ centroid
    return w2c


def iter_world_to_camera(
    rigs: list[Any], *, background: Background, name: str | None = None
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(timestamp, world_to_camera)`` aligned with ``background``.

    Wraps :func:`iter_world_to_camera_uncentered` and shifts the translation
    column by ``R_w2c @ background.tile_local_centroid`` so the resulting
    camera lines up with the gaussians stored on the background (which are
    re-centered to their tile-local centroid for numerical stability).
    """
    centroid = background.tile_local_centroid.detach().cpu().numpy().astype(np.float64)
    for ts, w2c in iter_world_to_camera_uncentered(rigs, name=name):
        yield ts, _apply_tile_local_centroid(w2c, centroid)


def iter_world_to_camera_interpolated(
    rigs: list[Any],
    *,
    background: Background,
    fps: float,
    name: str | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Interpolated counterpart of :func:`iter_world_to_camera`.

    Same SLERP/LERP sampling as
    :func:`iter_world_to_camera_interpolated_uncentered`, with the
    background centroid compensation applied so the result is ready to
    render against a :class:`Background`-loaded scene.
    """
    centroid = background.tile_local_centroid.detach().cpu().numpy().astype(np.float64)
    for ts, w2c in iter_world_to_camera_interpolated_uncentered(
        rigs, fps=fps, name=name
    ):
        yield ts, _apply_tile_local_centroid(w2c, centroid)


def iter_world_to_camera_interpolated_uncentered(
    rigs: list[Any], *, name: str | None = None, fps: float
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(timestamp_us, world_to_camera)`` sampled uniformly at ``fps``.

    Same world-frame convention as :func:`iter_world_to_camera_uncentered` —
    prefer :func:`iter_world_to_camera_interpolated` for rendering against
    a :class:`Background`-loaded scene.

    Spans the full GT timeline of the selected camera; rotations are SLERPed
    and translations are linearly interpolated between adjacent GT poses so
    that playback wall-clock matches the captured trajectory regardless of
    its native cadence.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    target_rig, target_cam = _find_rig_with_camera(rigs, name=name)
    if target_rig is None or target_cam is None:
        return

    timestamps = np.asarray(
        [pose.timestamp_us for pose in target_rig.poses], dtype=np.float64
    )
    translations = np.asarray(
        [pose.translation for pose in target_rig.poses], dtype=np.float64
    )
    quaternions = np.asarray(
        [pose.rotation for pose in target_rig.poses], dtype=np.float64
    )

    sensor_in_rig = np.asarray(target_cam.extrinsics.to_matrix(), dtype=np.float64)

    n_poses = timestamps.shape[0]
    if n_poses == 1:
        r_rig = _quat_to_matrix(tuple(quaternions[0]))
        yield float(timestamps[0]), _compose_w2c(sensor_in_rig, r_rig, translations[0])
        return

    t_first = float(timestamps[0])
    t_last = float(timestamps[-1])
    step_us = 1_000_000.0 / float(fps)
    n_samples = int(np.floor((t_last - t_first) / step_us)) + 1

    for k in range(n_samples):
        t = t_first + k * step_us
        i = int(np.searchsorted(timestamps, t, side="right")) - 1
        i = max(0, min(i, n_poses - 2))
        ts0 = timestamps[i]
        ts1 = timestamps[i + 1]
        alpha = (t - ts0) / (ts1 - ts0) if ts1 > ts0 else 0.0
        alpha = float(max(0.0, min(1.0, alpha)))

        t_rig = (1.0 - alpha) * translations[i] + alpha * translations[i + 1]
        q = _slerp(quaternions[i], quaternions[i + 1], alpha)
        r_rig = _quat_to_matrix(tuple(q))

        yield float(t), _compose_w2c(sensor_in_rig, r_rig, t_rig)


def _find_rig_with_camera(
    rigs: list[Any], *, name: str | None
) -> tuple[Any | None, Any | None]:
    """Return the first ``(rig, camera)`` matching ``name`` (or the first available)."""
    available: list[str] = []
    if name is None:
        camera = first_camera(rigs)
        if camera is None:
            return None, None
        for rig in rigs:
            if rig.poses and any(cam is camera for cam in rig.cameras or []):
                return rig, camera
        return None, None
    for rig in rigs:
        if not rig.poses or not rig.cameras:
            available.extend(c.name for c in rig.cameras or [])
            continue
        cam = next((c for c in rig.cameras if c.name == name), None)
        if cam is None:
            available.extend(c.name for c in rig.cameras)
            continue
        return rig, cam
    if name is not None:
        raise ValueError(
            f"camera {name!r} not found in rig_trajectories; available: {available}"
        )
    return None, None


def _mat4(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 rigid transform from a 3x3 rotation and a translation."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


def _rig_in_world(pose: Any) -> np.ndarray:
    """Return the 4x4 rig-in-world transform for a ``RigPose`` (ENU world)."""
    return _mat4(
        _quat_to_matrix(pose.rotation),
        np.asarray(pose.translation, dtype=np.float64),
    )


def _camera_in_world(pose: Any, cam: Any) -> np.ndarray:
    """Compose ``rig_in_world @ sensor_in_rig`` into a 4x4 camera-in-world."""
    return _rig_in_world(pose) @ np.asarray(
        cam.extrinsics.to_matrix(), dtype=np.float64
    )


def _compose_w2c(
    sensor_in_rig: np.ndarray, r_rig: np.ndarray, t_rig: np.ndarray
) -> np.ndarray:
    """Compose a rig pose with sensor-in-rig into a 4x4 world→camera matrix."""
    camera_in_world = _mat4(r_rig, t_rig) @ sensor_in_rig
    return np.linalg.inv(camera_in_world)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """SLERP between two ``(x, y, z, w)`` unit quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = (1.0 - t) * q0 + t * q1
        return result / np.linalg.norm(result)
    theta = float(np.arccos(dot))
    sin_theta = float(np.sin(theta))
    s0 = float(np.sin((1.0 - t) * theta) / sin_theta)
    s1 = float(np.sin(t * theta) / sin_theta)
    return s0 * q0 + s1 * q1


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


def load_rig_trajectories(usdz_path: str | Path) -> list[Any]:
    """Return the rigs referenced by ``scene.json.extras.rig_trajectories``.

    Returns ``[]`` when the archive has no scene metadata, no
    ``rig_trajectories`` extra, or the referenced sidecar is missing.
    """
    try:
        meta = read_scene_json(usdz_path)
    except (OSError, KeyError, ValueError):
        return []
    rig_uri = meta.get("extras", {}).get("rig_trajectories")
    if not rig_uri:
        return []
    try:
        return read_rig_trajectories(usdz_path, rig_uri)
    except (OSError, KeyError, ValueError):
        return []


def initial_camera_pose_from_rig_trajectories(
    rigs: list[Any],
    name: str | None = None,
) -> tuple[tuple[float, float, float], float] | None:
    """Compose the first rig pose with the chosen camera's extrinsics.

    If ``name`` is ``None``, picks the camera whose optical axis is most
    forward-facing in the rig. If ``name`` is given, picks the camera whose
    ``name`` attribute matches and uses the first pose of the rig that owns
    it; raises ``ValueError`` if no such camera exists.

    Returns ``(world_position, yaw_deg)`` where yaw is the rotation of the
    composed sensor-in-world transform around the +Z (up) axis; the world
    position is in the ENU world frame (the same frame as the gaussians).
    Returns ``None`` if there is no rig or no camera.
    """
    rig, cam = _find_rig_with_camera(rigs, name=name)
    if rig is None or cam is None:
        return None

    # sensor_in_world = rig_in_world @ sensor_in_rig
    camera_in_world = _camera_in_world(rig.poses[0], cam)
    r_sensor_world = camera_in_world[:3, :3]
    t_sensor_world = camera_in_world[:3, 3]

    # splatsim yaw rotates around +Z; at yaw=0 the camera looks along
    # world -Y. With OpenCV extrinsics the camera looks along sensor +Z.
    fwd_world = r_sensor_world @ np.array([0.0, 0.0, 1.0])
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


def _quat_to_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """Convert an ``(x, y, z, w)`` quaternion to a 3x3 rotation matrix.

    ``3dgs_io.parse_rig_trajectories`` exposes rotations in ``(x, y, z, w)``
    order (the same convention as scipy / Eigen / glTF).
    """
    x, y, z, w = q
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
    # LiDAR attributes concatenate only if every chunk carries them; otherwise
    # drop to None so the renderer falls back uniformly.
    if all(t.intensity_raw is not None for t in tensors):
        intensity_raw = torch.cat(
            [t.intensity_raw for t in tensors if t.intensity_raw is not None], dim=0
        )
    else:
        intensity_raw = None
    if all(t.raydrop_logit is not None for t in tensors):
        raydrop_logit = torch.cat(
            [t.raydrop_logit for t in tensors if t.raydrop_logit is not None], dim=0
        )
    else:
        raydrop_logit = None
    return GaussianTensors(
        means=torch.cat([t.means for t in tensors], dim=0),
        quats=torch.cat([t.quats for t in tensors], dim=0),
        scales=torch.cat([t.scales for t in tensors], dim=0),
        opacities=torch.cat([t.opacities for t in tensors], dim=0),
        colors=torch.cat([t.colors for t in tensors], dim=0),
        sh_degree=sh_degree,
        intensity_raw=intensity_raw,
        raydrop_logit=raydrop_logit,
    )
