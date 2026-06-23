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

import gzip
import importlib as _importlib
import json
import struct
import tempfile
import zipfile
from collections.abc import Iterator
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
        chunk_path = base_dir / chunk_uri
        cloud = _load_spz(str(chunk_path))
        if cloud.num_points == 0:
            continue
        tensors = cloud_to_tensors(cloud, device, use_sh=use_sh)
        # SPZ stores positions as 24-bit signed fixed-point, so positions
        # outside ±(2^24 / 2^fractionalBits / 2) wrap around. For scenes
        # authored in large local coordinates (e.g. MGRS-relative, ~10^5 m),
        # 3dgs_io's SPZ writer silently emits wrapped positions. Use the
        # child tile's boundingVolume center to detect the wrap offset and
        # restore root_local coordinates.
        bbox_center = _child_bbox_center(child)
        if bbox_center is not None:
            shift = _spz_wrap_shift(chunk_path, tensors.means, bbox_center)
            if shift is not None:
                tensors.means = tensors.means + shift.to(tensors.means)
        tensor_list.append(tensors)

    if not tensor_list:
        raise ValueError(f"{tileset_path}: no SPZ chunks found")

    merged = _concat_tensors(tensor_list)
    return merged, root_transform


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
                # r_sensor_rig is rig->sensor (memory: 3dgs_io extrinsic
                # convention). Sensor +Z expressed in rig coords is the
                # third column of r_sensor_rig.T, i.e. the third row of
                # r_sensor_rig. Vehicle forward is rig +X, so the X
                # component of that vector measures forwardness.
                try:
                    r_sr = _quat_to_matrix(cam.extrinsics.rotation)
                except Exception:
                    continue
                fwd_x_in_rig = float(r_sr[2, 0])
                if best is None or fwd_x_in_rig > best[0]:
                    best = (fwd_x_in_rig, cam)
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


def camera_intrinsics_K(camera: Any) -> tuple[np.ndarray, int, int]:
    """Return ``(K, width, height)`` for a pinhole rig camera.

    ``K`` is a 3x3 OpenCV intrinsic matrix in pixel units; ``width``/``height``
    are the image resolution. Raises ``ValueError`` if the camera is not
    pinhole or lacks the required parameters.
    """
    params = camera.camera_model.parameters or {}
    model_type = (camera.camera_model.type or "").lower()
    if model_type != "pinhole":
        raise ValueError(
            f"camera {camera.name!r}: K-matrix render requires a pinhole "
            f"camera, got {model_type!r}"
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


def iter_world_to_camera_root_local(
    rigs: list[Any], *, name: str | None = None
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(timestamp, world_to_camera)`` per rig pose for one camera.

    ``world_to_camera`` is a 4x4 OpenCV extrinsic (``+Z`` forward) that maps
    points in the *root-local* world frame (the frame the gaussians live in
    before :class:`Background` re-centers them) to the camera frame. Callers
    rendering against a tile-local scene must add ``R_w2c @ background.origin``
    to the translation column.

    The composition mirrors :func:`initial_camera_pose_from_rig_trajectories`:
    despite the ``T_sensor_rig`` field name, ``3dgs_io`` stores the rig→sensor
    matrix in OpenCV convention, so we invert it implicitly via
    ``R_w2c = R_rs @ R_rig.T`` and ``t_w2c = t_rs - R_w2c @ t_rig``.
    """
    available: list[str] = []
    for rig in rigs:
        if not rig.poses or not rig.cameras:
            available.extend(c.name for c in rig.cameras or [])
            continue
        if name is None:
            cam = rig.cameras[0]
        else:
            cam = next((c for c in rig.cameras if c.name == name), None)
            if cam is None:
                available.extend(c.name for c in rig.cameras)
                continue
        r_rs = _quat_to_matrix(cam.extrinsics.rotation)
        t_rs = np.asarray(cam.extrinsics.translation, dtype=np.float64)
        for pose in rig.poses:
            r_rig = _quat_to_matrix(pose.rotation)
            t_rig = np.asarray(pose.translation, dtype=np.float64)
            r_w2c = r_rs @ r_rig.T
            t_w2c = t_rs - r_w2c @ t_rig
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = r_w2c
            w2c[:3, 3] = t_w2c
            yield float(pose.timestamp_us), w2c
        return
    if name is not None:
        raise ValueError(
            f"camera {name!r} not found in rig_trajectories; available: {available}"
        )


def iter_world_to_camera_interpolated(
    rigs: list[Any], *, name: str | None = None, fps: float
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(timestamp_us, world_to_camera)`` sampled uniformly at ``fps``.

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

    r_rs = _quat_to_matrix(target_cam.extrinsics.rotation)
    t_rs = np.asarray(target_cam.extrinsics.translation, dtype=np.float64)

    n_poses = timestamps.shape[0]
    if n_poses == 1:
        r_rig = _quat_to_matrix(tuple(quaternions[0]))
        yield float(timestamps[0]), _compose_w2c(r_rs, t_rs, r_rig, translations[0])
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

        yield float(t), _compose_w2c(r_rs, t_rs, r_rig, t_rig)


def _find_rig_with_camera(
    rigs: list[Any], *, name: str | None
) -> tuple[Any | None, Any | None]:
    """Return the first ``(rig, camera)`` matching ``name`` (or the first available)."""
    available: list[str] = []
    for rig in rigs:
        if not rig.poses or not rig.cameras:
            available.extend(c.name for c in rig.cameras or [])
            continue
        if name is None:
            return rig, rig.cameras[0]
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


def _compose_w2c(
    r_rs: np.ndarray, t_rs: np.ndarray, r_rig: np.ndarray, t_rig: np.ndarray
) -> np.ndarray:
    """Compose rig→sensor (OpenCV) with rig pose into a 4x4 world→camera matrix."""
    r_w2c = r_rs @ r_rig.T
    t_w2c = t_rs - r_w2c @ t_rig
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = r_w2c
    w2c[:3, 3] = t_w2c
    return w2c


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


def initial_camera_pose_from_rig_trajectories(
    rigs: list[Any],
    name: str | None = None,
) -> tuple[tuple[float, float, float], float] | None:
    """Compose the first rig pose with the chosen camera's extrinsics.

    If ``name`` is ``None``, picks the first camera of the first rig that has
    any. If ``name`` is given, picks the camera whose ``name`` attribute
    matches and uses the first pose of the rig that owns it; raises
    ``ValueError`` if no such camera exists.

    Returns ``(world_position, yaw_deg)`` where yaw is the rotation of the
    composed sensor-in-world transform around the +Z (up) axis; the world
    position is in root-local frame (the same frame as the gaussians).
    Returns ``None`` if there is no rig or no camera.
    """
    available: list[str] = []
    for rig in rigs:
        if not rig.poses or not rig.cameras:
            if name is not None:
                available.extend(c.name for c in rig.cameras or [])
            continue
        if name is None:
            cam = rig.cameras[0]
        else:
            cam = next((c for c in rig.cameras if c.name == name), None)
            if cam is None:
                available.extend(c.name for c in rig.cameras)
                continue
        rig_pose = rig.poses[0]

        r_rig = _quat_to_matrix(rig_pose.rotation)
        t_rig = np.asarray(rig_pose.translation, dtype=np.float64)
        # Despite the ``T_sensor_rig`` field name, 3dgs_io rig_trajectories
        # store the rig→sensor (OpenCV/COLMAP, +Z forward) matrix — verified
        # empirically on Tier IV NuRec exports where the documented
        # sensor→rig reading puts the camera 1.3m underground. Invert to
        # get sensor pose in the rig frame.
        r_rs = _quat_to_matrix(cam.extrinsics.rotation)
        t_rs = np.asarray(cam.extrinsics.translation, dtype=np.float64)
        r_sensor = r_rs.T
        t_sensor = -r_rs.T @ t_rs

        # T_sensor_world = T_rig_world @ T_sensor_rig
        t_sensor_world = t_rig + r_rig @ t_sensor
        r_sensor_world = r_rig @ r_sensor

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
    if name is not None:
        raise ValueError(
            f"camera {name!r} not found in rig_trajectories; available: {available}"
        )
    return None


def _child_bbox_center(child: dict[str, Any]) -> np.ndarray | None:
    """Read the 3D Tiles ``boundingVolume.box`` center for a tile child."""
    box = child.get("boundingVolume", {}).get("box")
    if not box or len(box) < 3:
        return None
    return np.asarray(box[:3], dtype=np.float64)


def _spz_wrap_shift(
    chunk_path: Path,
    means: torch.Tensor,
    bbox_center: np.ndarray,
) -> torch.Tensor | None:
    """Compute the per-axis integer-wrap correction for an SPZ chunk.

    Niantic SPZ stores positions as ``int24`` scaled by ``2^fractionalBits``,
    so any coordinate outside ``±(2^23 / 2^fractionalBits) m`` wraps back
    into range when encoded. Returns the ``(dx, dy, dz)`` shift in meters
    that maps the centered cloud to the child tile's bbox center, snapped
    to integer multiples of the wrap period.
    """
    fractional_bits = _read_spz_fractional_bits(chunk_path)
    if fractional_bits is None:
        return None
    wrap_period = 2.0 ** (24 - fractional_bits)  # meters, e.g. 4096 at fb=12
    centroid = means.mean(dim=0).detach().cpu().numpy().astype(np.float64)
    raw_delta = bbox_center - centroid
    k = np.round(raw_delta / wrap_period)
    shift = k * wrap_period
    if not np.any(k):
        return None
    return torch.from_numpy(shift)


def _read_spz_fractional_bits(chunk_path: Path) -> int | None:
    """Return ``fractionalBits`` from an SPZ file header, or ``None`` on miss.

    SPZ v3 layout: magic (4) + version (4) + num_points (4) + sh_degree (1)
    + fractionalBits (1) + flags (1) + reserved (1). Files are gzipped.
    """
    with chunk_path.open("rb") as f:
        head = f.read(16)
    if len(head) < 16:
        return None
    if head[:2] == b"\x1f\x8b":
        # gzip — only need the first ~16 bytes; decompress a small prefix.
        with gzip.open(chunk_path, "rb") as f:
            head = f.read(16)
        if len(head) < 16:
            return None
    magic = struct.unpack("<I", head[:4])[0]
    if magic != 0x5053474E:  # b"NGSP" little-endian
        return None
    return head[13]


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
    return GaussianTensors(
        means=torch.cat([t.means for t in tensors], dim=0),
        quats=torch.cat([t.quats for t in tensors], dim=0),
        scales=torch.cat([t.scales for t in tensors], dim=0),
        opacities=torch.cat([t.opacities for t in tensors], dim=0),
        colors=torch.cat([t.colors for t in tensors], dim=0),
        sh_degree=sh_degree,
    )
