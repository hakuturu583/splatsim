from __future__ import annotations

import importlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from splatsim import _usdz
from splatsim._conversions import GaussianTensors
from splatsim._usdz import (
    initial_camera_pose_from_rig_trajectories,
    iter_world_to_camera_uncentered,
    read_rig_trajectories,
    read_scene_json,
)

_3dgs_io = importlib.import_module("3dgs_io")
FRAME_CONVENTION = _3dgs_io.FRAME_CONVENTION
Camera = _3dgs_io.Camera
CameraExtrinsics = _3dgs_io.CameraExtrinsics
CameraModel = _3dgs_io.CameraModel
RigPose = _3dgs_io.RigPose
RigTrajectory = _3dgs_io.RigTrajectory
serialize_rig_trajectories = _3dgs_io.serialize_rig_trajectories


def _rig():
    return RigTrajectory(
        rig_id="ego",
        poses=[
            RigPose(
                timestamp_us=123,
                translation=(10.0, 20.0, 30.0),
                rotation=(0.0, 0.0, 0.0, 1.0),
            )
        ],
        cameras=[
            Camera(
                name="front",
                camera_model=CameraModel.pinhole(
                    width=1920,
                    height=1080,
                    fx=1000.0,
                    fy=1000.0,
                    cx=960.0,
                    cy=540.0,
                ),
                extrinsics=CameraExtrinsics(
                    translation=(1.0, 2.0, 3.0),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                ),
            )
        ],
    )


def _scene_doc() -> dict[str, Any]:
    return {
        "schema": "splatsim.scene/v2",
        "world": {
            "frame_convention": FRAME_CONVENTION,
            "ecef_anchor": np.eye(4).tolist(),
        },
        "gaussians": {"frame": "world", "tileset": "tileset.json"},
        "extras": {"rig_trajectories": "rig_trajectories.json"},
    }


def test_read_rig_trajectories_accepts_v2_world_layout(tmp_path) -> None:
    usdz_path = tmp_path / "scene.usdz"
    rig_uri = "rig_trajectories.json"
    with zipfile.ZipFile(usdz_path, "w") as zf:
        zf.writestr(rig_uri, json.dumps(serialize_rig_trajectories([_rig()])))

    rigs = read_rig_trajectories(usdz_path, rig_uri)

    assert len(rigs) == 1
    assert rigs[0].rig_id == "ego"
    assert rigs[0].poses[0].timestamp_us == 123
    assert rigs[0].poses[0].translation == (10.0, 20.0, 30.0)


def test_v2_sensor_in_rig_is_composed_without_legacy_inversion() -> None:
    rigs = [_rig()]

    initial_pose = initial_camera_pose_from_rig_trajectories(rigs, name="front")
    assert initial_pose is not None
    position, _yaw = initial_pose
    [(timestamp, world_to_camera)] = iter_world_to_camera_uncentered(rigs, name="front")

    assert position == (11.0, 22.0, 33.0)
    assert timestamp == 123.0
    np.testing.assert_allclose(world_to_camera[:3, :3], np.eye(3))
    np.testing.assert_allclose(world_to_camera[:3, 3], [-11.0, -22.0, -33.0])


def test_read_scene_json_rejects_v1_schema(tmp_path) -> None:
    usdz_path = tmp_path / "scene.usdz"
    scene = _scene_doc()
    scene["schema"] = "splatsim.scene/v1"
    with zipfile.ZipFile(usdz_path, "w") as zf:
        zf.writestr("scene.json", json.dumps(scene))

    with pytest.raises(ValueError, match="splatsim.scene/v2"):
        read_scene_json(usdz_path)


def test_read_scene_json_accepts_frame_explicit_v2(tmp_path) -> None:
    usdz_path = tmp_path / "scene.usdz"
    scene = _scene_doc()
    with zipfile.ZipFile(usdz_path, "w") as zf:
        zf.writestr("scene.json", json.dumps(scene))

    assert read_scene_json(usdz_path) == scene


def test_load_spz_scene_reads_chunks_without_interpreting_tileset(
    tmp_path, monkeypatch
) -> None:
    usdz_path = tmp_path / "scene.usdz"
    scene = _scene_doc()
    anchor = np.eye(4)
    anchor[:3, 3] = [1.0, 2.0, 3.0]
    scene["world"]["ecef_anchor"] = anchor.tolist()
    with zipfile.ZipFile(usdz_path, "w") as zf:
        zf.writestr("scene.json", json.dumps(scene))
        zf.writestr("tileset.json", "not valid JSON and intentionally unused")
        zf.writestr("chunks/chunk_000001.spz", b"second")
        zf.writestr("chunks/chunk_000000.spz", b"first")

    loaded: list[bytes] = []

    def fake_load_spz(path):
        loaded.append(Path(path).read_bytes())
        return SimpleNamespace(num_points=1)

    def fake_cloud_to_tensors(_cloud, device, *, use_sh):
        value = float(len(loaded))
        return GaussianTensors(
            means=torch.tensor([[value, 0.0, 0.0]], device=device),
            quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
            scales=torch.ones((1, 3), device=device),
            opacities=torch.ones(1, device=device),
            colors=torch.zeros((1, 3), device=device),
            sh_degree=0,
        )

    monkeypatch.setattr(_usdz, "_load_spz", fake_load_spz)
    monkeypatch.setattr(_usdz, "cloud_to_tensors", fake_cloud_to_tensors)

    tensors, ecef_anchor = _usdz.load_spz_scene(usdz_path, torch.device("cpu"))

    assert loaded == [b"first", b"second"]
    assert tensors.means[:, 0].tolist() == [1.0, 2.0]
    np.testing.assert_allclose(ecef_anchor, anchor)


def test_load_spz_scene_restores_lidar_sidecars(tmp_path, monkeypatch) -> None:
    usdz_path = tmp_path / "scene.usdz"
    scene = _scene_doc()
    scene["gaussians"]["ext_attributes"] = {
        "extension": "EXT_gaussian_lidar",
        "sidecar_suffix": ".lidar",
        "attributes": ["lidar_intensity_raw", "lidar_raydrop_logit"],
    }
    sidecar = _3dgs_io.encode_lidar_sidecar(
        {
            "lidar_intensity_raw": np.array([0.0], dtype=np.float32),
            "lidar_raydrop_logit": np.array([-1.0], dtype=np.float32),
        },
        count=1,
    )
    with zipfile.ZipFile(usdz_path, "w") as zf:
        zf.writestr("scene.json", json.dumps(scene))
        zf.writestr("chunks/chunk_000000.spz", b"spz")
        zf.writestr("chunks/chunk_000000.lidar", sidecar)

    monkeypatch.setattr(_usdz, "_load_spz", lambda _path: SimpleNamespace(num_points=1))
    monkeypatch.setattr(
        _usdz,
        "cloud_to_tensors",
        lambda _cloud, device, *, use_sh: GaussianTensors(
            means=torch.zeros((1, 3), device=device),
            quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
            scales=torch.ones((1, 3), device=device),
            opacities=torch.ones(1, device=device),
            colors=torch.zeros((1, 3), device=device),
            sh_degree=0,
        ),
    )

    tensors, _anchor = _usdz.load_spz_scene(usdz_path, torch.device("cpu"))

    assert tensors.intensity_raw is not None
    assert tensors.raydrop_logit is not None
    torch.testing.assert_close(
        tensors.intensity_raw, torch.tensor([0.0]), atol=0.01, rtol=0
    )
    torch.testing.assert_close(
        tensors.raydrop_logit, torch.tensor([-1.0]), atol=0.02, rtol=0
    )
