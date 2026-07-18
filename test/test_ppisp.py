from __future__ import annotations

import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from splatsim.ppisp import (
    _knn_weights,
    apply_ppisp,
    build_ppisp_tables,
    load_ppisp_tables,
    read_ppisp,
)


def _fake_ppisp(
    cameras: dict[str, tuple[np.ndarray, np.ndarray]],
    frames: list[tuple[int, float, np.ndarray]],
):
    cam_objs = {
        name: SimpleNamespace(vignetting=vig, crf=crf)
        for name, (vig, crf) in cameras.items()
    }
    frame_objs = [
        SimpleNamespace(timestamp_us=ts, exposure=exp, color=color)
        for ts, exp, color in frames
    ]
    return SimpleNamespace(cameras=cam_objs, frames=frame_objs)


def _fake_rigs(pose_positions: dict[int, tuple[float, float, float]]):
    poses = [
        SimpleNamespace(timestamp_us=ts, translation=np.array(pos, dtype=np.float64))
        for ts, pos in pose_positions.items()
    ]
    return [SimpleNamespace(poses=poses)]


def test_build_ppisp_tables_matches_frame_positions_by_timestamp() -> None:
    cameras = {
        "cam_front": (
            np.zeros((3, 5), dtype=np.float32),
            np.zeros((3, 4), dtype=np.float32),
        ),
        "cam_back": (
            np.full((3, 5), 0.5, dtype=np.float32),
            np.full((3, 4), 0.25, dtype=np.float32),
        ),
    }
    frames = [
        (100, 0.5, np.arange(8, dtype=np.float32)),
        (200, 1.0, np.arange(8, dtype=np.float32) * 2),
        (300, -0.25, np.zeros(8, dtype=np.float32)),
    ]
    pose_positions = {
        100: (0.0, 0.0, 0.0),
        200: (1.0, 0.0, 0.0),
        300: (0.0, 1.0, 0.0),
        # 400 has no matching frame - should be ignored
        400: (5.0, 5.0, 5.0),
    }

    ppisp = _fake_ppisp(cameras, frames)
    rigs = _fake_rigs(pose_positions)

    tables = build_ppisp_tables(ppisp, rigs, device="cpu")
    assert tables is not None
    assert tables.num_cameras == 2
    # camera_names is sorted, so cam_back < cam_front
    assert tables.camera_names == ("cam_back", "cam_front")
    assert tables.camera_index["cam_front"] == 1
    assert tables.num_frames == 3
    # Frame positions in same order as frames list
    assert torch.allclose(
        tables.frame_positions,
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ),
    )
    assert tables.frame_exposure.tolist() == [0.5, 1.0, -0.25]


def test_build_ppisp_tables_returns_none_when_empty() -> None:
    assert build_ppisp_tables(_fake_ppisp({}, []), [], device="cpu") is None
    # No usable rig poses -> no frame positions -> None
    ppisp = _fake_ppisp(
        {
            "cam": (
                np.zeros((3, 5), dtype=np.float32),
                np.zeros((3, 4), dtype=np.float32),
            )
        },
        [(100, 0.0, np.zeros(8, dtype=np.float32))],
    )
    assert build_ppisp_tables(ppisp, _fake_rigs({999: (0, 0, 0)}), device="cpu") is None


def test_build_ppisp_tables_centroid_shift() -> None:
    cameras = {
        "cam": (np.zeros((3, 5), dtype=np.float32), np.zeros((3, 4), dtype=np.float32))
    }
    frames = [(100, 0.0, np.zeros(8, dtype=np.float32))]
    rigs = _fake_rigs({100: (10.0, -5.0, 0.5)})
    tables = build_ppisp_tables(
        _fake_ppisp(cameras, frames),
        rigs,
        device="cpu",
        centroid=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )
    assert tables is not None
    assert torch.allclose(
        tables.frame_positions,
        torch.tensor([[9.0, -6.0, -0.5]], dtype=torch.float32),
    )


def test_knn_weights_zero_distance_hits_singleton() -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    query = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    idx, w = _knn_weights(positions, query, k=2, eps=1e-9)
    # zero-distance frame gets ~1.0 weight, other much less
    assert idx[0].item() == 0
    assert w[0].item() > 0.99
    assert torch.allclose(w.sum(), torch.tensor(1.0))


def test_knn_weights_caps_k_at_available_frames() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    query = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    idx, w = _knn_weights(positions, query, k=8, eps=1e-6)
    assert idx.shape == (1,)
    assert torch.allclose(w, torch.tensor([1.0]))


def test_read_ppisp_returns_none_when_archive_missing_payload(tmp_path) -> None:
    usdz = tmp_path / "no_ppisp.usdz"
    with zipfile.ZipFile(usdz, "w") as zf:
        zf.writestr("some.usd", "")
    assert read_ppisp(usdz) is None


def test_read_ppisp_returns_none_on_malformed_json(tmp_path) -> None:
    usdz = tmp_path / "bad_json.usdz"
    with zipfile.ZipFile(usdz, "w") as zf:
        zf.writestr("ppisp.json", "{not valid json")
    assert read_ppisp(usdz) is None


def test_load_ppisp_tables_fallback_when_no_payload(tmp_path) -> None:
    usdz = tmp_path / "no_ppisp.usdz"
    with zipfile.ZipFile(usdz, "w") as zf:
        zf.writestr("scene.usd", "")
    assert load_ppisp_tables(usdz, [], device="cpu") is None


def test_apply_ppisp_returns_input_when_camera_name_unknown() -> None:
    cameras = {
        "cam_a": (
            np.zeros((3, 5), dtype=np.float32),
            np.zeros((3, 4), dtype=np.float32),
        )
    }
    frames = [(100, 0.0, np.zeros(8, dtype=np.float32))]
    tables = build_ppisp_tables(
        _fake_ppisp(cameras, frames),
        _fake_rigs({100: (0.0, 0.0, 0.0)}),
        device="cpu",
    )
    assert tables is not None
    rgb = torch.rand(4, 6, 3, dtype=torch.float32)
    query = torch.zeros(3, dtype=torch.float32)
    # Unknown camera -> untouched passthrough (no CUDA op invoked, so this
    # test runs on CPU).
    out = apply_ppisp(tables, rgb, "cam_missing", query)
    assert torch.equal(out, rgb)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ppisp_apply requires CUDA")
def test_apply_ppisp_cuda_matches_shape() -> None:
    cameras = {
        "cam_a": (
            np.array([[1.0, 0.0, 0.0, 0.0, 0.0]] * 3, dtype=np.float32),
            np.zeros((3, 4), dtype=np.float32),
        )
    }
    frames = [
        (100, 0.0, np.zeros(8, dtype=np.float32)),
        (200, 0.5, np.zeros(8, dtype=np.float32)),
    ]
    tables = build_ppisp_tables(
        _fake_ppisp(cameras, frames),
        _fake_rigs({100: (0.0, 0.0, 0.0), 200: (1.0, 0.0, 0.0)}),
        device="cuda",
    )
    assert tables is not None
    rgb = torch.rand(8, 12, 3, device="cuda", dtype=torch.float32)
    out = apply_ppisp(
        tables, rgb, "cam_a", torch.tensor([0.5, 0.0, 0.0], device="cuda"), k=2
    )
    assert out.shape == rgb.shape
    assert torch.isfinite(out).all()


def test_load_ppisp_tables_end_to_end(tmp_path) -> None:
    payload = {
        "schema": "splatsim.ppisp/v1",
        "pipeline": ["exposure", "vignetting", "color", "crf"],
        "cameras": {
            "cam_a": {
                "vignetting": [[0.5, 0.0, 0.0, 0.0, 0.0]] * 3,
                "crf": [[0.0, 0.0, 0.0, 0.0]] * 3,
            }
        },
        "frames": [
            {
                "timestamp_us": 100,
                "exposure": 0.25,
                "color": [0.0] * 8,
            }
        ],
    }
    usdz = tmp_path / "with_ppisp.usdz"
    with zipfile.ZipFile(usdz, "w") as zf:
        zf.writestr("ppisp.json", json.dumps(payload))
    rigs = _fake_rigs({100: (0.0, 0.0, 0.0)})
    tables = load_ppisp_tables(usdz, rigs, device="cpu")
    assert tables is not None
    assert tables.camera_names == ("cam_a",)
    assert tables.num_frames == 1
