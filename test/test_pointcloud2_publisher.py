from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cyclonedds")

from splatsim.cyclonedds.msg_types import Time
from splatsim.cyclonedds.pointcloud2_publisher import (  # noqa: E402
    _POINT_STEP,
    _make_pointcloud2_message,
)


def test_make_pointcloud2_message_packs_xyzi_fields() -> None:
    xyz = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    intensity = np.array([0.25, 0.75], dtype=np.float32)

    msg = _make_pointcloud2_message(
        xyz,
        intensity,
        stamp=Time(sec=12, nanosec=34),
        frame_id="top_lidar",
    )

    assert msg.header.stamp.sec == 12
    assert msg.header.stamp.nanosec == 34
    assert msg.header.frame_id == "top_lidar"
    assert msg.height == 1
    assert msg.width == 2
    assert msg.point_step == _POINT_STEP
    assert msg.row_step == _POINT_STEP * 2
    assert [(f.name, f.offset, f.datatype, f.count) for f in msg.fields] == [
        ("x", 0, 7, 1),
        ("y", 4, 7, 1),
        ("z", 8, 7, 1),
        ("intensity", 12, 7, 1),
    ]

    packed = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(-1, 4)
    expected = np.column_stack([xyz, intensity])
    np.testing.assert_allclose(packed, expected)


def test_make_pointcloud2_message_validates_shapes() -> None:
    xyz = np.zeros((2, 3), dtype=np.float32)
    intensity = np.zeros((2,), dtype=np.float32)
    stamp = Time(sec=0, nanosec=0)

    with pytest.raises(ValueError, match="xyz must be"):
        _make_pointcloud2_message(
            np.zeros((2, 2), dtype=np.float32),
            intensity,
            stamp=stamp,
            frame_id="lidar",
        )

    with pytest.raises(ValueError, match="intensity must be"):
        _make_pointcloud2_message(
            xyz,
            np.zeros((3,), dtype=np.float32),
            stamp=stamp,
            frame_id="lidar",
        )
