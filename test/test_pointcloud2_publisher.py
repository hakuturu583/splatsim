from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("cyclonedds")

from splatsim.cyclonedds.msg_types import Time
from splatsim.cyclonedds.pointcloud2_publisher import (  # noqa: E402
    _POINT_RECORD,
    _POINT_STEP,
    _make_pointcloud2_message,
)


def test_make_pointcloud2_message_packs_autoware_fields() -> None:
    xyz = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    # 0.25 -> 63, and 1.5 clamps to 1.0 -> 255 (out-of-range values are clipped).
    intensity = np.array([0.25, 1.5], dtype=np.float32)
    channel = np.array([3, 127], dtype=np.int64)

    msg = _make_pointcloud2_message(
        xyz,
        intensity,
        channel=channel,
        stamp=Time(sec=12, nanosec=34),
        frame_id="top_lidar",
    )

    assert msg.header.stamp.sec == 12
    assert msg.header.stamp.nanosec == 34
    assert msg.header.frame_id == "top_lidar"
    assert msg.height == 1
    assert msg.width == 2
    assert msg.point_step == _POINT_STEP == 16
    assert msg.row_step == _POINT_STEP * 2
    # sensor_msgs/PointField datatypes: FLOAT32=7, UINT8=2, UINT16=4.
    assert [(f.name, f.offset, f.datatype, f.count) for f in msg.fields] == [
        ("x", 0, 7, 1),
        ("y", 4, 7, 1),
        ("z", 8, 7, 1),
        ("intensity", 12, 2, 1),
        ("return_type", 13, 2, 1),
        ("channel", 14, 4, 1),
    ]

    # np.frombuffer is typed as returning a float64 array regardless of the
    # structured dtype= argument, so the field-name indexing below needs a cast.
    rec = cast(
        "np.ndarray[Any, np.dtype[np.void]]",
        np.frombuffer(bytes(msg.data), dtype=_POINT_RECORD),
    )
    np.testing.assert_allclose(np.column_stack([rec["x"], rec["y"], rec["z"]]), xyz)
    assert rec["intensity"].dtype == np.uint8
    assert rec["intensity"].tolist() == [63, 255]
    assert rec["return_type"].tolist() == [0, 0]
    assert rec["channel"].dtype == np.uint16
    assert rec["channel"].tolist() == [3, 127]


def test_make_pointcloud2_message_defaults_channel_to_zero() -> None:
    xyz = np.zeros((3, 3), dtype=np.float32)
    intensity = np.zeros((3,), dtype=np.float32)

    msg = _make_pointcloud2_message(
        xyz, intensity, stamp=Time(sec=0, nanosec=0), frame_id="lidar"
    )

    # np.frombuffer is typed as returning a float64 array regardless of the
    # structured dtype= argument, so the field-name indexing below needs a cast.
    rec = cast(
        "np.ndarray[Any, np.dtype[np.void]]",
        np.frombuffer(bytes(msg.data), dtype=_POINT_RECORD),
    )
    assert rec["channel"].tolist() == [0, 0, 0]


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

    with pytest.raises(ValueError, match="channel must be"):
        _make_pointcloud2_message(
            xyz,
            intensity,
            channel=np.zeros((3,), dtype=np.int64),
            stamp=stamp,
            frame_id="lidar",
        )
