"""Publish ``sensor_msgs/PointCloud2`` from a splatsim LiDAR panorama."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import numpy as np
from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic
from numpy.typing import NDArray

from splatsim.cyclonedds.msg_types import (
    Header,
    PointCloud2,
    PointField,
    Time,
)

if TYPE_CHECKING:
    from cyclonedds.domain import DomainParticipant


# sensor_msgs/PointField datatype constants.
_PF_FLOAT32 = 7

# Point layout: contiguous packed x, y, z, intensity (all float32) = 16 bytes.
_POINT_STEP = 16
_FIELDS: tuple[PointField, ...] = (
    PointField(name="x", offset=0, datatype=_PF_FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=_PF_FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=_PF_FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=_PF_FLOAT32, count=1),
)


class PointCloud2Publisher:
    """Publishes ``(N,3) xyz + (N,) intensity`` point clouds as ``sensor_msgs/PointCloud2``.

    The wire layout is the minimal XYZI packing (16 bytes/point). Extended
    Autoware fields (return_type, azimuth, elevation, time_stamp) can be
    added later without changing this class's public API.
    """

    def __init__(
        self,
        participant: DomainParticipant,
        *,
        topic_name: str = "/splatsim/lidar/pointcloud",
        frame_id: str = "splatsim_lidar",
    ) -> None:
        self._frame_id = frame_id
        topic = Topic(participant, _to_dds_topic(topic_name), PointCloud2)
        self._writer = DataWriter(participant, topic)

    def publish(
        self,
        xyz: NDArray[np.float32],
        intensity: NDArray[np.float32],
        *,
        stamp: Optional[Time] = None,
    ) -> None:
        """Publish one point cloud frame.

        Args:
            xyz: (N, 3) float32 coordinates in the sensor frame.
            intensity: (N,) float32 in [0, 1].
            stamp: Optional ROS 2 timestamp. Defaults to wall-clock now.
        """
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must be (N, 3); got shape {xyz.shape}")
        if intensity.ndim != 1 or intensity.shape[0] != xyz.shape[0]:
            raise ValueError(
                f"intensity must be (N,) matching xyz rows; got {intensity.shape} vs {xyz.shape}",
            )
        stamp = stamp if stamp is not None else _now()

        n = xyz.shape[0]
        # Pack into a contiguous (N, 4) float32 buffer: [x, y, z, intensity].
        packed = np.empty((n, 4), dtype=np.float32)
        packed[:, :3] = xyz.astype(np.float32, copy=False)
        packed[:, 3] = intensity.astype(np.float32, copy=False)

        msg = PointCloud2(
            header=Header(stamp=stamp, frame_id=self._frame_id),
            height=1,
            width=n,
            fields=list(_FIELDS),
            is_bigendian=False,
            point_step=_POINT_STEP,
            row_step=_POINT_STEP * n,
            data=packed.tobytes(),
            is_dense=True,
        )
        self._writer.write(msg)


def _to_dds_topic(ros_topic: str) -> str:
    return "rt/" + ros_topic.lstrip("/")


def _now() -> Time:
    sec, nanosec = divmod(time.time_ns(), 10**9)
    return Time(sec=sec, nanosec=nanosec)
