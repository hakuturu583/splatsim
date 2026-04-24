"""Publish ``sensor_msgs/Imu`` over CycloneDDS from CARLA IMU sensor data."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic

from splatsim.cyclonedds.msg_types import Header, Imu, Quaternion, Time, Vector3

if TYPE_CHECKING:
    from cyclonedds.domain import DomainParticipant

logger = logging.getLogger(__name__)


class ImuPublisher:
    """Publishes CARLA IMU data as ``sensor_msgs/Imu`` via CycloneDDS.

    Parameters
    ----------
    participant:
        CycloneDDS domain participant (caller manages its lifetime).
    topic_name:
        ROS 2 topic name (e.g. ``"/sensing/imu/imu_data"``).
        The ``rt/`` DDS prefix is added automatically.
    frame_id:
        ``frame_id`` written into each message header.
    """

    def __init__(
        self,
        participant: DomainParticipant,
        topic_name: str = "/sensing/imu/imu_data",
        frame_id: str = "imu_link",
    ) -> None:
        self._frame_id = frame_id
        dds_name = "rt/" + topic_name.lstrip("/")
        topic = Topic(participant, dds_name, Imu)
        self._writer = DataWriter(participant, topic)

    def publish(
        self,
        accelerometer: tuple[float, float, float],
        gyroscope: tuple[float, float, float],
        stamp: Optional[Time] = None,
    ) -> None:
        """Publish a single IMU measurement.

        Parameters
        ----------
        accelerometer:
            Linear acceleration ``(x, y, z)`` in m/s^2.
        gyroscope:
            Angular velocity ``(x, y, z)`` in rad/s.
        stamp:
            Timestamp for the message header.  When *None*, the current
            wall-clock time is used.
        """
        if stamp is None:
            sec, nanosec = divmod(time.time_ns(), 10**9)
            stamp = Time(sec=sec, nanosec=nanosec)

        msg = Imu(
            header=Header(stamp=stamp, frame_id=self._frame_id),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            orientation_covariance=(-1.0,) + (0.0,) * 8,  # -1 = orientation unknown
            angular_velocity=Vector3(
                x=gyroscope[0], y=gyroscope[1], z=gyroscope[2]
            ),
            angular_velocity_covariance=(0.0,) * 9,
            linear_acceleration=Vector3(
                x=accelerometer[0], y=accelerometer[1], z=accelerometer[2]
            ),
            linear_acceleration_covariance=(0.0,) * 9,
        )
        self._writer.write(msg)
