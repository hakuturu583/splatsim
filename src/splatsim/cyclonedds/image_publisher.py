"""Publish ``sensor_msgs/Image`` over CycloneDDS."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import numpy as np
from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic
from numpy.typing import NDArray

from splatsim.cyclonedds.msg_types import Header, Image, Time

if TYPE_CHECKING:
    from cyclonedds.domain import DomainParticipant


class ImagePublisher:
    """Publishes BGR images as ``sensor_msgs/Image`` via CycloneDDS.

    Parameters
    ----------
    participant:
        CycloneDDS domain participant (caller manages its lifetime).
    topic_name:
        DDS topic name (e.g. ``"/splatsim/image_raw"``).
    frame_id:
        ``frame_id`` written into each message header.
    """

    def __init__(
        self,
        participant: DomainParticipant,
        topic_name: str = "/splatsim/image_raw",
        frame_id: str = "camera",
    ) -> None:
        self._frame_id = frame_id
        topic = Topic(participant, topic_name, Image)
        self._writer = DataWriter(participant, topic)

    def publish(
        self,
        image: Optional[NDArray[np.uint8]],
        stamp: Optional[Time] = None,
    ) -> None:
        """Publish a single image frame.

        Parameters
        ----------
        image:
            ``H x W x 3`` BGR ``uint8`` array, or *None* to skip.
        stamp:
            Timestamp for the message header.  When *None*, the current
            wall-clock time is used.
        """
        if image is None:
            return

        if stamp is None:
            stamp = _now()

        height, width = image.shape[:2]
        msg = Image(
            header=Header(stamp=stamp, frame_id=self._frame_id),
            height=height,
            width=width,
            encoding="bgr8",
            is_bigendian=0,
            step=width * 3,
            data=np.ascontiguousarray(image).tobytes(),
        )
        self._writer.write(msg)


def _now() -> Time:
    """Return the current wall-clock time as a ROS 2 ``Time``."""
    sec, nanosec = divmod(time.time_ns(), 10**9)
    return Time(sec=sec, nanosec=nanosec)
