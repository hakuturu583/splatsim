"""Publish ``sensor_msgs/Image`` or ``sensor_msgs/CompressedImage`` over CycloneDDS."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic
from numpy.typing import NDArray

from splatsim.cyclonedds._util import to_dds_topic
from splatsim.cyclonedds.msg_types import CompressedImage, Header, Image, Time

if TYPE_CHECKING:
    from cyclonedds.domain import DomainParticipant


class ImagePublisher:
    """Publishes BGR images as ``sensor_msgs/Image`` or ``sensor_msgs/CompressedImage`` via CycloneDDS.

    Parameters
    ----------
    participant:
        CycloneDDS domain participant (caller manages its lifetime).
    topic_name:
        ROS 2 topic name (e.g. ``"/splatsim/image_raw"``).
        The ``rt/`` DDS prefix is added automatically.
        When *compress_format* is set, ``"/compressed"`` is appended
        automatically if not already present.
    frame_id:
        ``frame_id`` written into each message header.
    compress_format:
        Image compression format (e.g. ``"jpeg"``, ``"png"``).
        When empty (default), raw ``sensor_msgs/Image`` is published.
    compress_quality:
        JPEG quality (1-100). Only used when *compress_format* is ``"jpeg"``.
        Defaults to 95.
    """

    def __init__(
        self,
        participant: DomainParticipant,
        topic_name: str = "/splatsim/image_raw",
        frame_id: str = "camera",
        compress_format: str = "",
        compress_quality: int = 95,
    ) -> None:
        self._frame_id = frame_id
        self._compress_format = compress_format.lower().strip()
        self._compress_quality = compress_quality

        if self._compress_format:
            import cv2  # noqa: PLC0415

            self._cv2 = cv2
            if not topic_name.endswith("/compressed"):
                topic_name = topic_name.rstrip("/") + "/compressed"
            topic = Topic(participant, to_dds_topic(topic_name), CompressedImage)
        else:
            topic = Topic(participant, to_dds_topic(topic_name), Image)
        self._writer = DataWriter(participant, topic)

    def publish(
        self,
        image: Optional[NDArray[np.uint8]],
        stamp: Time,
    ) -> None:
        """Publish a single image frame.

        Parameters
        ----------
        image:
            ``H x W x 3`` BGR ``uint8`` array, or *None* to skip.
        stamp:
            Timestamp for the message header. Callers must supply the clock
            that matches the rest of the graph (sim time under CARLA co-sim,
            wall clock only for standalone dev viewers).
        """
        if image is None:
            return

        if self._compress_format:
            self._publish_compressed(image, stamp)
        else:
            self._publish_raw(image, stamp)

    def _publish_raw(self, image: NDArray[np.uint8], stamp: Time) -> None:
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

    def _publish_compressed(self, image: NDArray[np.uint8], stamp: Time) -> None:
        cv2 = self._cv2
        fmt = self._compress_format
        if fmt == "jpeg":
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, self._compress_quality]
            success, buf = cv2.imencode(".jpg", image, encode_param)
        elif fmt == "png":
            success, buf = cv2.imencode(".png", image)
        else:
            raise ValueError(
                f"Unsupported compress_format: {fmt!r} (expected 'jpeg' or 'png')"
            )

        if not success:
            return

        msg = CompressedImage(
            header=Header(stamp=stamp, frame_id=self._frame_id),
            format=fmt,
            data=buf.tobytes(),
        )
        self._writer.write(msg)
