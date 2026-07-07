"""Publish ``sensor_msgs/CameraInfo`` over CycloneDDS."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic

from splatsim.cyclonedds._util import _now, _to_dds_topic
from splatsim.cyclonedds.msg_types import CameraInfo, Header, RegionOfInterest, Time

if TYPE_CHECKING:
    from cyclonedds.domain import DomainParticipant


class CameraConfig(Protocol):
    """Anything that exposes pinhole-camera intrinsics."""

    @property
    def fx(self) -> float: ...
    @property
    def fy(self) -> float: ...
    @property
    def cx(self) -> float: ...
    @property
    def cy(self) -> float: ...
    @property
    def image_width(self) -> int: ...
    @property
    def image_height(self) -> int: ...


class CameraInfoPublisher:
    """Publishes ``sensor_msgs/CameraInfo`` via CycloneDDS.

    Intrinsic parameters are precomputed once from a
    :class:`CameraConfig`-compatible object and reused on every
    :meth:`publish` call—only the header timestamp changes.

    Parameters
    ----------
    participant:
        CycloneDDS domain participant (caller manages its lifetime).
    config:
        Camera configuration from which intrinsics are derived.
    topic_name:
        DDS topic name (e.g. ``"/splatsim/camera_info"``).
    frame_id:
        ``frame_id`` written into each message header.
    """

    def __init__(
        self,
        participant: DomainParticipant,
        config: CameraConfig,
        topic_name: str = "/splatsim/camera_info",
        frame_id: str = "camera",
    ) -> None:
        self._frame_id = frame_id

        topic = Topic(participant, _to_dds_topic(topic_name), CameraInfo)
        self._writer = DataWriter(participant, topic)

        # Precompute static intrinsic fields from config.
        fx, fy = config.fx, config.fy
        cx, cy = config.cx, config.cy

        self._height = config.image_height
        self._width = config.image_width

        # 3x3 intrinsic matrix K (row-major).
        self._k: list[float] = [
            fx,
            0.0,
            cx,
            0.0,
            fy,
            cy,
            0.0,
            0.0,
            1.0,
        ]

        # 3x3 rectification matrix R (identity for monocular).
        self._r: list[float] = [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]

        # 3x4 projection matrix P (monocular, no translation).
        self._p: list[float] = [
            fx,
            0.0,
            cx,
            0.0,
            0.0,
            fy,
            cy,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]

        self._roi = RegionOfInterest(
            x_offset=0,
            y_offset=0,
            height=self._height,
            width=self._width,
            do_rectify=False,
        )

    def publish(self, stamp: Optional[Time] = None) -> None:
        """Publish a ``CameraInfo`` message.

        Parameters
        ----------
        stamp:
            Timestamp for the message header.  When *None*, the current
            wall-clock time is used.
        """
        if stamp is None:
            stamp = _now()

        msg = CameraInfo(
            header=Header(stamp=stamp, frame_id=self._frame_id),
            height=self._height,
            width=self._width,
            distortion_model="plumb_bob",
            d=[],
            k=self._k,
            r=self._r,
            p=self._p,
            binning_x=0,
            binning_y=0,
            roi=self._roi,
        )
        self._writer.write(msg)
