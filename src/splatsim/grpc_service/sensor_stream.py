"""Deliver rendered sensor data over gRPC instead of DDS.

The rendering service normally publishes every camera image, camera info and
LiDAR point cloud over CycloneDDS. With ``--sensor-transport=grpc`` (or
``SPLATSIM_SENSOR_TRANSPORT=grpc``) it builds the publishers in this module
instead: they take exactly the calls the DDS publishers take, so the render
loops do not know which transport they feed, and hand each message to a
:class:`SensorDataHub`, which fans it out to the clients of the
``SubscribeSensorData`` stream.

Rendering must never wait on a client, so every subscriber gets a bounded
queue that drops its OLDEST message when full -- a slow reader sees the newest
frames, the way a DDS ``KEEP_LAST`` reader would -- and a message nobody has
subscribed to is never even built.
"""

from __future__ import annotations

import collections
import logging
import os
import threading
from typing import Callable, Iterable, Iterator, Mapping, Optional, Protocol

import numpy as np
from numpy.typing import NDArray

from splatsim.cyclonedds.pointcloud2_publisher import _POINT_FIELDS, _POINT_STEP
from splatsim.grpc_service._generated import rendering_service_pb2 as pb2

logger = logging.getLogger(__name__)

#: Environment variable selecting the transport when the CLI flag is absent.
ENV_VAR = "SPLATSIM_SENSOR_TRANSPORT"
SENSOR_TRANSPORT_DDS = "dds"
SENSOR_TRANSPORT_GRPC = "grpc"
SENSOR_TRANSPORTS = (SENSOR_TRANSPORT_DDS, SENSOR_TRANSPORT_GRPC)

#: Messages buffered per subscriber when the request leaves ``queue_size`` 0.
DEFAULT_QUEUE_SIZE = 16


def sensor_transport_from_env(environ: Mapping[str, str] | None = None) -> str:
    """The transport :data:`ENV_VAR` names, or ``"dds"`` when it is unset.

    An unknown value raises rather than falling back: a server that silently
    publishes over DDS while the client waits on gRPC looks healthy on both
    ends and delivers nothing.
    """
    raw = (environ if environ is not None else os.environ).get(ENV_VAR, "")
    return parse_sensor_transport(raw or SENSOR_TRANSPORT_DDS)


def parse_sensor_transport(value: str) -> str:
    """Normalise a transport name, raising ``ValueError`` on an unknown one."""
    transport = value.strip().lower()
    if transport not in SENSOR_TRANSPORTS:
        raise ValueError(
            f"unknown sensor transport {value!r}: expected one of "
            f"{', '.join(SENSOR_TRANSPORTS)}"
        )
    return transport


class _Stamp(Protocol):
    """What the render loops pass as ``stamp`` (``msg_types.Time``)."""

    @property
    def sec(self) -> int: ...
    @property
    def nanosec(self) -> int: ...


class SensorSubscription:
    """One ``SubscribeSensorData`` client: a topic filter and a bounded queue."""

    def __init__(self, topics: Iterable[str], queue_size: int) -> None:
        self.topics: frozenset[str] = frozenset(topics)
        self._queue: collections.deque[pb2.SensorData] = collections.deque(
            maxlen=max(1, queue_size)
        )
        self._cond = threading.Condition()
        self._closed = False
        #: Messages discarded because the client fell behind.
        self.dropped = 0

    def wants(self, topic: str) -> bool:
        return not self.topics or topic in self.topics

    @property
    def closed(self) -> bool:
        return self._closed

    def put(self, msg: pb2.SensorData) -> None:
        with self._cond:
            if self._closed:
                return
            if len(self._queue) == self._queue.maxlen:
                self.dropped += 1
                if self.dropped == 1 or self.dropped % 100 == 0:
                    logger.warning(
                        "Sensor subscriber falling behind: %d messages dropped",
                        self.dropped,
                    )
            self._queue.append(msg)
            self._cond.notify()

    def get(self, timeout: float | None = None) -> pb2.SensorData | None:
        """Next message, or ``None`` on timeout or once closed and drained."""
        with self._cond:
            if not self._queue and not self._closed:
                self._cond.wait(timeout)
            return self._queue.popleft() if self._queue else None

    def close(self) -> None:
        """Wake a blocked :meth:`get`; no further messages are accepted."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class SensorDataHub:
    """Fans sensor messages out to every subscription that wants their topic."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscriptions: tuple[SensorSubscription, ...] = ()

    def subscribe(
        self, topics: Iterable[str] = (), queue_size: int = 0
    ) -> SensorSubscription:
        sub = SensorSubscription(topics, queue_size or DEFAULT_QUEUE_SIZE)
        with self._lock:
            self._subscriptions = (*self._subscriptions, sub)
        logger.info(
            "Sensor subscriber added (topics=%s, queue=%d); %d active",
            sorted(sub.topics) or "all",
            sub._queue.maxlen,
            len(self._subscriptions),
        )
        return sub

    def unsubscribe(self, sub: SensorSubscription) -> None:
        sub.close()
        with self._lock:
            self._subscriptions = tuple(s for s in self._subscriptions if s is not sub)
        logger.info(
            "Sensor subscriber removed (%d dropped); %d active",
            sub.dropped,
            len(self._subscriptions),
        )

    def wants(self, topic: str) -> bool:
        """Whether any subscriber would receive *topic* -- lets publishers skip
        encoding a frame nobody is listening for."""
        return any(s.wants(topic) for s in self._subscriptions)

    def publish(self, topic: str, build: Callable[[], pb2.SensorData]) -> None:
        """Deliver ``build()`` to every subscriber of *topic*; build it only if
        there is one. The message is shared, not copied, between subscribers."""
        targets = [s for s in self._subscriptions if s.wants(topic)]
        if not targets:
            return
        msg = build()
        for sub in targets:
            sub.put(msg)

    def iter_messages(
        self,
        sub: SensorSubscription,
        is_active: Callable[[], bool],
        poll_s: float = 0.5,
    ) -> Iterator[pb2.SensorData]:
        """Yield *sub*'s messages until it is closed or ``is_active()`` turns
        false, then unsubscribe it."""
        try:
            while is_active() and not sub.closed:
                msg = sub.get(timeout=poll_s)
                if msg is not None:
                    yield msg
        finally:
            self.unsubscribe(sub)


def _header(stamp: _Stamp, frame_id: str) -> pb2.Header:
    return pb2.Header(
        stamp=pb2.Timestamp(sec=int(stamp.sec), nanosec=int(stamp.nanosec)),
        frame_id=frame_id,
    )


class GrpcImagePublisher:
    """gRPC counterpart of :class:`splatsim.cyclonedds.ImagePublisher`.

    Same topic naming (``/compressed`` appended when compressing) and the same
    payloads: a raw ``bgr8`` image, or a JPEG / PNG ``CompressedImage``.
    """

    def __init__(
        self,
        hub: SensorDataHub,
        topic_name: str = "/splatsim/image_raw",
        frame_id: str = "camera",
        compress_format: str = "",
        compress_quality: int = 95,
    ) -> None:
        self._hub = hub
        self._frame_id = frame_id
        self._compress_format = compress_format.lower().strip()
        self._compress_quality = compress_quality
        if self._compress_format:
            if self._compress_format not in ("jpeg", "png"):
                raise ValueError(
                    f"Unsupported compress_format: {self._compress_format!r} "
                    "(expected 'jpeg' or 'png')"
                )
            import cv2  # noqa: PLC0415

            self._cv2 = cv2
            if not topic_name.endswith("/compressed"):
                topic_name = topic_name.rstrip("/") + "/compressed"
        self.topic = topic_name

    def publish(self, image: Optional[NDArray[np.uint8]], stamp: _Stamp) -> None:
        """Publish one ``H x W x 3`` BGR ``uint8`` frame (``None`` skips)."""
        if image is None:
            return
        self._hub.publish(self.topic, lambda: self._build(image, stamp))

    def _build(self, image: NDArray[np.uint8], stamp: _Stamp) -> pb2.SensorData:
        header = _header(stamp, self._frame_id)
        if not self._compress_format:
            height, width = image.shape[:2]
            return pb2.SensorData(
                topic=self.topic,
                image=pb2.Image(
                    header=header,
                    height=height,
                    width=width,
                    encoding="bgr8",
                    is_bigendian=0,
                    step=width * 3,
                    data=np.ascontiguousarray(image).tobytes(),
                ),
            )
        cv2 = self._cv2
        if self._compress_format == "jpeg":
            ok, buf = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self._compress_quality]
            )
        else:
            ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError(f"{self._compress_format} encoding failed")
        return pb2.SensorData(
            topic=self.topic,
            compressed_image=pb2.CompressedImage(
                header=header, format=self._compress_format, data=buf.tobytes()
            ),
        )


class _CameraConfig(Protocol):
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


class GrpcCameraInfoPublisher:
    """gRPC counterpart of :class:`splatsim.cyclonedds.CameraInfoPublisher`."""

    def __init__(
        self,
        hub: SensorDataHub,
        config: _CameraConfig,
        topic_name: str = "/splatsim/camera_info",
        frame_id: str = "camera",
    ) -> None:
        self._hub = hub
        self._frame_id = frame_id
        self.topic = topic_name
        fx, fy, cx, cy = config.fx, config.fy, config.cx, config.cy
        self._height = config.image_height
        self._width = config.image_width
        self._k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self._r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self._p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

    def publish(self, stamp: _Stamp) -> None:
        self._hub.publish(self.topic, lambda: self._build(stamp))

    def _build(self, stamp: _Stamp) -> pb2.SensorData:
        return pb2.SensorData(
            topic=self.topic,
            camera_info=pb2.CameraInfo(
                header=_header(stamp, self._frame_id),
                height=self._height,
                width=self._width,
                distortion_model="plumb_bob",
                d=[],
                k=self._k,
                r=self._r,
                p=self._p,
                binning_x=0,
                binning_y=0,
                roi=pb2.RegionOfInterest(
                    x_offset=0,
                    y_offset=0,
                    height=self._height,
                    width=self._width,
                    do_rectify=False,
                ),
            ),
        )


_PB_POINT_FIELDS = tuple(
    pb2.PointField(name=f.name, offset=f.offset, datatype=f.datatype, count=f.count)
    for f in _POINT_FIELDS
)


class GrpcPointCloud2Publisher:
    """gRPC counterpart of :class:`splatsim.cyclonedds.PointCloud2Publisher`.

    Carries the same 16-byte point records, so a client decodes ``data``
    exactly as it would a ``sensor_msgs/PointCloud2`` from the DDS path.
    """

    def __init__(
        self,
        hub: SensorDataHub,
        *,
        topic_name: str = "/splatsim/lidar/pointcloud",
        frame_id: str = "splatsim_lidar",
    ) -> None:
        self._hub = hub
        self._frame_id = frame_id
        self.topic = topic_name

    def publish_packed(
        self, records: NDArray[np.uint8], count: int, *, stamp: _Stamp
    ) -> None:
        """Publish ``(count * 16,)`` pre-packed point records."""
        if records.size != count * _POINT_STEP:
            raise ValueError(
                f"records must be (count * {_POINT_STEP},) uint8; "
                f"got {records.size} bytes for {count} points"
            )
        self._hub.publish(self.topic, lambda: self._build(records, count, stamp))

    def _build(
        self, records: NDArray[np.uint8], count: int, stamp: _Stamp
    ) -> pb2.SensorData:
        return pb2.SensorData(
            topic=self.topic,
            point_cloud=pb2.PointCloud2(
                header=_header(stamp, self._frame_id),
                height=1,
                width=count,
                fields=_PB_POINT_FIELDS,
                is_bigendian=False,
                point_step=_POINT_STEP,
                row_step=_POINT_STEP * count,
                data=records.tobytes(),
                is_dense=False,
            ),
        )
