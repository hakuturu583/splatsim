"""Deliver rendered sensor data over gRPC instead of DDS.

The rendering service normally publishes every camera image, camera info and
LiDAR point cloud over CycloneDDS. With ``--sensor-transport=grpc`` (or
``SPLATSIM_SENSOR_TRANSPORT=grpc``) it hands the very same publishers a
:class:`SensorDataHub` in place of the DDS participant: each message they
would have written to DDS is converted to its protobuf mirror and fanned out
to the clients of the ``SubscribeSensorData`` stream, so both transports
carry identical payloads by construction.

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
from typing import Any, Callable, Iterable, Iterator, Mapping

from splatsim.cyclonedds import msg_types
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
            self._cond.wait_for(lambda: self._queue or self._closed, timeout)
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

    def create_writer(self, topic_name: str, msg_type: type) -> _HubWriter:
        """The :class:`~splatsim.cyclonedds._util.MessageSink` hook: a writer
        the DDS publishers write *msg_type* messages for *topic_name* to."""
        try:
            field, convert = _CONVERTERS[msg_type]
        except KeyError:
            raise TypeError(f"no gRPC mirror for {msg_type.__name__}") from None
        return _HubWriter(self, topic_name, field, convert)

    def publish(self, topic: str, build: Callable[[], pb2.SensorData]) -> None:
        """Deliver ``build()`` to every subscriber of *topic*; build it only if
        there is one. The message is shared, not copied, between subscribers."""
        targets = [s for s in self._subscriptions if s.wants(topic)]
        if not targets:
            return
        msg = build()
        for sub in targets:
            sub.put(msg)

    def iter_messages(self, sub: SensorSubscription) -> Iterator[pb2.SensorData]:
        """Yield *sub*'s messages until it is closed, then unsubscribe it."""
        try:
            while (msg := sub.get()) is not None:
                yield msg
        finally:
            self.unsubscribe(sub)


class _HubWriter:
    """Stands in for a DDS ``DataWriter``: converts each written message to
    protobuf -- only when someone subscribes to its topic -- and publishes it."""

    def __init__(
        self,
        hub: SensorDataHub,
        topic: str,
        field: str,
        convert: Callable[[Any], Any],
    ) -> None:
        self._hub = hub
        self.topic = topic
        self._field = field
        self._convert = convert

    def write(self, msg: Any) -> None:
        self._hub.publish(
            self.topic,
            lambda: pb2.SensorData(
                topic=self.topic, **{self._field: self._convert(msg)}
            ),
        )


def _header(h: msg_types.Header) -> pb2.Header:
    return pb2.Header(
        stamp=pb2.Timestamp(sec=h.stamp.sec, nanosec=h.stamp.nanosec),
        frame_id=h.frame_id,
    )


def _image(m: msg_types.Image) -> pb2.Image:
    return pb2.Image(
        header=_header(m.header),
        height=m.height,
        width=m.width,
        encoding=m.encoding,
        is_bigendian=m.is_bigendian,
        step=m.step,
        data=bytes(m.data),
    )


def _compressed_image(m: msg_types.CompressedImage) -> pb2.CompressedImage:
    return pb2.CompressedImage(
        header=_header(m.header), format=m.format, data=bytes(m.data)
    )


def _camera_info(m: msg_types.CameraInfo) -> pb2.CameraInfo:
    roi = m.roi
    return pb2.CameraInfo(
        header=_header(m.header),
        height=m.height,
        width=m.width,
        distortion_model=m.distortion_model,
        d=m.d,
        k=m.k,
        r=m.r,
        p=m.p,
        binning_x=m.binning_x,
        binning_y=m.binning_y,
        roi=pb2.RegionOfInterest(
            x_offset=roi.x_offset,
            y_offset=roi.y_offset,
            height=roi.height,
            width=roi.width,
            do_rectify=roi.do_rectify,
        ),
    )


def _point_cloud(m: msg_types.PointCloud2) -> pb2.PointCloud2:
    return pb2.PointCloud2(
        header=_header(m.header),
        height=m.height,
        width=m.width,
        fields=[
            pb2.PointField(
                name=f.name, offset=f.offset, datatype=f.datatype, count=f.count
            )
            for f in m.fields
        ],
        is_bigendian=m.is_bigendian,
        point_step=m.point_step,
        row_step=m.row_step,
        data=bytes(m.data),
        is_dense=m.is_dense,
    )


# DDS message type -> (SensorData payload field, converter).
_CONVERTERS: dict[type, tuple[str, Callable[[Any], Any]]] = {
    msg_types.Image: ("image", _image),
    msg_types.CompressedImage: ("compressed_image", _compressed_image),
    msg_types.CameraInfo: ("camera_info", _camera_info),
    msg_types.PointCloud2: ("point_cloud", _point_cloud),
}
