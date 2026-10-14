"""Sensor data over gRPC (``--sensor-transport=grpc``).

With a :class:`SensorDataHub` in place of the DDS participant, the regular
publishers must deliver what they would have written to DDS -- same topic
names, same payloads -- and the hub between them and the
``SubscribeSensorData`` stream must never make the render loop wait on a
client. The last tests drive the real servicer over an in-process channel.
"""

from __future__ import annotations

import threading
from concurrent import futures
from dataclasses import dataclass

import cv2
import grpc
import numpy as np
import pytest

from splatsim.cyclonedds import CameraInfoPublisher, ImagePublisher
from splatsim.cyclonedds.msg_types import Time
from splatsim.cyclonedds.pointcloud2_publisher import PointCloud2Publisher
from splatsim.grpc_service._generated import (
    rendering_service_pb2 as pb2,
    rendering_service_pb2_grpc as pb2_grpc,
)
from splatsim.grpc_service.sensor_stream import (
    DEFAULT_QUEUE_SIZE,
    SensorDataHub,
    sensor_transport_from_env,
)
from splatsim.grpc_service.server import RenderingServiceServicer

_STAMP = Time(sec=12, nanosec=345)


@dataclass
class _Pinhole:
    fx: float = 500.0
    fy: float = 510.0
    cx: float = 320.0
    cy: float = 240.0
    image_width: int = 640
    image_height: int = 480


def _image(h: int = 4, w: int = 6) -> np.ndarray:
    return np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)


# ── transport selection ──────────────────────────────────────────────


def test_transport_defaults_to_dds() -> None:
    assert sensor_transport_from_env({}) == "dds"
    assert sensor_transport_from_env({"SPLATSIM_SENSOR_TRANSPORT": ""}) == "dds"


def test_transport_from_env() -> None:
    assert sensor_transport_from_env({"SPLATSIM_SENSOR_TRANSPORT": "gRPC "}) == "grpc"


def test_unknown_transport_refused() -> None:
    with pytest.raises(ValueError, match="sensor transport"):
        sensor_transport_from_env({"SPLATSIM_SENSOR_TRANSPORT": "zenoh"})
    with pytest.raises(ValueError):
        RenderingServiceServicer(sensor_transport="zenoh")


# ── hub ──────────────────────────────────────────────────────────────


def test_nothing_built_without_subscribers() -> None:
    hub = SensorDataHub()
    built: list[str] = []
    hub.publish("/a", lambda: built.append("a") or pb2.SensorData(topic="/a"))
    assert built == []


def test_topic_filter_and_fan_out() -> None:
    hub = SensorDataHub()
    every = hub.subscribe()
    only_b = hub.subscribe(["/b"])
    hub.publish("/a", lambda: pb2.SensorData(topic="/a"))
    hub.publish("/b", lambda: pb2.SensorData(topic="/b"))

    got_every = [every.get(0), every.get(0)]
    assert [m.topic for m in got_every if m] == ["/a", "/b"]
    first = only_b.get(0)
    assert first is not None and first.topic == "/b"
    assert only_b.get(0) is None


def test_slow_subscriber_keeps_newest() -> None:
    hub = SensorDataHub()
    sub = hub.subscribe(queue_size=3)
    for i in range(5):
        hub.publish("/a", lambda i=i: pb2.SensorData(topic=f"/a{i}"))
    topics = []
    while (m := sub.get(0)) is not None:
        topics.append(m.topic)
    assert topics == ["/a2", "/a3", "/a4"]
    assert sub.dropped == 2


def test_default_queue_size() -> None:
    assert SensorDataHub().subscribe(queue_size=0)._queue.maxlen == DEFAULT_QUEUE_SIZE


def test_iter_messages_ends_on_close_and_unsubscribes() -> None:
    hub = SensorDataHub()
    sub = hub.subscribe()
    hub.publish("/a", lambda: pb2.SensorData(topic="/a"))
    received: list[str] = []

    def consume() -> None:
        for msg in hub.iter_messages(sub):
            received.append(msg.topic)

    t = threading.Thread(target=consume)
    t.start()
    sub.close()  # must end the stream once the queue is drained
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert received == ["/a"]
    assert hub._subscriptions == ()


# ── publishers ───────────────────────────────────────────────────────


def test_raw_image() -> None:
    hub = SensorDataHub()
    sub = hub.subscribe()
    img = _image()
    ImagePublisher(hub, topic_name="/cam/image", frame_id="cam").publish(
        img, stamp=_STAMP
    )
    msg = sub.get(0)
    assert msg is not None and msg.WhichOneof("payload") == "image"
    assert msg.topic == "/cam/image"
    im = msg.image
    assert (im.header.stamp.sec, im.header.stamp.nanosec) == (12, 345)
    assert im.header.frame_id == "cam"
    assert (im.height, im.width, im.step, im.encoding) == (4, 6, 18, "bgr8")
    np.testing.assert_array_equal(
        np.frombuffer(im.data, np.uint8).reshape(img.shape), img
    )


def test_compressed_image_topic_and_payload() -> None:
    hub = SensorDataHub()
    pub = ImagePublisher(hub, topic_name="/cam/image", compress_format="png")
    sub = hub.subscribe(["/cam/image/compressed"])
    img = _image()
    pub.publish(img, stamp=_STAMP)
    msg = sub.get(0)
    assert msg is not None and msg.WhichOneof("payload") == "compressed_image"
    assert msg.compressed_image.format == "png"
    decoded = cv2.imdecode(
        np.frombuffer(msg.compressed_image.data, np.uint8), cv2.IMREAD_COLOR
    )
    np.testing.assert_array_equal(decoded, img)


def test_unsupported_message_type_refused() -> None:
    with pytest.raises(TypeError, match="no gRPC mirror"):
        SensorDataHub().create_writer("/x", Time)


def test_camera_info() -> None:
    hub = SensorDataHub()
    sub = hub.subscribe()
    CameraInfoPublisher(hub, _Pinhole(), topic_name="/cam/info").publish(_STAMP)
    msg = sub.get(0)
    assert msg is not None and msg.WhichOneof("payload") == "camera_info"
    info = msg.camera_info
    assert (info.width, info.height) == (640, 480)
    assert list(info.k) == [500.0, 0.0, 320.0, 0.0, 510.0, 240.0, 0.0, 0.0, 1.0]
    assert len(info.p) == 12 and info.distortion_model == "plumb_bob"


def test_point_cloud_matches_dds_layout() -> None:
    hub = SensorDataHub()
    sub = hub.subscribe()
    records = np.arange(3 * 16, dtype=np.uint8)
    PointCloud2Publisher(hub, topic_name="/lidar", frame_id="top").publish_packed(
        records, 3, stamp=_STAMP
    )
    msg = sub.get(0)
    assert msg is not None and msg.WhichOneof("payload") == "point_cloud"
    pc = msg.point_cloud
    assert (pc.height, pc.width, pc.point_step, pc.row_step) == (1, 3, 16, 48)
    assert [f.name for f in pc.fields] == [
        "x",
        "y",
        "z",
        "intensity",
        "return_type",
        "channel",
    ]
    assert pc.data == records.tobytes()


# ── servicer ─────────────────────────────────────────────────────────


def test_grpc_transport_writes_to_the_hub() -> None:
    servicer = RenderingServiceServicer(sensor_transport="grpc")
    assert servicer._sink() is servicer._sensor_hub is not None
    assert servicer._dp is None


@pytest.fixture
def serve():
    """Start a servicer on an ephemeral port; yield (servicer, stub)."""
    started: list[grpc.Server] = []
    channels: list[grpc.Channel] = []

    def _serve(transport: str):
        servicer = RenderingServiceServicer(sensor_transport=transport)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        pb2_grpc.add_RenderingServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        started.append(server)
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        channels.append(channel)
        return servicer, pb2_grpc.RenderingServiceStub(channel)

    yield _serve
    for channel in channels:
        channel.close()
    for server in started:
        server.stop(grace=None)


def test_subscribe_refused_under_dds(serve) -> None:
    _, stub = serve("dds")
    with pytest.raises(grpc.RpcError) as err:
        next(iter(stub.SubscribeSensorData(pb2.SubscribeSensorDataRequest())))
    # The raised error is also the grpc.Call carrying the status.
    assert isinstance(err.value, grpc.Call)
    assert err.value.code() == grpc.StatusCode.FAILED_PRECONDITION


def test_subscribe_receives_published_frames(serve) -> None:
    servicer, stub = serve("grpc")
    hub = servicer._sensor_hub
    assert hub is not None
    call = stub.SubscribeSensorData(
        pb2.SubscribeSensorDataRequest(topics=["/lidar"]), timeout=10.0
    )
    stream = iter(call)

    # Wait for the server side to register the subscription before publishing.
    pause = threading.Event()
    for _ in range(200):
        if hub._subscriptions:
            break
        pause.wait(0.01)
    assert hub._subscriptions

    lidar = PointCloud2Publisher(servicer._sink(), topic_name="/lidar")
    camera = ImagePublisher(servicer._sink(), topic_name="/cam")
    camera.publish(_image(), stamp=_STAMP)  # filtered out
    lidar.publish_packed(np.zeros(32, np.uint8), 2, stamp=_STAMP)

    msg = next(stream)
    assert msg.topic == "/lidar"
    assert msg.point_cloud.width == 2

    call.cancel()
    for _ in range(200):
        if not hub._subscriptions:
            break
        pause.wait(0.01)
    assert not hub._subscriptions, "cancelled subscriber was not removed"
