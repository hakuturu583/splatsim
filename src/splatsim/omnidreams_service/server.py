"""gRPC servicer for the OmniDreams world-model rendering backend.

Implements the SAME ``splatsim.v1.RenderingService`` contract as the 3D Gaussian
Splatting backend (:class:`splatsim.grpc_service.server.RenderingServiceServicer`)
so clients are backend-agnostic. The only difference visible to a client is that
this backend reads the optional ``InitializeRequest.initial_image`` anchor frame,
which the 3DGS backend ignores.

Scope: OmniDreams is a monocular camera world model, so ``Initialize`` +
``StreamCameraData`` are the live path. The LiDAR / rig RPCs return
``UNIMPLEMENTED`` rather than pretending to synthesise point clouds.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import grpc
import numpy as np

from cyclonedds.domain import DomainParticipant

from splatsim._geometry import mat4, quat_to_matrix
from splatsim.cyclonedds import CameraInfoPublisher, ImagePublisher
from splatsim.cyclonedds.msg_types import Time
from splatsim.grpc_service._generated import (
    rendering_service_pb2 as pb2,
    rendering_service_pb2_grpc as pb2_grpc,
)
from splatsim.grpc_service.pose_buffer import PoseBuffer, TimestampedPose
from splatsim.omnidreams_service.renderer import OmniDreamsRenderer

logger = logging.getLogger(__name__)


@dataclass
class _PinholeConfig:
    """Minimal CameraConfig-protocol implementation for CameraInfoPublisher.

    Defined locally (rather than reused from ``grpc_service.server``) so this
    backend never imports the gsplat rasteriser, keeping the OmniDreams image
    free of the CUDA kernels it does not need.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int


def _cam_to_world(
    position: tuple[float, float, float],
    rotation_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    """Build a 4x4 camera-to-world matrix from a pose.

    The world model consumes the ego camera-to-world pose directly (it is the
    trajectory conditioning), so — unlike the gsplat viewmat path — there is no
    inversion. The rotation is a ``(w, x, y, z)`` quaternion in the same RDF
    convention the 3DGS backend expects, so a client streams identical poses to
    either backend.
    """
    return mat4(quat_to_matrix(rotation_wxyz, order="wxyz"), position)


class OmniDreamsServicer(pb2_grpc.RenderingServiceServicer):
    """RenderingService backed by the OmniDreams autoregressive world model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialized = False
        self._renderer: OmniDreamsRenderer | None = None
        self._dp: DomainParticipant | None = None
        self._image_pub: ImagePublisher | None = None
        self._camera_info_pub: CameraInfoPublisher | None = None
        self._frame_rate: float = 30.0
        self._render_count: int = 0

    # -- Initialize --------------------------------------------------------

    def Initialize(
        self,
        request: pb2.InitializeRequest,
        context: grpc.ServicerContext,
    ) -> pb2.InitializeResponse:
        with self._lock:
            try:
                if not request.HasField("initial_image"):
                    return pb2.InitializeResponse(
                        success=False,
                        message=(
                            "OmniDreams backend requires InitializeRequest."
                            "initial_image (the anchor RGB frame)."
                        ),
                    )

                intr = request.intrinsics
                device = request.device or "cuda"

                self._renderer = OmniDreamsRenderer(
                    width=intr.width,
                    height=intr.height,
                    device=device,
                )
                logger.info("Seeding OmniDreams rollout from initial_image ...")
                self._renderer.seed(
                    request.initial_image,
                    scene_path=request.scene_path or None,
                )

                dp = DomainParticipant()
                self._dp = dp  # keep DDS entities alive
                frame_id = request.frame_id or "camera"
                self._image_pub = ImagePublisher(
                    dp,
                    topic_name=request.image_topic or "/splatsim/image_raw",
                    frame_id=frame_id,
                    compress_format=request.compress_format or "",
                )
                cam_config = _PinholeConfig(
                    fx=intr.fx,
                    fy=intr.fy,
                    cx=intr.cx,
                    cy=intr.cy,
                    image_width=intr.width,
                    image_height=intr.height,
                )
                self._camera_info_pub = CameraInfoPublisher(
                    dp,
                    cam_config,
                    topic_name=request.camera_info_topic or "/splatsim/camera_info",
                    frame_id=frame_id,
                )

                self._frame_rate = request.frame_rate or 30.0
                self._initialized = True
                logger.info(
                    "OmniDreams backend initialized (%dx%d, %.1f fps)",
                    intr.width,
                    intr.height,
                    self._frame_rate,
                )
                # The world model has no Gaussian re-centring or ECEF anchor, so
                # scene_origin / ecef_* stay at their zero defaults; clients that
                # need geolocation should read those from the scene bundle.
                return pb2.InitializeResponse(success=True)

            except Exception as exc:  # pragma: no cover - runtime/env specific
                logger.exception("OmniDreams Initialize failed")
                return pb2.InitializeResponse(success=False, message=str(exc))

    # -- Camera streaming --------------------------------------------------

    def StreamCameraData(
        self,
        request_iterator: Iterator[pb2.CameraData],
        context: grpc.ServicerContext,
    ) -> pb2.StreamSummary:
        if not self._initialized:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Service not initialized. Call Initialize first.",
            )

        assert self._renderer is not None  # noqa: S101

        pose_buffer = PoseBuffer()
        stream_done = threading.Event()
        render_failed = threading.Event()
        frames_rendered = 0
        poses_received = 0
        frame_period_s = 1.0 / self._frame_rate

        def _render_loop() -> None:
            nonlocal frames_rendered
            try:
                while not stream_done.is_set():
                    if not pose_buffer.new_pose_event.wait(timeout=1.0):
                        continue
                    pose_buffer.new_pose_event.clear()
                    render_start = time.monotonic()

                    latest = pose_buffer.get_latest()
                    if latest is None:
                        continue

                    self._render_and_publish(latest)
                    frames_rendered += 1
                    pose_buffer.trim_before(latest.time_ns)

                    sleep_time = frame_period_s - (time.monotonic() - render_start)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            except Exception:
                logger.exception("OmniDreams render loop failed")
                render_failed.set()

        render_thread = threading.Thread(target=_render_loop, daemon=True)
        render_thread.start()

        try:
            for data in request_iterator:
                stamp = data.stamp
                time_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
                p = data.pose.position
                r = data.pose.rotation
                pose_buffer.append(
                    TimestampedPose(
                        time_ns=time_ns,
                        position=(p.x, p.y, p.z),
                        rotation=(r.w, r.x, r.y, r.z),
                    )
                )
                poses_received += 1
                if render_failed.is_set():
                    logger.error("Render thread died, stopping stream reader")
                    break
        finally:
            stream_done.set()
            render_thread.join(timeout=10.0)

        logger.info(
            "OmniDreams stream finished: poses=%d frames=%d",
            poses_received,
            frames_rendered,
        )
        return pb2.StreamSummary(
            frames_rendered=frames_rendered,
            poses_received=poses_received,
        )

    def _render_and_publish(self, pose: TimestampedPose) -> None:
        assert self._renderer is not None  # noqa: S101
        c2w = _cam_to_world(pose.position, pose.rotation)
        rgb = self._renderer.render(c2w)  # H x W x 3 uint8 RGB
        # Single per-frame copy: reverse the channel stride to BGR and compact.
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])

        sec, nanosec = divmod(pose.time_ns, 1_000_000_000)
        stamp = Time(sec=sec, nanosec=nanosec)
        if self._image_pub is not None:
            self._image_pub.publish(bgr, stamp=stamp)
        if self._camera_info_pub is not None:
            self._camera_info_pub.publish(stamp=stamp)

        self._render_count += 1
        if self._render_count <= 5 or self._render_count % 100 == 0:
            logger.info("OmniDreams frame #%d published", self._render_count)

    # -- Unsupported modalities -------------------------------------------

    def _unimplemented(self, context: grpc.ServicerContext, what: str):
        context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            f"{what} is not supported by the OmniDreams camera world-model "
            "backend; use the 3D Gaussian Splatting backend for LiDAR / rigs.",
        )

    def InitializeLidar(self, request, context):  # noqa: D102, N802
        self._unimplemented(context, "InitializeLidar")

    def StreamLidarData(self, request_iterator, context):  # noqa: D102, N802
        self._unimplemented(context, "StreamLidarData")

    def InitializeCameraRig(self, request, context):  # noqa: D102, N802
        self._unimplemented(context, "InitializeCameraRig")

    def StreamRigData(self, request_iterator, context):  # noqa: D102, N802
        self._unimplemented(context, "StreamRigData")

    def SetLod(self, request, context):  # noqa: D102, N802
        # No Level-of-Detail concept in a world model; report it as unavailable
        # rather than aborting, matching the 3DGS "no LoD index" response shape.
        return pb2.SetLodResponse(
            success=True,
            enabled=False,
            message="LoD is not applicable to the OmniDreams backend.",
        )
