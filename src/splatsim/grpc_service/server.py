"""gRPC servicer for the SplatSim rendering service."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import grpc
import torch

from cyclonedds.domain import DomainParticipant

from splatsim.background import Background
from splatsim.cyclonedds import CameraInfoPublisher, ImagePublisher
from splatsim.cyclonedds.msg_types import Time
from splatsim.grpc_service._generated import (
    rendering_service_pb2 as pb2,
    rendering_service_pb2_grpc as pb2_grpc,
)
from splatsim.grpc_service.pose_buffer import PoseBuffer, TimestampedPose
from splatsim.grpc_service.viewmat_builder import (
    build_intrinsics,
    build_viewmat_from_pose,
)
from splatsim.renderer import Renderer
from splatsim.scene import Scene

logger = logging.getLogger(__name__)


@dataclass
class _PinholeConfig:
    """Minimal CameraConfig-protocol implementation for CameraInfoPublisher."""

    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int


class RenderingServiceServicer(pb2_grpc.RenderingServiceServicer):
    """gRPC servicer that manages scene loading, rendering, and DDS publishing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialized = False

        self._scene: Scene | None = None
        self._renderer: Renderer | None = None
        self._K: torch.Tensor | None = None
        self._device: torch.device | None = None
        self._dp: DomainParticipant | None = None
        self._image_pub: ImagePublisher | None = None
        self._camera_info_pub: CameraInfoPublisher | None = None
        self._frame_rate: float = 30.0
        self._clock_initial_ns: int = 0
        self._render_count: int = 0

    def Initialize(
        self,
        request: pb2.InitializeRequest,
        context: grpc.ServicerContext,
    ) -> pb2.InitializeResponse:
        """Load scene, create renderer and DDS publishers."""
        with self._lock:
            try:
                device = torch.device(request.device or "cuda")
                self._device = device

                logger.info("Loading tileset: %s", request.tileset_path)
                background = Background(
                    request.tileset_path,
                    device=device,
                    use_sh=request.use_sh,
                )
                self._scene = Scene(background=background)
                logger.info("Scene loaded: %d Gaussians", background.num_gaussians)

                intr = request.intrinsics
                bg = request.background_color
                bg_color = (bg.x, bg.y, bg.z) if bg else (0.0, 0.0, 0.0)
                self._renderer = Renderer(
                    width=intr.width,
                    height=intr.height,
                    device=device,
                    background_color=bg_color,
                    near_plane=request.near_plane or 0.01,
                    far_plane=request.far_plane or 1000.0,
                )

                self._K = build_intrinsics(intr.fx, intr.fy, intr.cx, intr.cy, device)

                dp = DomainParticipant()
                self._dp = dp  # prevent GC from destroying DDS entities
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
                clk = request.clock_initial
                self._clock_initial_ns = (
                    clk.sec * 1_000_000_000 + clk.nanosec if clk else 0
                )

                self._initialized = True
                logger.info("Initialization complete (%.1f fps)", self._frame_rate)

                origin = background.origin
                scene_origin = pb2.Vector3(
                    x=float(origin[0]),
                    y=float(origin[1]),
                    z=float(origin[2]),
                )
                ecef_t = background._ecef_translation
                ecef_r = background._ecef_rotation
                return pb2.InitializeResponse(
                    success=True,
                    scene_origin=scene_origin,
                    ecef_translation=pb2.Vector3(
                        x=float(ecef_t[0]),
                        y=float(ecef_t[1]),
                        z=float(ecef_t[2]),
                    ),
                    ecef_rotation=ecef_r.flatten().tolist(),
                )

            except Exception as exc:
                logger.exception("Initialize failed")
                return pb2.InitializeResponse(success=False, message=str(exc))

    def StreamCameraData(
        self,
        request_iterator: Iterator[pb2.CameraData],
        context: grpc.ServicerContext,
    ) -> pb2.StreamSummary:
        """Consume timestamped camera poses, render, and publish via DDS.

        Reading from the gRPC stream and GPU rendering run on separate
        threads so that slow rendering never blocks pose ingestion.
        The render loop always uses the latest available pose, and old
        poses are automatically dropped from the buffer.
        """
        if not self._initialized:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Service not initialized. Call Initialize first.",
            )

        assert self._renderer is not None  # noqa: S101
        assert self._scene is not None  # noqa: S101
        assert self._K is not None  # noqa: S101
        assert self._device is not None  # noqa: S101

        pose_buffer = PoseBuffer()
        stream_done = threading.Event()
        render_failed = threading.Event()
        frames_rendered = 0
        poses_received = 0

        frame_period_s = 1.0 / self._frame_rate

        def _render_loop() -> None:
            """Render at frame_rate using the latest buffered pose."""
            nonlocal frames_rendered
            try:
                while not stream_done.is_set():
                    # Wait for a new pose; clear so we block again next iteration
                    if not pose_buffer.new_pose_event.wait(timeout=1.0):
                        continue
                    pose_buffer.new_pose_event.clear()

                    render_start = time.monotonic()

                    latest = pose_buffer.get_latest()
                    if latest is None:
                        continue

                    render_time_ns = latest.time_ns

                    if frames_rendered <= 3 or frames_rendered % 100 == 0:
                        logger.info(
                            "Render #%d: render_t=%d pos=(%.4f, %.4f, %.4f)",
                            frames_rendered,
                            render_time_ns,
                            latest.position[0],
                            latest.position[1],
                            latest.position[2],
                        )

                    self._render_and_publish(latest, render_time_ns)
                    frames_rendered += 1
                    pose_buffer.trim_before(render_time_ns)

                    elapsed = time.monotonic() - render_start
                    sleep_time = frame_period_s - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except Exception:
                logger.exception("Render loop failed")
                render_failed.set()

        render_thread = threading.Thread(target=_render_loop, daemon=True)
        render_thread.start()

        try:
            for camera_data in request_iterator:
                stamp = camera_data.stamp
                time_ns = stamp.sec * 1_000_000_000 + stamp.nanosec

                p = camera_data.pose.position
                r = camera_data.pose.rotation
                pose = TimestampedPose(
                    time_ns=time_ns,
                    position=(p.x, p.y, p.z),
                    rotation=(r.w, r.x, r.y, r.z),
                )
                pose_buffer.append(pose)
                poses_received += 1

                if poses_received <= 3 or poses_received % 100 == 0:
                    logger.info(
                        "Received pose #%d: t=%d ns pos=(%.4f, %.4f, %.4f)",
                        poses_received,
                        time_ns,
                        p.x,
                        p.y,
                        p.z,
                    )

                if render_failed.is_set():
                    logger.error("Render thread died, stopping stream reader")
                    break
        finally:
            stream_done.set()
            render_thread.join(timeout=10.0)

        logger.info(
            "Stream finished: poses_received=%d, frames_rendered=%d",
            poses_received,
            frames_rendered,
        )
        return pb2.StreamSummary(
            frames_rendered=frames_rendered,
            poses_received=poses_received,
        )

    def _render_and_publish(
        self,
        pose: TimestampedPose,
        render_time_ns: int,
    ) -> None:
        """Render a single frame at the interpolated pose and publish via DDS."""
        assert self._renderer is not None  # noqa: S101
        assert self._K is not None  # noqa: S101
        assert self._device is not None  # noqa: S101

        t0 = time.monotonic()
        viewmat = build_viewmat_from_pose(pose.position, pose.rotation, self._device)
        t_viewmat = time.monotonic()

        logger.debug(
            "Render pose: pos=(%.4f, %.4f, %.4f) rot_wxyz=(%.4f, %.4f, %.4f, %.4f)",
            *pose.position,
            *pose.rotation,
        )
        logger.debug("Viewmat:\n%s", viewmat.cpu().numpy())

        with torch.no_grad():
            rgb = self._renderer.render(viewmat, self._K, scene=self._scene)
        t_render = time.monotonic()

        # float32 RGB [H, W, 3] → uint8 BGR [H, W, 3] (flip on GPU before transfer)
        bgr_np = (rgb.clamp(0.0, 1.0) * 255).byte()[:, :, [2, 1, 0]].cpu().numpy()
        t_transfer = time.monotonic()

        sec, nanosec = divmod(render_time_ns, 1_000_000_000)
        stamp = Time(sec=sec, nanosec=nanosec)

        if self._image_pub is not None:
            self._image_pub.publish(bgr_np, stamp=stamp)
        if self._camera_info_pub is not None:
            self._camera_info_pub.publish(stamp=stamp)
        t_publish = time.monotonic()

        total_ms = (t_publish - t0) * 1000
        self._render_count += 1
        if self._render_count <= 5 or self._render_count % 100 == 0:
            logger.info(
                "Render timing #%d: total=%.1fms "
                "(viewmat=%.1f render=%.1f transfer=%.1f publish=%.1f)",
                self._render_count,
                total_ms,
                (t_viewmat - t0) * 1000,
                (t_render - t_viewmat) * 1000,
                (t_transfer - t_render) * 1000,
                (t_publish - t_transfer) * 1000,
            )
