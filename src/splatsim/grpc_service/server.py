"""gRPC servicer for the SplatSim rendering service."""

from __future__ import annotations

import logging
import threading
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
from splatsim.grpc_service.frame_scheduler import FrameScheduler
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
        self._image_pub: ImagePublisher | None = None
        self._camera_info_pub: CameraInfoPublisher | None = None
        self._frame_rate: float = 30.0
        self._clock_initial_ns: int = 0

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
                frame_id = request.frame_id or "camera"
                self._image_pub = ImagePublisher(
                    dp,
                    topic_name=request.image_topic or "/splatsim/image_raw",
                    frame_id=frame_id,
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
                return pb2.InitializeResponse(success=True, scene_origin=scene_origin)

            except Exception as exc:
                logger.exception("Initialize failed")
                return pb2.InitializeResponse(success=False, message=str(exc))

    def StreamCameraData(
        self,
        request_iterator: Iterator[pb2.CameraData],
        context: grpc.ServicerContext,
    ) -> pb2.StreamSummary:
        """Consume timestamped camera poses, render, and publish via DDS."""
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
        scheduler = FrameScheduler(self._clock_initial_ns, self._frame_rate)
        frames_rendered = 0
        poses_received = 0

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

            while scheduler.should_render(time_ns):
                render_time_ns = scheduler.next_render_time_ns

                interpolated = pose_buffer.interpolate(render_time_ns)
                if interpolated is None:
                    logger.warning(
                        "Cannot interpolate at t=%d ns; skipping frame %d",
                        render_time_ns,
                        scheduler.frame_count,
                    )
                    scheduler.advance()
                    continue

                self._render_and_publish(interpolated, render_time_ns)
                frames_rendered += 1
                scheduler.advance()
                pose_buffer.trim_before(render_time_ns)

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

        viewmat = build_viewmat_from_pose(pose.position, pose.rotation, self._device)

        logger.debug(
            "Render pose: pos=(%.4f, %.4f, %.4f) rot_wxyz=(%.4f, %.4f, %.4f, %.4f)",
            *pose.position,
            *pose.rotation,
        )
        logger.debug("Viewmat:\n%s", viewmat.cpu().numpy())

        with torch.no_grad():
            rgb = self._renderer.render(viewmat, self._K, scene=self._scene)

        # float32 RGB [H, W, 3] → uint8 BGR [H, W, 3] (flip on GPU before transfer)
        bgr_np = (rgb.clamp(0.0, 1.0) * 255).byte()[:, :, [2, 1, 0]].cpu().numpy()

        sec, nanosec = divmod(render_time_ns, 1_000_000_000)
        stamp = Time(sec=sec, nanosec=nanosec)

        if self._image_pub is not None:
            self._image_pub.publish(bgr_np, stamp=stamp)
        if self._camera_info_pub is not None:
            self._camera_info_pub.publish(stamp=stamp)
