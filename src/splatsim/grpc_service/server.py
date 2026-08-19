"""gRPC servicer for the SplatSim rendering service."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator

import grpc
import torch

from cyclonedds.domain import DomainParticipant

from splatsim.background import Background
from splatsim.cyclonedds import CameraInfoPublisher, ImagePublisher
from splatsim.cyclonedds.msg_types import Time
from splatsim.cyclonedds.pointcloud2_publisher import PointCloud2Publisher
from splatsim.dataclass.lidar_config import LidarConfig, sensor_defaults
from splatsim.dataclass.lod_config import LodConfig
from splatsim.lod import LodManager
from splatsim.grpc_service._generated import (
    rendering_service_pb2 as pb2,
    rendering_service_pb2_grpc as pb2_grpc,
)
from splatsim.grpc_service.pose_buffer import PoseBuffer, TimestampedPose
from splatsim.grpc_service.viewmat_builder import (
    build_base_to_world_from_pose,
    build_intrinsics,
    build_viewmat_from_pose,
)
from splatsim.lidar_renderer import (
    LidarRenderer,
    build_lidar_sensors_from_config,
    render_lidars_concurrent,
)
from splatsim.renderer import Renderer, render_cameras_concurrent
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


@dataclass
class _RigLidar:
    """One LiDAR on the rig, with everything the frame loop needs."""

    name: str
    renderer: LidarRenderer
    publisher: PointCloud2Publisher
    drop_threshold: float
    alpha_threshold: float


@dataclass
class _RigCamera:
    """One camera on the rig; ``cam_to_base`` is composed with the ego pose."""

    name: str
    renderer: Renderer
    K: torch.Tensor
    cam_to_base: torch.Tensor  # (4, 4) camera→base_link
    image_pub: ImagePublisher | None
    info_pub: CameraInfoPublisher | None


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

        # LiDAR state (populated by InitializeLidar; shares self._scene).
        self._lidar_renderer: LidarRenderer | None = None
        self._pointcloud_pub: PointCloud2Publisher | None = None
        self._lidar_frame_rate: float = 10.0
        self._lidar_drop_threshold: float = 0.5
        self._lidar_alpha_threshold: float = 0.1
        self._lidar_render_count: int = 0

        # Rig state: every LiDAR / camera driven off ONE base_link pose stream.
        # Rendering them together is what lets a frame share a single Gaussian
        # gather (see render_lidars_concurrent / render_cameras_concurrent), so
        # N sensors cost one multi-million-Gaussian transient buffer, not N.
        self._rig_lidars: list[_RigLidar] = []
        self._rig_cameras: list[_RigCamera] = []
        self._rig_frame_rate: float = 10.0
        self._rig_render_count: int = 0

    def _warmup(self, what: str, render_once: Callable[[], object]) -> None:
        """Render one throwaway frame so CUDA module loads / kernel JIT happen
        here instead of stalling the first streamed pose (measured ~320 ms of
        GIL hold on the first frame otherwise). Nothing is published. Failures
        are logged and swallowed: a broken warmup pose must not fail Initialize
        when the real pose stream would still work.
        """
        try:
            t0 = time.monotonic()
            with torch.no_grad():
                render_once()
            logger.info(
                "Warmup render (%s) done in %.0f ms",
                what,
                (time.monotonic() - t0) * 1000,
            )
        except Exception:
            logger.exception("Warmup render (%s) failed; continuing", what)

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

                # Build the LoD manager unconditionally so the octree LoD
                # index is pre-computed at load time, letting SetLod toggle LoD
                # on/off at runtime without re-loading the scene. This is also
                # what makes LoD active by default in the Docker / gRPC path,
                # which loads scenes directly rather than via Scene.from_config.
                lod_manager = LodManager(LodConfig())

                logger.info("Loading scene: %s", request.scene_path)
                background = Background(
                    request.scene_path,
                    device=device,
                    use_sh=request.use_sh,
                    lod_manager=lod_manager,
                )
                # Scene enables LoD by default whenever a manager is present;
                # only an explicit enable_lod=false overrides that initial state.
                self._scene = Scene(background=background, lod_manager=lod_manager)
                if request.HasField("enable_lod"):
                    self._scene.lod_enabled = request.enable_lod
                logger.info(
                    "Scene loaded: %d Gaussians (LoD %s)",
                    background.num_gaussians,
                    "on" if self._scene.lod_enabled else "off",
                )

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
                    radius_clip=getattr(request, "radius_clip", 0.0) or 0.0,
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

                renderer, K, scene = self._renderer, self._K, self._scene
                self._warmup(
                    "camera",
                    lambda: renderer.render(
                        torch.eye(4, device=device), K, scene=scene
                    ),
                )

                centroid = background.tile_local_centroid
                scene_origin = pb2.Vector3(
                    x=float(centroid[0]),
                    y=float(centroid[1]),
                    z=float(centroid[2]),
                )
                ecef_t = background.ecef_translation
                ecef_r = background.ecef_rotation
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

    def SetLod(
        self,
        request: pb2.SetLodRequest,
        context: grpc.ServicerContext,
    ) -> pb2.SetLodResponse:
        """Toggle Level-of-Detail filtering on the loaded scene at runtime.

        The render loop reads ``scene.lod_enabled`` per frame, so the new
        state takes effect on the next rendered frame. Enabling has no effect
        when the scene carries no pre-computed LoD index (the setter guards on
        the manager), in which case ``enabled`` in the response reflects the
        actual — still off — state.
        """
        with self._lock:
            if not self._initialized or self._scene is None:
                return pb2.SetLodResponse(
                    success=False,
                    enabled=False,
                    message="Scene not initialized. Call Initialize first.",
                )
            self._scene.lod_enabled = request.enabled
            effective = self._scene.lod_enabled
            logger.info(
                "SetLod: requested=%s effective=%s",
                request.enabled,
                effective,
            )
            message = ""
            if request.enabled and not effective:
                message = "LoD unavailable: no LoD index pre-computed for this scene."
            return pb2.SetLodResponse(
                success=True,
                enabled=effective,
                message=message,
            )

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

        return self._run_pose_stream(
            request_iterator,
            frame_rate=self._frame_rate,
            render_and_publish=self._render_and_publish,
        )

    def _run_pose_stream(
        self,
        request_iterator: Iterator[pb2.CameraData | pb2.LidarData | pb2.RigData],
        *,
        frame_rate: float,
        render_and_publish: Callable[[TimestampedPose, int], None],
    ) -> pb2.StreamSummary:
        """Shared two-thread pose-streaming loop for camera, LiDAR and rig.

        Reading from the gRPC stream and GPU rendering run on separate
        threads so slow rendering never blocks pose ingestion. The render
        loop always uses the latest available pose at ``frame_rate`` cadence,
        publishing through the supplied ``render_and_publish`` callback, and
        old poses are dropped from the buffer.
        """
        pose_buffer = PoseBuffer()
        stream_done = threading.Event()
        render_failed = threading.Event()
        frames_rendered = 0
        poses_received = 0

        frame_period_s = 1.0 / frame_rate

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

                    render_and_publish(latest, render_time_ns)
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
            for data in request_iterator:
                stamp = data.stamp
                time_ns = stamp.sec * 1_000_000_000 + stamp.nanosec

                p = data.pose.position
                r = data.pose.rotation
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

    # ── LiDAR ────────────────────────────────────────────────────────────

    def InitializeLidar(
        self,
        request: pb2.InitializeLidarRequest,
        context: grpc.ServicerContext,
    ) -> pb2.InitializeResponse:
        """Add a LiDAR sensor to the already-loaded scene.

        ``Initialize`` must have been called first — the LiDAR renderer shares
        the scene and DomainParticipant created there.
        """
        with self._lock:
            try:
                if not self._initialized or self._scene is None:
                    return pb2.InitializeResponse(
                        success=False,
                        message="Scene not initialized. Call Initialize first.",
                    )
                assert self._device is not None  # noqa: S101
                assert self._dp is not None  # noqa: S101

                # Rig form takes precedence; the single-sensor field stays
                # supported for existing clients.
                sensor_msgs = list(request.sensors) or [request.sensor]
                if len(sensor_msgs) > 1 or request.sensors:
                    return self._init_lidar_rig(sensor_msgs)

                s = sensor_msgs[0]
                ext = s.extrinsic
                pos = ext.position
                rot = ext.rotation  # wxyz
                elevation = tuple(s.elevation_deg) or None

                # Faithful hardware defaults for the requested model fill in
                # any field the client left at 0 (e.g. Velodyne HDL-64E gets
                # 64 beams / 2083 azimuth samples / 120 m range). Unknown
                # models fall back to the baseline literals below.
                d = sensor_defaults(s.sensor_type)
                cfg = LidarConfig(
                    name=s.name or "lidar",
                    sensor_type=s.sensor_type,
                    n_rows=int(s.n_rows) or int(d.get("n_rows", 128)),
                    n_columns=int(s.n_columns) or int(d.get("n_columns", 2048)),
                    fps=s.fps or d.get("fps", 10.0),
                    min_range_m=s.min_range_m or d.get("min_range_m", 0.3),
                    max_range_m=s.max_range_m or d.get("max_range_m", 120.0),
                    position=(pos.x, pos.y, pos.z),
                    rotation=(rot.w, rot.x, rot.y, rot.z),
                    elevation_deg=elevation,
                    pointcloud_topic=(
                        s.pointcloud_topic or "/splatsim/lidar/pointcloud"
                    ),
                    frame_id=s.frame_id or "splatsim_lidar",
                    drop_threshold=s.drop_threshold or 0.5,
                    alpha_threshold=s.alpha_threshold or 0.1,
                )

                spec = build_lidar_sensors_from_config([cfg])[0]
                lidar_renderer = LidarRenderer(
                    spec,
                    device=self._device,
                    min_range_m=cfg.min_range_m,
                    max_range_m=cfg.max_range_m,
                )
                self._lidar_renderer = lidar_renderer
                self._pointcloud_pub = PointCloud2Publisher(
                    self._dp,
                    topic_name=cfg.pointcloud_topic,
                    frame_id=cfg.frame_id,
                )
                self._lidar_frame_rate = cfg.fps
                self._lidar_drop_threshold = cfg.drop_threshold
                self._lidar_alpha_threshold = cfg.alpha_threshold

                logger.info(
                    "LiDAR initialized: name=%s rows=%d cols=%d %.1ffps topic=%s",
                    cfg.name,
                    self._lidar_renderer.n_rows,
                    self._lidar_renderer.n_columns,
                    cfg.fps,
                    cfg.pointcloud_topic,
                )

                lidar_scene = self._scene
                self._warmup(
                    f"lidar {cfg.name}",
                    lambda: lidar_renderer.panorama_to_pointcloud2_data(
                        lidar_renderer.render(
                            torch.eye(4, device=self._device), scene=lidar_scene
                        ),
                        drop_threshold=cfg.drop_threshold,
                        alpha_threshold=cfg.alpha_threshold,
                    ),
                )
                return pb2.InitializeResponse(success=True)

            except Exception as exc:
                logger.exception("InitializeLidar failed")
                return pb2.InitializeResponse(success=False, message=str(exc))

    def StreamLidarData(
        self,
        request_iterator: Iterator[pb2.LidarData],
        context: grpc.ServicerContext,
    ) -> pb2.StreamSummary:
        """Consume timestamped base_link poses, render LiDAR, and publish via DDS.

        Mirrors :meth:`StreamCameraData`: the gRPC ingestion thread and the GPU
        render loop run separately, and the render loop always uses the latest
        buffered pose at the sensor's spin rate.
        """
        if not self._initialized:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Service not initialized. Call Initialize first.",
            )
        if self._lidar_renderer is None or self._pointcloud_pub is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "LiDAR not initialized. Call InitializeLidar first.",
            )

        assert self._scene is not None  # noqa: S101
        assert self._device is not None  # noqa: S101

        return self._run_pose_stream(
            request_iterator,
            frame_rate=self._lidar_frame_rate,
            render_and_publish=self._render_and_publish_lidar,
        )

    # ── Rig: many LiDARs / cameras off one shared gather ─────────────────

    def _lidar_config_from_msg(self, s) -> LidarConfig:
        """Build a LidarConfig from a proto LidarSensorConfig (defaults filled)."""
        ext = s.extrinsic
        pos, rot = ext.position, ext.rotation  # rot is wxyz
        d = sensor_defaults(s.sensor_type)
        return LidarConfig(
            name=s.name or "lidar",
            sensor_type=s.sensor_type,
            n_rows=int(s.n_rows) or int(d.get("n_rows", 128)),
            n_columns=int(s.n_columns) or int(d.get("n_columns", 2048)),
            fps=s.fps or d.get("fps", 10.0),
            min_range_m=s.min_range_m or d.get("min_range_m", 0.3),
            max_range_m=s.max_range_m or d.get("max_range_m", 120.0),
            position=(pos.x, pos.y, pos.z),
            rotation=(rot.w, rot.x, rot.y, rot.z),
            elevation_deg=tuple(s.elevation_deg) or None,
            pointcloud_topic=(
                s.pointcloud_topic or f"/splatsim/{s.name or 'lidar'}/pointcloud"
            ),
            frame_id=s.frame_id or (s.name or "splatsim_lidar"),
            drop_threshold=s.drop_threshold or 0.5,
            alpha_threshold=s.alpha_threshold or 0.1,
        )

    def _init_lidar_rig(self, sensor_msgs) -> pb2.InitializeResponse:
        """Register several LiDARs that render together off one gather."""
        assert self._device is not None  # noqa: S101
        assert self._dp is not None  # noqa: S101

        cfgs = [self._lidar_config_from_msg(m) for m in sensor_msgs]
        names = [c.name for c in cfgs]
        if len(set(names)) != len(names):
            return pb2.InitializeResponse(
                success=False, message=f"duplicate LiDAR names: {names}"
            )

        specs = build_lidar_sensors_from_config(cfgs)
        rig: list[_RigLidar] = []
        for cfg, spec in zip(cfgs, specs):
            rig.append(
                _RigLidar(
                    name=cfg.name,
                    renderer=LidarRenderer(
                        spec,
                        device=self._device,
                        min_range_m=cfg.min_range_m,
                        max_range_m=cfg.max_range_m,
                    ),
                    publisher=PointCloud2Publisher(
                        self._dp,
                        topic_name=cfg.pointcloud_topic,
                        frame_id=cfg.frame_id,
                    ),
                    drop_threshold=cfg.drop_threshold,
                    alpha_threshold=cfg.alpha_threshold,
                )
            )
        self._rig_lidars = rig
        # The rig renders on one cadence; take the fastest sensor's rate so no
        # sensor is starved (slower ones simply repeat the latest pose).
        self._rig_frame_rate = max(c.fps for c in cfgs)
        logger.info(
            "LiDAR rig initialized: %d sensors (%s) at %.1f Hz",
            len(rig),
            ", ".join(names),
            self._rig_frame_rate,
        )

        assert self._scene is not None  # noqa: S101
        rig_scene = self._scene

        def _rig_once() -> None:
            panoramas = render_lidars_concurrent(
                [rl.renderer for rl in rig],
                torch.eye(4, device=self._device),
                rig_scene,
            )
            for rl, pano in zip(rig, panoramas):
                rl.renderer.panorama_to_pointcloud2_data(
                    pano,
                    drop_threshold=rl.drop_threshold,
                    alpha_threshold=rl.alpha_threshold,
                )

        self._warmup("lidar rig", _rig_once)
        return pb2.InitializeResponse(success=True)

    def InitializeCameraRig(
        self,
        request: pb2.InitializeCameraRigRequest,
        context: grpc.ServicerContext,
    ) -> pb2.InitializeResponse:
        """Register cameras driven off the shared base_link pose stream."""
        with self._lock:
            try:
                if not self._initialized or self._scene is None:
                    return pb2.InitializeResponse(
                        success=False,
                        message="Scene not initialized. Call Initialize first.",
                    )
                assert self._device is not None  # noqa: S101
                assert self._dp is not None  # noqa: S101

                names = [c.name for c in request.cameras]
                if len(set(names)) != len(names):
                    return pb2.InitializeResponse(
                        success=False, message=f"duplicate camera names: {names}"
                    )

                rig: list[_RigCamera] = []
                for c in request.cameras:
                    intr = c.intrinsics
                    ext = c.extrinsic
                    cam_to_base = build_base_to_world_from_pose(
                        (ext.position.x, ext.position.y, ext.position.z),
                        (
                            ext.rotation.w,
                            ext.rotation.x,
                            ext.rotation.y,
                            ext.rotation.z,
                        ),
                        self._device,
                    )
                    rig.append(
                        _RigCamera(
                            name=c.name or "camera",
                            renderer=Renderer(
                                width=intr.width,
                                height=intr.height,
                                device=self._device,
                            ),
                            K=build_intrinsics(
                                intr.fx, intr.fy, intr.cx, intr.cy, self._device
                            ),
                            cam_to_base=cam_to_base,
                            image_pub=ImagePublisher(
                                self._dp,
                                topic_name=(
                                    c.image_topic
                                    or f"/splatsim/{c.name or 'camera'}/image"
                                ),
                                frame_id=c.frame_id or (c.name or "splatsim_camera"),
                                compress_format=c.compress_format or "",
                            ),
                            info_pub=CameraInfoPublisher(
                                self._dp,
                                topic_name=(
                                    c.camera_info_topic
                                    or f"/splatsim/{c.name or 'camera'}/camera_info"
                                ),
                                frame_id=c.frame_id or (c.name or "splatsim_camera"),
                                config=_PinholeConfig(
                                    fx=intr.fx,
                                    fy=intr.fy,
                                    cx=intr.cx,
                                    cy=intr.cy,
                                    image_width=intr.width,
                                    image_height=intr.height,
                                ),
                            ),
                        )
                    )
                self._rig_cameras = rig
                if request.frame_rate:
                    self._rig_frame_rate = request.frame_rate
                logger.info(
                    "Camera rig initialized: %d cameras (%s)",
                    len(rig),
                    ", ".join(names),
                )

                def _cam_rig_once() -> None:
                    render_cameras_concurrent(
                        [rc.renderer for rc in rig],
                        [torch.linalg.inv(rc.cam_to_base) for rc in rig],
                        [rc.K for rc in rig],
                        scene=self._scene,
                        camera_names=[rc.name for rc in rig],
                    )

                self._warmup("camera rig", _cam_rig_once)
                return pb2.InitializeResponse(success=True)
            except Exception as exc:
                logger.exception("InitializeCameraRig failed")
                return pb2.InitializeResponse(success=False, message=str(exc))

    def StreamRigData(
        self,
        request_iterator: Iterator[pb2.RigData],
        context: grpc.ServicerContext,
    ) -> pb2.RigSummary:
        """Render every registered LiDAR and camera per streamed base_link pose."""
        if not self._initialized:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Service not initialized. Call Initialize first.",
            )
        if not self._rig_lidars and not self._rig_cameras:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "No rig registered. Call InitializeLidar (sensors) or "
                "InitializeCameraRig first.",
            )

        summary = self._run_pose_stream(
            request_iterator,
            frame_rate=self._rig_frame_rate,
            render_and_publish=self._render_and_publish_rig,
        )
        return pb2.RigSummary(
            frames_rendered=summary.frames_rendered,
            poses_received=summary.poses_received,
            lidars_rendered=summary.frames_rendered * len(self._rig_lidars),
            cameras_rendered=summary.frames_rendered * len(self._rig_cameras),
        )

    def _render_and_publish_rig(
        self,
        pose: TimestampedPose,
        render_time_ns: int,
    ) -> None:
        """One frame for the whole rig: one gather per modality, one stream each."""
        assert self._scene is not None  # noqa: S101
        assert self._device is not None  # noqa: S101

        t0 = time.monotonic()
        base_to_world = build_base_to_world_from_pose(
            pose.position, pose.rotation, self._device
        )
        sec, nanosec = divmod(render_time_ns, 1_000_000_000)
        stamp = Time(sec=sec, nanosec=nanosec)

        n_points = 0
        with torch.no_grad():
            if self._rig_lidars:
                panoramas = render_lidars_concurrent(
                    [rl.renderer for rl in self._rig_lidars],
                    base_to_world,
                    self._scene,
                )
                for rl, pano in zip(self._rig_lidars, panoramas):
                    records, n = rl.renderer.panorama_to_pointcloud2_data(
                        pano,
                        drop_threshold=rl.drop_threshold,
                        alpha_threshold=rl.alpha_threshold,
                    )
                    rl.publisher.publish_packed(records, n, stamp=stamp)
                    n_points += n
            t_lidar = time.monotonic()

            if self._rig_cameras:
                # world→camera = inv(base_to_world @ cam_to_base)
                viewmats = [
                    torch.linalg.inv(base_to_world @ rc.cam_to_base)
                    for rc in self._rig_cameras
                ]
                images = render_cameras_concurrent(
                    [rc.renderer for rc in self._rig_cameras],
                    viewmats,
                    [rc.K for rc in self._rig_cameras],
                    scene=self._scene,
                    camera_names=[rc.name for rc in self._rig_cameras],
                )
                for rc, rgb in zip(self._rig_cameras, images):
                    bgr = (rgb.clamp(0.0, 1.0) * 255).byte()[:, :, [2, 1, 0]]
                    if rc.image_pub is not None:
                        rc.image_pub.publish(bgr.cpu().numpy(), stamp=stamp)
                    if rc.info_pub is not None:
                        rc.info_pub.publish(stamp=stamp)
        t_end = time.monotonic()

        self._rig_render_count += 1
        if self._rig_render_count <= 5 or self._rig_render_count % 100 == 0:
            logger.info(
                "Rig frame #%d: %d LiDARs (%d pts) + %d cameras "
                "total=%.1fms (lidar=%.1f camera=%.1f)",
                self._rig_render_count,
                len(self._rig_lidars),
                n_points,
                len(self._rig_cameras),
                (t_end - t0) * 1000,
                (t_lidar - t0) * 1000,
                (t_end - t_lidar) * 1000,
            )

    def _render_and_publish_lidar(
        self,
        pose: TimestampedPose,
        render_time_ns: int,
    ) -> None:
        """Render one LiDAR panorama at *pose* and publish a PointCloud2."""
        assert self._lidar_renderer is not None  # noqa: S101
        assert self._pointcloud_pub is not None  # noqa: S101
        assert self._scene is not None  # noqa: S101
        assert self._device is not None  # noqa: S101

        t0 = time.monotonic()
        base_to_world = build_base_to_world_from_pose(
            pose.position, pose.rotation, self._device
        )
        with torch.no_grad():
            panorama = self._lidar_renderer.render(base_to_world, scene=self._scene)
            # Fast path: the point records are packed on the GPU and cross to
            # the host as one contiguous buffer (see panorama_to_pointcloud2_data).
            records, n_points = self._lidar_renderer.panorama_to_pointcloud2_data(
                panorama,
                drop_threshold=self._lidar_drop_threshold,
                alpha_threshold=self._lidar_alpha_threshold,
            )
        t_render = time.monotonic()

        sec, nanosec = divmod(render_time_ns, 1_000_000_000)
        stamp = Time(sec=sec, nanosec=nanosec)
        self._pointcloud_pub.publish_packed(records, n_points, stamp=stamp)
        t_publish = time.monotonic()

        self._lidar_render_count += 1
        if self._lidar_render_count <= 5 or self._lidar_render_count % 100 == 0:
            logger.info(
                "LiDAR render #%d: %d points total=%.1fms (render=%.1f publish=%.1f)",
                self._lidar_render_count,
                n_points,
                (t_publish - t0) * 1000,
                (t_render - t0) * 1000,
                (t_publish - t_render) * 1000,
            )
