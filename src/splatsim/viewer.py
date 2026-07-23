from __future__ import annotations

import argparse
import math
import sys
import time
import types
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QKeyEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

from splatsim.cyclonedds.msg_types import Time
from splatsim.renderer import Renderer

if TYPE_CHECKING:
    from splatsim.cyclonedds.camera_info_publisher import CameraInfoPublisher
    from splatsim.cyclonedds.image_publisher import ImagePublisher
    from splatsim.cyclonedds.pointcloud2_publisher import PointCloud2Publisher
    from splatsim.dataclass import LidarConfig
    from splatsim.lidar_renderer import LidarRenderer
    from splatsim.scene import Scene


def _wall_clock_stamp() -> Time:
    """Wall-clock timestamp for the standalone viewer.

    Callers running under a sim clock (e.g. CARLA co-sim) must build their
    own ``Time`` from the sim time source instead of calling this.
    """
    sec, nanosec = divmod(time.time_ns(), 10**9)
    return Time(sec=sec, nanosec=nanosec)


# When a LiDAR panorama is shown instead of a camera image, the window is
# resized to something readable (panoramas are typically 2048 x 128 which is
# too flat at native size). The aspect ratio is preserved on the horizontal
# axis; the vertical axis is stretched by ``_LIDAR_ROW_SCALE`` so individual
# scan rows remain visible.
_LIDAR_DISPLAY_WIDTH: int = 1600
_LIDAR_ROW_SCALE: int = 4


def _turbo_like_colormap(t: np.ndarray) -> np.ndarray:
    """Cheap jet/turbo-flavoured colormap: t in [0, 1] -> (H, W, 3) float in [0, 1]."""
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _lidar_panorama_to_rgb(
    panorama: dict[str, torch.Tensor],
    *,
    min_range_m: float,
    max_range_m: float,
    alpha_threshold: float = 0.1,
) -> np.ndarray:
    """Colourise a LiDAR panorama for GUI display.

    Distance is turbo-mapped (near = red, far = blue) and modulated by
    per-pixel intensity so lit surfaces read as brighter. Invalid pixels
    (low alpha or out-of-range) are rendered black.
    """
    distance = panorama["distance"].detach().cpu().numpy()
    intensity = panorama["intensity"].detach().cpu().numpy()
    alpha = panorama["alpha"].detach().cpu().numpy()

    valid = (
        (alpha > alpha_threshold)
        & (distance >= min_range_m)
        & (distance <= max_range_m)
    )

    range_norm = np.clip(
        (distance - min_range_m) / max(max_range_m - min_range_m, 1e-6), 0.0, 1.0
    )
    rgb = _turbo_like_colormap(range_norm)
    intensity_gain = 0.4 + 0.6 * np.clip(intensity, 0.0, 1.0)
    rgb = rgb * intensity_gain[..., None]
    rgb[~valid] = 0.0
    return np.ascontiguousarray((rgb * 255.0 + 0.5).astype(np.uint8))


class Viewer(QMainWindow):
    """Interactive real-time viewer using PyQt5."""

    def __init__(
        self,
        renderer: Renderer,
        scene: Scene | None = None,
        *,
        fov_y_deg: float = 60.0,
        move_speed: float = 5.0,
        rotate_speed: float = 1.5,
        initial_position: tuple[float, float, float] | None = None,
        initial_yaw_deg: float | None = None,
        image_publisher: ImagePublisher | None = None,
        camera_info_publisher: CameraInfoPublisher | None = None,
        lidar_renderer: LidarRenderer | None = None,
        pointcloud_publisher: PointCloud2Publisher | None = None,
        lidar_drop_threshold: float = 0.5,
        lidar_alpha_threshold: float = 0.1,
        camera_name: str | None = None,
    ) -> None:
        # QApplication must exist before QMainWindow.__init__
        self._app = QApplication.instance() or QApplication(sys.argv)
        super().__init__()

        self.renderer = renderer
        self.scene = scene
        self.fov_y_deg = fov_y_deg
        self.move_speed = move_speed
        self.rotate_speed = rotate_speed
        self.lidar_renderer = lidar_renderer
        self.camera_name = camera_name

        # Camera state (RUB: +X=right, +Y=up, -Z=forward)
        if initial_position is not None:
            self._position = torch.tensor(
                initial_position, device=renderer.device, dtype=torch.float32
            )
        else:
            self._position = torch.zeros(3, device=renderer.device, dtype=torch.float32)
        if initial_yaw_deg is not None:
            self._yaw: float = math.radians(initial_yaw_deg)
        else:
            self._yaw = self._estimate_initial_yaw()

        # Pre-compute intrinsics
        self._K = self._build_intrinsics()

        # Track pressed keys
        self._keys_pressed: set[int] = set()

        # Qt setup
        self._label = QLabel(self)
        self.setCentralWidget(self._label)
        if lidar_renderer is not None:
            display_h = max(
                1,
                int(
                    round(
                        _LIDAR_DISPLAY_WIDTH
                        * lidar_renderer.n_rows
                        * _LIDAR_ROW_SCALE
                        / lidar_renderer.n_columns
                    )
                ),
            )
            self.setFixedSize(_LIDAR_DISPLAY_WIDTH, display_h)
            self.setWindowTitle(
                f"splatsim viewer (LiDAR: {lidar_renderer.sensor_spec.name})"
            )
        else:
            self.setFixedSize(renderer.width, renderer.height)
            self.setWindowTitle("splatsim viewer")

        # FPS tracking
        self._last_tick_time: float = time.monotonic()
        self._fps: float = 0.0
        self._fps_alpha: float = 0.1  # EMA smoothing

        # HUD font
        self._hud_font = QFont("monospace", 12)

        # DDS publishers (optional)
        self._image_pub = image_publisher
        self._camera_info_pub = camera_info_publisher
        self._pointcloud_pub = pointcloud_publisher
        self._lidar_drop_threshold = lidar_drop_threshold
        self._lidar_alpha_threshold = lidar_alpha_threshold

        # Render timer (~30 FPS)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._dt: float = 1.0 / 30.0

    def _estimate_initial_yaw(self) -> float:
        """Estimate initial yaw via PCA on the XY (horizontal) plane.

        Z is up in this tile-local data. Finds the longest horizontal
        axis and orients the camera to look along it.
        """
        all_means: list[torch.Tensor] = []
        if self.scene is not None:
            if self.scene.background is not None:
                all_means.append(self.scene.background.tensors.means)
            for rb in self.scene.rigid_body_list:
                all_means.append(rb.tensors.means)
        if not all_means:
            return 0.0

        means = torch.cat(all_means, dim=0)  # [N, 3]
        xy = means[:, [0, 1]]  # [N, 2] horizontal plane (Z=up)
        xy = xy - xy.mean(dim=0)

        cov = (xy.T @ xy) / xy.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        principal = eigenvectors[:, -1]  # [2]

        # forward = (sin(yaw), -cos(yaw), 0); align with principal (px, py)
        px, py = principal[0].item(), principal[1].item()
        return math.atan2(px, -py)

    def _build_intrinsics(self) -> torch.Tensor:
        fov_y = math.radians(self.fov_y_deg)
        fy = self.renderer.height / (2.0 * math.tan(fov_y / 2.0))
        fx = fy
        cx = self.renderer.width / 2.0
        cy = self.renderer.height / 2.0
        return torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            device=self.renderer.device,
            dtype=torch.float32,
        )

    def _build_viewmat(self) -> torch.Tensor:
        """Build world-to-camera 4x4 matrix.

        gsplat camera: +X=right, +Y=down, +Z=forward (RDF).
        World (tile-local): +Z=up, yaw rotates around Z.
        """
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)

        # Yaw around world Z. At yaw=0 camera looks along world -Y.
        #   cam_X (right)   = (-cos, -sin, 0)
        #   cam_Y (down)    = (0, 0, -1)   = world -Z (down)
        #   cam_Z (forward) = (sin, -cos, 0)
        r_w2c = torch.tensor(
            [
                [-cos_y, -sin_y, 0.0],
                [0.0, 0.0, -1.0],
                [sin_y, -cos_y, 0.0],
            ],
            device=self.renderer.device,
            dtype=torch.float32,
        )
        t_w2c = -r_w2c @ self._position

        viewmat = torch.eye(4, device=self.renderer.device, dtype=torch.float32)
        viewmat[:3, :3] = r_w2c
        viewmat[:3, 3] = t_w2c
        return viewmat

    def _handle_input(self, dt: float) -> None:
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)

        # World directions: Z=up, yaw around Z
        forward = torch.tensor(
            [sin_y, -cos_y, 0.0], device=self.renderer.device, dtype=torch.float32
        )
        right = torch.tensor(
            [-cos_y, -sin_y, 0.0], device=self.renderer.device, dtype=torch.float32
        )
        up = torch.tensor(
            [0.0, 0.0, 1.0], device=self.renderer.device, dtype=torch.float32
        )

        speed = self.move_speed * dt

        if Qt.Key.Key_W in self._keys_pressed:
            self._position += forward * speed
        if Qt.Key.Key_S in self._keys_pressed:
            self._position -= forward * speed
        if Qt.Key.Key_A in self._keys_pressed:
            self._position -= right * speed
        if Qt.Key.Key_D in self._keys_pressed:
            self._position += right * speed

        if Qt.Key.Key_Up in self._keys_pressed:
            self._position += up * speed
        if Qt.Key.Key_Down in self._keys_pressed:
            self._position -= up * speed

        rot_speed = self.rotate_speed * dt
        if Qt.Key.Key_Left in self._keys_pressed:
            self._yaw += rot_speed
        if Qt.Key.Key_Right in self._keys_pressed:
            self._yaw -= rot_speed

    def _build_base_to_world(self) -> torch.Tensor:
        """Build the 4x4 base_link->world transform for LiDAR rendering.

        base_link convention: +X forward, +Y left, +Z up. World: +Z up, yaw
        rotates around Z. At yaw=0 the camera looks along world -Y, so base
        +X aligns with world -Y and base +Y aligns with world +X.
        """
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)
        r_b2w = torch.tensor(
            [
                [sin_y, cos_y, 0.0],
                [-cos_y, sin_y, 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=self.renderer.device,
            dtype=torch.float32,
        )
        base_to_world = torch.eye(4, device=self.renderer.device, dtype=torch.float32)
        base_to_world[:3, :3] = r_b2w
        base_to_world[:3, 3] = self._position
        return base_to_world

    def _render_camera_np(self) -> np.ndarray:
        viewmat = self._build_viewmat()
        with torch.no_grad():
            image = self.renderer.render(
                viewmat,
                self._K,
                scene=self.scene,
                camera_name=self.camera_name,
            )
        image_np = (image.clamp(0.0, 1.0) * 255).byte().cpu().numpy()
        image_np = np.ascontiguousarray(image_np)

        if self._image_pub is not None:
            stamp = _wall_clock_stamp()
            bgr_np = np.ascontiguousarray(image_np[:, :, ::-1])
            self._image_pub.publish(bgr_np, stamp=stamp)
            if self._camera_info_pub is not None:
                self._camera_info_pub.publish(stamp=stamp)

        return image_np

    def _render_lidar_np(self) -> np.ndarray:
        assert self.lidar_renderer is not None
        assert self.scene is not None
        base_to_world = self._build_base_to_world()
        with torch.no_grad():
            panorama = self.lidar_renderer.render(base_to_world, scene=self.scene)
            if self._pointcloud_pub is not None:
                point_cloud = self.lidar_renderer.panorama_to_point_cloud(
                    panorama,
                    drop_threshold=self._lidar_drop_threshold,
                    alpha_threshold=self._lidar_alpha_threshold,
                )
                self._pointcloud_pub.publish(
                    point_cloud["xyz"],
                    point_cloud["intensity"],
                    channel=point_cloud["channel"],
                    stamp=_wall_clock_stamp(),
                )
        return _lidar_panorama_to_rgb(
            panorama,
            min_range_m=self.lidar_renderer.min_range_m,
            max_range_m=self.lidar_renderer.max_range_m,
            alpha_threshold=self._lidar_alpha_threshold,
        )

    def _tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_tick_time
        self._last_tick_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps += self._fps_alpha * (instant_fps - self._fps)

        self._handle_input(self._dt)

        if self.lidar_renderer is not None:
            image_np = self._render_lidar_np()
        else:
            image_np = self._render_camera_np()

        # Qt display
        h, w, _ = image_np.shape
        qimg = QImage(image_np.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        if self.lidar_renderer is not None:
            pixmap = pixmap.scaled(
                self.width(),
                self.height(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # Draw HUD overlay
        painter = QPainter(pixmap)
        painter.setFont(self._hud_font)
        painter.setPen(Qt.GlobalColor.white)
        x, y, z = (
            self._position[0].item(),
            self._position[1].item(),
            self._position[2].item(),
        )
        yaw_deg = math.degrees(self._yaw)
        lines = [
            f"XYZ: {x:+8.2f} {y:+8.2f} {z:+8.2f}",
            f"Yaw: {yaw_deg:+7.1f} deg",
            f"FPS: {self._fps:5.1f}",
        ]
        if self.scene is not None and self.scene.lod_manager is not None:
            status = "ON" if self.scene.lod_enabled else "OFF"
            lines.append(f"LOD: {status} (L to toggle)")
        if self.lidar_renderer is not None:
            spec = self.lidar_renderer.sensor_spec
            lines.append(
                f"LiDAR: {spec.name} ({spec.sensor_type or 'uniform'}, "
                f"{self.lidar_renderer.n_rows}x{self.lidar_renderer.n_columns})"
            )
        if self._image_pub is not None or self._pointcloud_pub is not None:
            lines.append("DDS: publishing")
        for i, line in enumerate(lines):
            painter.drawText(10, 20 + i * 18, line)
        painter.end()

        self._label.setPixmap(pixmap)

    # --- Qt event overrides ---

    def keyPressEvent(  # ty: ignore[invalid-method-override]
        self, event: QKeyEvent | None
    ) -> None:
        if event is None:
            return
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        if event.key() == Qt.Key.Key_L and self.scene is not None:
            self.scene.lod_enabled = not self.scene.lod_enabled
            return
        self._keys_pressed.add(event.key())

    def keyReleaseEvent(  # ty: ignore[invalid-method-override]
        self, event: QKeyEvent | None
    ) -> None:
        if event is None:
            return
        self._keys_pressed.discard(event.key())

    def run(self) -> None:
        """Start the interactive viewer."""
        self.show()
        self._timer.start(int(self._dt * 1000))
        self._app.exec_()


def main() -> None:
    """Entry point: ``uv run viewer scene.usdz``."""
    parser = argparse.ArgumentParser(description="splatsim interactive viewer")
    parser.add_argument(
        "scene_source",
        type=Path,
        help="Path to a scene USDZ archive",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help=(
            "Name of the rig camera in a scene USDZ to seed intrinsics and "
            "initial pose (e.g. CAM_FRONT). Defaults to the first camera in "
            "the first rig."
        ),
    )
    parser.add_argument(
        "--dds",
        action="store_true",
        help="Publish images and camera info via CycloneDDS",
    )
    parser.add_argument("--topic-image", default="/splatsim/image_raw")
    parser.add_argument("--topic-camera-info", default="/splatsim/camera_info")
    parser.add_argument("--frame-id", default="camera")
    parser.add_argument(
        "--compress-format",
        default="",
        help="Compress format for published images (e.g. 'jpeg', 'png'). "
        "Empty string (default) publishes raw sensor_msgs/Image.",
    )
    parser.add_argument(
        "--pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Initial camera position in tile-local coordinates",
    )
    parser.add_argument(
        "--yaw",
        type=float,
        default=None,
        help="Initial camera yaw in degrees",
    )
    parser.add_argument(
        "--mp4",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Render the scene along the selected rig camera's GT trajectory "
            "and write it to this MP4 file instead of launching the viewer. "
            "Requires a scene USDZ with a rig_trajectories sidecar."
        ),
    )
    parser.add_argument(
        "--mp4-fps",
        type=int,
        default=30,
        help="Frame rate of the MP4 written by --mp4 (default: 30)",
    )
    parser.add_argument(
        "--lod",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable/disable level-of-detail (LoD) filtering. Pass --lod or "
            "--no-lod to override; without either flag the scene file's "
            "default is used (USDZ and the dataclass default to enabled)."
        ),
    )
    parser.add_argument(
        "--lidar",
        default=None,
        metavar="NAME",
        help=(
            "Render a LiDAR panorama for the named sensor from the scene "
            "YAML's ``lidar_sensors`` list instead of the RGB camera. "
            "Movement / rotation controls are unchanged."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default: WARNING)",
    )
    args = parser.parse_args()

    import logging

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from splatsim.dataclass import SceneConfig

    config = SceneConfig.from_source(
        args.scene_source, camera_name=args.camera, lod_enabled=args.lod
    )

    if args.mp4 and args.dds:
        raise SystemExit("Error: --mp4 and --dds cannot be combined")
    if args.lidar is not None and not args.lidar:
        raise SystemExit("Error: --lidar requires a sensor name; got an empty string")
    if args.lidar is not None and args.mp4:
        raise SystemExit("Error: --lidar and --mp4 cannot be combined")
    lidar_cfg = (
        _find_lidar_config(config, args.lidar) if args.lidar is not None else None
    )

    image_pub = None
    camera_info_pub = None
    pointcloud_pub = None
    if args.dds:
        try:
            from cyclonedds.domain import DomainParticipant
        except ImportError:
            raise SystemExit(
                "Error: --dds requires CycloneDDS.\n"
                "Install with: pip install splatsim[dds]"
            ) from None

        dp = DomainParticipant()
        if lidar_cfg is None:
            from splatsim.cyclonedds import CameraInfoPublisher, ImagePublisher

            image_pub = ImagePublisher(
                dp,
                topic_name=args.topic_image,
                frame_id=args.frame_id,
                compress_format=args.compress_format,
            )

            # Build a config-like object with the same intrinsics the Viewer uses.
            rc, vc = config.renderer, config.viewer
            fov_y = math.radians(vc.fov_y_deg)
            fy = rc.height / (2.0 * math.tan(fov_y / 2.0))
            cam_cfg = types.SimpleNamespace(
                fx=fy,
                fy=fy,
                cx=rc.width / 2.0,
                cy=rc.height / 2.0,
                image_width=rc.width,
                image_height=rc.height,
            )
            camera_info_pub = CameraInfoPublisher(
                dp,
                cam_cfg,
                topic_name=args.topic_camera_info,
                frame_id=args.frame_id,
            )
        else:
            from splatsim.cyclonedds import PointCloud2Publisher

            pointcloud_pub = PointCloud2Publisher(
                dp,
                topic_name=lidar_cfg.pointcloud_topic,
                frame_id=lidar_cfg.frame_id,
            )

    from splatsim.scene import Scene, print_progress

    device = torch.device(config.renderer.device)
    scene = Scene.from_config(config, device=device, progress=print_progress)

    rc = config.renderer
    renderer = Renderer(
        width=rc.width,
        height=rc.height,
        device=device,
        background_color=rc.background_color,
        near_plane=rc.near_plane,
        far_plane=rc.far_plane,
        radius_clip=rc.radius_clip,
        exposure=rc.exposure,
        ppisp_knn_k=rc.ppisp_knn_k,
    )

    if args.mp4:
        from splatsim._mp4 import render_trajectory_mp4

        n_frames = render_trajectory_mp4(
            scene,
            renderer,
            args.scene_source,
            output_path=args.mp4,
            camera_name=args.camera,
            fps=args.mp4_fps,
        )
        print(f"Wrote {n_frames} frames to {args.mp4}")
        return

    vc = config.viewer
    from splatsim.scene import resolve_initial_pose

    initial_position, initial_yaw_deg = resolve_initial_pose(
        config,
        scene.background,
        override_position=tuple(args.pos) if args.pos is not None else None,
        override_yaw_deg=args.yaw,
    )

    lidar_renderer = None
    if lidar_cfg is not None:
        lidar_renderer = _build_lidar_renderer(lidar_cfg, device)

    viewer = Viewer(
        renderer,
        scene=scene,
        fov_y_deg=vc.fov_y_deg,
        move_speed=vc.move_speed,
        rotate_speed=vc.rotate_speed,
        initial_position=initial_position,
        initial_yaw_deg=initial_yaw_deg,
        image_publisher=image_pub,
        camera_info_publisher=camera_info_pub,
        lidar_renderer=lidar_renderer,
        pointcloud_publisher=pointcloud_pub,
        lidar_drop_threshold=lidar_cfg.drop_threshold if lidar_cfg is not None else 0.5,
        lidar_alpha_threshold=(
            lidar_cfg.alpha_threshold if lidar_cfg is not None else 0.1
        ),
        camera_name=args.camera,
    )
    viewer.run()


def _find_lidar_config(config, lidar_name: str) -> LidarConfig:
    lidar_cfg = next((s for s in config.lidar_sensors if s.name == lidar_name), None)
    if lidar_cfg is None:
        available = ", ".join(sorted(s.name for s in config.lidar_sensors)) or "<none>"
        raise SystemExit(
            f"Error: LiDAR sensor '{lidar_name}' not found in scene. "
            f"Available sensors: {available}"
        )
    return lidar_cfg


def _build_lidar_renderer(
    lidar_cfg: LidarConfig, device: torch.device
) -> LidarRenderer:
    from splatsim.lidar_renderer import LidarRenderer, build_lidar_sensors_from_config

    specs = build_lidar_sensors_from_config([lidar_cfg])
    return LidarRenderer(
        specs[0],
        device=device,
        min_range_m=lidar_cfg.min_range_m,
        max_range_m=lidar_cfg.max_range_m,
    )


if __name__ == "__main__":
    main()
