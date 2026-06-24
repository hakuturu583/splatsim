from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QKeyEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

from splatsim.renderer import Renderer

if TYPE_CHECKING:
    from splatsim.cyclonedds.camera_info_publisher import CameraInfoPublisher
    from splatsim.cyclonedds.image_publisher import ImagePublisher
    from splatsim.scene import Scene


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
    ) -> None:
        # QApplication must exist before QMainWindow.__init__
        self._app = QApplication.instance() or QApplication(sys.argv)
        super().__init__()

        self.renderer = renderer
        self.scene = scene
        self.fov_y_deg = fov_y_deg
        self.move_speed = move_speed
        self.rotate_speed = rotate_speed

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
            self._yaw -= rot_speed
        if Qt.Key.Key_Right in self._keys_pressed:
            self._yaw += rot_speed

    def _tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_tick_time
        self._last_tick_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps += self._fps_alpha * (instant_fps - self._fps)

        self._handle_input(self._dt)
        viewmat = self._build_viewmat()

        with torch.no_grad():
            image = self.renderer.render(viewmat, self._K, scene=self.scene)

        # float32 RGB [H, W, 3] -> uint8 RGB [H, W, 3]
        image_np = (image.clamp(0.0, 1.0) * 255).byte().cpu().numpy()
        image_np = np.ascontiguousarray(image_np)

        # Publish to DDS
        if self._image_pub is not None:
            bgr_np = np.ascontiguousarray(image_np[:, :, ::-1])
            self._image_pub.publish(bgr_np)
            if self._camera_info_pub is not None:
                self._camera_info_pub.publish()

        # Qt display
        h, w, _ = image_np.shape
        qimg = QImage(image_np.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

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
        if self._image_pub is not None:
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
    """Entry point: ``uv run viewer scene.yaml``."""
    parser = argparse.ArgumentParser(description="splatsim interactive viewer")
    parser.add_argument(
        "scene_source",
        type=Path,
        help="Path to a scene YAML file or a scene USDZ archive",
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

    image_pub = None
    camera_info_pub = None
    if args.dds:
        import types

        try:
            from cyclonedds.domain import DomainParticipant
        except ImportError:
            raise SystemExit(
                "Error: --dds requires CycloneDDS.\n"
                "Install with: pip install splatsim[dds]"
            ) from None

        from splatsim.cyclonedds import CameraInfoPublisher, ImagePublisher

        dp = DomainParticipant()
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
            dp, cam_cfg, topic_name=args.topic_camera_info, frame_id=args.frame_id
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
    )
    viewer.run()


if __name__ == "__main__":
    main()
