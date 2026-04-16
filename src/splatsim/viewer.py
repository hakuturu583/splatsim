from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import time

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QKeyEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

from splatsim.renderer import Renderer

if TYPE_CHECKING:
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
        self._position = torch.zeros(3, device=renderer.device, dtype=torch.float32)
        self._yaw: float = self._estimate_initial_yaw()

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

        # Render timer (~30 FPS)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._dt: float = 1.0 / 30.0

    def _estimate_initial_yaw(self, k: int = 512) -> float:
        """Estimate initial yaw by looking towards nearby Gaussians."""
        all_means: list[torch.Tensor] = []
        if self.scene is not None:
            if self.scene.background is not None:
                all_means.append(self.scene.background.tensors.means)
            for rb in self.scene.rigid_body_list:
                all_means.append(rb.tensors.means)
        if not all_means:
            return 0.0

        means = torch.cat(all_means, dim=0)
        diffs = means - self._position.unsqueeze(0)
        dists = diffs.norm(dim=1)

        k = min(k, len(dists))
        _, indices = dists.topk(k, largest=False)
        nearby = diffs[indices]

        xz = nearby[:, [0, 2]]
        norms = xz.norm(dim=1, keepdim=True).clamp(min=1e-8)
        unit_dirs = xz / norms
        mean_dir = unit_dirs.mean(dim=0)
        if mean_dir.norm() < 1e-8:
            return 0.0

        mx, mz = mean_dir[0].item(), mean_dir[1].item()
        return math.atan2(-mx, -mz)

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
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)

        r_yaw = torch.tensor(
            [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]],
            device=self.renderer.device,
            dtype=torch.float32,
        )

        r_w2c = r_yaw.T
        t_w2c = -r_w2c @ self._position

        viewmat = torch.eye(4, device=self.renderer.device, dtype=torch.float32)
        viewmat[:3, :3] = r_w2c
        viewmat[:3, 3] = t_w2c
        return viewmat

    def _handle_input(self, dt: float) -> None:
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)

        forward = torch.tensor(
            [-sin_y, 0.0, -cos_y], device=self.renderer.device, dtype=torch.float32
        )
        right = torch.tensor(
            [cos_y, 0.0, -sin_y], device=self.renderer.device, dtype=torch.float32
        )
        up = torch.tensor(
            [0.0, 1.0, 0.0], device=self.renderer.device, dtype=torch.float32
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

        # GPU tensor -> QPixmap
        image_np = (image.clamp(0.0, 1.0) * 255).byte().cpu().numpy()  # [H, W, 3]
        image_np = np.ascontiguousarray(image_np)
        h, w, _ = image_np.shape
        qimg = QImage(bytes(image_np.data), w, h, w * 3, QImage.Format.Format_RGB888)
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
    parser.add_argument("scene_yaml", type=Path, help="Path to scene YAML file")
    args = parser.parse_args()

    from splatsim.scene import load_scene

    viewer = load_scene(args.scene_yaml)
    viewer.run()


if __name__ == "__main__":
    main()
