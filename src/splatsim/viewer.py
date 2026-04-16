from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pygame
import torch

from splatsim.renderer import Renderer

if TYPE_CHECKING:
    from splatsim.scene import Scene


class Viewer:
    """Interactive real-time viewer using pygame."""

    def __init__(
        self,
        renderer: Renderer,
        scene: Scene | None = None,
        *,
        fov_y_deg: float = 60.0,
        move_speed: float = 5.0,
        rotate_speed: float = 1.5,
    ) -> None:
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

    def _estimate_initial_yaw(self, k: int = 512) -> float:
        """Estimate initial yaw by looking towards nearby Gaussians.

        Collects the k nearest Gaussians from all sources, computes each
        direction on the XZ unit circle, then takes the mean direction
        (spherical average on the 1-sphere) to derive the yaw.
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
        diffs = means - self._position.unsqueeze(0)  # [N, 3]
        dists = diffs.norm(dim=1)  # [N]

        k = min(k, len(dists))
        _, indices = dists.topk(k, largest=False)
        nearby = diffs[indices]  # [k, 3]

        # Project onto XZ plane and normalize to unit circle directions
        xz = nearby[:, [0, 2]]  # [k, 2]  (x, z)
        norms = xz.norm(dim=1, keepdim=True).clamp(min=1e-8)
        unit_dirs = xz / norms  # [k, 2] unit vectors on XZ circle

        # Spherical mean: average unit vectors then re-normalize
        mean_dir = unit_dirs.mean(dim=0)  # [2]
        if mean_dir.norm() < 1e-8:
            return 0.0

        # RUB forward is -Z, so yaw = atan2(-x, -z)
        mx, mz = mean_dir[0].item(), mean_dir[1].item()
        return math.atan2(-mx, -mz)

    def _build_intrinsics(self) -> torch.Tensor:
        fov_y = math.radians(self.fov_y_deg)
        fy = self.renderer.height / (2.0 * math.tan(fov_y / 2.0))
        fx = fy  # square pixels
        cx = self.renderer.width / 2.0
        cy = self.renderer.height / 2.0
        return torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            device=self.renderer.device,
            dtype=torch.float32,
        )

    def _build_viewmat(self) -> torch.Tensor:
        """Build world-to-camera 4x4 matrix from current camera state."""
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)

        # Rotation matrix for yaw (around Y axis)
        r_yaw = torch.tensor(
            [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]],
            device=self.renderer.device,
            dtype=torch.float32,
        )

        # World-to-camera: R^T, -R^T @ t
        r_w2c = r_yaw.T
        t_w2c = -r_w2c @ self._position

        viewmat = torch.eye(4, device=self.renderer.device, dtype=torch.float32)
        viewmat[:3, :3] = r_w2c
        viewmat[:3, 3] = t_w2c
        return viewmat

    def _handle_input(self, keys: pygame.key.ScancodeWrapper, dt: float) -> None:
        """Process keyboard input for camera movement and rotation."""
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)

        # Direction vectors in RUB coordinates
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

        # WSAD: forward/backward/left/right
        if keys[pygame.K_w]:
            self._position += forward * speed
        if keys[pygame.K_s]:
            self._position -= forward * speed
        if keys[pygame.K_a]:
            self._position -= right * speed
        if keys[pygame.K_d]:
            self._position += right * speed

        # Arrow up/down: vertical movement
        if keys[pygame.K_UP]:
            self._position += up * speed
        if keys[pygame.K_DOWN]:
            self._position -= up * speed

        # Arrow left/right: yaw rotation
        rot_speed = self.rotate_speed * dt
        if keys[pygame.K_LEFT]:
            self._yaw -= rot_speed
        if keys[pygame.K_RIGHT]:
            self._yaw += rot_speed

    def run(self) -> None:
        """Start the interactive viewer event loop."""
        pygame.init()
        screen = pygame.display.set_mode((self.renderer.width, self.renderer.height))
        pygame.display.set_caption("splatsim viewer")
        clock = pygame.time.Clock()

        running = True
        while running:
            dt = clock.tick(30) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            keys = pygame.key.get_pressed()
            self._handle_input(keys, dt)

            viewmat = self._build_viewmat()

            with torch.no_grad():
                image = self.renderer.render(viewmat, self._K, scene=self.scene)

            # Convert GPU tensor to pygame surface
            image_np = (image.clamp(0.0, 1.0) * 255).byte().cpu().numpy()  # [H, W, 3]
            # pygame.surfarray expects [W, H, 3]
            surface = pygame.surfarray.make_surface(image_np.transpose(1, 0, 2))
            screen.blit(surface, (0, 0))

            fps = clock.get_fps()
            pygame.display.set_caption(f"splatsim viewer | {fps:.1f} FPS")
            pygame.display.flip()

        pygame.quit()


def main() -> None:
    """Entry point: ``python -m splatsim.viewer scene.yaml``."""
    parser = argparse.ArgumentParser(description="splatsim interactive viewer")
    parser.add_argument("scene_yaml", type=Path, help="Path to scene YAML file")
    args = parser.parse_args()

    from splatsim.scene import load_scene

    viewer = load_scene(args.scene_yaml)
    viewer.run()


if __name__ == "__main__":
    main()
