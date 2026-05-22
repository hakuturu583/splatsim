from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from splatsim._conversions import GaussianTensors, apply_rigid_transform
from splatsim.background import Background
from splatsim.dataclass import SceneConfig
from splatsim.lod import LodIndex, LodManager
from splatsim.renderer import Renderer
from splatsim.rigid_body import RigidBody

if TYPE_CHECKING:
    from splatsim.cyclonedds.camera_info_publisher import CameraInfoPublisher
    from splatsim.cyclonedds.image_publisher import ImagePublisher
    from splatsim.viewer import Viewer


class Scene:
    """Manages background and rigid bodies that compose a renderable scene.

    Rigid bodies are accessible by name for external pose manipulation::

        scene = Scene.from_config(cfg)
        scene["car_01"].set_pose((10.0, 0.0, -5.0))
    """

    def __init__(
        self,
        background: Background | None = None,
        rigid_bodies: dict[str, RigidBody] | None = None,
        lod_manager: LodManager | None = None,
    ) -> None:
        self.background = background
        self._rigid_bodies: dict[str, RigidBody] = rigid_bodies or {}
        self._lod_manager = lod_manager
        self._lod_enabled = lod_manager is not None

    # --- rigid body access ---------------------------------------------------

    def __getitem__(self, name: str) -> RigidBody:
        return self._rigid_bodies[name]

    def __contains__(self, name: str) -> bool:
        return name in self._rigid_bodies

    @property
    def rigid_bodies(self) -> dict[str, RigidBody]:
        return self._rigid_bodies

    @property
    def rigid_body_list(self) -> list[RigidBody]:
        return list(self._rigid_bodies.values())

    def add_rigid_body(self, name: str, rigid_body: RigidBody) -> None:
        self._rigid_bodies[name] = rigid_body

    def remove_rigid_body(self, name: str) -> RigidBody:
        return self._rigid_bodies.pop(name)

    # --- pose helpers --------------------------------------------------------

    def set_pose(
        self,
        name: str,
        position: tuple[float, float, float] | Tensor,
        rotation: tuple[float, float, float, float] | Tensor | None = None,
    ) -> None:
        """Set the pose of a rigid body by name."""
        self._rigid_bodies[name].set_pose(position, rotation)

    # --- LOD-aware tensor collection -----------------------------------------

    @property
    def lod_enabled(self) -> bool:
        return self._lod_enabled and self._lod_manager is not None

    @lod_enabled.setter
    def lod_enabled(self, value: bool) -> None:
        self._lod_enabled = value

    @property
    def lod_manager(self) -> LodManager | None:
        return self._lod_manager

    def collect_tensors(
        self, camera_position: tuple[float, float, float] | None = None
    ) -> list[GaussianTensors]:
        """Collect Gaussian tensors from all sources, applying LOD if enabled.

        When *camera_position* is provided and an :class:`LodManager` is
        configured, each source's tensors are filtered to the appropriate
        LOD tier based on camera-to-centroid distance.
        """
        result: list[GaussianTensors] = []

        if self.background is not None:
            tensors = self._filter_lod(
                self.background.tensors, self.background.lod_index, camera_position
            )
            result.append(tensors)

        for rb in self._rigid_bodies.values():
            if (
                self._lod_enabled
                and self._lod_manager is not None
                and camera_position is not None
                and rb.lod_index is not None
            ):
                # Slice base tensors *before* the rigid transform so we
                # only pay the matrix math for the Gaussians we keep.
                base = self._lod_manager.filter(
                    rb.base_tensors, rb.lod_index, camera_position
                )
                tensors = apply_rigid_transform(base, rb.position, rb.rotation)
            else:
                tensors = rb.tensors
            result.append(tensors)

        return result

    def _filter_lod(
        self,
        tensors: GaussianTensors,
        lod_index: LodIndex | None,
        camera_position: tuple[float, float, float] | None,
    ) -> GaussianTensors:
        """Apply LOD filtering if a manager and camera position are available."""
        if (
            self._lod_enabled
            and self._lod_manager is not None
            and camera_position is not None
            and lod_index is not None
        ):
            return self._lod_manager.filter(tensors, lod_index, camera_position)
        return tensors

    # --- construction --------------------------------------------------------

    @staticmethod
    def from_config(
        config: SceneConfig | str | Path,
        *,
        device: torch.device | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> Scene:
        """Build a Scene from a SceneConfig or YAML path.

        Args:
            progress: Optional callback ``(step, total, label)`` invoked
                after each major loading stage completes.
        """
        if not isinstance(config, SceneConfig):
            config = SceneConfig.from_yaml(config)

        if device is None:
            device = torch.device(config.renderer.device)

        has_bg = config.background_tileset is not None
        total = int(has_bg) + len(config.rigid_bodies)
        step = 0

        lod_manager: LodManager | None = None
        if config.lod.enabled:
            lod_manager = LodManager(config.lod)

        background: Background | None = None
        if config.background_tileset is not None:
            background = Background(
                config.background_tileset,
                device=device,
                use_sh=config.use_sh,
                lod_manager=lod_manager,
            )
            step += 1
            if progress is not None:
                progress(step, total, "background")
            if background.lod_index is not None:
                _log_lod_tiers("background", background.lod_index)

        rigid_bodies: dict[str, RigidBody] = {}
        for rb_cfg in config.rigid_bodies:
            rb = RigidBody(
                rb_cfg.source,
                device=device,
                use_sh=rb_cfg.use_sh,
                lod_manager=lod_manager,
            )
            rb.set_pose(rb_cfg.position, rb_cfg.rotation)
            rigid_bodies[rb_cfg.name] = rb
            step += 1
            if progress is not None:
                progress(step, total, rb_cfg.name)
            if rb.lod_index is not None:
                _log_lod_tiers(rb_cfg.name, rb.lod_index)

        return Scene(
            background=background,
            rigid_bodies=rigid_bodies,
            lod_manager=lod_manager,
        )


def _log_lod_tiers(name: str, lod_index: LodIndex) -> None:
    """Log the Gaussian distribution across LOD tiers."""
    total_n = lod_index.tier_counts[0] if lod_index.tier_counts else 0
    # tier_counts are cumulative (each is a prefix length), so find the
    # total from the largest tier.
    for c in lod_index.tier_counts:
        if c > total_n:
            total_n = c

    lines = [f"  LOD tiers for '{name}' (total: {total_n:,} Gaussians):"]
    for i, (count, max_d) in enumerate(
        zip(lod_index.tier_counts, lod_index.tier_max_distances)
    ):
        pct = 100.0 * count / total_n if total_n > 0 else 0.0
        dist_str = f"{max_d:.0f}m" if max_d < float("inf") else "inf"
        lines.append(
            f"    Tier {i}: {count:>10,} Gaussians ({pct:5.1f}%) | max_distance={dist_str}"
        )
    sys.stderr.write("\n".join(lines) + "\n")
    sys.stderr.flush()


def print_progress(step: int, total: int, label: str) -> None:
    """Print a terminal progress bar to *stderr*."""
    width = 40
    filled = int(width * step / total) if total > 0 else width
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    pct = 100.0 * step / total if total > 0 else 100.0
    sys.stderr.write(f"\r  Loading: |{bar}| {pct:5.1f}% ({label})")
    if step >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def load_scene(
    config: SceneConfig | str | Path,
    *,
    image_publisher: ImagePublisher | None = None,
    camera_info_publisher: CameraInfoPublisher | None = None,
) -> Viewer:
    """Build a Viewer from a SceneConfig or a YAML file path."""
    from splatsim.viewer import Viewer

    if not isinstance(config, SceneConfig):
        config = SceneConfig.from_yaml(config)

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
    )

    vc = config.viewer
    return Viewer(
        renderer,
        scene=scene,
        fov_y_deg=vc.fov_y_deg,
        move_speed=vc.move_speed,
        rotate_speed=vc.rotate_speed,
        image_publisher=image_publisher,
        camera_info_publisher=camera_info_publisher,
    )
