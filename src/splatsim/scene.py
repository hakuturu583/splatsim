from __future__ import annotations

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
                self._lod_manager is not None
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
            self._lod_manager is not None
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
    ) -> Scene:
        """Build a Scene from a SceneConfig or YAML path."""
        if not isinstance(config, SceneConfig):
            config = SceneConfig.from_yaml(config)

        if device is None:
            device = torch.device(config.renderer.device)

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

        return Scene(
            background=background,
            rigid_bodies=rigid_bodies,
            lod_manager=lod_manager,
        )


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
    scene = Scene.from_config(config, device=device)

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
