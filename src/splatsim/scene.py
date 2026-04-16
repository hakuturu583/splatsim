from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from splatsim.background import Background
from splatsim.dataclass import SceneConfig
from splatsim.renderer import Renderer
from splatsim.rigid_body import RigidBody
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
    ) -> None:
        self.background = background
        self._rigid_bodies: dict[str, RigidBody] = rigid_bodies or {}

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

        background: Background | None = None
        if config.background_tileset is not None:
            background = Background(
                config.background_tileset,
                device=device,
                use_sh=config.use_sh,
            )

        rigid_bodies: dict[str, RigidBody] = {}
        for rb_cfg in config.rigid_bodies:
            rb = RigidBody(rb_cfg.source, device=device, use_sh=rb_cfg.use_sh)
            rb.set_pose(rb_cfg.position, rb_cfg.rotation)
            rigid_bodies[rb_cfg.name] = rb

        return Scene(background=background, rigid_bodies=rigid_bodies)


def load_scene(config: SceneConfig | str | Path) -> Viewer:
    """Build a Viewer from a SceneConfig or a YAML file path."""
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
    )

    vc = config.viewer
    return Viewer(
        renderer,
        scene=scene,
        fov_y_deg=vc.fov_y_deg,
        move_speed=vc.move_speed,
        rotate_speed=vc.rotate_speed,
    )
