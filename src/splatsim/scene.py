from __future__ import annotations

from pathlib import Path

import torch

from splatsim.background import Background
from splatsim.dataclass import SceneConfig
from splatsim.renderer import Renderer
from splatsim.rigid_body import RigidBody
from splatsim.viewer import Viewer


def load_scene(config: SceneConfig | str | Path) -> Viewer:
    """Build a Viewer from a SceneConfig or a YAML file path."""
    if not isinstance(config, SceneConfig):
        config = SceneConfig.from_yaml(config)

    device = torch.device(config.renderer.device)

    # Background
    background: Background | None = None
    if config.background_tileset is not None:
        background = Background(
            config.background_tileset,
            device=device,
            use_sh=config.use_sh,
        )

    # Rigid bodies
    rigid_bodies: list[RigidBody] = []
    for rb_cfg in config.rigid_bodies:
        rb = RigidBody(rb_cfg.source, device=device, use_sh=rb_cfg.use_sh)
        rb.set_pose(rb_cfg.position, rb_cfg.rotation)
        rigid_bodies.append(rb)

    # Renderer
    rc = config.renderer
    renderer = Renderer(
        width=rc.width,
        height=rc.height,
        device=device,
        background_color=rc.background_color,
        near_plane=rc.near_plane,
        far_plane=rc.far_plane,
    )

    # Viewer
    vc = config.viewer
    return Viewer(
        renderer,
        background=background,
        rigid_bodies=rigid_bodies,
        fov_y_deg=vc.fov_y_deg,
        move_speed=vc.move_speed,
        rotate_speed=vc.rotate_speed,
    )
