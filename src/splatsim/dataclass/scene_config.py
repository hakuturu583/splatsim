from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from splatsim.dataclass.renderer_config import RendererConfig
from splatsim.dataclass.rigid_body_config import RigidBodyConfig
from splatsim.dataclass.viewer_config import ViewerConfig


@dataclass
class SceneConfig:
    """Top-level scene configuration loaded from YAML."""

    background_tileset: str | None = None
    use_sh: bool = False
    rigid_bodies: list[RigidBodyConfig] = field(default_factory=list)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> SceneConfig:
        """Load a SceneConfig from a YAML file.

        Paths in the YAML are resolved relative to the YAML file's directory.
        """
        path = Path(path)
        with path.open() as f:
            raw = yaml.safe_load(f)

        base_dir = path.parent

        # Background
        bg_tileset = raw.get("background_tileset")
        if bg_tileset is not None:
            bg_tileset = str(base_dir / bg_tileset)

        # Rigid bodies
        rigid_bodies: list[RigidBodyConfig] = []
        for rb in raw.get("rigid_bodies", []):
            source = str(base_dir / rb["source"])
            position = tuple(rb.get("position", [0.0, 0.0, 0.0]))
            rotation = tuple(rb.get("rotation", [1.0, 0.0, 0.0, 0.0]))
            rigid_bodies.append(
                RigidBodyConfig(
                    source=source,
                    position=position,
                    rotation=rotation,
                    use_sh=rb.get("use_sh", False),
                )
            )

        # Renderer
        renderer_raw = raw.get("renderer", {})
        bg_color = renderer_raw.get("background_color", [0.0, 0.0, 0.0])
        renderer = RendererConfig(
            width=renderer_raw.get("width", 960),
            height=renderer_raw.get("height", 540),
            background_color=tuple(bg_color),
            near_plane=renderer_raw.get("near_plane", 0.01),
            far_plane=renderer_raw.get("far_plane", 1000.0),
            device=renderer_raw.get("device", "cuda"),
        )

        # Viewer
        viewer_raw = raw.get("viewer", {})
        viewer = ViewerConfig(
            fov_y_deg=viewer_raw.get("fov_y_deg", 60.0),
            move_speed=viewer_raw.get("move_speed", 5.0),
            rotate_speed=viewer_raw.get("rotate_speed", 1.5),
        )

        return SceneConfig(
            background_tileset=bg_tileset,
            use_sh=raw.get("use_sh", False),
            rigid_bodies=rigid_bodies,
            renderer=renderer,
            viewer=viewer,
        )
