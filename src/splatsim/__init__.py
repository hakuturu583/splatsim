from splatsim.background import Background
from splatsim.dataclass import (
    RendererConfig,
    RigidBodyConfig,
    SceneConfig,
    ViewerConfig,
)
from splatsim.renderer import Renderer
from splatsim.rigid_body import RigidBody
from splatsim.scene import Scene, load_scene


def __getattr__(name: str):
    if name == "Viewer":
        from splatsim.viewer import Viewer

        return Viewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Background",
    "Renderer",
    "RendererConfig",
    "RigidBody",
    "RigidBodyConfig",
    "Scene",
    "SceneConfig",
    "Viewer",
    "ViewerConfig",
    "load_scene",
]
