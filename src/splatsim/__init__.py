"""SplatSim package.

Public names are exported lazily via :func:`__getattr__` so importing a light
subpackage (e.g. ``splatsim.omnidreams_service``, which needs no gsplat) does
not force the gsplat rasteriser stack to load. ``from splatsim import Renderer``
and ``import splatsim; splatsim.Renderer`` keep working unchanged — the first
access triggers the underlying import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    from splatsim.viewer import Viewer

# Public name -> (module, attribute) for lazy resolution.
_LAZY_EXPORTS = {
    "Background": ("splatsim.background", "Background"),
    "Renderer": ("splatsim.renderer", "Renderer"),
    "RigidBody": ("splatsim.rigid_body", "RigidBody"),
    "Scene": ("splatsim.scene", "Scene"),
    "load_scene": ("splatsim.scene", "load_scene"),
    "Viewer": ("splatsim.viewer", "Viewer"),
    "RendererConfig": ("splatsim.dataclass", "RendererConfig"),
    "RigidBodyConfig": ("splatsim.dataclass", "RigidBodyConfig"),
    "SceneConfig": ("splatsim.dataclass", "SceneConfig"),
    "ViewerConfig": ("splatsim.dataclass", "ViewerConfig"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    return getattr(module, target[1])


def __dir__() -> list[str]:
    return sorted(__all__)


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
