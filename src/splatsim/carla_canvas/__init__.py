"""splatsim as a carla_canvas renderer backend.

``renderer.backend: splatsim.carla_canvas:SplatsimRenderer`` in a carla_canvas
configuration renders a typesafe_carla program's CARLA cameras and LiDARs with
splatsim (see :mod:`splatsim.carla_canvas.renderer` and docs/carla_canvas.md).
"""

from __future__ import annotations


def __getattr__(name: str):
    if name == "SplatsimRenderer":  # imports carla_canvas, torch and gsplat
        from splatsim.carla_canvas.renderer import SplatsimRenderer

        return SplatsimRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["SplatsimRenderer"]
