from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RendererConfig:
    """Renderer settings."""

    width: int = 960
    height: int = 540
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    near_plane: float = 0.01
    far_plane: float = 1000.0
    device: str = "cuda"
    radius_clip: float = 0.0
    exposure: float = 1.0
