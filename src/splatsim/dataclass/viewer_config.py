from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ViewerConfig:
    """Viewer camera and control settings."""

    fov_y_deg: float = 60.0
    move_speed: float = 5.0
    rotate_speed: float = 1.5
    initial_position: tuple[float, float, float] | None = None
    initial_yaw_deg: float | None = None
