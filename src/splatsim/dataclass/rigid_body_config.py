from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RigidBodyConfig:
    """Configuration for a single rigid body in the scene."""

    source: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz
    use_sh: bool = False
