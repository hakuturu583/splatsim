from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LodTier:
    """A single LOD tier definition."""

    fraction: float = 1.0
    """Proportion of Gaussians to include (0.0 to 1.0)."""

    max_distance: float = float("inf")
    """Maximum camera distance (meters) for which this tier is used."""


@dataclass
class LodConfig:
    """Level-of-Detail configuration."""

    enabled: bool = True

    max_gaussians_per_cell: int = 50000
    """Octree leaf threshold. Gaussians are spatially partitioned into cells
    via recursive octree subdivision so that each leaf contains at most this
    many Gaussians."""

    tiers: list[LodTier] = field(
        default_factory=lambda: [
            LodTier(fraction=1.0, max_distance=50.0),
            LodTier(fraction=0.5, max_distance=150.0),
            LodTier(fraction=0.25, max_distance=300.0),
            LodTier(fraction=0.1, max_distance=float("inf")),
        ]
    )
