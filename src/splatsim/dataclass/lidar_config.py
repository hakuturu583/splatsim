from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LidarConfig:
    """LiDAR sensor entry loaded from scene YAML."""

    name: str = "lidar"
    enabled: bool = True
    sensor_type: str = "OT128"
    n_rows: int = 128
    n_columns: int = 2048
    fps: float = 10.0
    min_range_m: float = 0.3
    max_range_m: float = 120.0
    position: tuple[float, float, float] = (0.0, 0.0, 1.8)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pointcloud_topic: str = "/splatsim/lidar/pointcloud"
    frame_id: str = "splatsim_lidar"
    drop_threshold: float = 0.5
    alpha_threshold: float = 0.1
