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
    # Sensor orientation. Either RPY degrees ``(roll, pitch, yaw)`` (length 3)
    # or a unit quaternion ``(w, x, y, z)`` (length 4); the length selects the
    # format. Scene USDZ imports store the calibrated quaternion here.
    rotation: tuple[float, ...] = (0.0, 0.0, 0.0)
    # Explicit per-beam elevation table in degrees, ordered top→bottom. When
    # set it overrides the built-in ``sensor_type`` table and the uniform
    # fallback. Scene USDZ imports populate this from the rig's measured
    # LiDAR calibration (``lidar_model.parameters.elevation_deg``).
    elevation_deg: tuple[float, ...] | None = None
    pointcloud_topic: str = "/splatsim/lidar/pointcloud"
    frame_id: str = "splatsim_lidar"
    drop_threshold: float = 0.5
    alpha_threshold: float = 0.1
    # Transport for the rendered LiDAR data.
    #   "dds"  -> publish a sensor_msgs/PointCloud2 over CycloneDDS (default).
    #   "hils" -> emit raw Hesai UDP data packets that mimic the physical
    #             sensor (hardware-in-the-loop simulation). ``sensor_type``
    #             selects the wire format (OT128 / XT32).
    communication: str = "dds"
    # HILS UDP destination (only used when ``communication == "hils"``).
    hils_host: str = "127.0.0.1"
    hils_port: int = 2368
    # Wall-clock (Unix) time that simulation time 0 maps to, stamped into the
    # HILS packet date-time. ``None`` -> "now" when the sensor is created.
    hils_start_epoch: float | None = None
