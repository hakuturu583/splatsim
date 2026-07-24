from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LidarConfig:
    """LiDAR sensor entry (from a scene USDZ rig calibration or SceneConfig)."""

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

    @classmethod
    def for_sensor(cls, sensor_type: str, **overrides: object) -> "LidarConfig":
        """Build a config preloaded with faithful hardware defaults.

        For known models (see :data:`SENSOR_PRESETS`) this fills in realistic
        channel count, azimuth resolution, spin rate and range so the rendered
        scan matches the physical sensor. Any keyword ``overrides`` win over
        the preset. Unknown models fall back to the field defaults.
        """
        params: dict[str, object] = {"sensor_type": sensor_type}
        params.update(sensor_defaults(sensor_type))
        params.update(overrides)
        return cls(**params)  # ty: ignore[invalid-argument-type]


# Faithful per-model hardware defaults. Only fields that differ from the
# ``LidarConfig`` baseline (tuned for Hesai OT128) are listed; models without
# an entry keep the baseline field defaults.
SENSOR_PRESETS: dict[str, dict[str, float | int]] = {
    # Velodyne HDL-64E S3: 64 beams, 360° spin. Datasheet — vertical FOV
    # +2.0°/-24.9° (~26.9°), horizontal resolution 0.1728° at 10 Hz
    # (≈2083 azimuth samples per turn), range up to 120 m, 5-20 Hz spin
    # (10 Hz nominal), practical near clip ≈0.9 m. Per-beam elevations come
    # from the built-in "HDL64E" table in ``lidar_renderer``.
    "HDL64E": {
        "n_rows": 64,
        "n_columns": 2083,
        "fps": 10.0,
        "min_range_m": 0.9,
        "max_range_m": 120.0,
    },
}


def sensor_defaults(sensor_type: str) -> dict[str, float | int]:
    """Faithful hardware defaults for ``sensor_type`` (empty when unknown)."""
    return dict(SENSOR_PRESETS.get(sensor_type, {}))
