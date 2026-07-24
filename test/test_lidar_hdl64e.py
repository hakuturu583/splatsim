from __future__ import annotations

import math

from splatsim.dataclass.lidar_config import LidarConfig, sensor_defaults
from splatsim.lidar_renderer import (
    build_lidar_sensors_from_config,
    elevations_rad,
    is_known_sensor,
)


def test_hdl64e_is_registered() -> None:
    assert is_known_sensor("HDL64E")


def test_hdl64e_elevation_table() -> None:
    el = elevations_rad("HDL64E")
    # Velodyne HDL-64E: 64 beams in two stacked 32-laser blocks.
    assert len(el) == 64
    # Ordered top -> bottom (strictly descending).
    assert all(a > b for a, b in zip(el, el[1:]))
    # Datasheet nominal vertical FOV: +2.0 deg (top) to -24.33 deg (bottom).
    assert math.degrees(el[0]) == 2.0
    assert math.degrees(el[-1]) == -24.33
    # Block boundary: upper block bottom at -8.33 deg, lower block top -8.83.
    assert round(math.degrees(el[31]), 2) == -8.33
    assert round(math.degrees(el[32]), 2) == -8.83


def test_hdl64e_preset_defaults() -> None:
    d = sensor_defaults("HDL64E")
    assert d["n_rows"] == 64
    assert d["n_columns"] == 2083  # 0.1728 deg azimuth resolution at 10 Hz
    assert d["fps"] == 10.0
    assert d["max_range_m"] == 120.0
    # Unknown models get no preset overrides.
    assert sensor_defaults("OT128") == {}


def test_for_sensor_applies_faithful_defaults() -> None:
    cfg = LidarConfig.for_sensor("HDL64E")
    assert cfg.sensor_type == "HDL64E"
    assert cfg.n_rows == 64
    assert cfg.n_columns == 2083
    assert cfg.fps == 10.0
    assert cfg.max_range_m == 120.0


def test_for_sensor_overrides_win() -> None:
    cfg = LidarConfig.for_sensor("HDL64E", n_columns=1024, fps=20.0)
    assert cfg.n_columns == 1024
    assert cfg.fps == 20.0
    # Untouched fields keep the faithful preset value.
    assert cfg.n_rows == 64


def test_hdl64e_config_builds_spec() -> None:
    cfg = LidarConfig.for_sensor("HDL64E")
    specs = build_lidar_sensors_from_config([cfg])
    assert len(specs) == 1
    spec = specs[0]
    assert spec.sensor_type == "HDL64E"
    assert spec.n_columns == 2083
    assert spec.spinning_frequency_hz == 10.0
