from __future__ import annotations

import pytest

from splatsim.dataclass.lidar_config import LidarConfig
from splatsim.lidar_renderer import build_lidar_sensors_from_config


def test_build_lidar_sensors_from_config_uniform_rows() -> None:
    """A sensor_type-driven LidarConfig builds a uniform-row runtime spec."""
    lidar = LidarConfig(
        name="top",
        sensor_type="OT128",
        n_rows=128,
        n_columns=1024,
        fps=20.0,
        min_range_m=0.5,
        max_range_m=80.0,
        position=(0.0, 0.0, 1.9),
        rotation=(0.0, 0.0, 5.0),
        pointcloud_topic="/sensing/lidar/top/pointcloud",
        frame_id="top_lidar",
        drop_threshold=0.4,
        alpha_threshold=0.2,
    )

    # Communication defaults to DDS.
    assert lidar.communication == "dds"
    assert lidar.hils_host == "127.0.0.1"
    assert lidar.hils_port == 2368

    specs = build_lidar_sensors_from_config([lidar])
    assert len(specs) == 1
    assert specs[0].name == "top"
    assert specs[0].sensor_type == "OT128"
    assert specs[0].n_columns == 1024
    assert specs[0].n_rows_uniform == 128
    assert specs[0].s2b[2, 3] == 1.9


def test_lidar_sensors_from_usdz_rig_calibrations() -> None:
    """USDZ rig LiDAR calibrations map into LidarConfig sensors.

    3dgs_io stores sensor-in-rig extrinsics (translation is the mount
    position directly, rotation is an xyzw quaternion) and keeps the
    intrinsics under ``lidar_model.parameters``. The mapping must preserve
    the mount pose, reorder the quaternion to wxyz, and carry the per-beam
    elevation table sorted top→bottom.
    """
    import importlib
    import math

    import numpy as np

    from splatsim.dataclass.scene_config import _lidar_sensors_from_rigs
    from splatsim.lidar_renderer import build_lidar_sensors_from_config

    rt = importlib.import_module("3dgs_io.rig_trajectories")
    CameraExtrinsics = importlib.import_module("3dgs_io.cameras").CameraExtrinsics

    cal = rt.LidarCalibration(
        name="lidar_top",
        extrinsics=CameraExtrinsics(
            translation=(0.898, 0.00001, 2.180),
            rotation=(0.0, 0.0, 0.0, 1.0),  # xyzw identity
        ),
        lidar_model=rt.LidarModel(
            type="spinning",
            parameters={
                "n_rows": 128,
                "n_columns": 2048,
                "fps": 10.0,
                "min_range_m": 0.3,
                "max_range_m": 120.0,
                # deliberately unsorted; the mapping must sort descending
                "elevation_deg": [-5.0, 14.9, 0.0, -25.0, 3.2],
            },
        ),
    )
    rig = rt.RigTrajectory(rig_id="ego", poses=[], cameras=[], lidars=[cal])

    sensors = _lidar_sensors_from_rigs([rig])
    assert len(sensors) == 1
    s = sensors[0]
    assert s.name == "lidar_top"
    assert s.sensor_type == ""  # explicit elevation table drives geometry
    assert s.n_rows == 128 and s.n_columns == 2048 and s.fps == 10.0
    assert s.position == (0.898, 0.00001, 2.180)
    assert s.rotation == (1.0, 0.0, 0.0, 0.0)  # xyzw -> wxyz
    assert s.elevation_deg == (-5.0, 14.9, 0.0, -25.0, 3.2)
    assert s.pointcloud_topic == "/sensing/lidar/lidar_top/pointcloud"
    assert s.frame_id == "lidar_top"

    spec = build_lidar_sensors_from_config(sensors)[0]
    # translation preserved directly (sensor-in-rig, no inversion)
    assert np.allclose(spec.s2b[:3, 3], [0.898, 0.00001, 2.180])
    assert np.allclose(spec.s2b[:3, :3], np.eye(3))
    # elevation table sorted strictly descending (top -> bottom)
    elev_deg = [math.degrees(x) for x in spec.row_elevations_rad]
    assert elev_deg == sorted(elev_deg, reverse=True)
    assert elev_deg == pytest.approx([14.9, 3.2, 0.0, -5.0, -25.0])
