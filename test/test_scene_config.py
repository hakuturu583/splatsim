from __future__ import annotations

from textwrap import dedent

from splatsim.dataclass import SceneConfig
from splatsim.lidar_renderer import build_lidar_sensors_from_config


def test_scene_config_loads_lidar_sensors(tmp_path) -> None:
    scene_yaml = tmp_path / "scene.yaml"
    scene_yaml.write_text(
        dedent(
            """
            background_tileset: iteration_30000/tileset.json
            use_sh: false

            lidar_sensors:
              - name: top
                sensor_type: OT128
                n_rows: 128
                n_columns: 1024
                fps: 20.0
                min_range_m: 0.5
                max_range_m: 80.0
                position: [0.0, 0.0, 1.9]
                rotation: [0.0, 0.0, 5.0]
                pointcloud_topic: /sensing/lidar/top/pointcloud
                frame_id: top_lidar
                drop_threshold: 0.4
                alpha_threshold: 0.2
            """
        ),
        encoding="utf-8",
    )

    cfg = SceneConfig.from_yaml(scene_yaml)

    assert len(cfg.lidar_sensors) == 1
    lidar = cfg.lidar_sensors[0]
    assert lidar.name == "top"
    assert lidar.sensor_type == "OT128"
    assert lidar.n_columns == 1024
    assert lidar.fps == 20.0
    assert lidar.position == (0.0, 0.0, 1.9)
    assert lidar.rotation == (0.0, 0.0, 5.0)
    assert lidar.pointcloud_topic == "/sensing/lidar/top/pointcloud"
    assert lidar.frame_id == "top_lidar"
    assert lidar.drop_threshold == 0.4
    assert lidar.alpha_threshold == 0.2

    specs = build_lidar_sensors_from_config(cfg.lidar_sensors)
    assert len(specs) == 1
    assert specs[0].name == "top"
    assert specs[0].sensor_type == "OT128"
    assert specs[0].n_columns == 1024
    assert specs[0].n_rows_uniform == 128
    assert specs[0].s2b[2, 3] == 1.9
    # Communication defaults to DDS when not specified.
    assert lidar.communication == "dds"
    assert lidar.hils_host == "127.0.0.1"
    assert lidar.hils_port == 2368


def test_scene_config_loads_hils_lidar(tmp_path) -> None:
    scene_yaml = tmp_path / "scene.yaml"
    scene_yaml.write_text(
        dedent(
            """
            background_tileset: iteration_30000/tileset.json

            lidar_sensors:
              - name: top
                sensor_type: XT32
                n_rows: 32
                communication: hils
                hils_host: 192.168.1.201
                hils_port: 2368
            """
        ),
        encoding="utf-8",
    )

    cfg = SceneConfig.from_yaml(scene_yaml)
    lidar = cfg.lidar_sensors[0]
    assert lidar.communication == "hils"
    assert lidar.hils_host == "192.168.1.201"
    assert lidar.hils_port == 2368
