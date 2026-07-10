from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import carla
import numpy as np
import torch
from numpy.typing import NDArray

from autoware_carla_scenario.sensor import LidarSensorBase, LidarSensorConfig
from splatsim.lidar_renderer import LidarRenderer, LidarSensorSpec

if TYPE_CHECKING:
    from splatsim.carla_integration.geo_transform import GeoTransform
    from splatsim.dataclass import LidarConfig
    from splatsim.scene import Scene


@dataclass(frozen=True)
class SplatSimLidarSensorConfig(LidarSensorConfig):
    """LiDAR sensor config extended with SplatSim renderer parameters."""

    name: str = "lidar"
    sensor_type: str = "OT128"
    device: str = "cuda"
    drop_threshold: float = 0.5
    alpha_threshold: float = 0.1
    communication: str = "dds"
    hils_host: str = "127.0.0.1"
    hils_port: int = 2368
    hils_start_epoch: float | None = None

    @classmethod
    def from_scene_config(cls, config: LidarConfig) -> SplatSimLidarSensorConfig:
        """Build a runtime sensor config from a scene YAML LiDAR entry."""
        return cls(
            name=config.name,
            sensor_type=config.sensor_type,
            n_rows=config.n_rows,
            n_columns=config.n_columns,
            fps=config.fps,
            min_range_m=config.min_range_m,
            max_range_m=config.max_range_m,
            position_x=config.position[0],
            position_y=config.position[1],
            position_z=config.position[2],
            roll=config.rotation[0],
            pitch=config.rotation[1],
            yaw=config.rotation[2],
            drop_threshold=config.drop_threshold,
            alpha_threshold=config.alpha_threshold,
            communication=config.communication,
            hils_host=config.hils_host,
            hils_port=config.hils_port,
            hils_start_epoch=config.hils_start_epoch,
        )


class SplatSimLidarSensor(LidarSensorBase):
    """LiDAR sensor that renders SplatSim point clouds from a CARLA actor pose."""

    def __init__(
        self,
        config: SplatSimLidarSensorConfig,
        scene: Scene,
        *,
        geo_transform: GeoTransform,
    ) -> None:
        super().__init__(config)
        self._splatsim_config = config
        self._scene = scene
        self._geo_transform = geo_transform
        self._device = torch.device(config.device)
        self._renderer = LidarRenderer(
            _sensor_spec_from_config(config),
            device=self._device,
            min_range_m=config.min_range_m,
            max_range_m=config.max_range_m,
        )
        self._actor: carla.Actor | None = None

    @property
    def renderer(self) -> LidarRenderer:
        """The SplatSim LiDAR renderer used by this sensor."""
        return self._renderer

    @property
    def scene(self) -> Scene:
        """The Gaussian Splatting scene being rendered."""
        return self._scene

    def attach(self, world: carla.World, actor: carla.Actor) -> None:
        """Store the CARLA actor reference for per-frame pose tracking."""
        self._actor = actor
        self._attached = True

    def destroy(self) -> None:
        """Release the actor reference."""
        self._actor = None
        self._attached = False

    def get_point_cloud(self) -> Optional[NDArray[np.float32]]:
        """Render the latest point cloud as an ``(N, 4)`` XYZI array."""
        if self._actor is None:
            return None

        base_to_world = self._compute_base_to_world()
        with torch.no_grad():
            panorama = self._renderer.render(base_to_world, scene=self._scene)
            cloud = self._renderer.panorama_to_point_cloud(
                panorama,
                drop_threshold=self._splatsim_config.drop_threshold,
                alpha_threshold=self._splatsim_config.alpha_threshold,
            )

        xyz = cloud["xyz"]
        intensity = cloud["intensity"]
        point_cloud = np.empty((xyz.shape[0], 4), dtype=np.float32)
        point_cloud[:, :3] = xyz
        point_cloud[:, 3] = intensity
        return point_cloud

    def get_range_image(self) -> Optional[dict[str, NDArray]]:
        """Render the latest scan as a dense structured range image.

        Used by the HILS transport to build raw Hesai UDP packets. Returns
        ``None`` if the sensor is not attached yet, otherwise the dict from
        :meth:`LidarRenderer.panorama_to_range_image` with keys
        ``distance`` / ``intensity`` / ``valid`` (``(H, W)``) and
        ``azimuths`` (``(W,)``) / ``elevations`` (``(H,)``).
        """
        if self._actor is None:
            return None

        base_to_world = self._compute_base_to_world()
        with torch.no_grad():
            panorama = self._renderer.render(base_to_world, scene=self._scene)
            return self._renderer.panorama_to_range_image(
                panorama,
                drop_threshold=self._splatsim_config.drop_threshold,
                alpha_threshold=self._splatsim_config.alpha_threshold,
            )

    def _compute_base_to_world(self) -> torch.Tensor:
        """Build base_link-to-tile-local transform for the attached CARLA actor."""
        assert self._actor is not None  # noqa: S101

        T_world_actor = np.asarray(
            self._actor.get_transform().get_matrix(), dtype=np.float64
        )
        carla_pos = T_world_actor[:3, 3]
        R_carla = T_world_actor[:3, :3]

        tile_pos = self._geo_transform.carla_position_to_tile_local(
            carla_pos[0], carla_pos[1], carla_pos[2]
        )
        R_tile = self._geo_transform.carla_rotation_to_tile_local(R_carla)

        base_to_world = np.eye(4, dtype=np.float64)
        base_to_world[:3, :3] = np.column_stack(
            [
                R_tile[:, 0],
                -R_tile[:, 1],
                R_tile[:, 2],
            ]
        )
        base_to_world[:3, 3] = tile_pos
        return torch.tensor(base_to_world, device=self._device, dtype=torch.float32)


def _sensor_spec_from_config(config: SplatSimLidarSensorConfig) -> LidarSensorSpec:
    s2b = np.eye(4, dtype=np.float64)
    s2b[:3, :3] = _rpy_deg_to_matrix(config.roll, config.pitch, config.yaw)
    s2b[:3, 3] = np.array(
        [config.position_x, config.position_y, config.position_z],
        dtype=np.float64,
    )
    return LidarSensorSpec(
        name=config.name,
        sensor_type=config.sensor_type,
        s2b=s2b,
        n_columns=config.n_columns,
        spinning_frequency_hz=config.fps,
        n_rows_uniform=config.n_rows,
    )


def _rpy_deg_to_matrix(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> NDArray[np.float64]:
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
