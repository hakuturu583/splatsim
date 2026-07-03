from __future__ import annotations

from typing import TYPE_CHECKING, Union

from autoware_carla_scenario.actions import AttachLidarSensorAction, TickTiming
from autoware_carla_scenario.sensor import LidarSensorBase

from splatsim.carla_integration.splatsim_lidar import (
    SplatSimLidarSensor,
    SplatSimLidarSensorConfig,
)

if TYPE_CHECKING:
    from autoware_carla_scenario.conditions.base import BaseCondition
    from autoware_carla_scenario.entity_role import EntityRole
    from splatsim.carla_integration.geo_transform import GeoTransform
    from splatsim.scene import Scene


class AttachSplatSimLidarAction(AttachLidarSensorAction):
    """Attach a SplatSim Gaussian Splatting LiDAR to a CARLA actor."""

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        scene: Scene,
        geo_transform: GeoTransform,
        sensor_config: SplatSimLidarSensorConfig | None = None,
        condition: BaseCondition | None = None,
        timing: TickTiming = TickTiming.POST_TICK,
        *,
        label: str = "attach_splatsim_lidar",
        once: bool = True,
    ) -> None:
        if sensor_config is None:
            sensor_config = SplatSimLidarSensorConfig()
        self._splatsim_config = sensor_config
        self._scene = scene
        self._geo_transform = geo_transform
        super().__init__(
            entity_name,
            sensor_config,
            condition,
            timing,
            label=label,
            once=once,
        )

    def _create_sensor(self) -> LidarSensorBase:
        return SplatSimLidarSensor(
            self._splatsim_config,
            self._scene,
            geo_transform=self._geo_transform,
        )
