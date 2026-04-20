from __future__ import annotations

from typing import TYPE_CHECKING, Union

from autoware_carla_scenario.actions import AttachCameraSensorAction, TickTiming
from autoware_carla_scenario.sensor import CameraSensorBase

from splatsim.carla_integration.splatsim_camera import (
    SplatSimCameraSensor,
    SplatSimCameraSensorConfig,
)

if TYPE_CHECKING:
    from autoware_carla_scenario.conditions.base import BaseCondition
    from autoware_carla_scenario.entity_role import EntityRole
    from splatsim.carla_integration.geo_transform import GeoTransform
    from splatsim.scene import Scene


class AttachSplatSimCameraAction(AttachCameraSensorAction):
    """Attach a SplatSim Gaussian Splatting camera to a CARLA actor.

    On execution the action creates a :class:`SplatSimCameraSensor`,
    configures the gsplat renderer with the given parameters, and
    attaches it to the target entity so that subsequent calls to
    ``action.sensor.get_image()`` return Gaussian-Splatting-rendered
    images from the actor's perspective.

    Parameters
    ----------
    entity_name:
        Role name (e.g. ``EntityRole.ego()``) of the CARLA actor to
        attach the camera to.
    scene:
        The :class:`~splatsim.scene.Scene` containing the Gaussian
        Splatting background and rigid bodies to render.
    geo_transform:
        :class:`GeoTransform` for CARLA ↔ tile-local conversion.
    sensor_config:
        Camera and renderer configuration.  Falls back to
        :class:`SplatSimCameraSensorConfig` defaults when *None*.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        scene: Scene,
        geo_transform: GeoTransform,
        sensor_config: SplatSimCameraSensorConfig | None = None,
        condition: BaseCondition | None = None,
        timing: TickTiming = TickTiming.POST_TICK,
        *,
        label: str = "attach_splatsim_camera",
        once: bool = True,
    ) -> None:
        if sensor_config is None:
            sensor_config = SplatSimCameraSensorConfig()
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

    def _create_sensor(self) -> CameraSensorBase:
        return SplatSimCameraSensor(
            self._splatsim_config,
            self._scene,
            geo_transform=self._geo_transform,
        )
