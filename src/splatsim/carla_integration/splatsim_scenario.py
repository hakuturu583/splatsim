"""Base scenario class that combines CARLA APIs with SplatSim scene access."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

from autoware_carla_scenario.scenario_base import BaseScenario, EgoConfig

from splatsim.carla_integration.attach_splatsim_camera import (
    AttachSplatSimCameraAction,
)
from splatsim.carla_integration.splatsim_camera import SplatSimCameraSensorConfig
from splatsim.scene import Scene

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from autoware_carla_scenario.actions import TickTiming
    from autoware_carla_scenario.conditions.base import BaseCondition
    from autoware_carla_scenario.coordinate import (
        GroundProjectionConfig,
        Lanelet2Pose,
    )
    from autoware_carla_scenario.entity_role import EntityRole
    from splatsim.dataclass import SceneConfig


class SplatSimScenario(BaseScenario):
    """Abstract base class for CARLA scenarios with SplatSim rendering.

    Extends :class:`BaseScenario` so that subclasses have full access to
    the CARLA client, world, ego vehicle helpers **and** a SplatSim
    :class:`~splatsim.scene.Scene` for Gaussian Splatting rendering.

    Parameters
    ----------
    ego_config:
        Ego vehicle spawn configuration (forwarded to ``BaseScenario``).
    scene:
        A :class:`~splatsim.scene.Scene` instance, a
        :class:`~splatsim.dataclass.SceneConfig`, or a path to a YAML
        scene file.  Configs and paths are built into a ``Scene``
        automatically.
    spawn_pose:
        Optional Lanelet2 pose for ego spawn.
    ground_projection:
        Ground-projection settings for snapping to the CARLA road surface.
    random_seed:
        Seed for the CARLA TrafficManager random device.
    carla_to_splatsim:
        Optional 4 x 4 matrix that converts CARLA-world coordinates to
        the SplatSim scene coordinate frame.  Identity by default.
    sensor_config:
        Default camera / renderer configuration used by
        :meth:`attach_splatsim_camera`.  Falls back to
        :class:`SplatSimCameraSensorConfig` defaults when *None*.
    """

    def __init__(
        self,
        ego_config: EgoConfig,
        scene: Scene | SceneConfig | str | Path,
        *,
        spawn_pose: Lanelet2Pose | None = None,
        ground_projection: GroundProjectionConfig | None = None,
        random_seed: int = BaseScenario.DEFAULT_RANDOM_SEED,
        carla_to_splatsim: NDArray[np.float64] | None = None,
        sensor_config: SplatSimCameraSensorConfig | None = None,
    ) -> None:
        super().__init__(
            ego_config,
            spawn_pose=spawn_pose,
            ground_projection=ground_projection,
            random_seed=random_seed,
        )

        if isinstance(scene, Scene):
            self._scene = scene
        else:
            self._scene = Scene.from_config(scene)

        self._carla_to_splatsim = carla_to_splatsim
        self._sensor_config = sensor_config

    # -- public properties ---------------------------------------------------

    @property
    def scene(self) -> Scene:
        """The Gaussian Splatting scene used for rendering."""
        return self._scene

    # -- convenience helpers -------------------------------------------------

    def attach_splatsim_camera(
        self,
        entity_name: Union[EntityRole, str],
        *,
        sensor_config: SplatSimCameraSensorConfig | None = None,
        carla_to_splatsim: NDArray[np.float64] | None = None,
        condition: BaseCondition | None = None,
        timing: TickTiming | None = None,
        label: str = "attach_splatsim_camera",
        once: bool = True,
    ) -> AttachSplatSimCameraAction:
        """Create and register an action that attaches a SplatSim camera.

        The action is automatically registered via :meth:`register_pre_tick`
        or :meth:`register_post_tick` depending on *timing*.

        Parameters
        ----------
        entity_name:
            Role name of the CARLA actor to attach the camera to
            (e.g. ``EntityRole.ego()``).
        sensor_config:
            Per-call override.  Falls back to the instance-level
            ``sensor_config`` passed at construction, then to
            :class:`SplatSimCameraSensorConfig` defaults.
        carla_to_splatsim:
            Per-call override for the coordinate transform.  Falls back
            to the instance-level ``carla_to_splatsim``.
        condition:
            Optional trigger condition for the action.
        timing:
            When the action fires (pre-tick or post-tick).  Defaults to
            ``TickTiming.POST_TICK``.
        label:
            Human-readable action identifier.
        once:
            If *True* (default), the action fires at most once.

        Returns
        -------
        AttachSplatSimCameraAction
            The registered action (useful for accessing the sensor after
            attachment via ``action.sensor``).
        """
        from autoware_carla_scenario.actions import TickTiming as _TickTiming

        if timing is None:
            timing = _TickTiming.POST_TICK

        config = sensor_config or self._sensor_config
        transform = (
            carla_to_splatsim
            if carla_to_splatsim is not None
            else self._carla_to_splatsim
        )

        action = AttachSplatSimCameraAction(
            entity_name,
            self._scene,
            sensor_config=config,
            condition=condition,
            timing=timing,
            carla_to_splatsim=transform,
            label=label,
            once=once,
        )
        self.register_post_tick(action)
        return action
