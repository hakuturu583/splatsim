"""Base scenario class that combines CARLA APIs with SplatSim scene access."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Union

from autoware_carla_scenario.scenario_base import BaseScenario, EgoConfig

from splatsim.carla_integration.attach_splatsim_camera import (
    AttachSplatSimCameraAction,
)
from splatsim.carla_integration.geo_transform import GeoTransform, parse_geo_reference
from splatsim.carla_integration.splatsim_camera import SplatSimCameraSensorConfig
from splatsim.scene import Scene

if TYPE_CHECKING:
    from autoware_carla_scenario.actions import TickTiming
    from autoware_carla_scenario.conditions.base import BaseCondition
    from autoware_carla_scenario.coordinate import (
        GroundProjectionConfig,
        Lanelet2Pose,
    )
    from autoware_carla_scenario.entity_role import EntityRole
    from splatsim.dataclass import SceneConfig

logger = logging.getLogger(__name__)


class SplatSimScenario(BaseScenario):
    """Abstract base class for CARLA scenarios with SplatSim rendering.

    Extends :class:`BaseScenario` so that subclasses have full access to
    the CARLA client, world, ego vehicle helpers **and** a SplatSim
    :class:`~splatsim.scene.Scene` for Gaussian Splatting rendering.

    The geographic transform between CARLA world coordinates and the
    3DGS tile-local frame is computed automatically from the xodr
    ``<geoReference>`` and the tileset.json ECEF transform.

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

        self._sensor_config = sensor_config
        self._geo_transform: GeoTransform | None = None

    # -- public properties ---------------------------------------------------

    @property
    def scene(self) -> Scene:
        """The Gaussian Splatting scene used for rendering."""
        return self._scene

    @property
    def geo_transform(self) -> GeoTransform:
        """The CARLA ↔ tile-local coordinate transform.

        Created lazily on first access from the CARLA map's xodr
        ``<geoReference>`` and the scene's tileset ECEF transform.
        """
        if self._geo_transform is None:
            self._geo_transform = self._build_geo_transform()
        return self._geo_transform

    # -- convenience helpers -------------------------------------------------

    def attach_splatsim_camera(
        self,
        entity_name: Union[EntityRole, str],
        *,
        sensor_config: SplatSimCameraSensorConfig | None = None,
        condition: BaseCondition | None = None,
        timing: TickTiming | None = None,
        label: str = "attach_splatsim_camera",
        once: bool = True,
    ) -> AttachSplatSimCameraAction:
        """Create and register an action that attaches a SplatSim camera.

        The action is automatically registered via :meth:`register_post_tick`.

        Parameters
        ----------
        entity_name:
            Role name of the CARLA actor to attach the camera to
            (e.g. ``EntityRole.ego()``).
        sensor_config:
            Per-call override.  Falls back to the instance-level
            ``sensor_config`` passed at construction, then to
            :class:`SplatSimCameraSensorConfig` defaults.
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

        action = AttachSplatSimCameraAction(
            entity_name,
            self._scene,
            self.geo_transform,
            sensor_config=config,
            condition=condition,
            timing=timing,
            label=label,
            once=once,
        )
        self.register_post_tick(action)
        return action

    # -- internals -----------------------------------------------------------

    def _build_geo_transform(self) -> GeoTransform:
        """Build :class:`GeoTransform` from the CARLA map and scene data."""
        import numpy as _np  # noqa: PLC0415

        # 1. Parse PROJ string from CARLA xodr
        xodr_xml = self.world.get_map().to_opendrive()
        proj_string = parse_geo_reference(xodr_xml)
        logger.info("GeoReference proj string: %s", proj_string)

        # 2. Get tileset ECEF rotation/translation from Background
        bg = self._scene.background
        if bg is None:
            raise RuntimeError(
                "Cannot build GeoTransform: scene has no Background "
                "(no tileset.json ECEF data)"
            )

        # 3. Get tile origin (torch Tensor on GPU → numpy float64)
        tile_origin = bg.origin.cpu().numpy().astype(_np.float64)

        return GeoTransform(
            proj_string=proj_string,
            ecef_rotation=bg._ecef_rotation,
            ecef_translation=bg._ecef_translation,
            tile_origin=tile_origin,
        )
