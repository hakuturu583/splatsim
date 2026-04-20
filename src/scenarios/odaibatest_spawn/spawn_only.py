"""Spawn-only scenario with SplatSim camera on ego vehicle.

Spawns the ego vehicle at a lanelet coordinate, attaches a SplatSim
Gaussian Splatting camera, and publishes rendered images to ROS 2 topics
via CycloneDDS.

Usage::

    source .env
    uv run spawn-scenario ego.spawn_lanelet_id=2223437 ego.spawn_s=5.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cyclonedds.domain import DomainParticipant

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    EgoConfig,
    GroundProjectionConfig,
    Lanelet2Pose,
    TimeoutCondition,
)

from splatsim.carla_integration import SplatSimCameraSensor, SplatSimScenario
from splatsim.cyclonedds import CameraInfoPublisher, ImagePublisher

if TYPE_CHECKING:
    import carla

logger = logging.getLogger(__name__)


@dataclass
class SpawnOnlyConfig:
    """Parameters for the spawn-only scenario."""

    name: str = "spawn_only"
    timeout_seconds: float = 30.0


@dataclass
class SplatSimConfig:
    """SplatSim rendering and ROS 2 publishing parameters."""

    scene_yaml: str = ""
    image_topic: str = "/splatsim/image_raw"
    camera_info_topic: str = "/splatsim/camera_info"
    frame_id: str = "splatsim_camera"


class SpawnOnlyScenario(SplatSimScenario):
    """Spawn the ego at a Lanelet2 pose, attach SplatSim camera, publish to ROS 2."""

    def __init__(
        self,
        ego_config: EgoConfig,
        *,
        config: SpawnOnlyConfig,
        splatsim_config: SplatSimConfig,
        spawn_pose: Lanelet2Pose,
        ground_projection: GroundProjectionConfig | None = None,
    ) -> None:
        super().__init__(
            ego_config,
            splatsim_config.scene_yaml,
            spawn_pose=spawn_pose,
            ground_projection=ground_projection,
        )
        self._config = config
        self._splatsim_cfg = splatsim_config
        self._camera_action = None
        self._image_pub: ImagePublisher | None = None
        self._camera_info_pub: CameraInfoPublisher | None = None
        self._dds_participant: DomainParticipant | None = None

    def setup(self) -> None:
        """Snap ego spawn, attach SplatSim camera, set up ROS 2 publishers."""
        self._setup_ego_spawn()

        # Attach SplatSim camera to ego vehicle
        self._camera_action = self.attach_splatsim_camera(
            EGO_ROLE_NAME,
            label="ego_splatsim_camera",
        )

        # Set up CycloneDDS publishers for ROS 2 output
        scfg = self._splatsim_cfg
        self._dds_participant = DomainParticipant()
        self._image_pub = ImagePublisher(
            self._dds_participant,
            topic_name=scfg.image_topic,
            frame_id=scfg.frame_id,
        )

        # Register post-tick callback to render and publish
        self.register_post_tick(self._publish_splatsim_image)

        assert self._spawn_pose is not None  # noqa: S101
        logger.info(
            "Ego spawned on lanelet %d (s=%.1f). "
            "SplatSim camera attached -> publishing to %s. "
            "Waiting %.1f s ...",
            self._spawn_pose.lanelet_id,
            self._spawn_pose.s,
            scfg.image_topic,
            self._config.timeout_seconds,
        )

        self.register_pass_condition(
            TimeoutCondition(self._config.timeout_seconds, label="spawn_hold_timeout")
        )

    def _publish_splatsim_image(self, world: carla.World) -> None:
        """Render via SplatSim and publish image + camera_info to ROS 2."""
        if self._camera_action is None or self._camera_action.sensor is None:
            return

        sensor = self._camera_action.sensor
        assert isinstance(sensor, SplatSimCameraSensor)  # noqa: S101
        image = sensor.get_image()

        if self._image_pub is not None:
            self._image_pub.publish(image)

        # Lazily create camera_info publisher after sensor is attached
        # (need sensor config for intrinsics)
        if self._camera_info_pub is None and self._dds_participant is not None:
            self._camera_info_pub = CameraInfoPublisher(
                self._dds_participant,
                sensor.config,
                topic_name=self._splatsim_cfg.camera_info_topic,
                frame_id=self._splatsim_cfg.frame_id,
            )

        if self._camera_info_pub is not None:
            self._camera_info_pub.publish()

    def is_done(self) -> bool:
        """Always ``False`` -- termination is driven by the timeout condition."""
        return False
