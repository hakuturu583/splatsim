"""Spawn-only scenario with SplatSim camera on ego vehicle.

Spawns the ego vehicle at a lanelet coordinate, attaches a SplatSim
Gaussian Splatting camera, and publishes rendered images to ROS 2 topics
via CycloneDDS.

Usage::

    source .env
    uv run spawn-scenario ego.spawn_lanelet_id=2303321 ego.spawn_s=0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from cyclonedds.domain import DomainParticipant

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    EgoConfig,
    GroundProjectionConfig,
    Lanelet2Pose,
    TimeoutCondition,
)

from splatsim.carla_integration import (
    SplatSimCameraSensorConfig,
    SplatSimConfig,
    SplatSimScenario,
)

logger = logging.getLogger(__name__)


@dataclass
class SpawnOnlyConfig:
    """Parameters for the spawn-only scenario."""

    name: str = "spawn_only"
    timeout_seconds: float = 30.0


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

    def setup(self) -> None:
        """Snap ego spawn, attach SplatSim camera, set up ROS 2 publishers."""
        self._setup_ego_spawn()

        scfg = self._splatsim_cfg
        dds_participant = DomainParticipant()

        # Attach SplatSim camera to ego vehicle (1.5m above base_link)
        self.attach_splatsim_camera(
            EGO_ROLE_NAME,
            sensor_config=SplatSimCameraSensorConfig(position_z=1.5),
            label="ego_splatsim_camera",
            dds_participant=dds_participant,
            image_topic=scfg.image_topic,
            camera_info_topic=scfg.camera_info_topic,
            frame_id=scfg.frame_id,
        )

        # Register the shared publish callback
        self.register_post_tick(self.publish_ros_topics)

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

    def is_done(self) -> bool:
        """Always ``False`` -- termination is driven by the timeout condition."""
        return False
