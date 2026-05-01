"""Spawn-only scenario with SplatSim camera on ego vehicle.

Spawns the ego vehicle at a lanelet coordinate, attaches a SplatSim
Gaussian Splatting camera, and publishes rendered images to ROS 2 topics
via CycloneDDS.

Usage::

    source .env
    uv run spawn-scenario ego.spawn_lanelet_id=2303321 ego.spawn_s=0.0
"""

import logging
from dataclasses import dataclass
from typing import Optional

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    EgoConfig,
    GroundProjectionConfig,
    Lanelet2Pose,
    TimeoutCondition,
)
from autoware_carla_scenario.actions import AttachIMUSensorAction, IMUSensorConfig

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
        ground_projection: Optional[GroundProjectionConfig] = None,
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
        """Snap ego spawn, attach SplatSim camera & IMU, engage, set up ROS 2 publishers."""
        self._setup_ego_spawn()

        scfg = self._splatsim_cfg

        # Autoware integration: initialpose, localization, hand brake, engage
        self.setup_autoware()

        # Attach SplatSim camera to ego vehicle (1.5m above base_link)
        self.attach_splatsim_camera(
            EGO_ROLE_NAME,
            sensor_config=SplatSimCameraSensorConfig(position_z=1.5),
            label="ego_splatsim_camera",
            dds_participant=self.dds_participant,
            image_topic=scfg.image_topic,
            camera_info_topic=scfg.camera_info_topic,
            frame_id=scfg.frame_id,
            compress_format=scfg.compress_format,
        )

        # Attach CARLA IMU sensor and publish to /sensing/imu/imu_data
        imu_action = AttachIMUSensorAction(
            EGO_ROLE_NAME,
            sensor_config=IMUSensorConfig(sensor_tick=0.01, role_name="imu"),
            dds_participant=self.dds_participant,
            imu_topic="/sensing/imu/imu_data",
            frame_id="base_link",
            label="ego_imu_sensor",
        )
        self.register_post_tick(imu_action)

        # Register the shared publish callback
        self.register_post_tick(self.publish_ros_topics)

        # Diagnostic: log AutowareEntity state every 20 ticks
        assert (entity := self.ego_entity) is not None  # noqa: S101
        diag_state = {"tick": 0}

        def _log_autoware_state(world: object) -> None:
            diag_state["tick"] += 1
            if diag_state["tick"] % 20 != 0:
                return
            dds = entity._dds
            cmd = dds.current_ackermann_cmd
            vel = f"{cmd.longitudinal.velocity:.3f}" if cmd else "None"
            acc = f"{cmd.longitudinal.acceleration:.3f}" if cmd else "None"
            carla_vel = entity._vehicle.get_velocity() if entity._vehicle else None
            carla_speed = (
                f"{(carla_vel.x**2 + carla_vel.y**2 + carla_vel.z**2) ** 0.5:.3f}"
                if carla_vel
                else "None"
            )
            logger.info(
                "[diag] tick=%d engaged=%s gear=%s "
                "cmd_vel=%s cmd_acc=%s carla_speed=%s",
                diag_state["tick"],
                dds.is_engaged,
                getattr(dds.current_gear_cmd, "command", None),
                vel,
                acc,
                carla_speed,
            )

        self.register_post_tick(_log_autoware_state)

        # Suppress verbose position/tick logging from base scenario
        logging.getLogger("autoware_carla_scenario.scenario_base").setLevel(
            logging.WARNING
        )
        logging.getLogger("autoware_carla_scenario.scenario_runner").setLevel(
            logging.WARNING
        )

        assert self._spawn_pose is not None  # noqa: S101
        logger.info(
            "Ego spawned on lanelet %d (s=%.1f). "
            "SplatSim camera attached -> publishing to %s. "
            "IMU sensor attached -> publishing to /sensing/imu/imu_data. "
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
