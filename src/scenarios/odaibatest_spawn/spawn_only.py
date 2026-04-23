"""Spawn-only scenario with SplatSim camera on ego vehicle.

Spawns the ego vehicle at a lanelet coordinate, attaches a SplatSim
Gaussian Splatting camera, and publishes rendered images to ROS 2 topics
via CycloneDDS.

Usage::

    source .env
    uv run spawn-scenario ego.spawn_lanelet_id=2303321 ego.spawn_s=0.0

.. note::

   This module must **not** use ``from __future__ import annotations``
   because the CycloneDDS metaclass evaluates type hints at class-creation
   time.
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, float64, int32, uint32
from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    EgoConfig,
    GroundProjectionConfig,
    Lanelet2Pose,
    TimeoutCondition,
)
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.coordinate.transform import _interpolate_at_s

from splatsim.carla_integration import (
    SplatSimCameraSensorConfig,
    SplatSimConfig,
    SplatSimScenario,
)

logger = logging.getLogger(__name__)


# ── CycloneDDS message types for /initialpose3d ──────────────────────
# Defined here (not imported from autoware_carla_scenario.dds.msg)
# because that module uses ``from __future__ import annotations``
# which breaks CycloneDDS Topic creation.


@dataclass
class _Time(IdlStruct, typename="builtin_interfaces::msg::dds_::Time_"):
    sec: int32 = 0
    nanosec: uint32 = 0


@dataclass
class _Header(IdlStruct, typename="std_msgs::msg::dds_::Header_"):
    stamp: _Time = _Time()  # noqa: RUF009
    frame_id: str = ""


@dataclass
class _Point(IdlStruct, typename="geometry_msgs::msg::dds_::Point_"):
    x: float64 = 0.0
    y: float64 = 0.0
    z: float64 = 0.0


@dataclass
class _Quaternion(IdlStruct, typename="geometry_msgs::msg::dds_::Quaternion_"):
    x: float64 = 0.0
    y: float64 = 0.0
    z: float64 = 0.0
    w: float64 = 1.0


@dataclass
class _Pose(IdlStruct, typename="geometry_msgs::msg::dds_::Pose_"):
    position: _Point = _Point()  # noqa: RUF009
    orientation: _Quaternion = _Quaternion()  # noqa: RUF009


@dataclass
class _PoseWithCovariance(
    IdlStruct, typename="geometry_msgs::msg::dds_::PoseWithCovariance_"
):
    pose: _Pose = _Pose()  # noqa: RUF009
    covariance: array[float64, 36] = (0.0,) * 36  # type: ignore[assignment]


@dataclass
class _PoseWithCovarianceStamped(
    IdlStruct,
    typename="geometry_msgs::msg::dds_::PoseWithCovarianceStamped_",
):
    header: _Header = _Header()  # noqa: RUF009
    pose: _PoseWithCovariance = _PoseWithCovariance()  # noqa: RUF009


# ─────────────────────────────────────────────────────────────────────


@dataclass
class SpawnOnlyConfig:
    """Parameters for the spawn-only scenario."""

    name: str = "spawn_only"
    timeout_seconds: float = 30.0


def _publish_initialpose(
    participant: DomainParticipant,
    spawn_pose: Lanelet2Pose,
) -> None:
    """Publish the ego spawn position as /initialpose3d for simple_planning_simulator."""
    mm = MapManager.get_instance()
    lanelet = mm.lanelet_map.laneletLayer[spawn_pose.lanelet_id]
    points = [(p.x, p.y, p.z) for p in lanelet.centerline]
    x, y, z, heading = _interpolate_at_s(points, spawn_pose.s)

    # Compute quaternion from yaw (heading) - rotation around Z axis
    qw = math.cos(heading / 2.0)
    qz = math.sin(heading / 2.0)

    now_ns = time.time_ns()
    sec, nanosec = divmod(now_ns, 10**9)

    msg = _PoseWithCovarianceStamped(
        header=_Header(
            stamp=_Time(sec=sec, nanosec=nanosec),
            frame_id="map",
        ),
        pose=_PoseWithCovariance(
            pose=_Pose(
                position=_Point(x=x, y=y, z=z),
                orientation=_Quaternion(x=0.0, y=0.0, z=qz, w=qw),
            ),
            covariance=tuple([0.0] * 36),
        ),
    )

    topic = Topic(
        participant,
        "rt/initialpose3d",
        _PoseWithCovarianceStamped,
    )
    writer = DataWriter(participant, topic)
    # Publish multiple times to ensure delivery
    for _ in range(5):
        writer.write(msg)
        time.sleep(0.1)

    logger.info(
        "Published initialpose3d: map frame (%.2f, %.2f, %.2f) yaw=%.1f deg",
        x, y, z, math.degrees(heading),
    )


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
        """Snap ego spawn, attach SplatSim camera, set up ROS 2 publishers."""
        self._setup_ego_spawn()

        scfg = self._splatsim_cfg
        dds_participant = DomainParticipant()

        # Publish initial pose to Autoware's simple_planning_simulator
        assert self._spawn_pose is not None  # noqa: S101
        _publish_initialpose(dds_participant, self._spawn_pose)

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
