"""Autoware ↔ CARLA DDS bridge utilities for SplatSim scenarios.

Publishes DDS messages required for Autoware integration (initial pose,
localization state) and patches AutowareEntity quirks (hand brake).

.. note::

   This module must **not** use ``from __future__ import annotations``
   because the CycloneDDS metaclass evaluates type hints at class-creation
   time.
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, float64, uint16
from cyclonedds.pub import DataWriter
from cyclonedds.qos import Policy, Qos
from cyclonedds.topic import Topic

from autoware_carla_scenario.coordinate import Lanelet2Pose
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.coordinate.transform import _interpolate_at_s

from splatsim.cyclonedds.msg_types import Header as _Header, Time as _Time

logger = logging.getLogger(__name__)


# ── CycloneDDS message types ──────────────────────────────────────────
# _Time and _Header are re-used from splatsim.cyclonedds.msg_types.
# Geometry types below are specific to the Autoware bridge.


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


@dataclass
class _LocalizationInitializationState(
    IdlStruct,
    typename="autoware_adapi_v1_msgs::msg::dds_::LocalizationInitializationState_",
):
    stamp: _Time = _Time()  # noqa: RUF009
    state: uint16 = 0


# ── QoS profiles ──────────────────────────────────────────────────────

_TRANSIENT_LOCAL_QOS = Qos(
    Policy.Durability.TransientLocal,
    Policy.Reliability.Reliable(max_blocking_time=1_000_000_000),
    Policy.History.KeepLast(1),
)

_GEAR_PARK = 22  # autoware_auto_vehicle_msgs::msg::GearCommand::PARK
_LOCALIZATION_INITIALIZED = (
    3  # autoware_adapi_v1_msgs::msg::LocalizationInitializationState::INITIALIZED
)


# ── Public helpers ────────────────────────────────────────────────────


def publish_initialpose(
    participant: DomainParticipant,
    spawn_pose: Lanelet2Pose,
) -> None:
    """Publish the ego spawn position as ``/initialpose3d`` for simple_planning_simulator."""
    mm = MapManager.get_instance()
    lanelet = mm.lanelet_map.laneletLayer[spawn_pose.lanelet_id]
    points = [(p.x, p.y, p.z) for p in lanelet.centerline]
    x, y, z, heading = _interpolate_at_s(points, spawn_pose.s)

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

    topic = Topic(participant, "rt/initialpose3d", _PoseWithCovarianceStamped)
    writer = DataWriter(participant, topic)
    for _ in range(5):
        writer.write(msg)
        time.sleep(0.1)

    logger.info(
        "Published initialpose3d: map frame (%.2f, %.2f, %.2f) yaw=%.1f deg",
        x,
        y,
        z,
        math.degrees(heading),
    )


def publish_localization_initialized(participant: DomainParticipant) -> None:
    """Publish ``LocalizationInitializationState=INITIALIZED`` so Autoware exits INITIALIZING."""
    now_ns = time.time_ns()
    sec, nanosec = divmod(now_ns, 10**9)

    msg = _LocalizationInitializationState(
        stamp=_Time(sec=sec, nanosec=nanosec),
        state=_LOCALIZATION_INITIALIZED,
    )
    topic = Topic(
        participant,
        "rt/localization/initialization_state",
        _LocalizationInitializationState,
        qos=_TRANSIENT_LOCAL_QOS,
    )
    writer = DataWriter(participant, topic, qos=_TRANSIENT_LOCAL_QOS)
    for _ in range(5):
        writer.write(msg)
        time.sleep(0.1)

    logger.info("Published localization_initialization_state=INITIALIZED(3)")


def patch_motion_control(entity: Any) -> None:
    """Monkey-patch ``entity._apply_motion_control`` to fix two issues.

    1. **Hand brake**: The dependency sets ``hand_brake=True`` during PARK
       but never releases it when switching to DRIVE
       (``apply_ackermann_control`` does not touch ``hand_brake`` in CARLA).
       This patch releases the hand brake once at the transition point.

    2. **Steering sign**: CARLA ``VehicleAckermannControl.steer`` uses
       positive = right, while Autoware ``steering_tire_angle`` uses
       positive = left.  The dependency passes the value without negation,
       causing inverted steering.  This patch negates the steer value.
    """
    _state: dict[str, bool] = {"handbrake_released": False}

    def _patched(_carla: Any, gear_cmd: Any, is_reverse: bool) -> None:
        assert entity._vehicle is not None  # noqa: S101

        # -- PARK: hand brake, reset state --
        if gear_cmd is not None and gear_cmd.command == _GEAR_PARK:
            entity._vehicle.apply_control(_carla.VehicleControl(hand_brake=True))
            _state["handbrake_released"] = False
            return

        # -- Release hand brake once on PARK -> DRIVE transition --
        if not _state["handbrake_released"]:
            entity._vehicle.apply_control(_carla.VehicleControl(hand_brake=False))
            _state["handbrake_released"] = True
            logger.info("Released CARLA hand brake after PARK -> DRIVE transition")

        # -- Apply ackermann control with corrected steering sign --
        ackermann_cmd = (
            entity._dds.current_manual_ackermann_cmd
            if entity._dds.current_manual_ackermann_cmd is not None
            else entity._dds.current_ackermann_cmd
        )
        if ackermann_cmd is None:
            return

        speed = float(ackermann_cmd.longitudinal.velocity)
        if is_reverse:
            speed = -abs(speed)

        entity._vehicle.apply_ackermann_control(
            _carla.VehicleAckermannControl(
                steer=-float(ackermann_cmd.lateral.steering_tire_angle),
                steer_speed=float(ackermann_cmd.lateral.steering_tire_rotation_rate),
                speed=speed,
                acceleration=float(ackermann_cmd.longitudinal.acceleration),
                jerk=float(ackermann_cmd.longitudinal.jerk),
            )
        )

    entity._apply_motion_control = _patched  # type: ignore[attr-defined]
