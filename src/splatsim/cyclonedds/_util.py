"""Shared helpers for CycloneDDS publisher modules."""

from __future__ import annotations

import time

from splatsim.cyclonedds.msg_types import Time


def _to_dds_topic(ros_topic: str) -> str:
    """Convert a ROS 2 topic name to the DDS wire name (``rt/`` prefix)."""
    return "rt/" + ros_topic.lstrip("/")


def _now() -> Time:
    """Return the current wall-clock time as a ROS 2 ``Time``."""
    sec, nanosec = divmod(time.time_ns(), 10**9)
    return Time(sec=sec, nanosec=nanosec)
