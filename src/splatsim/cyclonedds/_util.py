"""Shared helpers for CycloneDDS publisher modules."""

from __future__ import annotations


def to_dds_topic(ros_topic: str) -> str:
    """Convert a ROS 2 topic name to the DDS wire name (``rt/`` prefix)."""
    return "rt/" + ros_topic.lstrip("/")
