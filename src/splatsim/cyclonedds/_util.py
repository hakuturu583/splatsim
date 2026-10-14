"""Shared helpers for CycloneDDS publisher modules."""

from __future__ import annotations

from typing import Any, Protocol

from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct
from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic


def to_dds_topic(ros_topic: str) -> str:
    """Convert a ROS 2 topic name to the DDS wire name (``rt/`` prefix)."""
    return "rt/" + ros_topic.lstrip("/")


class MessageWriter(Protocol):
    """Where a publisher's finished message goes (a DDS ``DataWriter``)."""

    def write(self, msg: Any, /) -> None: ...


class MessageSink(Protocol):
    """A non-DDS destination the publishers can write to instead of a
    ``DomainParticipant`` -- e.g. the gRPC service's sensor-data stream."""

    def create_writer(
        self, topic_name: str, msg_type: type[IdlStruct]
    ) -> MessageWriter: ...


def create_writer(
    participant: DomainParticipant | MessageSink,
    topic_name: str,
    msg_type: type[IdlStruct],
) -> MessageWriter:
    """A writer for the ROS 2 topic *topic_name* carrying *msg_type*: a DDS
    ``DataWriter`` on a participant, or whatever a :class:`MessageSink` makes."""
    if isinstance(participant, DomainParticipant):
        topic = Topic(participant, to_dds_topic(topic_name), msg_type)
        return DataWriter(participant, topic)
    return participant.create_writer(topic_name, msg_type)
