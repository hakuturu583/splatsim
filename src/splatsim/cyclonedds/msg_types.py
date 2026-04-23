"""ROS 2 message type definitions for CycloneDDS.

Provides CycloneDDS ``IdlStruct`` equivalents of the standard ROS 2 message
types used by camera publishers:

* ``builtin_interfaces/msg/Time``
* ``std_msgs/msg/Header``
* ``sensor_msgs/msg/RegionOfInterest``
* ``sensor_msgs/msg/Image``
* ``sensor_msgs/msg/CameraInfo``

.. note::

   This module must **not** use ``from __future__ import annotations``
   because the CycloneDDS metaclass evaluates type hints at class-creation
   time.
"""

from dataclasses import dataclass

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, float64, int32, sequence, uint8, uint32


@dataclass
class Time(IdlStruct, typename="builtin_interfaces::msg::dds_::Time_"):
    """builtin_interfaces/msg/Time."""

    sec: int32 = 0
    nanosec: uint32 = 0


@dataclass
class Header(IdlStruct, typename="std_msgs::msg::dds_::Header_"):
    """std_msgs/msg/Header."""

    stamp: Time = Time()  # noqa: RUF009
    frame_id: str = ""


@dataclass
class RegionOfInterest(IdlStruct, typename="sensor_msgs::msg::dds_::RegionOfInterest_"):
    """sensor_msgs/msg/RegionOfInterest."""

    x_offset: uint32 = 0
    y_offset: uint32 = 0
    height: uint32 = 0
    width: uint32 = 0
    do_rectify: bool = False


@dataclass
class Image(IdlStruct, typename="sensor_msgs::msg::dds_::Image_"):
    """sensor_msgs/msg/Image."""

    header: Header = Header()  # noqa: RUF009
    height: uint32 = 0
    width: uint32 = 0
    encoding: str = ""
    is_bigendian: uint8 = 0
    step: uint32 = 0
    data: sequence[uint8] = b""  # type: ignore[assignment]


@dataclass
class CameraInfo(IdlStruct, typename="sensor_msgs::msg::dds_::CameraInfo_"):
    """sensor_msgs/msg/CameraInfo."""

    header: Header = Header()  # noqa: RUF009
    height: uint32 = 0
    width: uint32 = 0
    distortion_model: str = ""
    d: sequence[float64] = ()  # type: ignore[assignment]
    k: array[float64, 9] = (0.0,) * 9  # type: ignore[assignment]
    r: array[float64, 9] = (0.0,) * 9  # type: ignore[assignment]
    p: array[float64, 12] = (0.0,) * 12  # type: ignore[assignment]
    binning_x: uint32 = 0
    binning_y: uint32 = 0
    roi: RegionOfInterest = RegionOfInterest()  # noqa: RUF009
