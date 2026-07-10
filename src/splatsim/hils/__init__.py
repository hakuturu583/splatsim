"""Hardware-in-the-loop (HILS) LiDAR transport.

Emits raw Hesai UDP data packets that mimic a physical LiDAR, as an
alternative to publishing a decoded ``sensor_msgs/PointCloud2`` over DDS.
Selected per sensor via ``communication: hils`` in the scene config.
"""

from splatsim.hils.hesai_packet import (
    MODELS,
    HesaiModel,
    build_frame_tensor,
    build_packets,
    get_model,
)
from splatsim.hils.udp_publisher import HesaiHilsPublisher

__all__ = [
    "MODELS",
    "HesaiHilsPublisher",
    "HesaiModel",
    "build_frame_tensor",
    "build_packets",
    "get_model",
]
