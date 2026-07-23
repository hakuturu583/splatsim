"""SplatSim rendering and ROS 2 publishing configuration.

This config is shared across all scenarios that use SplatSim cameras.
It is expected to grow as new features are added (e.g. manual
correction offsets, multi-camera support, per-camera topic namespacing,
renderer quality settings).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SplatSimConfig:
    """SplatSim rendering and ROS 2 publishing parameters.

    This dataclass is populated from the ``splatsim`` Hydra config group
    and passed to any scenario that integrates SplatSim cameras.

    .. note::

       This class is expected to be extended with additional fields
       (e.g. coordinate correction offsets, renderer quality knobs,
       multi-camera configuration) as the integration matures.
    """

    scene_source: str = ""
    image_topic: str = "/splatsim/image_raw"
    camera_info_topic: str = "/splatsim/camera_info"
    frame_id: str = "splatsim_camera"
    compress_format: str = ""
