try:
    import carla as _carla  # noqa: F401
except ImportError:
    raise ImportError(
        "splatsim.carla_integration requires CARLA dependencies.\n"
        "Install with: pip install splatsim[carla]"
    ) from None

from splatsim.carla_integration.attach_splatsim_camera import (
    AttachSplatSimCameraAction,
)
from splatsim.carla_integration.attach_splatsim_lidar import (
    AttachSplatSimLidarAction,
)
from splatsim.carla_integration.geo_transform import GeoTransform, parse_geo_reference
from splatsim.carla_integration.splatsim_config import SplatSimConfig
from splatsim.carla_integration.splatsim_camera import (
    SplatSimCameraSensor,
    SplatSimCameraSensorConfig,
)
from splatsim.carla_integration.splatsim_lidar import (
    SplatSimLidarSensor,
    SplatSimLidarSensorConfig,
)
from splatsim.carla_integration.splatsim_scenario import SplatSimScenario

__all__ = [
    "AttachSplatSimCameraAction",
    "AttachSplatSimLidarAction",
    "GeoTransform",
    "SplatSimCameraSensor",
    "SplatSimCameraSensorConfig",
    "SplatSimConfig",
    "SplatSimLidarSensor",
    "SplatSimLidarSensorConfig",
    "SplatSimScenario",
    "parse_geo_reference",
]
