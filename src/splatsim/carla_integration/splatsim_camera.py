from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import carla
import numpy as np
import torch
from numpy.typing import NDArray

from autoware_carla_scenario.sensor import CameraSensorBase, CameraSensorConfig
from splatsim.renderer import Renderer

if TYPE_CHECKING:
    from splatsim.carla_integration.geo_transform import GeoTransform
    from splatsim.scene import Scene

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplatSimCameraSensorConfig(CameraSensorConfig):
    """Camera sensor config extended with SplatSim renderer parameters."""

    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    near_plane: float = 0.01
    far_plane: float = 1000.0
    device: str = "cuda"


class SplatSimCameraSensor(CameraSensorBase):
    """Camera sensor that renders using SplatSim's Gaussian Splatting renderer.

    Instead of capturing images from CARLA's built-in camera sensor,
    this sensor renders the Gaussian Splatting scene from the
    perspective of a CARLA actor using gsplat rasterization.

    All coordinate conversion is done in float64 via :class:`GeoTransform`
    to avoid floating-point precision issues.  The final view matrix is
    cast to float32 only when passed to the gsplat rasteriser.
    """

    def __init__(
        self,
        config: SplatSimCameraSensorConfig,
        scene: Scene,
        *,
        geo_transform: GeoTransform,
    ) -> None:
        super().__init__(config)
        self._splatsim_config = config
        self._scene = scene
        self._device = torch.device(config.device)
        self._geo_transform = geo_transform

        self._renderer = Renderer(
            width=config.image_width,
            height=config.image_height,
            device=self._device,
            background_color=config.background_color,
            near_plane=config.near_plane,
            far_plane=config.far_plane,
        )

        # Intrinsic matrix from CameraSensorConfig (derived from fov + resolution).
        self._K = torch.tensor(
            config.intrinsic_matrix,
            device=self._device,
            dtype=torch.float32,
        )

        self._actor: carla.Actor | None = None
        self._frame_count = 0

    # -- public properties ---------------------------------------------------

    @property
    def renderer(self) -> Renderer:
        """The SplatSim renderer used by this sensor."""
        return self._renderer

    @property
    def scene(self) -> Scene:
        """The Gaussian Splatting scene being rendered."""
        return self._scene

    # -- CameraSensorBase interface ------------------------------------------

    def attach(self, world: carla.World, actor: carla.Actor) -> None:
        """Store the CARLA actor reference for per-frame pose tracking."""
        self._actor = actor
        self._attached = True

    def destroy(self) -> None:
        """Release the actor reference."""
        self._actor = None
        self._attached = False

    def get_image(self) -> Optional[NDArray[np.uint8]]:
        """Render the scene from the actor's current pose.

        Returns
        -------
        NDArray[np.uint8] | None
            H x W x 3 BGR image (OpenCV convention), or *None* if the
            sensor is not attached.
        """
        if self._actor is None:
            return None

        viewmat = self._compute_viewmat()

        with torch.no_grad():
            rgb = self._renderer.render(viewmat, self._K, scene=self._scene)

        # float32 RGB [H, W, 3] -> uint8 BGR [H, W, 3]
        rgb_np: NDArray[np.uint8] = (rgb.clamp(0.0, 1.0) * 255).byte().cpu().numpy()
        bgr_np: NDArray[np.uint8] = np.ascontiguousarray(rgb_np[:, :, ::-1])
        return bgr_np

    # -- internals -----------------------------------------------------------

    def _compute_viewmat(self) -> torch.Tensor:
        """Build a gsplat-compatible 4x4 view matrix from the CARLA actor pose.

        All intermediate computation is in float64 (via :class:`GeoTransform`)
        to avoid precision loss with large ECEF / UTM coordinates.
        The final 4x4 matrix is small-valued (tile-local, re-centered) and
        safe to cast to float32.

        Pipeline
        --------
        1. Camera pose in CARLA world (actor * sensor offset)
        2. Position: CARLA → xodr → LLA → ECEF → tile-local − origin  (float64)
        3. Rotation: CARLA → ENU → ECEF → tile-local                   (float64)
        4. Remap camera axes (X=fwd, Y=right, Z=up) to gsplat RDF
        5. Invert camera-to-world → world-to-camera view matrix
        """
        assert self._actor is not None  # noqa: S101
        cfg = self.config

        # 1. Camera pose in CARLA world
        T_world_actor = np.asarray(
            self._actor.get_transform().get_matrix(), dtype=np.float64
        )
        sensor_tf = carla.Transform(
            carla.Location(x=cfg.position_x, y=cfg.position_y, z=cfg.position_z),
            carla.Rotation(roll=cfg.roll, pitch=cfg.pitch, yaw=cfg.yaw),
        )
        T_actor_camera = np.asarray(sensor_tf.get_matrix(), dtype=np.float64)
        T_carla_cam = T_world_actor @ T_actor_camera

        carla_pos = T_carla_cam[:3, 3]
        R_carla = T_carla_cam[:3, :3]

        # 2. Position in re-centered tile-local (float64)
        tile_pos = self._geo_transform.carla_position_to_tile_local(
            carla_pos[0], carla_pos[1], carla_pos[2]
        )

        # 3. Rotation in tile-local (float64)
        R_tile = self._geo_transform.carla_rotation_to_tile_local(R_carla)

        # Debug: log tile-local position on first frame
        self._frame_count += 1
        if self._frame_count <= 3:
            import math as _m  # noqa: PLC0415

            fwd = R_tile[:, 0]
            _yaw = _m.atan2(fwd[0], -fwd[1])
            logger.info(
                "[frame %d] CARLA=(%.1f, %.1f, %.1f) -> "
                "tile_local=(%.2f, %.2f, %.2f) yaw=%.1f°",
                self._frame_count,
                carla_pos[0],
                carla_pos[1],
                carla_pos[2],
                tile_pos[0],
                tile_pos[1],
                tile_pos[2],
                _m.degrees(_yaw),
            )

        # 4. Remap camera axes (X=fwd, Y=right, Z=up) to gsplat RDF:
        #    gsplat right (X)   = camera right   (Y) = R_tile[:, 1]
        #    gsplat down  (Y)   = camera -up    (-Z) = -R_tile[:, 2]
        #    gsplat forward (Z) = camera forward (X) = R_tile[:, 0]
        R_rdf = np.column_stack([R_tile[:, 1], -R_tile[:, 2], R_tile[:, 0]])

        # 5. Invert camera-to-world → world-to-camera
        viewmat = np.eye(4, dtype=np.float64)
        viewmat[:3, :3] = R_rdf.T
        viewmat[:3, 3] = -(R_rdf.T @ tile_pos)

        return torch.tensor(viewmat, device=self._device, dtype=torch.float32)
