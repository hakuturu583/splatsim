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
    """

    def __init__(
        self,
        config: SplatSimCameraSensorConfig,
        scene: Scene,
        *,
        carla_to_splatsim: NDArray[np.float64] | None = None,
    ) -> None:
        super().__init__(config)
        self._splatsim_config = config
        self._scene = scene
        self._device = torch.device(config.device)

        # Track whether the user explicitly provided the transform.
        self._transform_provided = carla_to_splatsim is not None

        # Coordinate transform: CARLA world -> SplatSim world (4x4).
        self._carla_to_splatsim: NDArray[np.float64] = (
            np.asarray(carla_to_splatsim, dtype=np.float64)
            if carla_to_splatsim is not None
            else np.eye(4, dtype=np.float64)
        )

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
        self._auto_aligned = False

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

        # Auto-compute carla_to_splatsim on first call if not provided.
        if not self._transform_provided and not self._auto_aligned:
            self._auto_align()

        viewmat = self._compute_viewmat()

        with torch.no_grad():
            rgb = self._renderer.render(viewmat, self._K, scene=self._scene)

        # float32 RGB [H, W, 3] -> uint8 BGR [H, W, 3]
        rgb_np: NDArray[np.uint8] = (rgb.clamp(0.0, 1.0) * 255).byte().cpu().numpy()
        bgr_np: NDArray[np.uint8] = np.ascontiguousarray(rgb_np[:, :, ::-1])
        return bgr_np

    # -- internals -----------------------------------------------------------

    def _scene_centroid(self) -> NDArray[np.float64] | None:
        """Compute the centroid of all Gaussian means in the scene."""
        all_means: list[torch.Tensor] = []
        if self._scene.background is not None:
            all_means.append(self._scene.background.tensors.means)
        for rb in self._scene.rigid_body_list:
            all_means.append(rb.tensors.means)
        if not all_means:
            return None
        centroid: NDArray[np.float64] = (
            torch.cat(all_means, dim=0).mean(dim=0).cpu().numpy().astype(np.float64)
        )
        return centroid

    def _auto_align(self) -> None:
        """Auto-compute a translation-only carla_to_splatsim transform.

        Maps the current CARLA camera position to the scene centroid so
        that the rendered image is not blank.  This is an approximation;
        for accurate alignment provide an explicit ``carla_to_splatsim``.
        """
        self._auto_aligned = True

        centroid = self._scene_centroid()
        if centroid is None:
            logger.warning("Cannot auto-align: scene has no Gaussians")
            return

        assert self._actor is not None  # noqa: S101
        carla_pos = np.asarray(
            [
                self._actor.get_location().x,
                self._actor.get_location().y,
                self._actor.get_location().z,
            ],
            dtype=np.float64,
        )

        offset = centroid - carla_pos
        self._carla_to_splatsim = np.eye(4, dtype=np.float64)
        self._carla_to_splatsim[:3, 3] = offset

        logger.warning(
            "carla_to_splatsim not provided — auto-aligned with translation only. "
            "CARLA pos=(%.1f, %.1f, %.1f) -> scene centroid=(%.1f, %.1f, %.1f), "
            "offset=(%.1f, %.1f, %.1f). "
            "For accurate results, provide an explicit carla_to_splatsim transform.",
            *carla_pos,
            *centroid,
            *offset,
        )

    def _compute_viewmat(self) -> torch.Tensor:
        """Build a gsplat-compatible 4x4 view matrix from the CARLA actor pose.

        Coordinate pipeline
        --------------------
        1. Actor world pose in CARLA frame  (``T_world_actor``)
        2. + sensor offset in actor-local   (``T_actor_camera``)
        3. * ``carla_to_splatsim``           → camera in SplatSim world
        4. Remap CARLA camera axes (X=fwd, Y=right, Z=up)
           to gsplat RDF axes     (X=right, Y=down,  Z=fwd)
        5. Invert to get world-to-camera    → ``viewmat``
        """
        assert self._actor is not None  # noqa: S101
        cfg = self.config

        # 1. Actor world pose in CARLA coordinates
        T_world_actor = np.asarray(
            self._actor.get_transform().get_matrix(), dtype=np.float64
        )

        # 2. Sensor offset in actor-local frame (CARLA convention)
        sensor_tf = carla.Transform(
            carla.Location(x=cfg.position_x, y=cfg.position_y, z=cfg.position_z),
            carla.Rotation(roll=cfg.roll, pitch=cfg.pitch, yaw=cfg.yaw),
        )
        T_actor_camera = np.asarray(sensor_tf.get_matrix(), dtype=np.float64)

        # 3. Camera world pose in SplatSim coordinates
        T_cam = self._carla_to_splatsim @ T_world_actor @ T_actor_camera

        # 4. Remap CARLA camera axes to gsplat RDF:
        #    gsplat right (X)   = CARLA right   (Y) = R[:, 1]
        #    gsplat down  (Y)   = CARLA -up    (-Z) = -R[:, 2]
        #    gsplat forward (Z) = CARLA forward (X) = R[:, 0]
        R = T_cam[:3, :3]
        t = T_cam[:3, 3]

        R_rdf = np.column_stack([R[:, 1], -R[:, 2], R[:, 0]])

        # 5. Invert camera-to-world -> world-to-camera
        viewmat = np.eye(4, dtype=np.float64)
        viewmat[:3, :3] = R_rdf.T
        viewmat[:3, 3] = -(R_rdf.T @ t)

        return torch.tensor(viewmat, device=self._device, dtype=torch.float32)
