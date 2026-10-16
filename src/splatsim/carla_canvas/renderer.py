"""splatsim as a carla_canvas renderer.

carla_canvas runs a typesafe_carla program unchanged and has CARLA's cameras
and LiDARs rendered by a renderer service; this module makes splatsim that
renderer. carla_canvas keeps everything CARLA-specific (sensor attributes,
CARLA's LiDAR model and measurement layouts, gRPC, the ROS 2 topics); splatsim
renders:

* cameras: :class:`splatsim.renderer.Renderer` (gsplat), from the CARLA pose;
* LiDARs: :class:`splatsim.lidar_renderer.LidarRenderer`'s panorama, rendered
  at exactly the CARLA lasers' elevations and sampled per CARLA ray, with
  splatsim's learned intensity and ray drop;
* the world: a scene USDZ per map, its actor bank posing a splat asset at
  each CARLA actor matching a blueprint pattern.

Configuration (carla_canvas's file, ``CARLA_CANVAS_CONFIG``)::

    renderer:
      backend: splatsim.carla_canvas:SplatsimRenderer
      assets:
        maps:
          "Town10HD*":                  # glob on the map name (last part or whole)
            scene: scenes/town10.usdz
            alignment: geo              # geo (default): the map's OpenDRIVE geo-reference
            # matrix: [[...], ...]      # or a 4x4 CARLA-world -> tile-local matrix
        actors:                         # blueprint glob -> the bundle's actor asset id
          "vehicle.*": sedan
          "walker.*": {asset: pedestrian, anchor: origin}
      options:
        device: cuda
        background_color: [0, 0, 0]     # 0-1
        near_plane: 0.01
        far_plane: 1000.0
        exposure: 1.0
        lidar: {drop_threshold: 0.5, alpha_threshold: 0.1, min_range: 0.3, n_columns: auto}

``alignment: geo`` needs the ``carla_canvas`` extra (simple-lanelet2's
MGRS projector and pyproj, :mod:`splatsim.geo_transform`). An actor's asset is placed
at its bounding box's center (``anchor: bounding_box``, default) or at its
CARLA origin (``anchor: origin``), x forward / y left / z up.
"""

from __future__ import annotations

import fnmatch
import itertools
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from splatsim.scene import Scene

try:
    from carla_canvas import (
        CameraFrame,
        CameraView,
        LidarView,
        RayHits,
        Renderer,
        WorldInfo,
    )
    from carla_canvas.geometry import Transform
    from carla_canvas.scene import ActorInfo
    from carla_canvas.sensors import LidarSpec
except ImportError:
    raise ImportError(
        "splatsim.carla_canvas requires carla_canvas.\n"
        "Install with: pip install carla-canvas"
    ) from None

from splatsim.carla_canvas.frames import (
    SceneFrame,
    default_n_columns,
    quaternion_wxyz,
    rays_to_panorama,
)

log = logging.getLogger("splatsim.carla_canvas")


def _version() -> str:
    try:
        return metadata.version("splatsim")
    except metadata.PackageNotFoundError:
        return "0"


@dataclass(frozen=True)
class _ActorAsset:
    pattern: str
    asset_id: str
    anchor: str  # bounding_box | origin


def _actor_assets(spec: dict[str, Any]) -> list[_ActorAsset]:
    out = []
    for pattern, value in (spec or {}).items():
        if isinstance(value, str):
            out.append(_ActorAsset(pattern, value, "bounding_box"))
        else:
            anchor = str(value.get("anchor", "bounding_box"))
            if anchor not in ("bounding_box", "origin"):
                raise ValueError(
                    f"actors[{pattern!r}].anchor must be bounding_box or origin"
                )
            out.append(_ActorAsset(pattern, str(value["asset"]), anchor))
    return out


class SplatsimRenderer(Renderer):
    name = "splatsim"
    version = _version()
    camera_outputs = frozenset({"rgb"})
    semantic_lidar = False
    # Views hand poses, ray azimuths and actor poses over right-handed (CARLA's
    # with y negated, bodies FLU): SceneFrame only maps them into the tile.
    coordinate_frame = "right_handed"

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self._scene: Scene | None = None
        self._frame: SceneFrame | None = None
        self._actor_assets: list[_ActorAsset] = []
        self._asset_of: dict[int, _ActorAsset | None] = {}
        self._posed_frame = -1
        self._cameras: dict[tuple[int, int], Any] = {}
        self._lidars: dict[int, tuple[tuple, Any]] = {}

    @property
    def scene(self) -> Scene:
        if self._scene is None:
            raise RuntimeError("no world loaded (load_world has not run)")
        return self._scene

    @property
    def frame(self) -> SceneFrame:
        if self._frame is None:
            raise RuntimeError("no world loaded (load_world has not run)")
        return self._frame

    # --- world -------------------------------------------------------------------

    def load_world(self, world: WorldInfo) -> None:
        import torch

        from splatsim.dataclass import SceneConfig
        from splatsim.scene import Scene

        assets = world.config.get("assets") or {}
        spec = None
        for pattern, entry in (assets.get("maps") or {}).items():
            if fnmatch.fnmatchcase(world.map_basename, pattern) or fnmatch.fnmatchcase(
                world.map_name, pattern
            ):
                spec = entry
                break
        if spec is None:
            raise ValueError(
                f"no splatsim scene for map {world.map_name} (renderer.assets.maps)"
            )
        path = world.resolve(spec["scene"])
        self.device = torch.device(str(self.options.get("device", "cuda")))
        self._scene = Scene.from_config(
            SceneConfig.from_source(path), device=self.device
        )
        self._frame = self._scene_frame(spec, world)
        self._actor_assets = _actor_assets(assets.get("actors") or {})
        self._asset_of.clear()
        self._posed_frame = -1
        self._cameras.clear()
        self._lidars.clear()
        log.info("map %s: splatsim scene %s", world.map_name, path)

    def _scene_frame(self, spec: dict[str, Any], world: WorldInfo) -> SceneFrame:
        if "matrix" in spec:
            return SceneFrame.from_matrix(np.asarray(spec["matrix"], dtype=np.float64))
        if str(spec.get("alignment", "geo")) != "geo":
            raise ValueError("alignment must be geo, or give a 4x4 matrix")
        try:
            from splatsim.geo_transform import (
                GeoTransform,
                parse_geo_reference,
            )
        except ImportError as e:
            raise ImportError(
                "alignment: geo needs simple-lanelet2 and pyproj "
                "(pip install 'splatsim[carla_canvas]'), "
                f"or give the map an explicit matrix: {e}"
            ) from e
        if not world.opendrive:
            raise ValueError(
                "alignment: geo needs the map's OpenDRIVE (carla_canvas protocol 2)"
            )
        background = self.scene.background
        if background is None:
            raise ValueError(
                "alignment: geo needs a scene with a background (its ECEF anchor)"
            )
        geo = GeoTransform(
            proj_origin=parse_geo_reference(world.opendrive),
            ecef_rotation=background.ecef_rotation,
            ecef_translation=background.ecef_translation,
            tile_origin=background.tile_local_centroid.detach()
            .cpu()
            .numpy()
            .astype(np.float64),
        )
        return SceneFrame.from_geo(geo)

    # --- actors ------------------------------------------------------------------

    def _asset(self, info: ActorInfo) -> _ActorAsset | None:
        if info.id not in self._asset_of:
            self._asset_of[info.id] = next(
                (a for a in self._actor_assets if info.matches(a.pattern)), None
            )
        return self._asset_of[info.id]

    @staticmethod
    def _body_name(actor_id: int) -> str:
        return f"carla_{actor_id}"

    def _pose_actors(self, view: CameraView | LidarView) -> None:
        """Spawn, pose and remove the splat actors of this frame's CARLA actors
        (once per frame: every view of the frame shares the poses)."""
        scene = view.scene
        if scene.frame == self._posed_frame or not self._actor_assets:
            return
        self._posed_frame = scene.frame
        wanted = set()
        for actor_id, info in scene.actors.items():
            asset = self._asset(info)
            if (
                asset is None
                or actor_id not in scene.states
                or not info.is_visible_body
            ):
                continue
            pose = view.actor_matrix(actor_id)
            if asset.anchor == "bounding_box":
                box = info.bounding_box
                pose = pose @ Transform(box.location, box.rotation).matrix(view.frame)
            m = self.frame.flu_pose(pose)
            position = (float(m[0, 3]), float(m[1, 3]), float(m[2, 3]))
            rotation = quaternion_wxyz(m[:3, :3])
            name = self._body_name(actor_id)
            if name in self.scene.rigid_bodies:
                self.scene.set_pose(name, position, rotation)
            else:
                self.scene.spawn_actor(
                    asset.asset_id,
                    name,
                    position=position,
                    rotation=rotation,
                    world_position=False,
                )
            wanted.add(name)
        for name in list(self.scene.rigid_bodies):
            if name.startswith("carla_") and name not in wanted:
                self.scene.remove_rigid_body(name)

    @contextmanager
    def _hidden(self, actor_ids):
        """Leaves the actors' splats out (a LiDAR does not see its vehicle)."""
        hidden = {}
        for actor_id in actor_ids:
            name = self._body_name(actor_id)
            if name in self.scene.rigid_bodies:
                hidden[name] = self.scene.remove_rigid_body(name)
        try:
            yield
        finally:
            for name, body in hidden.items():
                self.scene.add_rigid_body(name, body)

    # --- cameras ---------------------------------------------------------------------

    def _camera_renderer(self, width: int, height: int):
        key = (width, height)
        renderer = self._cameras.get(key)
        if renderer is None:
            from splatsim.renderer import Renderer as SplatRenderer

            o = self.options
            renderer = self._cameras[key] = SplatRenderer(
                width,
                height,
                device=self.device,
                background_color=tuple(o.get("background_color", (0.0, 0.0, 0.0))),
                near_plane=float(o.get("near_plane", 0.01)),
                far_plane=float(o.get("far_plane", 1000.0)),
                radius_clip=float(o.get("radius_clip", 0.0)),
                exposure=float(o.get("exposure", 1.0)),
            )
        return renderer

    def _ppisp_camera(self, view: CameraView) -> str | None:
        """The scene's PPISP camera profile for this sensor (options.ppisp_camera:
        a name, or role_name glob -> name); None renders without PPISP."""
        profiles = self.options.get("ppisp_camera")
        if profiles is None or isinstance(profiles, str):
            return profiles
        role = view.attributes.get("role_name", "")
        for pattern, name in dict(profiles).items():
            if fnmatch.fnmatchcase(role, str(pattern)):
                return str(name)
        return None

    def render_camera(self, view: CameraView) -> CameraFrame:
        import torch

        cam = view.camera
        self._pose_actors(view)
        renderer = self._camera_renderer(cam.width, cam.height)
        viewmat = torch.tensor(
            self.frame.viewmat(view.pose_matrix()),
            device=self.device,
            dtype=torch.float32,
        )
        k = torch.tensor(cam.intrinsics(), device=self.device, dtype=torch.float32)
        with torch.no_grad(), self._hidden(view.ignore_actors):
            rgb = renderer.render(
                viewmat, k, scene=self.scene, camera_name=self._ppisp_camera(view)
            )
        return CameraFrame(rgb=(rgb.clamp(0.0, 1.0) * 255.0 + 0.5).byte().cpu().numpy())

    # --- LiDARs --------------------------------------------------------------------

    def _lidar_renderer(self, view: LidarView):
        """splatsim's LiDAR for this sensor: rows at the CARLA lasers' elevations."""
        from splatsim.lidar_renderer import LidarRenderer, LidarSensorSpec

        o = dict(self.options.get("lidar") or {})
        # the blueprint's attributes with CARLA's defaults, as carla_canvas fires them
        carla = LidarSpec.from_actor(view.scene.actors[view.sensor_id])
        columns = o.get("n_columns", "auto")
        if columns in (None, "auto"):
            columns = default_n_columns(
                carla.points_per_second,
                len(view.elevations),
                carla.rotation_frequency,
                carla.horizontal_fov,
            )
        scale = self.frame.scale  # ranges in scene units
        key = (
            tuple(float(e) for e in view.elevations),
            int(columns),
            float(view.max_distance) * scale,
            float(o.get("min_range", 0.3)) * scale,
        )
        cached = self._lidars.get(view.sensor_id)
        if cached is not None and cached[0] == key:
            return cached[1]
        elevations, n_columns, max_range, min_range = key
        if any(b >= a_ for a_, b in itertools.pairwise(elevations)):
            raise ValueError("the LiDAR's lasers must have distinct elevations")
        spec = LidarSensorSpec(
            name=f"carla_{view.sensor_id}",
            sensor_type="",
            s2b=np.eye(4),
            n_columns=n_columns,
            spinning_frequency_hz=carla.rotation_frequency,
            row_elevations_rad=tuple(float(np.radians(e)) for e in elevations),
        )
        renderer = LidarRenderer(
            spec, device=self.device, min_range_m=min_range, max_range_m=max_range
        )
        self._lidars[view.sensor_id] = (key, renderer)
        return renderer

    def cast_lidar_rays(self, view: LidarView) -> RayHits:
        import torch

        self._pose_actors(view)
        renderer = self._lidar_renderer(view)
        o = dict(self.options.get("lidar") or {})
        sensor_to_world = torch.tensor(
            self.frame.flu_pose(view.pose_matrix()),
            device=self.device,
            dtype=torch.float32,
        )
        with torch.no_grad(), self._hidden(view.ignore_actors):
            panorama = renderer.render(sensor_to_world, scene=self.scene)
            image = renderer.panorama_to_range_image(
                panorama,
                drop_threshold=float(o.get("drop_threshold", 0.5)),
                alpha_threshold=float(o.get("alpha_threshold", 0.1)),
            )
        rows, cols = rays_to_panorama(
            view.elevations, view.channel, view.body_azimuths(), renderer.n_columns
        )
        rows_t = torch.as_tensor(rows, device=self.device)
        cols_t = torch.as_tensor(cols, device=self.device)
        valid = image["valid"][rows_t, cols_t].cpu().numpy()
        distance = image["distance"][rows_t, cols_t].cpu().numpy().astype(np.float64)
        distance /= self.frame.scale  # scene units -> CARLA metres
        intensity = image["intensity"][rows_t, cols_t].clamp(0.0, 1.0).cpu().numpy()
        return RayHits(np.where(valid, distance, np.inf), intensity=intensity)
