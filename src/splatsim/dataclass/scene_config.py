from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from splatsim.dataclass.lod_config import LodConfig, LodTier
from splatsim.dataclass.lidar_config import LidarConfig
from splatsim.dataclass.renderer_config import RendererConfig
from splatsim.dataclass.rigid_body_config import RigidBodyConfig
from splatsim.dataclass.viewer_config import ViewerConfig


def _lidar_sensors_from_rigs(rigs) -> list[LidarConfig]:
    """Build :class:`LidarConfig` entries from a scene USDZ's rig calibrations.

    3dgs_io stores each LiDAR pose as sensor-in-rig: ``translation`` is the
    mount position in the ego/base frame directly (no inversion needed), and
    ``rotation`` is an ``xyzw`` unit quaternion — reordered here to the
    ``wxyz`` form :func:`build_lidar_sensors_from_config` expects. The
    intrinsics (row/column counts, spin rate, range, and the per-beam
    ``elevation_deg`` table) live in the free-form ``lidar_model.parameters``.
    """
    sensors: list[LidarConfig] = []
    for rig in rigs:
        for cal in getattr(rig, "lidars", None) or []:
            ext = cal.extrinsics
            tx, ty, tz = (float(v) for v in ext.translation)
            qx, qy, qz, qw = (float(v) for v in ext.rotation)  # xyzw
            model = getattr(cal, "lidar_model", None)
            params = dict(model.parameters) if model is not None else {}
            elevation = params.get("elevation_deg")
            sensors.append(
                LidarConfig(
                    name=cal.name,
                    # Geometry is driven by the explicit elevation table below,
                    # so no built-in named table (OT128/XT32) is assumed here.
                    sensor_type="",
                    n_rows=int(params.get("n_rows", 128)),
                    n_columns=int(params.get("n_columns", 2048)),
                    fps=float(params.get("fps", 10.0)),
                    min_range_m=float(params.get("min_range_m", 0.3)),
                    max_range_m=float(params.get("max_range_m", 120.0)),
                    position=(tx, ty, tz),
                    rotation=(qw, qx, qy, qz),
                    elevation_deg=(
                        tuple(float(e) for e in elevation) if elevation else None
                    ),
                    pointcloud_topic=f"/sensing/lidar/{cal.name}/pointcloud",
                    frame_id=cal.name,
                )
            )
    return sensors


@dataclass
class SceneConfig:
    """Top-level scene configuration loaded from YAML."""

    background_tileset: str | None = None
    use_sh: bool = True
    rigid_bodies: list[RigidBodyConfig] = field(default_factory=list)
    lidar_sensors: list[LidarConfig] = field(default_factory=list)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    lod: LodConfig = field(default_factory=LodConfig)
    # World-frame camera pose seeded from a scene USDZ. Translated into the
    # viewer's tile-local frame after the background is loaded (see
    # :func:`splatsim.scene.resolve_initial_pose`).
    initial_camera_world_position: tuple[float, float, float] | None = None
    initial_camera_yaw_deg: float | None = None

    @staticmethod
    def from_source(
        path: str | Path,
        *,
        camera_name: str | None = None,
        lod_enabled: bool | None = None,
    ) -> SceneConfig:
        """Build a SceneConfig from either a scene YAML or a scene USDZ.

        ``camera_name`` selects which rig camera seeds intrinsics and initial
        pose for scene USDZ inputs; ignored for YAML.

        ``lod_enabled`` overrides ``cfg.lod.enabled`` after loading when not
        ``None``. Pass ``True``/``False`` from the CLI to force LoD on/off
        regardless of the file's default.
        """
        path = Path(path)
        if path.suffix.lower() == ".usdz":
            cfg = SceneConfig.from_usdz(path, camera_name=camera_name)
        else:
            cfg = SceneConfig.from_yaml(path)
        if lod_enabled is not None:
            cfg.lod.enabled = lod_enabled
        return cfg

    @staticmethod
    def from_usdz(path: str | Path, *, camera_name: str | None = None) -> SceneConfig:
        """Build a SceneConfig from a scene USDZ's embedded ``scene.json``.

        Only metadata is read here; the heavy SPZ chunks are loaded later
        by :class:`Background` when it sees the same ``.usdz`` path.

        If the USDZ ships a ``rig_trajectories.json`` sidecar containing
        cameras, the most forward-facing camera's intrinsics seed
        ``renderer.width/height`` and ``viewer.fov_y_deg``, and the
        composed ``RigPose × CameraExtrinsics`` at the first timestamp
        seeds ``initial_camera_world_position`` and
        ``initial_camera_yaw_deg``.
        """
        from splatsim._usdz import (
            camera_to_viewer_intrinsics,
            first_camera,
            initial_camera_pose_from_rig_trajectories,
            read_rig_trajectories,
            read_scene_json,
        )

        path = Path(path)
        meta = read_scene_json(path)

        rd = meta.get("render_defaults", {})
        renderer = RendererConfig(
            near_plane=rd.get("near_plane", RendererConfig.near_plane),
            far_plane=rd.get("far_plane", 60.0),
            exposure=rd.get("exposure", 1.0),
            # Match gaussian_factory's RGB reference render: discard
            # sub-pixel splats instead of accumulating their low-contribution
            # tails. USDZ scenes do not currently serialize this option.
            radius_clip=1.0,
        )
        viewer = ViewerConfig()
        initial_pos: tuple[float, float, float] | None = None
        initial_yaw: float | None = None
        lidar_sensors: list[LidarConfig] = []

        rig_uri = meta.get("extras", {}).get("rig_trajectories")
        if rig_uri:
            rigs = read_rig_trajectories(path, rig_uri)
            cam = first_camera(rigs, name=camera_name)
            if cam is not None:
                width, height, fov_y_deg = camera_to_viewer_intrinsics(cam)
                if width and height:
                    renderer.width = width
                    renderer.height = height
                if fov_y_deg is not None:
                    viewer.fov_y_deg = fov_y_deg

            pose = initial_camera_pose_from_rig_trajectories(rigs, name=camera_name)
            if pose is not None:
                initial_pos, initial_yaw = pose

            lidar_sensors = _lidar_sensors_from_rigs(rigs)

        return SceneConfig(
            background_tileset=str(path),
            use_sh=True,
            rigid_bodies=[],
            lidar_sensors=lidar_sensors,
            renderer=renderer,
            viewer=viewer,
            lod=LodConfig(),
            initial_camera_world_position=initial_pos,
            initial_camera_yaw_deg=initial_yaw,
        )

    @staticmethod
    def from_yaml(path: str | Path) -> SceneConfig:
        """Load a SceneConfig from a YAML file.

        Paths in the YAML are resolved relative to the YAML file's directory.
        """
        path = Path(path)
        with path.open() as f:
            raw = yaml.safe_load(f)

        base_dir = path.parent

        # Background
        bg_tileset = raw.get("background_tileset")
        if bg_tileset is not None:
            bg_tileset = str(base_dir / bg_tileset)

        # Rigid bodies
        rigid_bodies: list[RigidBodyConfig] = []
        for rb in raw.get("rigid_bodies", []):
            source = str(base_dir / rb["source"])
            position = tuple(rb.get("position", [0.0, 0.0, 0.0]))
            rotation = tuple(rb.get("rotation", [1.0, 0.0, 0.0, 0.0]))
            name = rb.get("name", Path(source).stem)
            rigid_bodies.append(
                RigidBodyConfig(
                    source=source,
                    name=name,
                    position=position,
                    rotation=rotation,
                    use_sh=rb.get("use_sh", False),
                )
            )

        # LOD
        lod_raw = raw.get("lod", {})
        lod_tiers: list[LodTier] = []
        for tier in lod_raw.get("tiers", []):
            lod_tiers.append(
                LodTier(
                    fraction=tier.get("fraction", 1.0),
                    max_distance=tier.get("max_distance", float("inf")),
                )
            )
        lod = LodConfig(enabled=lod_raw.get("enabled", True))
        if "max_gaussians_per_cell" in lod_raw:
            lod.max_gaussians_per_cell = lod_raw["max_gaussians_per_cell"]
        if lod_tiers:
            lod.tiers = lod_tiers

        # Renderer
        renderer_raw = raw.get("renderer", {})
        bg_color = renderer_raw.get("background_color", [0.0, 0.0, 0.0])
        renderer = RendererConfig(
            width=renderer_raw.get("width", 960),
            height=renderer_raw.get("height", 540),
            background_color=tuple(bg_color),
            near_plane=renderer_raw.get("near_plane", 0.01),
            far_plane=renderer_raw.get("far_plane", 1000.0),
            device=renderer_raw.get("device", "cuda"),
            radius_clip=renderer_raw.get("radius_clip", 0.0),
        )

        # Viewer
        viewer_raw = raw.get("viewer", {})
        viewer = ViewerConfig(
            fov_y_deg=viewer_raw.get("fov_y_deg", 60.0),
            move_speed=viewer_raw.get("move_speed", 5.0),
            rotate_speed=viewer_raw.get("rotate_speed", 1.5),
        )

        # LiDAR sensors
        lidar_sensors: list[LidarConfig] = []
        for lidar in raw.get("lidar_sensors", []):
            pos = lidar.get("position", [0.0, 0.0, 1.8])
            rot = lidar.get("rotation", [0.0, 0.0, 0.0])
            lidar_sensors.append(
                LidarConfig(
                    name=lidar.get("name", "lidar"),
                    enabled=lidar.get("enabled", True),
                    sensor_type=lidar.get("sensor_type", "OT128"),
                    n_rows=lidar.get("n_rows", 128),
                    n_columns=lidar.get("n_columns", 2048),
                    fps=lidar.get("fps", 10.0),
                    min_range_m=lidar.get("min_range_m", 0.3),
                    max_range_m=lidar.get("max_range_m", 120.0),
                    position=tuple(pos),
                    rotation=tuple(rot),
                    pointcloud_topic=lidar.get(
                        "pointcloud_topic", "/splatsim/lidar/pointcloud"
                    ),
                    frame_id=lidar.get("frame_id", "splatsim_lidar"),
                    drop_threshold=lidar.get("drop_threshold", 0.5),
                    alpha_threshold=lidar.get("alpha_threshold", 0.1),
                    communication=lidar.get("communication", "dds"),
                    hils_host=lidar.get("hils_host", "127.0.0.1"),
                    hils_port=lidar.get("hils_port", 2368),
                    hils_start_epoch=lidar.get("hils_start_epoch"),
                )
            )

        return SceneConfig(
            background_tileset=bg_tileset,
            use_sh=raw.get("use_sh", True),
            rigid_bodies=rigid_bodies,
            lidar_sensors=lidar_sensors,
            renderer=renderer,
            viewer=viewer,
            lod=lod,
        )
