"""The carla_canvas backend through carla_canvas's renderer service, with
stand-ins for splatsim's GPU renderers and scene (no CUDA, no scene USDZ):
the requests carla_canvas sends become the poses, view matrices and
panorama samples splatsim is asked for."""

from __future__ import annotations

import math
import sys
import types
from typing import ClassVar

import numpy as np
import pytest

carla_canvas = pytest.importorskip("carla_canvas")
torch = pytest.importorskip("torch")

from carla_canvas.geometry import Transform  # noqa: E402
from carla_canvas.server.service import RendererService  # noqa: E402
from carla_canvas.v1 import renderer_pb2 as pb  # noqa: E402

FLIP = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
VEHICLE, CAMERA, LIDAR, OTHER = 10, 11, 12, 20


class FakeScene:
    def __init__(self) -> None:
        self.rigid_bodies: dict[str, dict] = {}
        self.background = None

    def spawn_actor(self, asset_id, name, *, position, rotation, world_position):
        assert world_position is False
        self.rigid_bodies[name] = {
            "asset": asset_id,
            "position": position,
            "rotation": rotation,
        }

    def set_pose(self, name, position, rotation):
        self.rigid_bodies[name].update(position=position, rotation=rotation)

    def remove_rigid_body(self, name):
        return self.rigid_bodies.pop(name)

    def add_rigid_body(self, name, body):
        self.rigid_bodies[name] = body


class FakeCameraRenderer:
    calls: ClassVar[list] = []

    def __init__(self, width, height, **kwargs):
        self.width, self.height = width, height

    def render(self, viewmat, k, *, scene, camera_name=None):
        FakeCameraRenderer.calls.append(
            (viewmat.clone(), k.clone(), set(scene.rigid_bodies), camera_name)
        )
        img = torch.zeros(self.height, self.width, 3)
        img[..., 0] = 1.0  # red
        return img


class FakeLidarSensorSpec:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeLidarRenderer:
    calls: ClassVar[list] = []

    def __init__(self, spec, *, device, min_range_m, max_range_m):
        self.spec, self.max_range_m = spec, max_range_m
        self.n_columns = spec.n_columns

    def render(self, sensor_to_world, *, scene):
        FakeLidarRenderer.calls.append(
            (sensor_to_world.clone(), set(scene.rigid_bodies))
        )
        return {}

    def panorama_to_range_image(self, panorama, *, drop_threshold, alpha_threshold):
        h, w = len(self.spec.row_elevations_rad), self.n_columns
        rows = torch.arange(h, dtype=torch.float32)[:, None].expand(h, w)
        cols = torch.arange(w, dtype=torch.float32)[None, :].expand(h, w)
        valid = torch.ones(h, w, dtype=torch.bool)
        valid[:, w // 4] = False  # FLU azimuth +90 (CARLA yaw -90, left): dropped
        return {"distance": 10.0 + rows, "intensity": cols / w, "valid": valid}


@pytest.fixture
def service(monkeypatch):
    modules = {
        "splatsim.renderer": types.SimpleNamespace(Renderer=FakeCameraRenderer),
        "splatsim.lidar_renderer": types.SimpleNamespace(
            LidarRenderer=FakeLidarRenderer, LidarSensorSpec=FakeLidarSensorSpec
        ),
        "splatsim.scene": types.SimpleNamespace(
            Scene=types.SimpleNamespace(from_config=lambda cfg, device: FakeScene())
        ),
        "splatsim.dataclass": types.SimpleNamespace(
            SceneConfig=types.SimpleNamespace(from_source=lambda path: path)
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    FakeCameraRenderer.calls.clear()
    FakeLidarRenderer.calls.clear()
    from splatsim.carla_canvas.renderer import SplatsimRenderer

    renderer = SplatsimRenderer({"device": "cpu"})
    svc = RendererService(renderer)
    context = types.SimpleNamespace(
        abort=lambda code, msg: pytest.fail(f"{code}: {msg}")
    )
    config = {
        "assets": {
            "maps": {"Town*": {"scene": "town.usdz", "matrix": FLIP}},
            "actors": {"vehicle.*": "sedan"},
        }
    }
    request = pb.LoadWorldRequest(map_name="Carla/Maps/Town01", episode_id=1)
    request.renderer_config.update(config)
    svc.LoadWorld(request, context)
    return svc, renderer, context


def actor(actor_id, type_id, parent_id=0, extent=(0, 0, 0), **attributes):
    a = pb.Actor(id=actor_id, type_id=type_id, parent_id=parent_id)
    a.bounding_box.location.z = extent[2]
    a.bounding_box.extent.x, a.bounding_box.extent.y, a.bounding_box.extent.z = extent
    for k, v in attributes.items():
        a.attributes.add(id=k, value=str(v))
    return a


def render_request(frame, poses, new=(), sensors=()):
    r = pb.RenderRequest()
    r.snapshot.timestamp.frame = frame
    r.snapshot.timestamp.elapsed_seconds = frame * 0.05
    r.snapshot.timestamp.delta_seconds = 0.05
    for actor_id, t in poses.items():
        t.to_proto(r.snapshot.actors.add(id=actor_id).transform)
    r.new_actors.extend(new)
    for sensor_id in sensors:
        r.sensors.add(sensor_id=sensor_id, deliver=True)
    return r


def test_camera_and_actors(service) -> None:
    svc, renderer, ctx = service
    svc.SpawnSensor(
        pb.SpawnSensorRequest(
            sensor=actor(
                CAMERA, "sensor.camera.rgb", VEHICLE, image_size_x=8, image_size_y=4
            )
        ),
        ctx,
    )
    poses = {
        VEHICLE: Transform((0, 0, 0)),
        CAMERA: Transform((1.5, 0, 2.0)),
        OTHER: Transform((10, 3, 0), (0, 90, 0)),
    }
    r = svc.Render(
        render_request(
            1,
            poses,
            new=[
                actor(VEHICLE, "vehicle.ego", extent=(2, 1, 0.8)),
                actor(OTHER, "vehicle.npc", extent=(2, 1, 0.7)),
            ],
            sensors=[CAMERA],
        ),
        ctx,
    )
    image = np.frombuffer(r.measurements[0].image.raw_data, np.uint8).reshape(4, 8, 4)
    assert (image[..., 2] == 255).all() and (image[..., 0] == 0).all()  # BGRA of red
    viewmat, k, bodies, ppisp = FakeCameraRenderer.calls[0]
    assert ppisp is None  # no ppisp_camera option
    assert bodies == {"carla_10", "carla_20"}  # cameras see the ego vehicle
    ahead = viewmat.double().numpy() @ np.array([6.5, 0.0, 2.0, 1.0])  # 5 m ahead
    np.testing.assert_allclose(ahead[:3], [0, 0, 5], atol=1e-5)
    assert k[0, 0].item() == pytest.approx(4.0) and k[0, 2].item() == 4.0
    other = renderer.scene.rigid_bodies["carla_20"]
    assert other["asset"] == "sedan"
    # at its bounding box's center (z 0.7), y mirrored; yaw +90 (south) -> facing -y
    np.testing.assert_allclose(other["position"], (10, -3, 0.7), atol=1e-6)
    w, _x, _y, z = other["rotation"]
    assert w == pytest.approx(math.cos(math.pi / 4)) and z == pytest.approx(
        -math.sin(math.pi / 4)
    )
    # the NPC leaves: its splats go
    del poses[OTHER]
    req = render_request(2, poses, sensors=[CAMERA])
    req.removed_actors.append(OTHER)
    svc.Render(req, ctx)
    assert set(renderer.scene.rigid_bodies) == {"carla_10"}


def test_lidar_samples_the_panorama(service) -> None:
    svc, renderer, ctx = service
    svc.SpawnSensor(
        pb.SpawnSensorRequest(
            sensor=actor(
                LIDAR,
                "sensor.lidar.ray_cast",
                VEHICLE,
                channels=4,
                range=80,
                upper_fov=5,
                lower_fov=-25,
                points_per_second=8000,
                rotation_frequency=10,
                dropoff_general_rate=0,
                dropoff_intensity_limit=0,
            )
        ),
        ctx,
    )
    poses = {VEHICLE: Transform((0, 0, 0)), LIDAR: Transform((0, 0, 2.5))}
    r = svc.Render(
        render_request(
            1,
            poses,
            new=[actor(VEHICLE, "vehicle.ego", extent=(2, 1, 0.8))],
            sensors=[LIDAR],
        ),
        ctx,
    )
    m = r.measurements[0].lidar
    pts = np.frombuffer(m.raw_data, "<f4").reshape(-1, 4)
    sensor_to_world, bodies = FakeLidarRenderer.calls[0]
    assert bodies == set()  # the LiDAR's own vehicle is hidden
    spec = renderer._lidars[LIDAR][1].spec
    np.testing.assert_allclose(spec.row_elevations_rad, np.radians([5, -5, -15, -25]))
    assert spec.n_columns == 360 and renderer._lidars[LIDAR][1].max_range_m == 80
    # range = 10 + laser index (the stand-in's row), intensity from the column
    counts = list(m.points_per_channel)
    rng = np.linalg.norm(pts[:, :3], axis=1)
    start = 0
    for channel, n in enumerate(counts):
        np.testing.assert_allclose(rng[start : start + n], 10 + channel, rtol=1e-5)
        start += n
    assert ((pts[:, 3] >= 0) & (pts[:, 3] <= 1)).all() and len(np.unique(pts[:, 3])) > 1
    # 8000 pts/s * 0.05 s / 4 = 100 rays per laser, sweeping yaw -180..0 in the
    # first tick: the ray at -90 (the dropped column) is missing on each laser
    assert counts == [99, 99, 99, 99]
    np.testing.assert_allclose(sensor_to_world[:3, 3].numpy(), [0, 0, 2.5])


def test_ppisp_camera_profiles(service) -> None:
    _svc, renderer, _ctx = service
    view = types.SimpleNamespace(attributes={"role_name": "front_left"})
    renderer.options["ppisp_camera"] = "cam0"
    assert renderer._ppisp_camera(view) == "cam0"
    renderer.options["ppisp_camera"] = {"front*": "cam_front", "*": "cam_other"}
    assert renderer._ppisp_camera(view) == "cam_front"
    view.attributes["role_name"] = "rear"
    assert renderer._ppisp_camera(view) == "cam_other"
