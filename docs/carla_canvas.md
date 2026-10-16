# splatsim as a carla_canvas renderer

[carla_canvas](https://github.com/hakuturu583/carla_canvas) runs a
[typesafe_carla](https://github.com/hakuturu583/typesafe_carla) program
**unchanged** and has its CARLA cameras and LiDARs rendered by a renderer
service over gRPC. `splatsim.carla_canvas.SplatsimRenderer` makes splatsim that
renderer:

```
typesafe_carla program ── carla_canvas plugin ──gRPC──▶ carla_canvas service
   (unchanged)               (claims sensors)             ├─ CARLA's LiDAR model, encodings
                                                          ├─ ROS 2 topics (CARLA's names, DDS)
                                                          └─ SplatsimRenderer
                                                               ├─ cameras: splatsim.Renderer (gsplat)
                                                               ├─ LiDARs: LidarRenderer panorama
                                                               └─ scene USDZ + actor bank
```

carla_canvas owns everything CARLA-specific, so this backend does not publish
DDS itself: `sensor.enable_for_ros()` in the program gets CARLA's topics from
carla_canvas's ROS bridge.

## What it renders

| CARLA sensor | splatsim |
|---|---|
| `sensor.camera.rgb` | `splatsim.renderer.Renderer` from the camera's CARLA pose (gsplat, SH, PPISP/exposure as configured) |
| `sensor.lidar.ray_cast` | `LidarRenderer` panorama with one row per CARLA laser (its exact elevation) and `points_per_second / channels / rotation_frequency` columns; each CARLA ray reads its row and nearest column. splatsim's learned intensity and ray drop are used; carla_canvas then applies CARLA's drop-off, noise and layout |
| actors | each CARLA actor matching a blueprint pattern gets an instance of a bundle actor asset, posed every frame at its bounding box's center (or its origin) |

Not rendered (they stay CARLA's, or are rejected for this backend): depth,
segmentation cameras and the semantic LiDAR. A LiDAR does not see its own
vehicle's splats.

## Use

```sh
uv sync --extra carla_canvas                # carla-canvas, simple-lanelet2 and pyproj (alignment: geo)
carla-canvas serve --config canvas.yaml     # on the GPU machine
# on the client: CARLA_CANVAS_CONFIG=canvas.yaml typesafe-codon run main.py
```

```yaml
renderer:
  backend: splatsim.carla_canvas:SplatsimRenderer
  endpoint: "gpu-host:50051"
  assets:
    maps:
      "Town10HD*":                    # glob on the CARLA map name
        scene: scenes/town10.usdz
        alignment: geo                # the map's OpenDRIVE geo-reference (default)
        # matrix: [[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]   # or CARLA -> tile-local
    actors:                           # blueprint glob -> the bundle's actor asset id
      "vehicle.*": sedan
      "walker.*": {asset: pedestrian, anchor: origin}
  options:
    device: cuda
    background_color: [0, 0, 0]       # 0-1
    near_plane: 0.01
    far_plane: 1000.0
    exposure: 1.0
    ppisp_camera: front               # the scene's PPISP camera profile, or {role_name glob: profile}
    lidar: {drop_threshold: 0.5, alpha_threshold: 0.1, min_range: 0.3, n_columns: auto}
sensors:
  - sensor.camera.rgb
  - sensor.lidar.ray_cast
ros:
  backend: auto
```

### Docker

The splatsim image runs this backend: its command is `carla-canvas serve
--endpoint 0.0.0.0:50051` with `CARLA_CANVAS_CONFIG=/config/canvas.yaml`.

```sh
# canvas.yaml (as above) and its scene USDZs in the current directory
docker run --rm --gpus all -p 50051:50051 -v $PWD:/config ghcr.io/hakuturu583/splatsim:latest-sm89
```

`enable_for_ros()` sensors are published over CycloneDDS from the container
(the `carla_canvas` extra includes `carla-canvas[ros]`); use `--network host`
instead of `-p` so DDS discovery reaches other hosts' ROS 2 nodes.

The client uses its own copy of the configuration with `endpoint:
"<gpu-host>:50051"` (and no `command`, so it does not try to start a renderer
itself). Asset paths are relative to the configuration file. `alignment: geo`
works in the image (simple-lanelet2).

## Frames

`alignment: geo` maps CARLA positions through `splatsim.geo_transform`
(`GeoTransform`: CARLA → OpenDRIVE → MGRS → LLA →
ECEF → tile-local, re-centered), using the OpenDRIVE that carla_canvas sends
with the map (protocol 2) and the scene's ECEF anchor. The renderer declares
carla_canvas's `coordinate_frame = "right_handed"`, so poses arrive already
right-handed (CARLA with y negated, bodies x forward / y left / z up) and
LiDAR azimuths as FLU angles; they are turned into the tile with the
position map's own (orthonormalized) Jacobian at the map origin: CARLA's x is
the MGRS grid's east, which differs from true east by the grid convergence
(0.7° in Tokyo, up to ~1.7° at a zone edge), so an ENU-based rotation would
turn every pose against its position. A `matrix` maps CARLA world to the
re-centered tile-local frame; with a uniform scale, LiDAR ranges are
converted back to metres (actor assets are not scaled). `splatsim.carla_canvas.frames`
(`SceneFrame`) only maps that frame into tile-local; the gsplat RDF view matrix
comes from carla_canvas's `view_matrix_cv` (numpy only, tested in `test/test_carla_canvas_frames.py`).

## Limits

* The LiDAR renders a full panorama each measurement, although a CARLA tick
  sweeps only part of a turn (splatsim's sector rendering could cut this).
* Instances are placed by CARLA's bounding box center; asset frames are taken
  as x forward, y left, z up.
