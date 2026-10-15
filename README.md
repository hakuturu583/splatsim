# splatsim

3D Gaussian Splatting based simulator. It renders CARLA cameras and LiDARs as a
[carla_canvas](https://github.com/hakuturu583/carla_canvas) renderer backend.

## Prerequisites

- Python 3.10+
- CUDA 12.8
- [uv](https://docs.astral.sh/uv/)
- (Docker builds) Docker with [BuildKit](https://docs.docker.com/build/buildkit/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## carla_canvas backend

splatsim can render the cameras and LiDARs of an unchanged
[typesafe_carla](https://github.com/hakuturu583/typesafe_carla) program through
[carla_canvas](https://github.com/hakuturu583/carla_canvas)
(`renderer.backend: splatsim.carla_canvas:SplatsimRenderer`); carla_canvas
provides the CARLA sensor models and the ROS 2 output. The Docker image runs
it as a carla_canvas renderer service:

```bash
# canvas.yaml (renderer.backend: splatsim.carla_canvas:SplatsimRenderer) and
# its scene USDZs in the current directory
docker run --rm --gpus all -p 50051:50051 -v $PWD:/config ghcr.io/tier4/splatsim:latest-sm89
# on the CARLA client: renderer.endpoint: "<gpu-host>:50051" in its canvas.yaml
CARLA_CANVAS_CONFIG=canvas.yaml typesafe-codon run main.py
```

See [docs/carla_canvas.md](docs/carla_canvas.md).

## Installation

### Local

```bash
# Core only (headless)
uv sync

# With GUI viewer
uv sync --extra gui

# As a carla_canvas renderer (`carla-canvas serve`)
uv sync --extra carla_canvas

# `splatsim-viewer --mp4` (the viewer module needs the GUI extra too)
uv sync --extra gui --extra mp4

# All optional dependencies
uv sync --extra all
```

### Docker

#### Pull a published image

Released images are published to the GitHub Container Registry. A **separate
image is built per GPU architecture** (one CUDA compute capability each), so
pick the tag that matches your GPU. Each tag is
`<version>-cuda<cuda>-sm<XX>`, plus a rolling `latest-sm<XX>`.

| GPU | Compute capability | Tag suffix |
| --- | --- | --- |
| RTX 20xx (Turing)     | 7.5  | `sm75`  |
| A100 (Ampere)         | 8.0  | `sm80`  |
| RTX 30xx (Ampere)     | 8.6  | `sm86`  |
| RTX 40xx (Ada)        | 8.9  | `sm89`  |
| H100 (Hopper)         | 9.0  | `sm90`  |
| RTX 50xx (Blackwell)  | 12.0 | `sm120` |

The tag suffix names the native SASS target (`sm_XX`). Every image also embeds
`compute_XX` PTX, so the driver can JIT-run it on GPU generations newer than
the one it was built for — PTX is uniform across images and is not a separate
tag dimension.

```bash
# A specific version for an RTX 40xx (Ada, sm_89), CUDA 12.8.1
docker pull ghcr.io/tier4/splatsim:1.0.0-cuda12.8.1-sm89

# Rolling latest for an RTX 50xx (Blackwell, sm_120)
docker pull ghcr.io/tier4/splatsim:latest-sm120

# Run the carla_canvas renderer service (requires NVIDIA Container Toolkit):
# mount the directory holding canvas.yaml and its scenes at /config
docker run --rm --gpus all -p 50051:50051 -v $PWD:/config ghcr.io/tier4/splatsim:latest-sm89
```

The image's command is `carla-canvas serve --endpoint 0.0.0.0:50051` with
`CARLA_CANVAS_CONFIG=/config/canvas.yaml`; set `-e CARLA_CANVAS_CONFIG=...` for
another file. Maps align through their OpenDRIVE geo-reference
(`alignment: geo`, with [simple_lanelet2](https://github.com/hakuturu583/simple_lanelet2)'s
MGRS projector) or an explicit `matrix` (docs/carla_canvas.md).

#### Build locally

The Dockerfile uses a multi-stage build with the `carla_canvas` extra to keep the image minimal.

```bash
# Build
docker buildx build \
  -f docker/Dockerfile \
  -t splatsim .

# Run (requires NVIDIA Container Toolkit)
docker run --rm --gpus all -p 50051:50051 -v $PWD:/config splatsim
```

To customize CUDA or Ubuntu versions:

```bash
docker buildx build \
  -f docker/Dockerfile \
  --build-arg CUDA_VERSION=12.8.1 \
  --build-arg UBUNTU_VERSION=22.04 \
  -t splatsim .
```

## Usage

```bash
# Launch the viewer
splatsim-viewer

# Show a configured LiDAR's panorama
splatsim-viewer scene.usdz --lidar top

# Render the rig camera's trajectory to an MP4 (the `gui` and `mp4` extras)
splatsim-viewer scene.usdz --mp4 out.mp4
```

## Dynamic objects (actor assets)

A scene USDZ built with 3dgs_io >= v2.1.0 can carry a bank of **rigid
dynamic-object assets** (`splatsim.actor_assets/v1`): cars, trucks, trailers,
cones — anything whose shape does not change over time. Each is a Gaussian
cloud authored in a canonical object-local frame (`+x` forward, `+y` left,
`+z` up, origin at the box centre, metric scale), in the same SPZ container the
background chunks use, so an actor's Gaussians and its per-Gaussian LiDAR
attributes go through the readers splatsim already has.

**Poses come from outside.** splatsim hands you a `RigidBody` per instance and
the scenario drives it — e.g. the carla_canvas backend, which poses CARLA's actors.
The bundle's own `sequence_tracks.json` is metadata here, exposed for callers
that want to replay it, not a playback engine.

```python
from splatsim import Scene
from splatsim.actor_assets import pose_from_track_frame

scene = Scene.from_config("scene.usdz")

# What the bundle ships
library = scene.actor_library
print(library.asset_ids, library.info("sedan_0007").size)

# Spawn as many instances as the scenario needs — they share one upload
scene.spawn_actor("sedan_0007", name="car_01", position=(113.6, -58.5, 1.9))
scene.spawn_actor("sedan_0007", name="car_02", position=(120.0, -58.5, 1.9))

# ...then drive them
scene.set_pose("car_01", (114.1, -58.5, 1.9), (0.966, 0.0, 0.0, 0.259))
```

Two conventions worth getting right, both handled for you if you use the
helpers:

- **Frame.** `Background` re-centres the scene on its own Gaussian centroid for
  numerical stability, so an ENU world-frame pose (a track pose, a map
  waypoint) must have that centroid subtracted. `spawn_actor` /
  `ActorAssetLibrary.spawn` do this by default; pass `world_position=False`
  when your pose is already tile-local.
- **Quaternion order.** Everything inside a scene bundle is `xyzw`; every
  splatsim pose is `wxyz`. Route bundle poses through
  `splatsim.actor_assets.pose_from_track_frame`.

### View-dependent bands, camera and LiDAR

Actors re-express **both** their view-dependent band sets when posed, unlike the
pre-existing `RigidBodyConfig` bodies:

- **colour SH** — leaving the specular bands in the object frame makes
  highlights spin with the car;
- **`raydrop_sh`** — the LiDAR renderer evaluates drop probability along the
  world-space sensor→Gaussian ray, so unrotated bands sample a car facing east
  with the pattern it had facing north. The scalar `raydrop_logit` is the
  band-0 term and is rotation-invariant, so it needs nothing.

Both go through the same closed-form `+Z` rotation off a single yaw, which is
exact for the yaw-only poses road vehicles have.
`RigidBody.sh_rotation_tilt` / `.sh_rotation_is_exact` report when a pose is
tilted far enough out of the ground plane for it to be an approximation.

Everything else on the LiDAR path already treats an actor like any other
source: `lidar_mask` is applied per-source before the scene concat (so an asset
without one participates fully), assets that carry no raydrop bands are
zero-padded to the scene's SH width and contribute only their scalar logit, and
the LOD `lidar_view` gather thins the actor's object-frame Gaussians before the
pose is applied.

Actors can also be declared in a scene config, for static props and spawn
points:

```yaml
actors:
  - asset_id: sedan_0007
    name: parked_car
    position: [113.62, -58.55, 1.92]   # ENU world frame
    rotation: [0.966, 0.0, 0.0, 0.259] # wxyz
```

## Hesai HILS packets

`splatsim.hils` encodes a rendered LiDAR range image as raw Hesai UDP data
packets (**OT128**, **XT32**) that an unmodified LiDAR driver (e.g. Autoware
`nebula`) can consume, on the GPU when the renderer runs on CUDA
(`HesaiHilsPublisher`, `build_packets`). Packet timestamps are
`start_epoch_s + <simulation elapsed time>`.

## LiDAR evaluation

`src/eval` scores the LiDAR the simulator renders from a reconstructed `.usdz`
scene against the ground-truth LiDAR of the matching WebAuto / T4 dataset, at the
same ego pose, and logs everything to a single [Rerun](https://rerun.io) `.rrd`
(both point clouds + every metric's time series on a shared timeline). It is a
small metric-plugin framework — each evaluation item is its own class under
`src/eval/metrics/` implementing `LidarEvalMetric`, so the per-frame render/mask
happens once and is shared across metrics:

- **`chamfer`** — symmetric Chamfer distance (raw + range-aware), in metres.
- **`bev`** — OnePlanner **BEV-encoder** feature similarity: both clouds are
  pushed through the BEV encoder and the resulting `[512, 180, 180]`
  bird's-eye-view feature maps are compared (per-cell / global cosine, relative
  L2). A shared-basis PCA(512→3) RGB view of each map and a cosine heatmap are
  logged as images so the learned representations can be compared by eye. This
  answers *"would a downstream planner perceive the reconstructed scene the same
  way it perceives the real one?"* rather than raw geometric distance.

```bash
uv sync --extra eval                       # chamfer metric (t4-devkit + rerun)
uv run python -m eval.eval_lidar \
    --scene scene.usdz --data-root ~/.webauto/datasets --dataset-id <id> \
    --metrics chamfer --output outputs/eval_lidar.rrd
```

### BEV-encoder metric

The BEV encoder is the LiDAR branch of OnePlanner's BEVFusion, shipped as
`oneplanner_bev_encoder.onnx`. Its ONNX embeds `autoware` sparse-convolution
custom ops (`GetIndicePairsImplicitGemm` / `ImplicitGemm`) that **plain
`onnxruntime` cannot execute** — their implementation lives only in the
`autoware_tensorrt_plugins` (which wrap [spconv](https://github.com/traveller59/spconv)).
Two backends run the ONNX:

- **`spconv` (default).** `onnx2torch` converts the whole graph to a
  `torch.nn.Module` and loads all weights from the ONNX; the two custom ops are
  supplied by `spconv` (the same library the plugins wrap) via lightweight
  converters. Pure-pip and CUDA-aligned with the project's torch — no TensorRT,
  no plugin `.so`, no version matching. This is the recommended path.
- **`tensorrt` (`--bev-backend tensorrt`).** Loads the `autoware_tensorrt_plugins`
  `.so`, builds (and disk-caches) a TensorRT engine, and runs inference using
  torch tensors as the CUDA buffers. The production path, but its `tensorrt`
  wheel + CUDA must match the prebuilt plugin's `libnvinfer` (here the plugin
  links `libnvinfer.so.10`, TensorRT 10.16.1), so install a matching `tensorrt`
  yourself and pass `--bev-plugins`.

```bash
uv sync --extra eval --extra bev           # onnx2torch + spconv (default backend)
export ONEPLANNER_BEV_ONNX=/path/to/oneplanner_bev_encoder.onnx
uv run python -m eval.eval_lidar \
    --scene scene.usdz --data-root ~/.webauto/datasets --dataset-id <id> \
    --metrics chamfer,bev --output outputs/eval_lidar.rrd
```

The encoder ONNX is an environment-specific artifact supplied via `--bev-onnx`
(or `$ONEPLANNER_BEV_ONNX`) rather than vendored. `spconv-cu126` pairs
`cumm-cu126` (CUDA 12.6 wheels, compatible with the CUDA 12.8 torch here).

## Development

```bash
uv sync --dev
pre-commit install
```
