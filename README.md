# splatsim

3D Gaussian Splatting based simulator with DDS and CARLA integration.

## Prerequisites

- Python 3.10+
- CUDA 12.8
- [uv](https://docs.astral.sh/uv/)
- (Docker builds) Docker with [BuildKit](https://docs.docker.com/build/buildkit/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Installation

### Local

```bash
# Core only (headless)
uv sync

# With GUI viewer
uv sync --extra gui

# With DDS support
uv sync --extra dds

# With CARLA support (includes DDS)
uv sync --extra carla

# All optional dependencies
uv sync --extra all
```

> CARLA extra requires system libraries for lanelet2. On Ubuntu 22.04:
>
> ```bash
> sudo apt-get install -y \
>   libboost-dev libboost-serialization-dev libboost-filesystem-dev \
>   libboost-program-options-dev libboost-python-dev libboost-system-dev \
>   libeigen3-dev libpugixml-dev libgeographic-dev librange-v3-dev python3-dev
> ```

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

```bash
# A specific version for an RTX 40xx (Ada, sm_89), CUDA 12.8.1
docker pull ghcr.io/tier4/splatsim:1.0.0-cuda12.8.1-sm89

# Rolling latest for an RTX 50xx (Blackwell, sm_120)
docker pull ghcr.io/tier4/splatsim:latest-sm120

# Run (requires NVIDIA Container Toolkit)
docker run --rm -it --gpus all ghcr.io/tier4/splatsim:latest-sm89
```

#### Build locally

The Dockerfile uses a multi-stage build with the `dds` extra to keep the image minimal.

```bash
# Build
docker buildx build \
  -f docker/Dockerfile \
  -t splatsim .

# Run (requires NVIDIA Container Toolkit)
docker run --rm -it --gpus all splatsim
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

# Render a configured LiDAR and publish sensor_msgs/PointCloud2 via DDS
splatsim-viewer scene.usdz --lidar top --dds

# Run a scenario
spawn-scenario
```

## LiDAR transport (DDS vs. HILS)

Each LiDAR sensor in a scene renders a point cloud that can be delivered two ways,
selected per sensor with the `communication` field:

- `dds` (default): publishes a `sensor_msgs/PointCloud2` over CycloneDDS on
  `pointcloud_topic`.
- `hils`: hardware-in-the-loop mode. Emits raw Hesai UDP data packets that mimic
  the physical sensor's wire format, so an unmodified LiDAR driver (e.g. Autoware
  `nebula`) can consume them. The `sensor_type` selects the packet format;
  currently supported models are **OT128** (Pandar OT128) and **XT32**
  (PandarXT-32).

```yaml
lidar_sensors:
  - name: top
    sensor_type: XT32        # OT128 | XT32
    communication: hils      # dds (default) | hils
    hils_host: 192.168.1.201 # UDP destination (unicast or broadcast)
    hils_port: 2368          # Hesai point-cloud port
    # hils_start_epoch: 1700000000  # optional: Unix time that sim-time 0
    #                               # maps to; omit to start "now".
```

Packet timestamps use the **simulation clock**: the date-time/timestamp written
into each packet is `hils_start_epoch + <simulation elapsed time>`. When
`hils_start_epoch` is omitted it defaults to the wall-clock time at which the
sensor is created (simulation starts "now").

Packets are encoded with `torch` directly from the rendered range image, so when
the renderer runs on CUDA the entire encode stays on the GPU and only the packed
byte buffer crosses to the host (once per frame) for the UDP send.

## Development

```bash
uv sync --dev
pre-commit install
```
