# splatsim

3D Gaussian Splatting based simulator with DDS and CARLA integration.

## Prerequisites

- Python 3.10+
- CUDA 12.4
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

The Dockerfile uses a multi-stage build with the `dds` extra to keep the image minimal.

Private git dependencies (e.g. `3dgs-io`) require a GitHub token passed as a build secret.

```bash
# 1. Create a .gitconfig with your token
printf '[url "https://x-access-token:%s@github.com/"]\n\tinsteadOf = https://github.com/\n' \
  "$(gh auth token)" > /tmp/.gitconfig

# 2. Build
docker buildx build \
  -f docker/Dockerfile \
  --secret id=gitconfig,src=/tmp/.gitconfig \
  -t splatsim .

# 3. Run (requires NVIDIA Container Toolkit)
docker run --rm -it --gpus all splatsim
```

To customize CUDA or Ubuntu versions:

```bash
docker buildx build \
  -f docker/Dockerfile \
  --build-arg CUDA_VERSION=12.4.1 \
  --build-arg UBUNTU_VERSION=22.04 \
  --secret id=gitconfig,src=/tmp/.gitconfig \
  -t splatsim .
```

## Usage

```bash
# Launch the viewer
viewer

# Run a scenario
spawn-scenario
```

## Development

```bash
uv sync --dev
pre-commit install
```
