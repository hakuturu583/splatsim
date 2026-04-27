ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for lanelet2 and PyQt5
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    python3-dev \
    libboost-dev \
    libboost-serialization-dev \
    libboost-filesystem-dev \
    libboost-program-options-dev \
    libboost-python-dev \
    libboost-system-dev \
    libeigen3-dev \
    libpugixml-dev \
    libgeographic-dev \
    librange-v3-dev \
    # PyQt5 / GUI dependencies
    libgl1-mesa-glx \
    libegl1 \
    libxkbcommon0 \
    libdbus-1-3 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace/splatsim
COPY . .

# Install project dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --dev

# Activate the virtualenv by default
ENV VIRTUAL_ENV=/workspace/splatsim/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

CMD ["bash"]
