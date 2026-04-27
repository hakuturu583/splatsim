ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

# =============================================================================
# Build stage
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace/splatsim

# Install dependencies first (layer cache optimization)
COPY pyproject.toml uv.lock .python-version ./
COPY resource/ resource/
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=gitconfig,target=/root/.gitconfig \
    uv sync --extra dds --no-dev --no-install-project

# Copy source and install project
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=gitconfig,target=/root/.gitconfig \
    uv sync --extra dds --no-dev

# =============================================================================
# Runtime stage
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libgl1-mesa-glx \
    libegl1 \
    libxkbcommon0 \
    libdbus-1-3 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/splatsim

COPY --from=builder /workspace/splatsim/.venv .venv/
COPY --from=builder /workspace/splatsim/src/ src/
COPY --from=builder /workspace/splatsim/pyproject.toml .
COPY launch/ launch/
COPY package.xml .env ./

ENV VIRTUAL_ENV=/workspace/splatsim/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

CMD ["bash"]
