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
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/python

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

WORKDIR /workspace/splatsim

COPY --from=builder /opt/python /opt/python
COPY --from=builder /workspace/splatsim/.venv .venv/
COPY --from=builder /workspace/splatsim/src/ src/
COPY --from=builder /workspace/splatsim/pyproject.toml .
COPY launch/ launch/
COPY package.xml .env ./

ENV VIRTUAL_ENV=/workspace/splatsim/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

CMD ["bash"]
