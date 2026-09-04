"""OmniDreams world-model backend for the shared SplatSim RenderingService.

This package is a *separate* rendering backend from the 3D Gaussian Splatting
one (:mod:`splatsim.grpc_service`). It speaks the exact same gRPC contract
(``proto/splatsim/v1/rendering_service.proto``) so a client cannot tell the two
apart, but instead of rasterising Gaussians it drives NVIDIA's OmniDreams /
Cosmos-Dreams autoregressive video world model (via the FlashDreams inference
runtime) to synthesise each camera frame.

The only protocol extension is the optional ``initial_image`` field on
``InitializeRequest`` (proto field 16): OmniDreams needs one real RGB frame to
anchor scene appearance before it can generate. The 3DGS backend ignores that
field, which is why it is optional.

Deliberately, this package imports NONE of the gsplat / USDZ / SplatAD stack —
only the pure-Python gRPC glue (``pose_buffer``, ``_serve``), the DDS
publishers, and (lazily, at Initialize time) the FlashDreams pipeline. That
keeps the OmniDreams Docker image free of the CUDA rasteriser kernels it does
not need.
"""

from __future__ import annotations

from splatsim.omnidreams_service.renderer import OmniDreamsRenderer
from splatsim.omnidreams_service.server import OmniDreamsServicer

__all__ = ["OmniDreamsRenderer", "OmniDreamsServicer"]
