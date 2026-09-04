"""gRPC-based rendering service for SplatSim.

``RenderingServiceServicer`` is exported lazily so that importing sibling
subpackages (e.g. ``splatsim.grpc_service.pose_buffer`` or the OmniDreams
backend, which reuse the pure-Python gRPC glue) does not force the gsplat
rasteriser stack that ``server`` depends on to load.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from splatsim.grpc_service.server import RenderingServiceServicer

__all__ = ["RenderingServiceServicer"]


def __getattr__(name: str):
    if name == "RenderingServiceServicer":
        from splatsim.grpc_service.server import RenderingServiceServicer

        return RenderingServiceServicer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
