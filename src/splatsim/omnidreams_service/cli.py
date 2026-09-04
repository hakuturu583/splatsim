"""CLI entry point for the OmniDreams gRPC rendering backend.

Same port and service as :mod:`splatsim.grpc_service.cli` (they share
:mod:`splatsim.grpc_service._serve`), so the two backends are drop-in
interchangeable. Run with ``splatsim-omnidreams-server`` or, without installing
the package (to avoid pulling the gsplat stack), with
``python -m splatsim.omnidreams_service.cli``.
"""

from __future__ import annotations

from splatsim.grpc_service._serve import run_cli, run_server
from splatsim.omnidreams_service.server import OmniDreamsServicer

_NAME = "OmniDreams RenderingService"


def serve(port: int = 50051, max_workers: int = 4) -> None:
    """Start the OmniDreams gRPC server and block until interrupted."""
    run_server(OmniDreamsServicer(), name=_NAME, port=port, max_workers=max_workers)


def main() -> None:
    """CLI entry point."""
    run_cli(
        OmniDreamsServicer,
        name=_NAME,
        description="splatsim OmniDreams world-model gRPC rendering server",
    )


if __name__ == "__main__":
    main()
