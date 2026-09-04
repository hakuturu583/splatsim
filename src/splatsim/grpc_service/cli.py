"""CLI entry point for the gRPC rendering service."""

from __future__ import annotations

from splatsim.grpc_service._serve import run_cli, run_server
from splatsim.grpc_service.server import RenderingServiceServicer

_NAME = "RenderingService"


def serve(port: int = 50051, max_workers: int = 4) -> None:
    """Start the gRPC server and block until interrupted."""
    run_server(
        RenderingServiceServicer(), name=_NAME, port=port, max_workers=max_workers
    )


def main() -> None:
    """CLI entry point."""
    run_cli(
        RenderingServiceServicer,
        name=_NAME,
        description="splatsim gRPC rendering server",
    )


if __name__ == "__main__":
    main()
