"""CLI entry point for the gRPC rendering service."""

from __future__ import annotations

import argparse
import logging
import os
from concurrent import futures

import grpc

from splatsim.grpc_service._generated import rendering_service_pb2_grpc as pb2_grpc
from splatsim.grpc_service.server import RenderingServiceServicer

logger = logging.getLogger(__name__)


def serve(port: int = 50051, max_workers: int = 4) -> None:
    """Start the gRPC server and block until interrupted."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb2_grpc.add_RenderingServiceServicer_to_server(RenderingServiceServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("RenderingService listening on port %d", port)
    server.wait_for_termination()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="splatsim gRPC rendering server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="port to listen on (default: 50051)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="max gRPC thread pool workers (default: 4)",
    )
    args = parser.parse_args()

    log_level = os.environ.get("SPLATSIM_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    serve(port=args.port, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
