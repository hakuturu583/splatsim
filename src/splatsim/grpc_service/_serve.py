"""Shared gRPC serve / CLI boilerplate for RenderingService backends.

Kept torch-free (only ``grpc`` + the generated stubs) so a lightweight backend
such as :mod:`splatsim.omnidreams_service` can reuse it without pulling the
gsplat stack that :mod:`splatsim.grpc_service.server` imports. The servicer is
supplied by the caller — as an instance to :func:`run_server`, or as a factory
to :func:`run_cli` so its (possibly heavy) construction is deferred until after
argument parsing.
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent import futures
from typing import Callable

import grpc

from splatsim.grpc_service._generated import rendering_service_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


def run_server(
    servicer: pb2_grpc.RenderingServiceServicer,
    *,
    name: str,
    port: int = 50051,
    max_workers: int = 4,
) -> None:
    """Start a RenderingService gRPC server and block until interrupted."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb2_grpc.add_RenderingServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("%s listening on port %d", name, port)
    server.wait_for_termination()


def run_cli(
    servicer_factory: Callable[[], pb2_grpc.RenderingServiceServicer],
    *,
    name: str,
    description: str,
) -> None:
    """Parse ``--port`` / ``--max-workers``, configure logging, and serve."""
    parser = argparse.ArgumentParser(description=description)
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
    run_server(
        servicer_factory(),
        name=name,
        port=args.port,
        max_workers=args.max_workers,
    )
