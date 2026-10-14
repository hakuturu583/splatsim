"""CLI entry point for the gRPC rendering service."""

from __future__ import annotations

import argparse
import logging
import os
from concurrent import futures

import grpc

from splatsim.grpc_service._generated import rendering_service_pb2_grpc as pb2_grpc
from splatsim.grpc_service.sensor_stream import (
    ENV_VAR as SENSOR_TRANSPORT_ENV_VAR,
    SENSOR_TRANSPORTS,
    sensor_transport_from_env,
)
from splatsim.grpc_service.server import RenderingServiceServicer

logger = logging.getLogger(__name__)


def serve(
    port: int = 50051, max_workers: int = 4, sensor_transport: str = "dds"
) -> None:
    """Start the gRPC server and block until interrupted."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb2_grpc.add_RenderingServiceServicer_to_server(
        RenderingServiceServicer(sensor_transport=sensor_transport), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(
        "RenderingService listening on port %d (sensor data over %s)",
        port,
        sensor_transport,
    )
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
    parser.add_argument(
        "--sensor-transport",
        type=str.lower,
        choices=SENSOR_TRANSPORTS,
        default=None,
        help=(
            "where rendered sensor data goes: 'dds' publishes it over "
            "CycloneDDS, 'grpc' returns it through the SubscribeSensorData "
            f"stream instead (default: ${SENSOR_TRANSPORT_ENV_VAR}, else dds)"
        ),
    )
    args = parser.parse_args()
    sensor_transport = args.sensor_transport or sensor_transport_from_env()

    log_level = os.environ.get("SPLATSIM_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    serve(
        port=args.port,
        max_workers=args.max_workers,
        sensor_transport=sensor_transport,
    )


if __name__ == "__main__":
    main()
