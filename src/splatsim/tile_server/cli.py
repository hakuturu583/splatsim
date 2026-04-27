"""CLI entry point for the 3D Tiles HTTP server."""

from __future__ import annotations

import argparse
import logging

from splatsim.tile_server.server import serve


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Serve Cesium 3D Tiles over HTTP with CORS",
    )
    parser.add_argument(
        "directory",
        help="directory containing tileset.json and tile data",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--bind",
        default="0.0.0.0",
        help="address to bind to (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    serve(directory=args.directory, port=args.port, bind=args.bind)


if __name__ == "__main__":
    main()
