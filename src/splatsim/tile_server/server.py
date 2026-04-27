"""HTTP static file server for Cesium 3D Tiles with CORS support."""

from __future__ import annotations

import logging
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

logger = logging.getLogger(__name__)

# Additional MIME types for 3D Tiles
_EXTRA_EXTENSIONS_MAP = {
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".b3dm": "application/octet-stream",
    ".i3dm": "application/octet-stream",
    ".pnts": "application/octet-stream",
    ".cmpt": "application/octet-stream",
}


class CORSRequestHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with CORS headers for Cesium viewers."""

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        **_EXTRA_EXTENSIONS_MAP,
    }

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Handle CORS preflight requests."""
        self.send_response(204)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        logger.info(format, *args)


def serve(
    directory: str,
    port: int = 8080,
    bind: str = "0.0.0.0",  # noqa: S104
) -> None:
    """Start serving *directory* over HTTP and block until interrupted."""
    handler = partial(CORSRequestHandler, directory=directory)
    server = HTTPServer((bind, port), handler)
    logger.info("Serving %s on http://%s:%d", directory, bind, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down tile server")
    finally:
        server.server_close()
