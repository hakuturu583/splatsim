"""Geographic coordinate transform between CARLA world and 3DGS tile-local frame.

Conversion pipeline (all in float64 to avoid precision loss)::

    CARLA (x=East, y=South, z=Up)
      → xodr  (flip y: y_xodr = −y_carla)
      → MGRS  (add MGRS offset from lanelet2 projector)
      → LLA   (MGRSProjector.reverse — matches the actual map projection)
      → ECEF  (WGS84 via pyproj)
      → tile-local  (inverse of tileset.json transform)
      → re-centered (subtract Background.tile_local_centroid)

The rotation is converted via:
    R_carla → S @ R_carla (flip y axis: CARLA South → ENU North)
            → R_enu_to_ecef @ ... → R_ecef_to_tile @ ...
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET

import lanelet2.core  # ty: ignore[unresolved-import]
import lanelet2.io  # ty: ignore[unresolved-import]
import numpy as np
from autoware_lanelet2_extension_python.projection import (
    MGRSProjector,
)
from numpy.typing import NDArray
from pyproj import Transformer

logger = logging.getLogger(__name__)


class GeoTransform:
    """Convert CARLA world coordinates to re-centered tile-local coordinates.

    Uses the lanelet2 :class:`MGRSProjector` for projected → geographic
    conversion so that the result is consistent with the original map
    data.  (pyproj's Transverse Mercator does **not** match the
    MGRSProjector, causing ~10-50 m offsets.)

    Parameters
    ----------
    proj_origin:
        ``(latitude, longitude)`` of the xodr GeoReference origin in
        decimal degrees (parsed from ``+lat_0`` / ``+lon_0``).
    ecef_rotation:
        3×3 rotation from tile-local to ECEF (from tileset.json).
    ecef_translation:
        3-vector ECEF translation (from tileset.json).
    tile_origin:
        3-vector subtracted from tile-local positions for re-centering
        (``Background.tile_local_centroid``).
    """

    def __init__(
        self,
        proj_origin: tuple[float, float],
        ecef_rotation: NDArray[np.float64],
        ecef_translation: NDArray[np.float64],
        tile_origin: NDArray[np.float64],
    ) -> None:
        lat_0, lon_0 = proj_origin

        # Lanelet2 MGRSProjector — the authoritative map projection.
        self._projector = MGRSProjector(lanelet2.io.Origin(lat_0, lon_0))

        # MGRS offset: projected coords of the geographic origin.
        origin_gps = lanelet2.core.GPSPoint(lat_0, lon_0, 0.0)
        origin_local = self._projector.forward(origin_gps)
        self._mgrs_offset_x = origin_local.x
        self._mgrs_offset_y = origin_local.y

        # pyproj: geographic → ECEF (WGS84)
        self._lla_to_ecef = Transformer.from_crs(
            "EPSG:4326", "EPSG:4978", always_xy=True
        )

        # Tileset: tile_local → ECEF: p_ecef = R @ p_tile + t
        self._tile_R = np.asarray(ecef_rotation, dtype=np.float64)
        self._tile_t = np.asarray(ecef_translation, dtype=np.float64)
        self._tile_R_inv = self._tile_R.T  # R^T = R^{-1} (orthogonal)

        self._tile_origin = np.asarray(tile_origin, dtype=np.float64)

        # Precompute ENU → tile-local rotation at the map origin.
        R_enu_to_ecef = _enu_to_ecef_rotation(lat_0, lon_0)
        self._R_enu_to_tile = self._tile_R_inv @ R_enu_to_ecef

        logger.info(
            "GeoTransform: origin=(lat=%.6f, lon=%.6f), "
            "mgrs_offset=(%.1f, %.1f), tile_origin=(%.1f, %.1f, %.1f)",
            lat_0,
            lon_0,
            self._mgrs_offset_x,
            self._mgrs_offset_y,
            *self._tile_origin,
        )

    # ------------------------------------------------------------------
    # Position conversion
    # ------------------------------------------------------------------

    def carla_position_to_tile_local(
        self,
        carla_x: float,
        carla_y: float,
        carla_z: float,
    ) -> NDArray[np.float64]:
        """Convert a CARLA world position to re-centered tile-local (float64)."""
        # 1. CARLA → xodr (flip y)
        xodr_x = carla_x
        xodr_y = -carla_y

        # 2. xodr → MGRS absolute (add offset)
        mgrs_x = xodr_x + self._mgrs_offset_x
        mgrs_y = xodr_y + self._mgrs_offset_y

        # 3. MGRS → LLA via lanelet2 MGRSProjector
        mgrs_pt = lanelet2.core.BasicPoint3d(mgrs_x, mgrs_y, carla_z)
        gps = self._projector.reverse(mgrs_pt)

        # 4. LLA → ECEF
        ex, ey, ez = self._lla_to_ecef.transform(gps.lon, gps.lat, carla_z)
        ecef = np.array([ex, ey, ez], dtype=np.float64)

        # 5. ECEF → tile-local
        tile_local = self._tile_R_inv @ (ecef - self._tile_t)

        # 6. Re-center
        return tile_local - self._tile_origin

    # ------------------------------------------------------------------
    # Rotation conversion
    # ------------------------------------------------------------------

    def carla_rotation_to_tile_local(
        self,
        R_carla: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert a CARLA world rotation matrix to tile-local frame.

        Parameters
        ----------
        R_carla:
            3×3 rotation (actor-local → CARLA-world).  Columns are the
            actor's local axes in CARLA world (x=East, y=South, z=Up).
        """
        # CARLA → ENU: flip y (South → North)
        _S = np.diag([1.0, -1.0, 1.0])
        R_enu = _S @ R_carla

        # ENU → tile-local (precomputed at map origin)
        return self._R_enu_to_tile @ R_enu


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _enu_to_ecef_rotation(lat_deg: float, lon_deg: float) -> NDArray[np.float64]:
    """Rotation matrix from ENU to ECEF at the given lat/lon (degrees)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    return np.array(
        [
            [-slon, -slat * clon, clat * clon],
            [clon, -slat * slon, clat * slon],
            [0.0, clat, slat],
        ],
        dtype=np.float64,
    )


def parse_geo_reference(xodr_xml: str) -> tuple[float, float]:
    """Extract ``(lat_0, lon_0)`` from the xodr ``<geoReference>`` PROJ string.

    Returns
    -------
    tuple[float, float]
        ``(latitude, longitude)`` in decimal degrees.

    Raises
    ------
    ValueError
        If the element or required parameters are missing.
    """
    # Extract raw PROJ string
    match = re.search(
        r"<geoReference>\s*<!\[CDATA\[(.*?)\]\]>\s*</geoReference>",
        xodr_xml,
        re.DOTALL,
    )
    if match:
        proj_string = match.group(1).strip()
    else:
        root = ET.fromstring(xodr_xml)  # noqa: S314
        geo_ref = root.find(".//geoReference")
        if geo_ref is not None and geo_ref.text:
            proj_string = geo_ref.text.strip()
        else:
            raise ValueError("No <geoReference> found in OpenDRIVE XML")

    # Extract lat_0 / lon_0
    lat_match = re.search(r"\+lat_0=([0-9eE.+-]+)", proj_string)
    lon_match = re.search(r"\+lon_0=([0-9eE.+-]+)", proj_string)
    if lat_match is None or lon_match is None:
        raise ValueError(
            f"Cannot extract +lat_0/+lon_0 from GeoReference: {proj_string}"
        )

    lat_0 = float(lat_match.group(1))
    lon_0 = float(lon_match.group(1))
    logger.info("GeoReference: lat_0=%.8f, lon_0=%.8f", lat_0, lon_0)
    return lat_0, lon_0
