"""Geographic coordinate transform between CARLA world and 3DGS tile-local frame.

Conversion pipeline (all in float64 to avoid precision loss)::

    CARLA (x=East, y=South, z=Up)
      → xodr  (flip y: y_xodr = −y_carla)
      → LLA   (pyproj inverse of GeoReference proj string)
      → ECEF  (WGS84)
      → tile-local  (inverse of tileset.json transform)
      → re-centered (subtract Background._origin)

The rotation is converted similarly:
    R_carla → S @ R_carla (flip y axis for ENU)
            → R_enu_to_ecef @ ... → R_ecef_to_tile @ ...
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray
from pyproj import Transformer

logger = logging.getLogger(__name__)


class GeoTransform:
    """Convert CARLA world coordinates to re-centered tile-local coordinates.

    Parameters
    ----------
    proj_string:
        PROJ string from xodr ``<geoReference>``
        (e.g. ``"+proj=utm +zone=54 +lat_0=... +lon_0=... +datum=WGS84 ..."``)
    ecef_rotation:
        3x3 rotation from tile-local to ECEF (from tileset.json transform).
    ecef_translation:
        3-vector ECEF translation (from tileset.json transform).
    tile_origin:
        3-vector subtracted from tile-local positions for re-centering
        (i.e. ``Background._origin``).
    """

    def __init__(
        self,
        proj_string: str,
        ecef_rotation: NDArray[np.float64],
        ecef_translation: NDArray[np.float64],
        tile_origin: NDArray[np.float64],
    ) -> None:
        # pyproj transformers (thread-safe, reusable)
        self._proj_to_lla = Transformer.from_proj(
            proj_string, "EPSG:4326", always_xy=True
        )
        self._lla_to_ecef = Transformer.from_crs(
            "EPSG:4326", "EPSG:4978", always_xy=True
        )

        # Tileset: tile_local → ECEF: p_ecef = R @ p_tile + t
        self._tile_R = np.asarray(ecef_rotation, dtype=np.float64)
        self._tile_t = np.asarray(ecef_translation, dtype=np.float64)
        # Inverse: p_tile = R^T @ (p_ecef − t)
        self._tile_R_inv = self._tile_R.T

        self._tile_origin = np.asarray(tile_origin, dtype=np.float64)

        # Precompute rotation: ENU (at xodr origin) → tile-local
        # xodr origin (0, 0) in projected coords → geographic
        lon_0, lat_0 = self._proj_to_lla.transform(0.0, 0.0)
        R_enu_to_ecef = _enu_to_ecef_rotation(lat_0, lon_0)
        self._R_enu_to_tile = self._tile_R_inv @ R_enu_to_ecef

        logger.info(
            "GeoTransform: xodr origin → (lat=%.6f, lon=%.6f), "
            "tile_origin=(%.1f, %.1f, %.1f)",
            lat_0,
            lon_0,
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
        xodr_x, xodr_y = carla_x, -carla_y

        # 2. xodr → LLA (longitude, latitude)
        lon, lat = self._proj_to_lla.transform(xodr_x, xodr_y)

        # 3. LLA → ECEF
        ex, ey, ez = self._lla_to_ecef.transform(lon, lat, carla_z)
        ecef = np.array([ex, ey, ez], dtype=np.float64)

        # 4. ECEF → tile-local
        tile_local = self._tile_R_inv @ (ecef - self._tile_t)

        # 5. Re-center
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
            3x3 rotation (actor-local → CARLA-world).  Columns are the
            actor's local axes expressed in CARLA world coordinates
            (x=East, y=South, z=Up).
        """
        # CARLA world → ENU (xodr): flip y (South→North)
        # v_enu = diag(1, -1, 1) @ v_carla  →  R_enu = S @ R_carla
        _S = np.diag([1.0, -1.0, 1.0])
        R_enu = _S @ R_carla

        # ENU → tile-local (precomputed)
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


def parse_geo_reference(xodr_xml: str) -> str:
    """Extract the PROJ string from an OpenDRIVE XML string.

    Parameters
    ----------
    xodr_xml:
        Full OpenDRIVE XML content (e.g. from ``carla_map.to_opendrive()``).

    Returns
    -------
    str
        The raw PROJ string found inside ``<geoReference>``.

    Raises
    ------
    ValueError
        If the ``<geoReference>`` element is missing or empty.
    """
    match = re.search(
        r"<geoReference>\s*<!\[CDATA\[(.*?)\]\]>\s*</geoReference>",
        xodr_xml,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    # Fallback: try plain element text
    root = ET.fromstring(xodr_xml)  # noqa: S314
    geo_ref = root.find(".//geoReference")
    if geo_ref is not None and geo_ref.text:
        return geo_ref.text.strip()

    raise ValueError("No <geoReference> found in OpenDRIVE XML")
