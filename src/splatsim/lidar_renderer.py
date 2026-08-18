"""gsplat-native LiDAR simulation for splatsim.

Ported from tier4/gaussian_factory (``lidar_sim.py``). Renders a per-frame
LiDAR panorama from a 3DGS scene using ``gsplat.rasterization`` in
``camera_model='lidar'`` mode. Training-time helpers (GT projection, loss
functions) are intentionally omitted — splatsim only consumes the scene.

Public surface:

* :class:`LidarSensorSpec` — sensor mounting + optics (OT128 / XT32 or custom).
* :func:`render_lidar_panorama` — low-level renderer taking raw tensors.
* :class:`LidarRenderer` — high-level wrapper that pulls Gaussians from a
  :class:`splatsim.scene.Scene` and applies SH-derived intensity / fixed
  raydrop fallback when per-Gaussian LiDAR attributes are absent.

gsplat's spinning-lidar model wants ``RowOffsetStructuredSpinningLidar
ModelParameters`` (elevations table + per-row azimuth offsets + spinning
frequency / direction) wrapped in an ``Ext`` cache that carries the
angles→columns map + tile assignment. We build that once per sensor at
init time via :func:`LidarSensorSpec.coeffs` and reuse across every
render call.

Sensor-frame convention matches gsplat:

* ``+x`` forward, ``+y`` left, ``+z`` up.
* Azimuth ``= atan2(y, x)``, range ``[-π, +π)``.
* Elevation ``= atan2(z, sqrt(x²+y²))``, range ``(-π/2, +π/2)``.
* The panorama is ``(n_rows, n_columns)`` with row 0 = top elevation
  (descending elevations table) and col 0 = first azimuth of the spin
  (depends on ``spinning_direction``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple, Sequence

import numpy as np
import torch

from splatsim import _lidar_cull_ext as _cuda_cull_ext
from splatsim._conversions import SH_C0
from splatsim._geometry import mat4, quat_to_matrix, rpy_deg_to_matrix

if TYPE_CHECKING:
    from splatsim.scene import Scene

# Re-exported angle tables. Values taken verbatim from
# tier4/gaussian_factory (``lidar_raster.scan_patterns``).

_OT128_DEG: tuple[float, ...] = (
    14.9850,
    13.2830,
    11.7580,
    10.4830,
    9.8360,
    9.1710,
    8.4960,
    7.8120,
    7.4620,
    7.1150,
    6.7670,
    6.4160,
    6.0640,
    5.7100,
    5.3550,
    4.9980,
    4.6430,
    4.2820,
    3.9210,
    3.5580,
    3.1940,
    2.8290,
    2.4630,
    2.0950,
    1.9740,
    1.8540,
    1.7290,
    1.6090,
    1.4870,
    1.3620,
    1.2420,
    1.1200,
    0.9950,
    0.8750,
    0.7500,
    0.6250,
    0.5000,
    0.3750,
    0.2500,
    0.1250,
    0.0000,
    -0.1250,
    -0.2500,
    -0.3750,
    -0.5000,
    -0.6260,
    -0.7510,
    -0.8760,
    -1.0010,
    -1.1260,
    -1.2510,
    -1.3770,
    -1.5020,
    -1.6270,
    -1.7510,
    -1.8760,
    -2.0010,
    -2.1260,
    -2.2510,
    -2.3760,
    -2.5010,
    -2.6260,
    -2.7510,
    -2.8760,
    -3.0010,
    -3.1260,
    -3.2510,
    -3.3760,
    -3.5010,
    -3.6260,
    -3.7510,
    -3.8760,
    -4.0010,
    -4.1260,
    -4.2500,
    -4.3750,
    -4.5010,
    -4.6260,
    -4.7510,
    -4.8760,
    -5.0010,
    -5.1260,
    -5.2520,
    -5.3770,
    -5.5020,
    -5.6260,
    -5.7520,
    -5.8770,
    -6.0020,
    -6.3780,
    -6.7540,
    -7.1300,
    -7.5070,
    -7.8820,
    -8.2570,
    -8.6320,
    -9.0030,
    -9.3760,
    -9.7490,
    -10.1210,
    -10.4930,
    -10.8640,
    -11.2340,
    -11.6030,
    -11.9750,
    -12.3430,
    -12.7090,
    -13.0750,
    -13.4390,
    -13.8030,
    -14.1640,
    -14.5250,
    -14.8790,
    -15.2370,
    -15.5930,
    -15.9480,
    -16.2990,
    -16.6510,
    -17.0000,
    -17.3470,
    -17.7010,
    -18.3860,
    -19.0630,
    -19.7300,
    -20.3760,
    -21.6530,
    -23.0440,
    -24.7650,
)

_XT32_DEG: tuple[float, ...] = (
    15.0,
    14.0,
    13.0,
    12.0,
    11.0,
    10.0,
    9.0,
    8.0,
    7.0,
    6.0,
    5.0,
    4.0,
    3.0,
    2.0,
    1.0,
    0.0,
    -1.0,
    -2.0,
    -3.0,
    -4.0,
    -5.0,
    -6.0,
    -7.0,
    -8.0,
    -9.0,
    -10.0,
    -11.0,
    -12.0,
    -13.0,
    -14.0,
    -15.0,
    -16.0,
)


def _linspace_deg(hi_deg: float, lo_deg: float, n: int) -> tuple[float, ...]:
    """Inclusive top→bottom elevation ramp of ``n`` beams in degrees."""
    step = (hi_deg - lo_deg) / (n - 1)
    return tuple(hi_deg - step * i for i in range(n))


# Velodyne HDL-64E S3 — 64 beams across two stacked 32-laser blocks,
# ordered top→bottom. Physical units ship with a per-laser factory
# calibration (db.xml) that varies unit-to-unit; absent that we use the
# datasheet nominal two-block design: an upper block from +2.0° to -8.33°
# (~1/3° spacing) and a lower block from -8.83° to -24.33° (~1/2° spacing),
# i.e. a +2.0°/-24.33° vertical field of view (~26.9° per the datasheet).
_HDL64E_DEG: tuple[float, ...] = (
    *_linspace_deg(2.0, -8.33, 32),
    *_linspace_deg(-8.83, -24.33, 32),
)


_TABLES_RAD: dict[str, tuple[float, ...]] = {
    "OT128": tuple(math.radians(d) for d in _OT128_DEG),
    "XT32": tuple(math.radians(d) for d in _XT32_DEG),
    "HDL64E": tuple(math.radians(d) for d in _HDL64E_DEG),
}


def elevations_rad(sensor_type: str) -> tuple[float, ...]:
    """Return the descending beam-elevation table for ``sensor_type``."""
    return _TABLES_RAD[sensor_type]


def is_known_sensor(sensor_type: str) -> bool:
    return sensor_type in _TABLES_RAD


# ── Sensor pose / spec ──────────────────────────────────────────────


def sensor_to_base_4x4(
    translation: Sequence[float],
    rotation_wxyz: Sequence[float],
) -> np.ndarray:
    """4×4 sensor→base_link rigid transform from a YAML pose entry."""
    return mat4(quat_to_matrix(rotation_wxyz, order="wxyz"), translation)


@dataclass
class LidarSensorSpec:
    """Per-physical-LiDAR sensor spec wrapping a gsplat lidar params object.

    The dataclass is intentionally cheap to construct (just 4×4 + a few
    floats + the sensor_type string). Heavy gsplat pre-processing
    (``compute_angles_to_columns_map`` + ``compute_tiling`` + the
    ``RowOffsetStructuredSpinningLidarModelParametersExt`` cache) is
    lazily built on first ``coeffs(...)`` access and reused.
    """

    name: str
    sensor_type: str  # "OT128" | "XT32" (or empty for "uniform")
    s2b: np.ndarray  # (4, 4) float64 sensor → base_link
    n_columns: int = 2048  # panorama azimuth bins
    spinning_frequency_hz: float = 10.0
    # Optional uniform-elevation fallback (used when sensor_type is
    # unknown). Same defaults as the legacy config.
    el_lo_rad: float = math.radians(-25.0)
    el_hi_rad: float = math.radians(15.0)
    n_rows_uniform: int = 128
    # Explicit per-beam elevation table (radians, strictly descending /
    # top→bottom). When non-empty it overrides both ``sensor_type`` and the
    # uniform fallback. Kept as a tuple so it stays hashable for the
    # ``_build_lidar_coeffs`` LRU cache key.
    row_elevations_rad: tuple[float, ...] = ()

    def coeffs(self, device: torch.device):
        """Build the gsplat ``...ParametersExt`` once per device."""
        return _build_lidar_coeffs(
            self.sensor_type,
            self.n_columns,
            self.spinning_frequency_hz,
            self.el_lo_rad,
            self.el_hi_rad,
            self.n_rows_uniform,
            self.row_elevations_rad,
            str(device),
        )


@lru_cache(maxsize=32)
def _build_lidar_coeffs(
    sensor_type: str,
    n_columns: int,
    spinning_frequency_hz: float,
    el_lo_rad: float,
    el_hi_rad: float,
    n_rows_uniform: int,
    row_elevations_rad: tuple[float, ...],
    device_str: str,
):
    """gsplat lidar params + cached preprocessing. Memoised per device."""
    from gsplat.cuda._lidar import (
        SpinningDirection,
        RowOffsetStructuredSpinningLidarModelParameters,
        compute_tiling,
        compute_angles_to_columns_map,
    )
    import gsplat

    device = torch.device(device_str)
    if row_elevations_rad:
        # Explicit calibrated table (e.g. from a scene USDZ). Already sorted
        # strictly descending by the caller, matching the panorama's
        # row 0 = top-elevation convention used by the named tables below.
        elevs = torch.tensor(row_elevations_rad, dtype=torch.float32, device=device)
    elif sensor_type in _TABLES_RAD:
        elevs = torch.tensor(
            _TABLES_RAD[sensor_type], dtype=torch.float32, device=device
        )
    else:
        # Uniform spec fallback. linspace including both endpoints
        # would land the last entry exactly at el_lo; clip slightly to
        # keep the sorted-strictly-descending invariant gsplat asserts.
        elevs = torch.linspace(
            el_hi_rad,
            el_lo_rad,
            n_rows_uniform,
            dtype=torch.float32,
            device=device,
        )
        # Nudge to enforce strict descending (gsplat asserts torch.diff > 0
        # on the CW-relative angles which is satisfied by linspace, but
        # we add a tiny epsilon to defend against numerical ties).
        elevs[0] = elevs[0] - 1e-6
        elevs[-1] = elevs[-1] - 1e-6
    n_rows = int(elevs.shape[0])

    # Spinning lidar columns sweep from +π toward -π (CW) over the full
    # azimuth circle. The endpoints are exclusive of wrap-around so the
    # last column doesn't collide with the first.
    column_azimuths = torch.linspace(
        math.pi - 1e-4,
        -math.pi + 1e-4,
        int(n_columns),
        dtype=torch.float32,
        device=device,
    )
    row_azimuth_offsets = torch.zeros(n_rows, dtype=torch.float32, device=device)
    params = RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=elevs,
        column_azimuths_rad=column_azimuths,
        row_azimuth_offsets_rad=row_azimuth_offsets,
        spinning_frequency_hz=float(spinning_frequency_hz),
        spinning_direction=SpinningDirection.CLOCKWISE,
    )
    a2c = compute_angles_to_columns_map(params)
    tiling = compute_tiling(params)
    return gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(
        params, a2c, tiling
    )


def build_lidar_sensors_from_config(cfg_sensors) -> list[LidarSensorSpec]:
    """Convert scene/config LiDAR sensor entries to :class:`LidarSensorSpec`."""
    out: list[LidarSensorSpec] = []
    for s in cfg_sensors or []:
        translation = list(
            getattr(s, "translation", getattr(s, "position", (0.0, 0.0, 0.0)))
        )
        rotation = list(getattr(s, "rotation", (1.0, 0.0, 0.0, 0.0)))
        if len(rotation) == 4:
            r = quat_to_matrix(rotation, order="wxyz")
        elif len(rotation) == 3:
            r = rpy_deg_to_matrix(rotation)
        else:
            raise ValueError(
                f"LiDAR sensor {getattr(s, 'name', '<unnamed>')}: "
                "rotation must be quaternion [w,x,y,z] or RPY [roll,pitch,yaw]"
            )
        s2b = mat4(r, translation)
        elevation_deg = getattr(s, "elevation_deg", None)
        if elevation_deg:
            # Sort strictly descending (top→bottom) so the panorama's row 0
            # is the highest beam regardless of the source table's order.
            row_elevations_rad = tuple(
                math.radians(v)
                for v in sorted((float(d) for d in elevation_deg), reverse=True)
            )
        else:
            row_elevations_rad = ()
        out.append(
            LidarSensorSpec(
                name=str(s.name),
                sensor_type=str(getattr(s, "sensor_type", "") or ""),
                s2b=s2b,
                n_columns=int(getattr(s, "n_columns", 2048)),
                spinning_frequency_hz=float(getattr(s, "fps", 10.0)),
                n_rows_uniform=int(getattr(s, "n_rows", 128)),
                row_elevations_rad=row_elevations_rad,
            )
        )
    return out


# ── Rendering ───────────────────────────────────────────────────────

# Default raydrop_logit for Gaussians without a learned attribute.
# sigmoid(-6.0) ≈ 0.0025 → very low drop probability.
DEFAULT_RAYDROP_LOGIT: float = -6.0


def _raydrop_sh_degree_from_coefs(coefs: int) -> int:
    """Return the SH degree for ``coefs`` higher-order raydrop bands.

    ``coefs == (degree + 1)**2 - 1`` (the DC/band-0 term is excluded — it lives
    in the scalar ``raydrop_logit``). Raises ``ValueError`` if ``coefs`` is not
    of that form for any non-negative integer degree.
    """
    root = math.isqrt(coefs + 1)
    if root * root != coefs + 1:
        raise ValueError(
            f"raydrop_sh coefs {coefs} is not (deg+1)**2 - 1 for any integer degree"
        )
    return root - 1


def _eval_view_dependent_raydrop(
    means: torch.Tensor,  # (N, 3) world
    view_pos: torch.Tensor,  # (3,) sensor origin in world
    raydrop_logit: torch.Tensor,  # (N,) scalar band-0 (DC) logit
    raydrop_sh: torch.Tensor | None,  # (N, (deg+1)**2-1) higher-order bands, or None
) -> torch.Tensor:
    """Evaluate the view-dependent raydrop logit at each Gaussian's view ray.

    Mirrors colour SH: the drop logit is evaluated along the direction from the
    sensor origin to the Gaussian, exactly like gsplat evaluates colour SH along
    the camera→Gaussian direction. The scalar ``raydrop_logit`` is the band-0
    (DC) contribution and ``raydrop_sh`` carries the higher bands (its width
    determines the SH degree); when there are none the scalar is returned
    unchanged (backward compatible).
    """
    if raydrop_sh is None:
        return raydrop_logit

    # Fast path: a dedicated single-pass CUDA kernel. gsplat's
    # spherical_harmonics takes colour-shaped (N, K, 3) coefficients, so the
    # fallback below has to materialise a throwaway (N, K, 3) buffer (306 MiB at
    # N=3.2M, K=9) and zero-fill + scatter into it, then evaluate three channels
    # to use one -- ~1.8 ms/frame/sensor of pure packing. The kernel reads the
    # scalar logit and the higher bands directly.
    if means.is_cuda and means.shape[0] > 0:
        ext = _cuda_cull_ext._try_load()
        if ext is not None and hasattr(ext, "raydrop_sh_eval"):
            return ext.raydrop_sh_eval(
                means.float(),
                view_pos.to(means.device, torch.float32).reshape(3),
                raydrop_logit.float(),
                raydrop_sh.float(),
            )

    import gsplat

    n = means.shape[0]
    degree = _raydrop_sh_degree_from_coefs(raydrop_sh.shape[1])
    k = (degree + 1) ** 2
    # gsplat.spherical_harmonics wants coeffs shaped (..., K, 3). We only need a
    # scalar channel, so pack the raydrop coefficients into channel 0 (zeros
    # elsewhere) and read channel 0 back. Band 0 is scaled by 1/SH_C0 so the DC
    # contribution equals the scalar raydrop_logit.
    coeffs = torch.zeros((n, k, 3), dtype=torch.float32, device=means.device)
    coeffs[:, 0, 0] = raydrop_logit.float() / SH_C0
    coeffs[:, 1:k, 0] = raydrop_sh.float()
    dirs = means.float() - view_pos.to(means.device, torch.float32).reshape(3)
    return gsplat.spherical_harmonics(degree, dirs, coeffs)[:, 0]


@lru_cache(maxsize=32)
def _lidar_intrinsics(h: int, w: int, device_str: str) -> torch.Tensor:
    """Cached (1, 3, 3) intrinsics for gsplat's lidar path.

    gsplat's ``camera_model='lidar'`` ignores the K matrix (it uses the
    precomputed angle-to-column mapping from ``lidar_coeffs``), so a
    fixed placeholder is fine and needs to be built only once.
    """
    focal = float(w)
    return torch.tensor(
        [[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=torch.device(device_str),
    ).unsqueeze(0)


# Testing hook. Set to ``False`` to force the PyTorch fallback in
# _lidar_cull_keep, so tests can compare CUDA-cull output to the reference
# expression without spinning up a separate process. Not part of the
# public API.
_USE_CUDA_CULL: bool = True


def _lidar_cull_keep(
    *,
    means: torch.Tensor,
    scales: torch.Tensor,
    sensor_to_world: torch.Tensor,
    min_range_m: float,
    max_range_m: float | None,
    cull_scale_sigmas: float,
    elev_fov_cull: bool,
    sin_min: float,
    cos_min: float,
    sin_max: float,
    cos_max: float,
) -> torch.Tensor:
    """Combined shell (range) + elevation-FOV keep mask.

    Prefers the fused CUDA extension when available (single kernel launch,
    ~4× faster than the PyTorch chain at N=4M on sm_89). Falls back to the
    equivalent PyTorch expression otherwise — matches bit-for-bit on
    float32 inputs and preserves the original semantics on non-CUDA or
    non-float32 tensors.

    Notes on the linear-in-margin elevation test:
        sin(elev_gaussian) ≷ sin(elev_bound ± ang_margin) with
        ang_margin ≈ margin/dist yields, after multiplying by dist:
          z_s + margin·cos(elev_min) ≥ dist·sin(elev_min)   (below-FOV cut)
          z_s - margin·cos(elev_max) ≤ dist·sin(elev_max)   (above-FOV cut)
        Uses the first-order Taylor of sin(θ+δ) around the FOV edge;
        residual is second-order and biased toward *keeping* extra
        Gaussians (never drops one the exact atan2 test would keep).
    """
    device = means.device
    sensor_pos = sensor_to_world[:3, 3].to(device=device, dtype=means.dtype)
    up_world = sensor_to_world[:3, 2].to(device=device, dtype=means.dtype)

    if (
        _USE_CUDA_CULL
        and means.is_cuda
        and means.dtype == torch.float32
        and scales.dtype == torch.float32
        and means.is_contiguous()
        and scales.is_contiguous()
        and _cuda_cull_ext.is_available()
    ):
        try:
            return _cuda_cull_ext.lidar_cull_mask(
                means,
                scales,
                sensor_pos,
                up_world,
                min_range=min_range_m,
                max_range=max_range_m,
                cull_scale_sigmas=cull_scale_sigmas,
                use_elev=elev_fov_cull,
                sin_min=sin_min,
                cos_min=cos_min,
                sin_max=sin_max,
                cos_max=cos_max,
            )
        except Exception:
            # Fall through to PyTorch chain. Deliberately broad: we never
            # want a build/runtime issue in the accelerator to stop a
            # rendering pipeline.
            pass

    delta = means - sensor_pos
    dist = torch.linalg.vector_norm(delta, dim=-1)
    # NaN scales would poison the mask and silently drop every Gaussian;
    # treat them as zero-margin so an unlabeled bad splat is at worst
    # missed for its own extent, not for every neighbour.
    max_scale = torch.nan_to_num(scales.amax(dim=-1), nan=0.0)
    margin = cull_scale_sigmas * max_scale
    keep = dist + margin >= min_range_m
    if max_range_m is not None:
        keep = keep & (dist - margin <= max_range_m)
    if elev_fov_cull:
        z_s = delta @ up_world
        keep = keep & (z_s + margin * cos_min >= dist * sin_min)
        keep = keep & (z_s - margin * cos_max <= dist * sin_max)
    return keep


def _rigid_inverse_4x4(m: torch.Tensor) -> torch.Tensor:
    """Inverse of a rigid 4×4 transform ``[R | t; 0 0 0 1]``.

    Uses ``R^T`` and ``-R^T @ t`` — dtype-stable, allocation-cheap, and
    numerically better than ``torch.linalg.inv`` on rigid inputs.
    """
    r = m[:3, :3]
    t = m[:3, 3]
    inv = torch.eye(4, dtype=m.dtype, device=m.device)
    r_t = r.transpose(-1, -2)
    inv[:3, :3] = r_t
    inv[:3, 3] = -(r_t @ t)
    return inv


# ── SplatAD spherical LiDAR kernel (vendored) ───────────────────────
# Non-uniform elevation tile geometry for the SplatAD rasterizer.
#
# The rasterizer is tail-bound by a few very long per-tile Gaussian lists (p95
# ~66k, max ~480k entries at the upstream 4x64), and every pixel in a tile walks
# its whole list -- so the cost is dominated by testing Gaussians whose tile
# bbox covers the pixel but whose ray never hits them. Smaller tiles cut those
# lists; on a 27M-Gaussian driving scene (RTX 3090, sm_86) the 5-sensor rig
# measures, as a 4-frame mean over light and heavy poses:
#
#   4x64 (upstream)  ~400 ms      2x16   159 ms
#   4x32              201 ms      4x8    147 ms
#   1x32              165 ms      1x16   141 ms   <- here
#
# Note 1x16 is 16 threads, i.e. a half-warp per block: shorter azimuth tiles win
# even at the cost of idle lanes (1x32, a full warp over the same beam row, is
# clearly worse). Tiling only regroups Gaussians -- it never changes a pixel's
# front-to-back order -- so the render moves only by the handful of boundary
# cells where the 3-sigma bbox binning and the ~3.7-sigma alpha cutoff disagree
# (measured IoU >= 0.99998, p99 distance diff 0).
# Annotated (rather than inferred as Literal) because benchmarks and tests
# rebind them to explore the tiling.
_SPLATAD_TILE_HEIGHT: int = 1
_SPLATAD_TILE_WIDTH: int = 16
_SPLATAD_RAST = None

# Static per-(sensor, device) rasterization geometry. Everything here depends
# only on the sensor spec (beam tables) and the tile constants, so rebuilding it
# per frame (meshgrid + stack + two .item() device syncs per render) was pure
# overhead in the streaming path. Keyed by the spec's hashable geometry fields.
_PANO_GEOM_CACHE: dict = {}


class _PanoGeom(NamedTuple):
    elevs_desc: torch.Tensor  # (H,) descending beam elevations [rad], device
    azs_cw: torch.Tensor  # (W,) clockwise column azimuths [rad], device
    raster_pts: torch.Tensor  # (1, H, W, 4) ascending az/el grid, device
    tile_boundaries: torch.Tensor  # (H//th + 1,) ascending deg, device
    dirs: torch.Tensor  # (H, W, 3) unit beam directions (desc/CW grid), device
    min_el_deg: float  # ascending-grid elevation bounds (CPU floats,
    max_el_deg: float  # cached to avoid per-frame .item() syncs)


def _panorama_geometry(lidar_spec: LidarSensorSpec, device) -> _PanoGeom:
    key = (
        str(device),
        lidar_spec.name,
        lidar_spec.sensor_type,
        lidar_spec.n_columns,
        lidar_spec.row_elevations_rad,
        lidar_spec.el_lo_rad,
        lidar_spec.el_hi_rad,
        lidar_spec.n_rows_uniform,
        _SPLATAD_TILE_HEIGHT,
        _SPLATAD_TILE_WIDTH,
    )
    geom = _PANO_GEOM_CACHE.get(key)
    if geom is not None:
        return geom

    elevs_desc_cpu = _sensor_row_elevations(lidar_spec)
    elevs_desc = elevs_desc_cpu.to(device)  # (H,) desc rad
    azs_cw = _sensor_column_azimuths(lidar_spec).to(device)  # (W,) CW rad

    # SplatAD requires ASCENDING elevation + ASCENDING azimuth. Build the
    # panorama grid in that order (the render output is flipped back).
    el_deg = torch.rad2deg(torch.flip(elevs_desc, [0]))  # (H,) ascending deg
    az_deg = torch.rad2deg(torch.flip(azs_cw, [0]))  # (W,) ascending deg
    grid_el, grid_az = torch.meshgrid(el_deg, az_deg, indexing="ij")  # (H, W)
    raster_pts = torch.stack(
        [grid_az, grid_el, torch.ones_like(grid_az), torch.zeros_like(grid_az)],
        dim=-1,
    ).unsqueeze(0)  # (1, H, W, 4) = [azimuth_deg, elevation_deg, range=1, time=0]

    # Non-uniform elevation tile boundaries (tile_height beams/tile), asc deg.
    th = _SPLATAD_TILE_HEIGHT
    tile_boundaries = torch.cat(
        [
            el_deg[0:1] - 1.0,
            (el_deg[th::th] + el_deg[th - 1 : -1 : th]) / 2,
            el_deg[-1:] + 1.0,
        ]
    )

    # Unit beam directions on the output (descending-elevation, CW-azimuth)
    # grid -- the same values _panorama_to_points recomputed per frame.
    g_el, g_az = torch.meshgrid(elevs_desc, azs_cw, indexing="ij")
    dirs = torch.stack(
        [
            torch.cos(g_el) * torch.cos(g_az),
            torch.cos(g_el) * torch.sin(g_az),
            torch.sin(g_el),
        ],
        dim=-1,
    )  # (H, W, 3)

    el_deg_cpu = torch.rad2deg(torch.flip(elevs_desc_cpu, [0]))
    geom = _PanoGeom(
        elevs_desc=elevs_desc,
        azs_cw=azs_cw,
        raster_pts=raster_pts,
        tile_boundaries=tile_boundaries,
        dirs=dirs,
        min_el_deg=float(el_deg_cpu.min()),
        max_el_deg=float(el_deg_cpu.max()),
    )
    _PANO_GEOM_CACHE[key] = geom
    return geom


def _splatad_lidar_rasterization():
    """Return the vendored SplatAD ``lidar_rasterization`` entry point, building
    the ``splatad_lidar_cuda`` CUDA extension on first use.

    LiDAR rasterization goes exclusively through this kernel — there is no gsplat
    fallback — so a missing CUDA toolkit / build failure raises rather than
    silently degrading. Cached after the first successful load.
    """
    global _SPLATAD_RAST
    if _SPLATAD_RAST is None:
        try:
            from .splatad_lidar.cuda._backend import _C
            from .splatad_lidar.rendering import lidar_rasterization
        except Exception as e:  # noqa: BLE001 - surface any import/build failure
            raise RuntimeError(
                "SplatAD LiDAR kernel could not be imported/built "
                "(splatsim.splatad_lidar). LiDAR rendering requires this CUDA "
                "kernel and has no gsplat fallback; ensure a CUDA toolkit with "
                f"nvcc is on PATH. Original error: {e}"
            ) from e
        if _C is None:
            raise RuntimeError(
                "SplatAD LiDAR CUDA extension 'splatad_lidar_cuda' failed to "
                "build (no nvcc / CUDA toolkit detected). LiDAR rendering "
                "requires this kernel and has no gsplat fallback."
            )
        _SPLATAD_RAST = lidar_rasterization
    return _SPLATAD_RAST


@dataclass
class LidarGaussians:
    """The LiDAR-ready Gaussian set handed to the rasterizer.

    Produced by :meth:`LidarRenderer.gather` / :func:`gather_lidar_rig` and
    consumed by :meth:`LidarRenderer.render` via its ``shared=`` argument. Every
    field is already LOD-filtered, lidar_mask-applied and concatenated across
    scene sources; ``colors`` is deliberately absent (the LiDAR path never reads
    it when per-Gaussian ``intensity_raw`` exists).
    """

    means: torch.Tensor  # (N, 3) world
    quats: torch.Tensor  # (N, 4) wxyz
    scales: torch.Tensor  # (N, 3)
    opacities: torch.Tensor  # (N,)
    intensity_sig: torch.Tensor  # (N,)
    raydrop_logit: torch.Tensor  # (N,)
    raydrop_sh: torch.Tensor | None  # (N, (deg+1)**2-1) or None

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    def _tensors(self) -> "list[torch.Tensor]":
        ts = [
            self.means,
            self.quats,
            self.scales,
            self.opacities,
            self.intensity_sig,
            self.raydrop_logit,
        ]
        if self.raydrop_sh is not None:
            ts.append(self.raydrop_sh)
        return ts

    def nbytes(self) -> int:
        """Device bytes held by this set (for VRAM accounting / logging)."""
        return int(sum(t.numel() * t.element_size() for t in self._tensors()))

    def record_stream(self, stream: "torch.cuda.Stream") -> None:
        """Mark these buffers as in use by *stream*.

        They are allocated on the gathering stream but read by the per-sensor
        rasterizations on side streams; without this the caching allocator may
        hand the memory to another stream while those reads are still pending.
        """
        for t in self._tensors():
            if t.is_cuda:
                t.record_stream(stream)


def gather_lidar_rig(
    renderers: "Sequence[LidarRenderer]",
    base_to_world: torch.Tensor,
    scene: "Scene",
    *,
    base_to_world_end: torch.Tensor | None = None,
) -> "LidarGaussians | None":
    """One LOD gather covering every LiDAR on a rig.

    Each sensor's own :meth:`LidarRenderer.gather` selects LOD tiers by distance
    from THAT sensor's mount, so an N-sensor rig paid N gathers and — worse for
    the streaming path — held N transient copies of a multi-million-Gaussian set
    at once. On a driving scene that is the dominant VRAM term and it is what
    makes per-sensor CUDA streams lose to sequential rendering (they overlap the
    rasterizers but multiply the peak).

    This gathers ONCE from the rig's base_link, with the LOD cell cull widened to
    ``max(sensor max_range + that sensor's mount offset)`` so no sensor loses a
    Gaussian it would otherwise see. The per-sensor radial/FOV cull still runs
    inside each rasterization, so each sensor's own range limits are respected
    exactly.

    LOD tiers are selected per cell from the NEAREST sensor mount (the filter
    takes an ``[S, 3]`` position set), so the shared selection is a superset of
    every sensor's own: no sensor is ever handed a coarser tier than it would
    have picked alone. Keying on a single rig origin instead would decimate the
    near field of whichever sensor sits furthest from it — measured at 5.3% of
    cells changed for a roof LiDAR against a ground-level base_link.

    The result is still not bit-identical to per-sensor gathers: a sensor can
    now receive Gaussians a *neighbouring* sensor's proximity kept, i.e. strictly
    finer LOD than it asked for. Pass ``shared=None`` to
    :meth:`LidarRenderer.render` for the exact per-sensor behaviour.

    Returns ``None`` when the scene contributes nothing.
    """
    if not renderers:
        return None
    ref = renderers[0]
    device = ref.device
    b2w = base_to_world.to(device)

    # Cell cull bound: the longest range on the rig. Cell distances below are
    # measured from the nearest mount, so no mount-offset slack is needed.
    max_dist: float | None = 0.0
    for r in renderers:
        if r.max_range_m is None:
            max_dist = None
            break
        assert max_dist is not None  # noqa: S101 - guarded by the break above
        max_dist = max(max_dist, r.max_range_m)
    if max_dist is not None and base_to_world_end is not None:
        max_dist += 0.5 * float(
            torch.norm(base_to_world_end.to(device)[:3, 3] - b2w[:3, 3])
        )

    # [S, 3] world positions of every sensor mount this frame.
    sensor_positions = torch.stack(
        [(b2w @ r._s2b_t.to(device))[:3, 3] for r in renderers], dim=0
    ).detach()

    lod_scale = float(os.environ.get("SPLATSIM_LIDAR_LOD_SCALE", "0.5"))
    ignore_mask = any(r.ignore_lidar_mask for r in renderers)
    tensor_list = scene.collect_tensors(
        sensor_positions,
        lod_count_scale=lod_scale,
        lidar_view=not ignore_mask,
        lod_max_distance=max_dist,
    )
    if not tensor_list:
        return None

    sh_degrees = {t.sh_degree for t in tensor_list}
    if len(sh_degrees) != 1:
        raise ValueError(f"Mixed SH degrees across scene sources: {sh_degrees}")

    means_l, quats_l, scales_l, opac_l, inten_l, drop_l = [], [], [], [], [], []
    sh_l: list[torch.Tensor | None] = []
    for t in tensor_list:
        if not ignore_mask and t.lidar_mask is not None:
            t = t[t.lidar_mask]
        i_sig, r_logit = _resolve_lidar_attrs(t)
        means_l.append(t.means)
        quats_l.append(t.quats)
        scales_l.append(t.scales)
        opac_l.append(t.opacities)
        inten_l.append(i_sig)
        drop_l.append(r_logit)
        sh_l.append(t.raydrop_sh)

    means = torch.cat(means_l, dim=0).to(device)
    if means.shape[0] == 0:
        return None
    return LidarGaussians(
        means=means,
        quats=torch.cat(quats_l, dim=0).to(device),
        scales=torch.cat(scales_l, dim=0).to(device),
        opacities=torch.cat(opac_l, dim=0).to(device),
        intensity_sig=torch.cat(inten_l, dim=0),
        raydrop_logit=torch.cat(drop_l, dim=0),
        raydrop_sh=ref._concat_raydrop_sh(sh_l, [m.shape[0] for m in means_l], device),
    )


# Side streams for the rig paths, reused across frames. Allocating fresh
# torch.cuda.Stream objects per frame is NOT free: their per-stream blocks churn
# the caching allocator, and the cost lands as a bimodal stall -- measured on a
# 27M-Gaussian scene at 5 sensors, every other frame jumped 151 -> 498 ms (and on
# a light frame 37 -> 479 ms, 13x). Holding the streams removes it entirely
# (frame-to-frame spread 1.02x) and is what makes the concurrent path actually
# beat sequential: 383 -> 151 ms on that same heavy frame.
_STREAM_POOL: dict = {}


def _side_streams(count: int, device) -> "list[torch.cuda.Stream]":
    """Return ``count`` cached side streams for *device*, growing the pool."""
    key = torch.device(device).index if torch.device(device).index is not None else 0
    pool = _STREAM_POOL.setdefault(key, [])
    while len(pool) < count:
        pool.append(torch.cuda.Stream(device=device))
    return pool[:count]


def render_lidars_concurrent(
    renderers: "Sequence[LidarRenderer]",
    base_to_world: torch.Tensor,
    scene: "Scene",
    base_to_world_end: torch.Tensor | None = None,
    *,
    shared_gather: bool = True,
) -> list[dict]:
    """Render every LiDAR on a rig for one frame, sharing one Gaussian gather.

    Two things make an N-sensor rig cheap here:

    * **One gather instead of N** (``shared_gather``, default on). The sensors
      sit within a couple of metres of each other on the same vehicle, so
      :func:`gather_lidar_rig` collects the union once and every sensor
      rasterizes it. That removes N-1 LOD gathers AND, decisively, N-1
      multi-million-Gaussian transient buffers — the peak-VRAM term that used to
      make the concurrent path lose to sequential rendering on big scenes. See
      :func:`gather_lidar_rig` for the LOD tier approximation this implies.
    * **One CUDA stream per sensor.** With the memory blow-up gone, the
      per-sensor cull / projection / launch pipelines (latency- and CPU-bound at
      aggressive LOD, not SM-bound) overlap instead of serializing.

    ``SPLATSIM_LIDAR_CONCURRENT=0`` forces one stream (the shared gather still
    applies); ``shared_gather=False`` restores per-sensor gathers. Falls back to
    sequential on CUDA OOM, which concurrent streams can still provoke on very
    large scenes since all sensors' rasterizer scratch is live at once.
    """
    if not renderers:
        return []

    shared = None
    if shared_gather:
        shared = gather_lidar_rig(
            renderers, base_to_world, scene, base_to_world_end=base_to_world_end
        )
        if shared is None:
            return [r._empty_panorama() for r in renderers]

    def _sequential() -> list[dict]:
        return [
            r.render(
                base_to_world,
                scene=scene,
                base_to_world_end=base_to_world_end,
                shared=shared,
            )
            for r in renderers
        ]

    if (
        len(renderers) <= 1
        or not torch.cuda.is_available()
        or os.environ.get("SPLATSIM_LIDAR_CONCURRENT", "1") == "0"
    ):
        return _sequential()

    try:
        current = torch.cuda.current_stream()
        streams = _side_streams(len(renderers), renderers[0].device)
        # The shared gather was produced on the CURRENT stream; entering a side
        # stream does not inherit that dependency, so each side stream must wait
        # for it explicitly. Without this the first (and largest) sensor races
        # the tail of the gather and rasterizes a partially-written buffer --
        # observed as an empty panorama in 11 of 12 runs.
        for st in streams:
            st.wait_stream(current)
        outs: dict[int, dict] = {}
        for i, (r, st) in enumerate(zip(renderers, streams)):
            with torch.cuda.stream(st):
                outs[i] = r.render(
                    base_to_world,
                    scene=scene,
                    base_to_world_end=base_to_world_end,
                    shared=shared,
                )
            # Keep the caching allocator from recycling the shared buffers (and
            # each sensor's outputs) into another stream before this one is done.
            if shared is not None:
                shared.record_stream(st)
            for t in outs[i].values():
                if torch.is_tensor(t):
                    t.record_stream(current)
        for st in streams:
            current.wait_stream(st)
        torch.cuda.synchronize()
        return [outs[i] for i in range(len(renderers))]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return _sequential()


def _panorama_to_points(
    distance: torch.Tensor,  # (H, W) range [m] from the sensor origin
    elevs_desc: torch.Tensor,  # (H,) descending beam elevations [rad]
    azs_cw: torch.Tensor,  # (W,) clockwise column azimuths [rad]
) -> torch.Tensor:
    """Back-project a ``(H, W)`` range panorama to a ``(H, W, 3)`` sensor-frame
    point cloud via the per-beam ``(elevation, azimuth)`` directions.

    ``point = range * (cosEL·cosAZ, cosEL·sinAZ, sinEL)``. Pixels with no return
    (range 0) map to the origin; mask them with the ``alpha`` map when consuming.
    """
    el = elevs_desc.to(device=distance.device, dtype=distance.dtype)
    az = azs_cw.to(device=distance.device, dtype=distance.dtype)
    grid_el, grid_az = torch.meshgrid(el, az, indexing="ij")  # (H, W)
    dirs = torch.stack(
        [
            torch.cos(grid_el) * torch.cos(grid_az),
            torch.cos(grid_el) * torch.sin(grid_az),
            torch.sin(grid_el),
        ],
        dim=-1,
    )  # (H, W, 3) unit beam directions in the sensor frame
    return distance.unsqueeze(-1) * dirs


def render_lidar_panorama(
    *,
    means: torch.Tensor,  # (N, 3) world
    quats: torch.Tensor,  # (N, 4) wxyz, will be renormalised
    scales: torch.Tensor,  # (N, 3) post-exp (positive)
    opacities: torch.Tensor,  # (N,) post-sigmoid in [0, 1]
    intensity_sig: torch.Tensor,  # (N,) sigmoid(lidar_intensity_raw)
    raydrop_logit: torch.Tensor,  # (N,) raw lidar_raydrop_logit (SH band-0/DC)
    sensor_to_world: torch.Tensor,  # (4, 4) sweep-start pose, same device
    lidar_spec: LidarSensorSpec,
    raydrop_sh: torch.Tensor | None = None,  # (N, (deg+1)**2-1) higher SH bands
    sensor_to_world_end: torch.Tensor | None = None,  # (4, 4) sweep-end pose
    min_range_m: float = 0.3,
    max_range_m: float | None = 120.0,
    packed: bool = False,
    radius_clip: float = 0.0,
    frustum_cull: bool = False,
    cull_scale_sigmas: float = 3.0,
    elev_fov_cull: bool = True,
    with_ut: bool = True,
    with_eval3d: bool = True,
) -> dict[str, torch.Tensor]:
    """Run the SplatAD spherical lidar raster once for one sensor at one frame.

    Returns:
        ``{"alpha": (H, W), "distance": (H, W),
            "intensity": (H, W), "raydrop_logit": (H, W)}``

    All maps are on the same device as ``means``. ``distance`` is the
    alpha-weighted expected hit distance from the sensor origin (so it
    matches the canonical LiDAR "range" reading); the value is 0 where
    no Gaussians intersect that bin.

    View-dependent raydrop
    ----------------------
    ``raydrop_logit`` is the band-0 (DC) drop logit. When ``raydrop_sh`` (the
    higher-order SH bands, shape ``(N, (deg+1)**2 - 1)`` where its width fixes
    the degree) is given, the drop logit is re-evaluated per Gaussian along the
    sensor→Gaussian ray — exactly like colour SH — so a Gaussian can drop for one
    sensor view and return for another. With ``raydrop_sh=None`` the scalar logit
    is used directly (the previous behaviour).

    Acceleration knobs
    ------------------
    ``min_range_m``/``max_range_m`` gate what gsplat needs to touch.
    They are also forwarded to gsplat as ``near_plane``/``far_plane``
    so Gaussians outside the sensor's usable shell are dropped inside
    the projection kernel. ``max_range_m=None`` disables the far cut.

    ``packed`` is inert: it selected gsplat's packed rasterization path,
    which the vendored SplatAD kernel does not have. Kept for call-site
    compatibility with the gsplat-backed signature.

    ``radius_clip`` drops Gaussians whose projected 2D radius is below
    the given pixel count; useful for skipping sub-pixel dust.

    ``frustum_cull`` runs a spherical-shell + elevation-FOV test
    (``|means - sensor_pos| ± sigmas * max(scale)``) and gathers the survivors
    before they reach the rasterizer. It defaults to OFF, and measurably so:

    * It no longer removes much. The LOD gather already drops whole octree
      cells beyond the sensor's range (``lod_max_distance``), so this pass cuts
      only ~15% of what reaches it (9.6M -> 8.2M on a driving frame) -- yet it
      pays for 7 full-array gathers to do it.
    * Those gathers cost more than they save: the 5-sensor rig measures
      124.4 ms with the cull and 116.6 ms without, at 0.6 GiB LOWER peak VRAM
      (no gathered copies).
    * It was also dropping real returns. The projection kernel rejects on the
      exact projected extent; this pass approximates with a linearized
      elevation band, so it can discard Gaussians the rasterizer would have
      kept. Measured against cull-off across the rig, every differing cell was
      a return the cull had thrown away and none was one it invented
      (only-OFF 3-9 cells per sensor, only-ON 0).

    Turn it on only when memory pressure makes the shorter arrays worth the
    gather and the lost returns.

    ``elev_fov_cull`` layers a splatAD-style vertical-FOV test on top
    of the radial shell (sensor-frame elevation ± ``sigmas * scale /
    dist``). Gaussians whose entire angular extent falls above or
    below the sensor's row-elevation range can never touch a ray, so
    dropping them shrinks what gsplat needs to project. Cheap
    (one 3×N matmul + an ``atan2``) and orthogonal to the radial cull.

    ``with_ut``/``with_eval3d`` are likewise inert. They toggled gsplat's
    unscented-transform projection and 3D-evaluated blending; the SplatAD
    kernel projects and blends unconditionally in its own spherical model,
    so neither flag reaches a kernel argument. Kept (defaulting to True, the
    behaviour they described) so existing call sites keep working.

    ``sensor_to_world_end`` is a motion-during-sweep *approximation*, NOT true
    rolling shutter. ``sensor_to_world`` is the pose at the first azimuth column
    and ``sensor_to_world_end`` the pose at the last, and the whole panorama is
    rendered from the translational midpoint of the two (the frustum cull and
    the view-dependent raydrop evaluation use that same midpoint, so all three
    stay consistent). Every column therefore shares one pose: the intra-sweep
    skew a real spinning LiDAR accumulates is NOT reproduced — only the average
    displacement over the sweep is. The gsplat backend modelled this properly
    via ``viewmats_rs`` + ``RollingShutterType``; the vendored SplatAD kernel
    instead expects per-Gaussian ``velocities``, which this path does not yet
    supply (it passes ``velocities=None``). Wiring that up is what would make
    this real rolling shutter. When ``None`` (default) the panorama is rendered
    from the single ``sensor_to_world`` pose.
    """
    device = means.device

    if frustum_cull and means.shape[0] > 0:
        if elev_fov_cull:
            # Use the sensor's ACTUAL beam-elevation extent (honouring an explicit
            # row_elevations_rad table), not the OT128/uniform defaults. Sensors
            # with a custom table (sensor_type="") previously fell back to the
            # default [-25, +15] deg, leaving the corner LiDARs (real [-16, +15])
            # under-culled by ~9 deg. _sensor_row_elevations mirrors the beam-table
            # precedence, so this tightens the vertical-FOV cull without dropping
            # anything a beam could actually see (the scale/dist margin still
            # covers boundary Gaussians -> bit-identical output).
            _elevs = _sensor_row_elevations(lidar_spec)
            _e_lo = float(_elevs.min())
            _e_hi = float(_elevs.max())
            sin_min, cos_min = math.sin(_e_lo), math.cos(_e_lo)
            sin_max, cos_max = math.sin(_e_hi), math.cos(_e_hi)
        else:
            sin_min = cos_min = sin_max = cos_max = 0.0
        # For a rolling-shutter sweep the sensor moves; pre-cull from the sweep
        # midpoint so a boundary Gaussian isn't dropped for either endpoint (the
        # exact per-column near/far gating still happens inside gsplat).
        cull_s2w = sensor_to_world
        if sensor_to_world_end is not None:
            cull_s2w = sensor_to_world.clone()
            cull_s2w[:3, 3] = 0.5 * (
                sensor_to_world[:3, 3]
                + sensor_to_world_end.to(cull_s2w.device, cull_s2w.dtype)[:3, 3]
            )
        keep = _lidar_cull_keep(
            means=means,
            scales=scales,
            sensor_to_world=cull_s2w,
            min_range_m=float(min_range_m),
            max_range_m=None if max_range_m is None else float(max_range_m),
            cull_scale_sigmas=float(cull_scale_sigmas),
            elev_fov_cull=bool(elev_fov_cull),
            sin_min=float(sin_min),
            cos_min=float(cos_min),
            sin_max=float(sin_max),
            cos_max=float(cos_max),
        )
        # Index once, unconditionally: sharing the index buffer across the
        # six tensors avoids six independent mask scans, and skipping the
        # `keep.all()` short-circuit avoids a device→host sync per frame.
        idx = keep.nonzero(as_tuple=False).squeeze(-1)
        means = means.index_select(0, idx)
        quats = quats.index_select(0, idx)
        scales = scales.index_select(0, idx)
        opacities = opacities.index_select(0, idx)
        intensity_sig = intensity_sig.index_select(0, idx)
        raydrop_logit = raydrop_logit.index_select(0, idx)
        if raydrop_sh is not None:
            raydrop_sh = raydrop_sh.index_select(0, idx)

    # View-dependent raydrop: fold the higher SH bands into the per-Gaussian
    # scalar logit before it is composited by the rasterizer. Evaluated after
    # culling so the (potentially large) SH matmul only runs over kept Gaussians.
    # The view ray uses the sweep-midpoint sensor origin (matching the cull) so a
    # single evaluation approximates the whole rolling-shutter spin.
    if raydrop_sh is not None and means.shape[0] > 0:
        view_pos = sensor_to_world[:3, 3]
        if sensor_to_world_end is not None:
            view_pos = 0.5 * (
                view_pos
                + sensor_to_world_end.to(view_pos.device, view_pos.dtype)[:3, 3]
            )
        raydrop_logit = _eval_view_dependent_raydrop(
            means, view_pos, raydrop_logit, raydrop_sh
        )

    # Beam table straight from the sensor spec (gsplat-independent): rows are the
    # descending elevations, columns sweep +pi -> -pi (clockwise) -- the exact
    # panorama layout the point-cloud / range-image back-projection expects.
    # All static per-sensor geometry (raster grid, tile boundaries, beam
    # directions) comes from the per-(spec, device) cache.
    geom = _panorama_geometry(lidar_spec, device)
    elevs_desc = geom.elevs_desc  # (H,) desc rad
    azs_cw = geom.azs_cw  # (W,) CW rad
    H, W = int(elevs_desc.shape[0]), int(azs_cw.shape[0])

    if means.shape[0] == 0:
        zero = torch.zeros((H, W), dtype=torch.float32, device=device)
        return {
            "alpha": zero.clone(),
            "distance": zero.clone(),
            "points": torch.zeros(
                (*zero.shape, 3), dtype=zero.dtype, device=zero.device
            ),
            "intensity": zero.clone(),
            "raydrop_logit": torch.full_like(zero, DEFAULT_RAYDROP_LOGIT),
        }

    # SplatAD renders on an ASCENDING elevation/azimuth grid; the cached
    # raster_pts/tile_boundaries are in that order and the panorama is flipped
    # back to our (descending-elevation, CW-azimuth) grid after the raster.
    th, tw = _SPLATAD_TILE_HEIGHT, _SPLATAD_TILE_WIDTH
    raster_pts = geom.raster_pts
    tile_boundaries = geom.tile_boundaries

    # Rolling shutter: SplatAD models intra-sweep skew via per-Gaussian
    # velocities, not a sweep-end pose. Until that path is wired, approximate the
    # sweep by rendering from the midpoint pose (matching the cull midpoint).
    render_s2w = sensor_to_world
    if sensor_to_world_end is not None:
        render_s2w = sensor_to_world.clone()
        render_s2w[:3, 3] = 0.5 * (
            sensor_to_world[:3, 3]
            + sensor_to_world_end.to(sensor_to_world.device, sensor_to_world.dtype)[
                :3, 3
            ]
        )
    render_s2w = render_s2w.to(device=device, dtype=torch.float32)
    viewmats = _rigid_inverse_4x4(render_s2w).unsqueeze(0)  # (C=1, 4, 4)

    # lidar_features = [intensity_sig, raydrop_logit] -> rendered feature channels.
    lidar_features = torch.stack(
        [intensity_sig.float(), raydrop_logit.float()], dim=-1
    ).unsqueeze(0)  # (1, N, 2)
    quats_n = torch.nn.functional.normalize(quats.float(), p=2, dim=-1)

    lidar_rasterization = _splatad_lidar_rasterization()
    render, alpha, _alpha_sum, meta = lidar_rasterization(
        means=means.float(),
        quats=quats_n,
        scales=scales.float(),
        opacities=opacities.float(),
        lidar_features=lidar_features,
        velocities=None,
        viewmats=viewmats,
        raster_pts=raster_pts,
        tile_elevation_boundaries=tile_boundaries.clone(),
        min_azimuth=-180.0,
        max_azimuth=180.0,
        min_elevation=geom.min_el_deg,
        max_elevation=geom.max_el_deg + 1e-3,
        n_elevation_channels=H,
        azimuth_resolution=360.0 / float(W),
        tile_width=tw,
        tile_height=th,
        near_plane=float(min_range_m),
        far_plane=1e10 if max_range_m is None else float(max_range_m),
        radius_clip=float(radius_clip),
        # The alpha-sum-until-point map is a training-time feature (raydrop
        # supervision against real returns); this inference path discards it,
        # so skip its per-Gaussian-per-pixel accumulation in the kernel.
        compute_alpha_sum_until_points=False,
        use_depth_compensation=False,
    )

    # Unpack on the ascending grid, then flip rows+cols back to our grid. SplatAD's
    # median range (meta["median_depths"]) gives sharp rings where gsplat's
    # expected-depth panorama smears them.
    def _to_grid(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, [0, 1])

    dist_grid = _to_grid(meta["median_depths"][0, ..., 0])
    return {
        "alpha": _to_grid(alpha[0, ..., 0] if alpha.dim() == 4 else alpha[0]),
        "distance": dist_grid,
        # (H, W, 3) sensor-frame point cloud from the median range, back-projected
        # along the (cached) unit beam directions.
        "points": dist_grid.unsqueeze(-1) * geom.dirs,
        "intensity": _to_grid(render[0, ..., 0]),
        "raydrop_logit": _to_grid(render[0, ..., 1]),
    }


# ── splatsim integration wrapper ────────────────────────────────────

# Luminance weights for SH-derived intensity fallback (Rec. 709).
_LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


def _sensor_row_elevations(spec: LidarSensorSpec) -> torch.Tensor:
    """Return the (H,) descending elevation table used by ``spec``.

    Mirrors the precedence of :func:`_build_lidar_coeffs` (explicit calibrated
    table, then known-sensor lookup, then uniform linspace fallback) so
    reconstruction uses the exact same per-row elevations as beam emission and
    consumers can reconstruct per-pixel directions without touching gsplat
    internals.
    """
    if spec.row_elevations_rad:
        # Explicit calibrated table (e.g. from a scene USDZ). Consumed exactly
        # as ``_build_lidar_coeffs`` consumes it: already sorted strictly
        # descending by the caller (row 0 = top elevation).
        return torch.tensor(spec.row_elevations_rad, dtype=torch.float32)
    if spec.sensor_type in _TABLES_RAD:
        return torch.tensor(_TABLES_RAD[spec.sensor_type], dtype=torch.float32)
    elevs = torch.linspace(
        spec.el_hi_rad,
        spec.el_lo_rad,
        spec.n_rows_uniform,
        dtype=torch.float32,
    )
    elevs[0] = elevs[0] - 1e-6
    elevs[-1] = elevs[-1] - 1e-6
    return elevs


def _sensor_column_azimuths(spec: LidarSensorSpec) -> torch.Tensor:
    """Return the (W,) azimuth table (radians) used by ``spec``.

    Matches ``_build_lidar_coeffs``: columns sweep +π → -π (clockwise).
    """
    return torch.linspace(
        math.pi - 1e-4,
        -math.pi + 1e-4,
        int(spec.n_columns),
        dtype=torch.float32,
    )


@lru_cache(maxsize=32)
def _lidar_elev_fov_rad(
    sensor_type: str,
    el_lo_rad: float,
    el_hi_rad: float,
    n_rows_uniform: int,
) -> tuple[float, float]:
    """Return ``(elev_min, elev_max)`` in radians for a sensor spec.

    Uses the same table / linspace logic as :func:`_sensor_row_elevations`
    but exposes the extrema as plain Python floats so callers can drop
    Gaussians outside the sensor's vertical FOV without paying a device
    → host sync every frame.
    """
    if sensor_type in _TABLES_RAD:
        tab = _TABLES_RAD[sensor_type]
        return (float(min(tab)), float(max(tab)))
    # Uniform-spec fallback mirrors ``_build_lidar_coeffs`` (linspace hi→lo
    # with a 1e-6 rad nudge on both endpoints).
    return (float(el_lo_rad) - 1e-6, float(el_hi_rad) - 1e-6)


@lru_cache(maxsize=32)
def _lidar_elev_fov_sincos(
    sensor_type: str,
    el_lo_rad: float,
    el_hi_rad: float,
    n_rows_uniform: int,
) -> tuple[float, float, float, float]:
    """Return ``(sin_min, cos_min, sin_max, cos_max)`` for the sensor's
    vertical FOV. Cached so per-frame elev-cull avoids the trig cost."""
    lo, hi = _lidar_elev_fov_rad(sensor_type, el_lo_rad, el_hi_rad, n_rows_uniform)
    return (math.sin(lo), math.cos(lo), math.sin(hi), math.cos(hi))


def _sh_dc_to_luminance(colors: torch.Tensor, sh_degree: int) -> torch.Tensor:
    """Return (N,) luminance in [0, 1] from a colors tensor.

    - ``sh_degree == 0``: ``colors`` is already RGB in [0, 1].
    - ``sh_degree > 0``: DC band is ``colors[:, 0, :]`` in SH0-normalized space;
      apply the standard 3DGS decoding (``0.2820947918 * dc + 0.5``).
    """
    if sh_degree == 0:
        rgb = colors.clamp(0.0, 1.0)
    else:
        # SH DC → RGB: c = 0.2820947918 * dc + 0.5 (see 3DGS paper).
        dc = colors[:, 0, :]
        rgb = (0.2820947918 * dc + 0.5).clamp(0.0, 1.0)
    w = torch.tensor(_LUMA_WEIGHTS, dtype=rgb.dtype, device=rgb.device)
    return (rgb * w).sum(dim=-1)


def _resolve_lidar_attrs(
    tensors,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(intensity_sig, raydrop_logit)`` for one GaussianTensors group.

    Uses per-Gaussian ``intensity_raw`` / ``raydrop_logit`` when the scene
    carries them (gaussian_factory-trained), otherwise derives intensity from
    SH luminance and holds raydrop at :data:`DEFAULT_RAYDROP_LOGIT`.
    """
    n = tensors.means.shape[0]
    device = tensors.means.device
    if tensors.intensity_raw is not None:
        intensity_sig = torch.sigmoid(tensors.intensity_raw.to(device))
    else:
        intensity_sig = _sh_dc_to_luminance(tensors.colors, tensors.sh_degree).to(
            device
        )
    if tensors.raydrop_logit is not None:
        raydrop_logit = tensors.raydrop_logit.to(device)
    else:
        raydrop_logit = torch.full(
            (n,), DEFAULT_RAYDROP_LOGIT, dtype=torch.float32, device=device
        )
    return intensity_sig, raydrop_logit


class LidarRenderer:
    """High-level LiDAR renderer bound to a :class:`splatsim.scene.Scene`.

    Parameters
    ----------
    sensor_spec:
        Sensor mounting + optics. Held immutable — build a new
        :class:`LidarRenderer` if the sensor changes.
    device:
        Torch device for CUDA rasterization (must match the scene).
    min_range_m / max_range_m:
        Range gating passed through to :func:`render_lidar_panorama`.
    ignore_lidar_mask:
        When True, ignore each group's per-Gaussian ``lidar_mask`` and let
        every Gaussian participate in the LiDAR pass (A/B eval knob). When
        False (default), Gaussians with ``lidar_mask == False`` are
        hard-excluded from the LiDAR geometry pass.
    """

    def __init__(
        self,
        sensor_spec: LidarSensorSpec,
        *,
        device: torch.device | str,
        min_range_m: float = 0.3,
        max_range_m: float | None = 120.0,
        packed: bool = False,
        radius_clip: float = 0.0,
        frustum_cull: bool = False,
        cull_scale_sigmas: float = 3.0,
        elev_fov_cull: bool = True,
        with_ut: bool = True,
        with_eval3d: bool = True,
        ignore_lidar_mask: bool = False,
    ) -> None:
        self.sensor_spec = sensor_spec
        self.device = torch.device(device)
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m) if max_range_m is not None else None
        self.packed = bool(packed)
        self.radius_clip = float(radius_clip)
        self.frustum_cull = bool(frustum_cull)
        self.cull_scale_sigmas = float(cull_scale_sigmas)
        self.elev_fov_cull = bool(elev_fov_cull)
        self.with_ut = bool(with_ut)
        self.with_eval3d = bool(with_eval3d)
        # When True, the per-Gaussian ``lidar_mask`` (appearance-only /
        # far-field exclusion) is ignored and every Gaussian participates in
        # the LiDAR pass. Useful for A/B evaluation of masked vs unmasked.
        self.ignore_lidar_mask = bool(ignore_lidar_mask)
        # Precompute the (H,) elevation table once (used by point-cloud conv).
        self._elevs = _sensor_row_elevations(sensor_spec).to(self.device)
        self._azimuths = _sensor_column_azimuths(sensor_spec).to(self.device)
        # Cache the sensor→base extrinsic on-device — it's fixed for the
        # life of the renderer, so no need to rebuild it every frame.
        self._s2b_t = torch.from_numpy(sensor_spec.s2b.astype(np.float32)).to(
            self.device
        )
        # Prime the coeffs cache so the first render skips tile-assign cost.
        _ = sensor_spec.coeffs(self.device)

    @property
    def n_rows(self) -> int:
        return int(self._elevs.shape[0])

    @property
    def n_columns(self) -> int:
        return int(self.sensor_spec.n_columns)

    def _empty_panorama(self) -> dict[str, torch.Tensor]:
        """Zero-filled panorama output (no LiDAR returns): alpha/distance/intensity
        are 0 and every ray drops (raydrop_logit = DEFAULT_RAYDROP_LOGIT)."""
        zero = torch.zeros(
            (self.n_rows, self.n_columns), dtype=torch.float32, device=self.device
        )
        return {
            "alpha": zero.clone(),
            "distance": zero.clone(),
            "points": torch.zeros(
                (*zero.shape, 3), dtype=zero.dtype, device=zero.device
            ),
            "intensity": zero.clone(),
            "raydrop_logit": torch.full_like(zero, DEFAULT_RAYDROP_LOGIT),
        }

    def render(
        self,
        base_to_world: torch.Tensor,
        *,
        scene: "Scene",
        base_to_world_end: torch.Tensor | None = None,
        shared: "LidarGaussians | None" = None,
    ) -> dict[str, torch.Tensor]:
        """Render one LiDAR panorama.

        Args:
            base_to_world: (4, 4) float32 tensor. Ego/base pose at the start of
                the azimuth sweep (the whole panorama when no end pose is given).
            scene: The splatsim Scene providing Gaussians.
            base_to_world_end: Optional (4, 4) ego/base pose at the end of the
                sweep. When given, the panorama is rendered from the midpoint of
                the two poses — a motion-during-sweep approximation, not true
                rolling shutter (see :func:`render_lidar_panorama`).
            shared: Pre-gathered Gaussians from :func:`gather_lidar_rig`. When
                given, this sensor skips its own LOD gather and rasterizes the
                shared set — the point of the rig path: one gather and one
                transient buffer for N sensors instead of N of each. ``None``
                gathers per-sensor, as before.

        Returns:
            ``{"alpha", "distance", "intensity", "raydrop_logit"}`` — each an
            (H, W) float32 tensor. ``H`` = sensor row count,
            ``W`` = ``sensor_spec.n_columns``.
        """
        sensor_to_world = base_to_world.to(self.device) @ self._s2b_t
        sensor_to_world_end = None
        if base_to_world_end is not None:
            sensor_to_world_end = base_to_world_end.to(self.device) @ self._s2b_t

        if shared is None:
            shared = self.gather(
                base_to_world, scene, base_to_world_end=base_to_world_end
            )
        if shared is None or shared.means.shape[0] == 0:
            return self._empty_panorama()

        return render_lidar_panorama(
            means=shared.means,
            quats=shared.quats,
            scales=shared.scales,
            opacities=shared.opacities,
            intensity_sig=shared.intensity_sig,
            raydrop_logit=shared.raydrop_logit,
            sensor_to_world=sensor_to_world,
            lidar_spec=self.sensor_spec,
            raydrop_sh=shared.raydrop_sh,
            sensor_to_world_end=sensor_to_world_end,
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
            packed=self.packed,
            radius_clip=self.radius_clip,
            frustum_cull=self.frustum_cull,
            cull_scale_sigmas=self.cull_scale_sigmas,
            elev_fov_cull=self.elev_fov_cull,
            with_ut=self.with_ut,
            with_eval3d=self.with_eval3d,
        )

    def gather(
        self,
        base_to_world: torch.Tensor,
        scene: "Scene",
        *,
        base_to_world_end: torch.Tensor | None = None,
    ) -> "LidarGaussians | None":
        """Collect the LiDAR-ready Gaussian set this sensor would rasterize.

        Split out of :meth:`render` so a whole rig can share one gather — see
        :func:`gather_lidar_rig`. Returns ``None`` when the scene contributes
        nothing.
        """
        sensor_to_world = base_to_world.to(self.device) @ self._s2b_t
        cam_pos = sensor_to_world[:3, 3].detach()

        # LiDAR-specific LOD: a spinning LiDAR sees the full 360°, so (unlike a
        # camera) it cannot azimuth-cull and the camera-tuned tiers can leave far
        # too many Gaussians for the non-packed rasterizer (OOM / low FPS on dense
        # near-field scenes). SPLATSIM_LIDAR_LOD_SCALE thins every LOD cell further
        # (keeping each cell's top-`scale` importance-sorted Gaussians), cutting
        # memory + time roughly in proportion. Default 0.5: production scenes are
        # now ~60M Gaussians, which OOM the rasterizer at full density; 0.5 fits
        # them at full azimuth resolution with no measurable quality loss (the
        # dropped Gaussians are the least important per cell). Set to 1.0 to keep
        # every Gaussian, or lower (0.25) for more headroom / speed.
        lod_scale = float(os.environ.get("SPLATSIM_LIDAR_LOD_SCALE", "0.5"))
        # lidar_view fuses the static lidar_mask into the LOD gather and skips
        # the colors block (one gather instead of two); sources that bypass the
        # LOD filter still carry their mask and are handled per-source below.
        # Whole-cell LOD max-range cull: drop octree cells provably beyond the
        # sensor's radial cull before the gather. Under a sweep-end pose the
        # render/cull origin is the sweep midpoint, up to half the base motion
        # away from cam_pos — widen the bound by that slack so the cell cull
        # stays a strict superset of the per-Gaussian cull.
        lod_max_dist = self.max_range_m
        if lod_max_dist is not None and base_to_world_end is not None:
            lod_max_dist = lod_max_dist + 0.5 * float(
                torch.norm(
                    base_to_world_end.to(self.device)[:3, 3]
                    - base_to_world.to(self.device)[:3, 3]
                )
            )
        tensor_list = scene.collect_tensors(
            cam_pos,
            lod_count_scale=lod_scale,
            lidar_view=not self.ignore_lidar_mask,
            lod_max_distance=lod_max_dist,
        )
        if not tensor_list:
            return None

        sh_degrees = {t.sh_degree for t in tensor_list}
        if len(sh_degrees) != 1:
            raise ValueError(f"Mixed SH degrees across scene sources: {sh_degrees}")

        means_list, quats_list, scales_list, opacities_list = [], [], [], []
        intensity_list, raydrop_list = [], []
        # Per-group SH raydrop bands (or None). Collected verbatim, then reconciled
        # to a single scene-wide degree below so groups that lack the higher bands
        # contribute only their scalar (DC) logit.
        raydrop_sh_list: list[torch.Tensor | None] = []
        for t in tensor_list:
            # Hard-exclude appearance-only / far-field Gaussians (lidar_mask == False)
            # from the LiDAR geometry pass BEFORE resolving attrs, so the sigmoid /
            # luminance fallback only runs over kept Gaussians. __getitem__ filters
            # every field (geometry + intensity_raw/raydrop_logit/raydrop_sh)
            # consistently, so it also carries any future per-Gaussian field.
            # ``None`` (or the ``ignore_lidar_mask`` override) keeps every Gaussian.
            if not self.ignore_lidar_mask and t.lidar_mask is not None:
                t = t[t.lidar_mask]
            i_sig, r_logit = _resolve_lidar_attrs(t)
            means_list.append(t.means)
            quats_list.append(t.quats)
            scales_list.append(t.scales)
            opacities_list.append(t.opacities)
            intensity_list.append(i_sig)
            raydrop_list.append(r_logit)
            raydrop_sh_list.append(t.raydrop_sh)

        means = torch.cat(means_list, dim=0).to(self.device)
        quats = torch.cat(quats_list, dim=0).to(self.device)
        scales = torch.cat(scales_list, dim=0).to(self.device)
        opacities = torch.cat(opacities_list, dim=0).to(self.device)
        intensity_sig = torch.cat(intensity_list, dim=0)
        raydrop_logit = torch.cat(raydrop_list, dim=0)
        raydrop_sh = self._concat_raydrop_sh(
            raydrop_sh_list, [m.shape[0] for m in means_list], self.device
        )

        # Every Gaussian masked out of the LiDAR pass -> caller emits a
        # zero-output panorama.
        if means.shape[0] == 0:
            return None

        return LidarGaussians(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            intensity_sig=intensity_sig,
            raydrop_logit=raydrop_logit,
            raydrop_sh=raydrop_sh,
        )

    @staticmethod
    def _concat_raydrop_sh(
        raydrop_sh_list: list[torch.Tensor | None],
        counts: list[int],
        device: torch.device,
    ) -> torch.Tensor | None:
        """Concatenate per-group SH raydrop bands into a single scene tensor.

        Returns the concatenated ``raydrop_sh`` (its width fixes the SH degree),
        or ``None`` when no group carries higher bands (scalar-only scene).
        Groups that lack the higher bands are padded with zeros (so they
        contribute only their scalar DC logit), which lets a scene mix SH-trained
        Gaussians (e.g. the background) with groups that carry only the scalar
        raydrop (e.g. dynamic objects). All groups that do carry bands must share
        the same coefficient count (SH degree).
        """
        widths = {t.shape[1] for t in raydrop_sh_list if t is not None}
        if not widths:
            return None
        if len(widths) != 1:
            raise ValueError(
                f"Mixed raydrop_sh SH widths across scene sources: {widths}"
            )
        coefs = widths.pop()
        parts: list[torch.Tensor] = []
        for t, n in zip(raydrop_sh_list, counts):
            if t is None:
                parts.append(
                    torch.zeros((n, coefs), dtype=torch.float32, device=device)
                )
            else:
                parts.append(t.to(device, torch.float32))
        return torch.cat(parts, dim=0)

    def _validity_mask(
        self,
        panorama: dict[str, torch.Tensor],
        *,
        drop_threshold: float,
        alpha_threshold: float,
    ) -> torch.Tensor:
        """Per-cell return mask shared by the point-cloud / range-image paths.

        A cell is a valid return when it has enough alpha coverage, a low
        enough raydrop probability, and an in-range distance.
        """
        mask = (
            (panorama["alpha"] > alpha_threshold)
            & (torch.sigmoid(panorama["raydrop_logit"]) < drop_threshold)
            & (panorama["distance"] > self.min_range_m)
        )
        # ``max_range_m`` is optional: ``None`` means "no far clip".
        if self.max_range_m is not None:
            mask = mask & (panorama["distance"] < self.max_range_m)
        return mask

    def panorama_to_point_cloud(
        self,
        panorama: dict[str, torch.Tensor],
        *,
        drop_threshold: float = 0.5,
        alpha_threshold: float = 0.1,
    ) -> dict[str, np.ndarray]:
        """Convert a rendered panorama into a sparse point cloud in sensor frame.

        Args:
            panorama: Output of :meth:`render`.
            drop_threshold: sigmoid(raydrop_logit) above this drops the sample.
            alpha_threshold: Minimum alpha coverage to keep a sample.

        Returns:
            ``{"xyz": (N, 3) float32, "intensity": (N,) float32,
            "channel": (N,) uint16}``. Coordinates are in the sensor frame
            (+x forward, +y left, +z up — gsplat lidar convention).
            ``channel`` is the panorama row index of each point, i.e. the
            ring / laser-beam id (row 0 = topmost beam).
        """
        distance = panorama["distance"]
        intensity = panorama["intensity"]
        valid = self._validity_mask(
            panorama, drop_threshold=drop_threshold, alpha_threshold=alpha_threshold
        )

        row_grid = torch.arange(self.n_rows, device=distance.device)[:, None].expand(
            -1, self.n_columns
        )  # (H, W) ring / laser-beam index

        # ``panorama["points"]`` is the same back-projection (range x unit beam
        # direction) already produced by the render, so reuse it instead of
        # re-deriving the trig grids per frame.
        xyz = panorama["points"][valid]
        intensity_valid = intensity[valid]
        channel_valid = row_grid[valid]

        return {
            "xyz": xyz.detach().cpu().numpy().astype(np.float32),
            "intensity": intensity_valid.detach().cpu().numpy().astype(np.float32),
            "channel": channel_valid.detach().cpu().numpy().astype(np.uint16),
        }

    def panorama_to_pointcloud2_data(
        self,
        panorama: dict[str, torch.Tensor],
        *,
        drop_threshold: float = 0.5,
        alpha_threshold: float = 0.1,
    ) -> tuple[np.ndarray, int]:
        """Pack the valid returns straight into PointCloud2 point records.

        Produces the exact 16-byte little-endian record layout used by
        :mod:`splatsim.cyclonedds.pointcloud2_publisher` (x, y, z float32 |
        intensity uint8 | return_type uint8 = 0 | channel uint16) — but packs
        it on the GPU and moves ONE contiguous buffer to the host, instead of
        transferring xyz / intensity / channel separately and re-packing them
        on the CPU per frame.

        Returns:
            ``(records, count)`` — ``records`` is a ``(count * 16,)`` uint8
            array viewing the packed points, ready for ``PointCloud2.data``.
        """
        valid = self._validity_mask(
            panorama, drop_threshold=drop_threshold, alpha_threshold=alpha_threshold
        )
        xyz = panorama["points"][valid]  # (N, 3) float32, sensor frame
        n = int(xyz.shape[0])

        intensity_u8 = (
            (panorama["intensity"][valid].clamp(0.0, 1.0) * 255.0)
            .to(torch.uint8)
            .to(torch.int32)
        )
        row_grid = torch.arange(self.n_rows, device=xyz.device, dtype=torch.int32)[
            :, None
        ].expand(-1, self.n_columns)
        channel = row_grid[valid]
        # Little-endian byte layout 12..15: [intensity, return_type=0, ch_lo,
        # ch_hi] == int32 word (intensity | channel << 16), bit-cast to float32
        # so it can ride in the same (N, 4) float32 record tensor.
        word = intensity_u8 | (channel << 16)
        rec = torch.cat([xyz, word.view(torch.float32).unsqueeze(1)], dim=1)
        return rec.contiguous().cpu().numpy().view(np.uint8).reshape(-1), n

    def panorama_to_range_image(
        self,
        panorama: dict[str, torch.Tensor],
        *,
        drop_threshold: float = 0.5,
        alpha_threshold: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Convert a rendered panorama into a dense structured range image.

        Unlike :meth:`panorama_to_point_cloud` (which drops invalid cells and
        returns a sparse cloud), this keeps the full ``(H, W)`` grid so it can
        be encoded directly into per-channel / per-azimuth LiDAR packets. The
        same validity gate is applied, exposed as a boolean mask; invalid
        cells keep their raw distance but should be treated as "no return".

        Everything is returned **on the render device** (no host transfer) so a
        GPU consumer — e.g. :func:`splatsim.hils.build_frame_tensor` — can
        encode packets without a per-frame round-trip of the grids.

        Args:
            panorama: Output of :meth:`render`.
            drop_threshold: sigmoid(raydrop_logit) above this drops the sample.
            alpha_threshold: Minimum alpha coverage to keep a sample.

        Returns:
            ``{"distance", "intensity", "valid", "azimuths", "elevations"}`` —
            device tensors. ``distance`` / ``intensity`` / ``valid`` are
            ``(H, W)`` (``float32`` / ``float32`` / ``bool``); ``azimuths`` is
            ``(W,)`` and ``elevations`` is ``(H,)`` in radians.
        """
        valid = self._validity_mask(
            panorama, drop_threshold=drop_threshold, alpha_threshold=alpha_threshold
        )

        return {
            "distance": panorama["distance"].detach(),
            "intensity": panorama["intensity"].detach(),
            "valid": valid.detach(),
            "azimuths": self._azimuths,
            "elevations": self._elevs,
        }


__all__ = [
    "DEFAULT_RAYDROP_LOGIT",
    "LidarRenderer",
    "LidarSensorSpec",
    "elevations_rad",
    "is_known_sensor",
    "render_lidar_panorama",
    "sensor_to_base_4x4",
]
