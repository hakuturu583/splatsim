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
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from splatsim import _lidar_cull_ext as _cuda_cull_ext
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


def render_lidar_panorama(
    *,
    means: torch.Tensor,  # (N, 3) world
    quats: torch.Tensor,  # (N, 4) wxyz, will be renormalised
    scales: torch.Tensor,  # (N, 3) post-exp (positive)
    opacities: torch.Tensor,  # (N,) post-sigmoid in [0, 1]
    intensity_sig: torch.Tensor,  # (N,) sigmoid(lidar_intensity_raw)
    raydrop_logit: torch.Tensor,  # (N,) raw lidar_raydrop_logit
    sensor_to_world: torch.Tensor,  # (4, 4) sweep-start pose, same device
    lidar_spec: LidarSensorSpec,
    sensor_to_world_end: torch.Tensor | None = None,  # (4, 4) sweep-end pose
    min_range_m: float = 0.3,
    max_range_m: float | None = 120.0,
    packed: bool = False,
    radius_clip: float = 0.0,
    frustum_cull: bool = True,
    cull_scale_sigmas: float = 3.0,
    elev_fov_cull: bool = True,
    with_ut: bool = True,
    with_eval3d: bool = True,
) -> dict[str, torch.Tensor]:
    """Run gsplat's lidar raster once for one sensor at one frame.

    Returns:
        ``{"alpha": (H, W), "distance": (H, W),
            "intensity": (H, W), "raydrop_logit": (H, W)}``

    All maps are on the same device as ``means``. ``distance`` is the
    alpha-weighted expected hit distance from the sensor origin (so it
    matches the canonical LiDAR "range" reading); the value is 0 where
    no Gaussians intersect that bin.

    Acceleration knobs
    ------------------
    ``min_range_m``/``max_range_m`` gate what gsplat needs to touch.
    They are also forwarded to gsplat as ``near_plane``/``far_plane``
    so Gaussians outside the sensor's usable shell are dropped inside
    the projection kernel. ``max_range_m=None`` disables the far cut.

    ``packed`` toggles gsplat's packed rasterization path. gsplat
    currently disallows packed whenever ``with_ut`` or ``with_eval3d``
    is on, and the ``RGB-Ed`` hit-distance render mode we use requires
    ``with_eval3d=True``. As a result ``packed=True`` is not usable
    with the default output and it stays off by default.

    ``radius_clip`` drops Gaussians whose projected 2D radius is below
    the given pixel count; useful for skipping sub-pixel dust.

    ``frustum_cull`` runs a cheap Python-side spherical-shell test
    (``|means - sensor_pos| ± sigmas * max(scale)``) that trims the
    input tensors before they reach gsplat. This is the biggest
    single-source win for driving-scale scenes, where most Gaussians
    live far outside the sensor's usable range.

    ``elev_fov_cull`` layers a splatAD-style vertical-FOV test on top
    of the radial shell (sensor-frame elevation ± ``sigmas * scale /
    dist``). Gaussians whose entire angular extent falls above or
    below the sensor's row-elevation range can never touch a ray, so
    dropping them shrinks what gsplat needs to project. Cheap
    (one 3×N matmul + an ``atan2``) and orthogonal to the radial cull.

    ``with_ut``/``with_eval3d`` toggle gsplat's unscented-transform
    projection and 3D-evaluated blending. Both default to True (the
    correct choice under the spherical lidar camera model). Turn them
    off in exchange for ``packed=True`` when speed matters more than
    covariance fidelity.

    ``sensor_to_world_end`` enables rolling-shutter (motion-during-sweep)
    rendering: ``sensor_to_world`` is the pose at the first azimuth column and
    ``sensor_to_world_end`` the pose at the last, with gsplat interpolating the
    sensor pose across the spin. This reproduces the skew a real spinning LiDAR
    accumulates while the vehicle moves (and, conversely, lets a consumer feed
    de-skewing poses). Requires ``with_ut=True``. When ``None`` (default) the
    whole panorama is rendered from the single ``sensor_to_world`` pose
    (``RollingShutterType.GLOBAL``), i.e. the previous behaviour.
    """
    import gsplat

    device = means.device

    if sensor_to_world_end is not None and not with_ut:
        raise ValueError(
            "sensor_to_world_end (rolling shutter) requires with_ut=True; "
            "gsplat's unscented-transform path is the only one that interpolates "
            "the per-column sensor pose."
        )

    if frustum_cull and means.shape[0] > 0:
        if elev_fov_cull:
            sin_min, cos_min, sin_max, cos_max = _lidar_elev_fov_sincos(
                lidar_spec.sensor_type,
                lidar_spec.el_lo_rad,
                lidar_spec.el_hi_rad,
                lidar_spec.n_rows_uniform,
            )
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

    coeffs = lidar_spec.coeffs(device)
    H, W = coeffs.n_rows, coeffs.n_columns

    if means.shape[0] == 0:
        zero = torch.zeros((H, W), dtype=torch.float32, device=device)
        return {
            "alpha": zero.clone(),
            "distance": zero.clone(),
            "intensity": zero.clone(),
            "raydrop_logit": torch.full_like(zero, DEFAULT_RAYDROP_LOGIT),
        }

    # gsplat rasterization expects (..., C, 4, 4) view matrices. Use
    # world_to_sensor (= inv(sensor_to_world)) for the single "camera".
    sensor_to_world_f32 = sensor_to_world.to(device=device, dtype=torch.float32)
    viewmats = _rigid_inverse_4x4(sensor_to_world_f32).unsqueeze(0)  # (C=1, 4, 4)

    # Rolling shutter: interpolate sensor pose start->end across the azimuth spin.
    # Our columns sweep +pi -> -pi (col 0 -> col W-1), i.e. left-to-right.
    viewmats_rs = None
    rolling_shutter = None
    if sensor_to_world_end is not None:
        from gsplat.cuda._wrapper import RollingShutterType

        end_f32 = sensor_to_world_end.to(device=device, dtype=torch.float32)
        viewmats_rs = _rigid_inverse_4x4(end_f32).unsqueeze(0)
        rolling_shutter = RollingShutterType.ROLLING_LEFT_TO_RIGHT

    Ks = _lidar_intrinsics(int(H), int(W), str(device))

    # Pack (intensity_sig, raydrop_logit) into a 2-channel ``colors``.
    # render_mode='RGB-Ed' returns (intensity, raydrop_logit, distance)
    # as the trailing channel + an alpha map.
    colors = torch.stack(
        [intensity_sig.float(), raydrop_logit.float()], dim=-1
    ).unsqueeze(0)  # (C=1, N, 2)
    quats_n = torch.nn.functional.normalize(quats.float(), p=2, dim=-1)

    # gsplat requires global_z_order=True when with_ut is off.
    rast_kwargs: dict = dict(
        means=means.float(),
        quats=quats_n,
        scales=scales.float(),
        opacities=opacities.float(),
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        camera_model="lidar",
        lidar_coeffs=coeffs,
        with_ut=bool(with_ut),
        with_eval3d=bool(with_eval3d),
        global_z_order=not bool(with_ut),
        render_mode="RGB-Ed",
        packed=bool(packed),
        radius_clip=float(radius_clip),
        near_plane=float(min_range_m),
    )
    if max_range_m is not None:
        rast_kwargs["far_plane"] = float(max_range_m)
    if viewmats_rs is not None:
        rast_kwargs["viewmats_rs"] = viewmats_rs
        rast_kwargs["rolling_shutter"] = rolling_shutter

    render_colors, render_alphas, _meta = gsplat.rasterization(**rast_kwargs)
    rc = render_colors[0]  # (H, W, 3) = intensity, raydrop_logit, distance
    alpha = render_alphas[0, ..., 0]  # (H, W)
    return {
        "alpha": alpha,
        "distance": rc[..., 2],
        "intensity": rc[..., 0],
        "raydrop_logit": rc[..., 1],
    }


# ── splatsim integration wrapper ────────────────────────────────────

# Luminance weights for SH-derived intensity fallback (Rec. 709).
_LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


def _sensor_row_elevations(spec: LidarSensorSpec) -> torch.Tensor:
    """Return the (H,) descending elevation table used by ``spec``.

    Mirrors the logic in :func:`_build_lidar_coeffs` (known-sensor lookup vs
    uniform linspace fallback) so consumers can reconstruct per-pixel
    directions without touching gsplat internals.
    """
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
        frustum_cull: bool = True,
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
            "intensity": zero.clone(),
            "raydrop_logit": torch.full_like(zero, DEFAULT_RAYDROP_LOGIT),
        }

    def render(
        self,
        base_to_world: torch.Tensor,
        *,
        scene: "Scene",
        base_to_world_end: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Render one LiDAR panorama.

        Args:
            base_to_world: (4, 4) float32 tensor. Ego/base pose at the start of
                the azimuth sweep (the whole panorama when no end pose is given).
            scene: The splatsim Scene providing Gaussians.
            base_to_world_end: Optional (4, 4) ego/base pose at the end of the
                sweep. When given, enables rolling-shutter (motion-during-sweep)
                rendering — gsplat interpolates the sensor pose across the spin.

        Returns:
            ``{"alpha", "distance", "intensity", "raydrop_logit"}`` — each an
            (H, W) float32 tensor. ``H`` = sensor row count,
            ``W`` = ``sensor_spec.n_columns``.
        """
        sensor_to_world = base_to_world.to(self.device) @ self._s2b_t
        sensor_to_world_end = None
        if base_to_world_end is not None:
            sensor_to_world_end = base_to_world_end.to(self.device) @ self._s2b_t
        cam_pos = sensor_to_world[:3, 3].detach()

        tensor_list = scene.collect_tensors(cam_pos)
        if not tensor_list:
            return self._empty_panorama()

        sh_degrees = {t.sh_degree for t in tensor_list}
        if len(sh_degrees) != 1:
            raise ValueError(f"Mixed SH degrees across scene sources: {sh_degrees}")

        means_list, quats_list, scales_list, opacities_list = [], [], [], []
        intensity_list, raydrop_list = [], []
        for t in tensor_list:
            # Hard-exclude appearance-only / far-field Gaussians (lidar_mask == False)
            # from the LiDAR geometry pass BEFORE resolving attrs, so the sigmoid /
            # luminance fallback only runs over kept Gaussians. __getitem__ filters
            # every field (geometry + intensity_raw/raydrop_logit) consistently, so it
            # also carries any future per-Gaussian field. ``None`` (or the
            # ``ignore_lidar_mask`` override) keeps every Gaussian.
            if not self.ignore_lidar_mask and t.lidar_mask is not None:
                t = t[t.lidar_mask]
            i_sig, r_logit = _resolve_lidar_attrs(t)
            means_list.append(t.means)
            quats_list.append(t.quats)
            scales_list.append(t.scales)
            opacities_list.append(t.opacities)
            intensity_list.append(i_sig)
            raydrop_list.append(r_logit)

        means = torch.cat(means_list, dim=0).to(self.device)
        quats = torch.cat(quats_list, dim=0).to(self.device)
        scales = torch.cat(scales_list, dim=0).to(self.device)
        opacities = torch.cat(opacities_list, dim=0).to(self.device)
        intensity_sig = torch.cat(intensity_list, dim=0)
        raydrop_logit = torch.cat(raydrop_list, dim=0)

        # Every Gaussian masked out of the LiDAR pass -> zero-output panorama.
        if means.shape[0] == 0:
            return self._empty_panorama()

        return render_lidar_panorama(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            intensity_sig=intensity_sig,
            raydrop_logit=raydrop_logit,
            sensor_to_world=sensor_to_world,
            lidar_spec=self.sensor_spec,
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

        el_grid = self._elevs[:, None].expand(-1, self.n_columns)  # (H, W)
        az_grid = self._azimuths[None, :].expand(self.n_rows, -1)  # (H, W)
        row_grid = torch.arange(self.n_rows, device=distance.device)[:, None].expand(
            -1, self.n_columns
        )  # (H, W) ring / laser-beam index

        cos_el = torch.cos(el_grid)
        x = distance * cos_el * torch.cos(az_grid)
        y = distance * cos_el * torch.sin(az_grid)
        z = distance * torch.sin(el_grid)

        xyz = torch.stack([x, y, z], dim=-1)[valid]
        intensity_valid = intensity[valid]
        channel_valid = row_grid[valid]

        return {
            "xyz": xyz.detach().cpu().numpy().astype(np.float32),
            "intensity": intensity_valid.detach().cpu().numpy().astype(np.float32),
            "channel": channel_valid.detach().cpu().numpy().astype(np.uint16),
        }

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
