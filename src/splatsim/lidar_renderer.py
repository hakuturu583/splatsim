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


_TABLES_RAD: dict[str, tuple[float, ...]] = {
    "OT128": tuple(math.radians(d) for d in _OT128_DEG),
    "XT32": tuple(math.radians(d) for d in _XT32_DEG),
}


def elevations_rad(sensor_type: str) -> tuple[float, ...]:
    """Return the descending beam-elevation table for ``sensor_type``."""
    return _TABLES_RAD[sensor_type]


def is_known_sensor(sensor_type: str) -> bool:
    return sensor_type in _TABLES_RAD


# ── Sensor pose / spec ──────────────────────────────────────────────


def _quat_wxyz_to_rotmat_np(qwxyz: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = (float(c) for c in qwxyz)
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
            ],
            [
                2 * (qx * qy + qw * qz),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qw * qx),
            ],
            [
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _rpy_deg_to_rotmat_np(rpy_deg: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (math.radians(float(c)) for c in rpy_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def sensor_to_base_4x4(
    translation: Sequence[float],
    rotation_wxyz: Sequence[float],
) -> np.ndarray:
    """4×4 sensor→base_link rigid transform from a YAML pose entry."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_wxyz_to_rotmat_np(rotation_wxyz)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


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

    def coeffs(self, device: torch.device):
        """Build the gsplat ``...ParametersExt`` once per device."""
        return _build_lidar_coeffs(
            self.sensor_type,
            self.n_columns,
            self.spinning_frequency_hz,
            self.el_lo_rad,
            self.el_hi_rad,
            self.n_rows_uniform,
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
    if sensor_type in _TABLES_RAD:
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
        s2b = np.eye(4, dtype=np.float64)
        s2b[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
        if len(rotation) == 4:
            s2b[:3, :3] = _quat_wxyz_to_rotmat_np(rotation)
        elif len(rotation) == 3:
            s2b[:3, :3] = _rpy_deg_to_rotmat_np(rotation)
        else:
            raise ValueError(
                f"LiDAR sensor {getattr(s, 'name', '<unnamed>')}: "
                "rotation must be quaternion [w,x,y,z] or RPY [roll,pitch,yaw]"
            )
        out.append(
            LidarSensorSpec(
                name=str(s.name),
                sensor_type=str(getattr(s, "sensor_type", "") or ""),
                s2b=s2b,
                n_columns=int(getattr(s, "n_columns", 2048)),
                spinning_frequency_hz=float(getattr(s, "fps", 10.0)),
                n_rows_uniform=int(getattr(s, "n_rows", 128)),
            )
        )
    return out


# ── Rendering ───────────────────────────────────────────────────────


def render_lidar_panorama(
    *,
    means: torch.Tensor,  # (N, 3) world
    quats: torch.Tensor,  # (N, 4) wxyz, will be renormalised
    scales: torch.Tensor,  # (N, 3) post-exp (positive)
    opacities: torch.Tensor,  # (N,) post-sigmoid in [0, 1]
    intensity_sig: torch.Tensor,  # (N,) sigmoid(lidar_intensity_raw)
    raydrop_logit: torch.Tensor,  # (N,) raw lidar_raydrop_logit
    sensor_to_world: torch.Tensor,  # (4, 4) on the same device
    lidar_spec: LidarSensorSpec,
) -> dict[str, torch.Tensor]:
    """Run gsplat's lidar raster once for one sensor at one frame.

    Returns:
        ``{"alpha": (H, W), "distance": (H, W),
            "intensity": (H, W), "raydrop_logit": (H, W)}``

    All maps are on the same device as ``means``. ``distance`` is the
    alpha-weighted expected hit distance from the sensor origin (so it
    matches the canonical LiDAR "range" reading); the value is 0 where
    no Gaussians intersect that bin.
    """
    import gsplat

    device = means.device
    coeffs = lidar_spec.coeffs(device)
    # gsplat rasterization expects (..., C, 4, 4) view matrices. Use
    # world_to_sensor (= inv(sensor_to_world)) for the single "camera".
    world_to_sensor = torch.linalg.inv(sensor_to_world).to(
        device=device, dtype=torch.float32
    )
    viewmats = world_to_sensor.unsqueeze(0)  # (C=1, 4, 4)

    H, W = coeffs.n_rows, coeffs.n_columns
    focal = float(W)
    Ks = torch.tensor(
        [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    # Pack (intensity_sig, raydrop_logit) into a 2-channel ``colors``.
    # render_mode='RGB-Ed' returns (intensity, raydrop_logit, distance)
    # as the trailing channel + an alpha map.
    colors = torch.stack(
        [intensity_sig.float(), raydrop_logit.float()], dim=-1
    ).unsqueeze(0)  # (C=1, N, 2)
    quats_n = torch.nn.functional.normalize(quats.float(), p=2, dim=-1)
    render_colors, render_alphas, _meta = gsplat.rasterization(
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
        with_ut=True,
        with_eval3d=True,
        global_z_order=False,
        render_mode="RGB-Ed",
        packed=False,
    )
    rc = render_colors[0]  # (H, W, 3) = intensity, raydrop_logit, distance
    alpha = render_alphas[0, ..., 0]  # (H, W)
    return {
        "alpha": alpha,
        "distance": rc[..., 2],
        "intensity": rc[..., 0],
        "raydrop_logit": rc[..., 1],
    }


# ── splatsim integration wrapper ────────────────────────────────────

# Default raydrop_logit for Gaussians without a learned attribute.
# sigmoid(-6.0) ≈ 0.0025 → very low drop probability.
DEFAULT_RAYDROP_LOGIT: float = -6.0

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
    """

    def __init__(
        self,
        sensor_spec: LidarSensorSpec,
        *,
        device: torch.device | str,
        min_range_m: float = 0.3,
        max_range_m: float = 120.0,
    ) -> None:
        self.sensor_spec = sensor_spec
        self.device = torch.device(device)
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        # Precompute the (H,) elevation table once (used by point-cloud conv).
        self._elevs = _sensor_row_elevations(sensor_spec).to(self.device)
        self._azimuths = _sensor_column_azimuths(sensor_spec).to(self.device)
        # Prime the coeffs cache so the first render skips tile-assign cost.
        _ = sensor_spec.coeffs(self.device)

    @property
    def n_rows(self) -> int:
        return int(self._elevs.shape[0])

    @property
    def n_columns(self) -> int:
        return int(self.sensor_spec.n_columns)

    def render(
        self,
        base_to_world: torch.Tensor,
        *,
        scene: "Scene",
    ) -> dict[str, torch.Tensor]:
        """Render one LiDAR panorama.

        Args:
            base_to_world: (4, 4) float32 tensor. Ego/base pose in world.
            scene: The splatsim Scene providing Gaussians.

        Returns:
            ``{"alpha", "distance", "intensity", "raydrop_logit"}`` — each an
            (H, W) float32 tensor. ``H`` = sensor row count,
            ``W`` = ``sensor_spec.n_columns``.
        """
        s2b_t = torch.from_numpy(self.sensor_spec.s2b.astype(np.float32)).to(
            self.device
        )
        sensor_to_world = base_to_world.to(self.device) @ s2b_t
        cam_pos = sensor_to_world[:3, 3].detach()

        tensor_list = scene.collect_tensors(cam_pos)
        if not tensor_list:
            zero = torch.zeros(
                (self.n_rows, self.n_columns), dtype=torch.float32, device=self.device
            )
            return {
                "alpha": zero.clone(),
                "distance": zero.clone(),
                "intensity": zero.clone(),
                "raydrop_logit": torch.full_like(zero, DEFAULT_RAYDROP_LOGIT),
            }

        sh_degrees = {t.sh_degree for t in tensor_list}
        if len(sh_degrees) != 1:
            raise ValueError(f"Mixed SH degrees across scene sources: {sh_degrees}")

        means = torch.cat([t.means for t in tensor_list], dim=0).to(self.device)
        quats = torch.cat([t.quats for t in tensor_list], dim=0).to(self.device)
        scales = torch.cat([t.scales for t in tensor_list], dim=0).to(self.device)
        opacities = torch.cat([t.opacities for t in tensor_list], dim=0).to(self.device)

        intensity_list, raydrop_list = [], []
        for t in tensor_list:
            i_sig, r_logit = _resolve_lidar_attrs(t)
            intensity_list.append(i_sig)
            raydrop_list.append(r_logit)
        intensity_sig = torch.cat(intensity_list, dim=0)
        raydrop_logit = torch.cat(raydrop_list, dim=0)

        return render_lidar_panorama(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            intensity_sig=intensity_sig,
            raydrop_logit=raydrop_logit,
            sensor_to_world=sensor_to_world,
            lidar_spec=self.sensor_spec,
        )

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
            ``{"xyz": (N, 3) float32, "intensity": (N,) float32}``. Coordinates
            in sensor frame (+x forward, +y left, +z up — gsplat lidar convention).
        """
        distance = panorama["distance"]
        intensity = panorama["intensity"]
        alpha = panorama["alpha"]
        raydrop = torch.sigmoid(panorama["raydrop_logit"])

        valid = (
            (alpha > alpha_threshold)
            & (raydrop < drop_threshold)
            & (distance > self.min_range_m)
            & (distance < self.max_range_m)
        )

        el_grid = self._elevs[:, None].expand(-1, self.n_columns)  # (H, W)
        az_grid = self._azimuths[None, :].expand(self.n_rows, -1)  # (H, W)

        cos_el = torch.cos(el_grid)
        x = distance * cos_el * torch.cos(az_grid)
        y = distance * cos_el * torch.sin(az_grid)
        z = distance * torch.sin(el_grid)

        xyz = torch.stack([x, y, z], dim=-1)[valid]
        intensity_valid = intensity[valid]

        return {
            "xyz": xyz.detach().cpu().numpy().astype(np.float32),
            "intensity": intensity_valid.detach().cpu().numpy().astype(np.float32),
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
