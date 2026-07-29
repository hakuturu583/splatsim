"""Render-sensor construction + the sim's range/FOV coverage envelope.

One :class:`Lidar` is built per USDZ rig LiDAR (mount + beam table from the
scene's own calibration) and every sensor is rendered + unioned to mirror the
GT ``LIDAR_CONCAT``. :func:`coverage_mask` marks which GT points the sim could
physically return, which the range-aware Chamfer metric uses.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from splatsim.lidar_renderer import (
    LidarRenderer,
    build_lidar_sensors_from_config,
)


class Lidar:
    """A render sensor: its renderer, sensor->base mount, and coverage envelope.

    ``min_range``/``max_range`` and ``el_min``/``el_max`` (radians) describe the
    range shell + vertical FOV the sim can return, used by :func:`coverage_mask`
    to build the range-aware Chamfer metric.
    """

    __slots__ = (
        "name",
        "renderer",
        "s2b",
        "min_range",
        "max_range",
        "el_min",
        "el_max",
    )

    def __init__(
        self,
        name: str,
        renderer: LidarRenderer,
        s2b: np.ndarray,
        el_min: float,
        el_max: float,
    ) -> None:
        self.name = name
        self.renderer = renderer
        self.s2b = s2b
        self.min_range = renderer.min_range_m
        self.max_range = (
            renderer.max_range_m if renderer.max_range_m is not None else float("inf")
        )
        self.el_min = el_min
        self.el_max = el_max


def coverage_mask(gt_base: np.ndarray, lidars: list[Lidar]) -> np.ndarray:
    """Boolean mask of GT points the LiDAR simulation could actually return.

    A GT point (in base_link) is *coverable* if, expressed in some render
    sensor's frame, it lies within that sensor's range shell ``[min, max]`` and
    its vertical FOV ``[el_min, el_max]`` (azimuth is a full 360 deg spin, so it
    imposes no constraint). Points outside every sensor's envelope -- e.g. beyond
    the sim's ``max_range`` or above/below the beam fan -- can never be rendered,
    so excluding them isolates reconstruction quality from sensor-model limits.
    """
    if gt_base.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    mask = np.zeros(gt_base.shape[0], dtype=bool)
    for ld in lidars:
        # p_sensor = R^T (p - t), with s2b = [R | t] (sensor -> base).
        p = (gt_base - ld.s2b[:3, 3]) @ ld.s2b[:3, :3]
        rng = np.linalg.norm(p, axis=1)
        el = np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1]))
        mask |= (
            (rng >= ld.min_range)
            & (rng <= ld.max_range)
            & (el >= ld.el_min)
            & (el <= ld.el_max)
        )
    return mask


def _spec_el_bounds(spec) -> tuple[float, float]:
    """(min, max) beam elevation in radians for a spec.

    The explicit per-beam table wins over the uniform-span fallback -- the same
    precedence :mod:`splatsim.lidar_renderer` uses to build the beam pattern.
    """
    if spec.row_elevations_rad:
        return float(min(spec.row_elevations_rad)), float(max(spec.row_elevations_rad))
    return float(spec.el_lo_rad), float(spec.el_hi_rad)


def _estimate_gt_azimuth_columns(
    gt_base: np.ndarray, s2b: np.ndarray, el_min_rad: float, el_max_rad: float
) -> int | None:
    """Estimate a spinning LiDAR's azimuth column count from GT point density.

    GT points (base_link) are expressed in the sensor frame and restricted to its
    vertical FOV; within thin elevation rings the fundamental azimuth step is the
    20th-percentile gap between consecutive returns (robust to occlusion gaps and
    the rare dual return). ``round(360 deg / step)`` is the samples per revolution.
    Returns None when the GT is too sparse in this FOV to estimate (e.g. a
    narrow-FOV auxiliary LiDAR that the concat barely populates).
    """
    p = (gt_base - s2b[:3, 3]) @ s2b[:3, :3]  # base -> sensor (R orthonormal)
    el = np.degrees(np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1])))
    lo, hi = np.degrees(el_min_rad), np.degrees(el_max_rad)
    in_fov = (el >= lo) & (el <= hi)
    el = el[in_fov]  # work on the in-FOV subset only
    az = np.degrees(np.arctan2(p[in_fov, 1], p[in_fov, 0]))
    ring_steps: list[float] = []
    for center in np.linspace(lo + 2.0, hi - 2.0, 40):
        ring = np.abs(el - center) < 0.04
        if ring.sum() < 300:
            continue
        gaps = np.diff(np.sort(az[ring]))
        # Exclude dual-return duplicates (~0) and occlusion gaps (>1 deg); what
        # remains is the firing interval within the ring.
        gaps = gaps[(gaps > 0.01) & (gaps < 1.0)]
        if gaps.size > 50:
            ring_steps.append(float(np.median(gaps)))
    if not ring_steps:
        return None
    # The true grid step is the finest consistent ring (occlusion only widens
    # gaps, never narrows them); the 10th percentile is a robust "finest".
    step = float(np.percentile(ring_steps, 10))
    return int(round(360.0 / step))


def _resolve_n_columns(args, specs, gt_base) -> int | None:
    """Resolve one azimuth column count applied to every render sensor (or None).

    ``--n-columns``: an integer overrides directly; ``usdz`` keeps each sensor's
    stored value (``None``); ``auto`` (default) derives one resolution from the
    GT density. Because ``LIDAR_CONCAT`` carries no per-sensor labels, a single
    value measured from the densest (highest-beam) LiDAR is applied to all --
    per-sensor estimates from the merged cloud are unreliable. Logs its choice.
    """
    keep_msg = "[n_columns] keeping per-sensor USDZ value"
    mode = args.n_columns
    if mode == "usdz":
        print(f"{keep_msg} (usdz)")
        return None
    if mode != "auto":
        print(f"[n_columns] {int(mode)} (override)")
        return int(mode)
    if gt_base is None:
        print(f"{keep_msg} (no GT)")
        return None
    ref = max(specs, key=lambda s: len(s.row_elevations_rad) or s.n_rows_uniform)
    est = _estimate_gt_azimuth_columns(gt_base, ref.s2b, *_spec_el_bounds(ref))
    if est is None:
        print(f"{keep_msg} (GT too sparse)")
        return None
    print(f"[n_columns] {est} (gt-density from {ref.name})")
    return est


def build_lidar_renderers(config, args, device, gt_base=None) -> list[Lidar]:
    """Build one renderer per USDZ rig LiDAR, to be aggregated at eval time.

    The T4 GT is ``LIDAR_CONCAT`` -- the point cloud of *all* physical LiDARs
    merged. To compare like-for-like the render side must likewise enable every
    LiDAR the scene knows about and union their scans. Each sensor's mount
    (height / orientation) and per-beam table come from the scene USDZ's own rig
    calibration (``config.lidar_sensors``, via the production
    ``build_lidar_sensors_from_config`` path); the T4 ``LIDAR_CONCAT``
    calibrated_sensor sits at base_link (ground) and must NOT be used as a mount.

    Azimuth resolution (``n_columns``) is resolved per :func:`_resolve_n_columns`
    -- by default measured from the GT density, since the scene's stored value is
    typically a library default (e.g. 2048) rather than the real sensor's.

    ``--lidar-name`` (comma-separated) restricts to a subset; the default is all.

    Returns a list of :class:`Lidar`.
    """
    sensors = list(config.lidar_sensors or [])
    if not sensors:
        raise SystemExit(
            "Scene USDZ carries no LiDAR calibration (config.lidar_sensors is "
            "empty); cannot determine the sensor mounts. Use a scene exported "
            "with rig LiDAR extrinsics."
        )
    if args.lidar_name:
        wanted = {n.strip() for n in args.lidar_name.split(",") if n.strip()}
        sensors = [c for c in sensors if c.name in wanted]
        if not sensors:
            avail = [c.name for c in (config.lidar_sensors or [])]
            raise SystemExit(
                f"--lidar-name {args.lidar_name!r} matched none of {avail}"
            )

    specs = build_lidar_sensors_from_config(sensors)
    n_columns = _resolve_n_columns(args, specs, gt_base)

    out: list[Lidar] = []
    for cfg_sensor, spec in zip(sensors, specs):
        el_min, el_max = _spec_el_bounds(spec)
        if n_columns is not None:
            spec = dataclasses.replace(spec, n_columns=n_columns)

        min_range = (
            args.min_range if args.min_range is not None else cfg_sensor.min_range_m
        )
        max_range = (
            args.max_range if args.max_range is not None else cfg_sensor.max_range_m
        )
        renderer = LidarRenderer(
            spec,
            device=device,
            min_range_m=float(min_range),
            max_range_m=float(max_range),
        )
        out.append(
            Lidar(spec.name, renderer, spec.s2b.astype(np.float32), el_min, el_max)
        )
    return out
