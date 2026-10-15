"""Sector streaming: render a LiDAR revolution as azimuth wedges.

Two contracts are pinned here:

* **Azimuth-wedge cull parity** — the fused CUDA cull's wedge test matches the
  PyTorch fallback bit-for-bit, including wedges touching the ±180° seam.
* **Sector equivalence** — concatenating the S sector renders reproduces the
  full-frame panorama EXACTLY when the pose is static (the tile grid stays the
  full ring; a sector only rasterizes its own tile columns), and closely under
  translation (rolling shutter per sector vs one whole-sweep pose pair are
  different — per-sector is the better — approximations, so only near-equality
  can be asserted).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ── azimuth-wedge cull parity ────────────────────────────────────────────


@cuda
@pytest.mark.parametrize("az_center", [0.0, 2.0, math.pi - 0.05, -math.pi + 0.05])
@pytest.mark.parametrize("use_elev", [False, True])
def test_azimuth_wedge_cull_cuda_matches_pytorch(
    az_center: float, use_elev: bool
) -> None:
    from splatsim import lidar_renderer as lr
    from splatsim._lidar_cull_ext import is_available

    if not is_available():
        pytest.skip("CUDA cull extension unavailable")

    g = torch.Generator(device="cuda").manual_seed(0)
    n = 200_000
    means = (
        (torch.rand(n, 3, device="cuda", generator=g) * 2 - 1) * 300.0
    ).contiguous()
    scales = (0.05 + torch.rand(n, 3, device="cuda", generator=g) * 0.5).contiguous()

    s2w = torch.eye(4, device="cuda")
    s2w[:3, 3] = torch.tensor([1.0, -2.0, 1.5], device="cuda")

    kwargs: dict[str, Any] = dict(
        means=means,
        scales=scales,
        sensor_to_world=s2w,
        min_range_m=0.5,
        max_range_m=150.0,
        cull_scale_sigmas=3.0,
        elev_fov_cull=use_elev,
        sin_min=math.sin(-0.4),
        cos_min=math.cos(-0.4),
        sin_max=math.sin(0.3),
        cos_max=math.cos(0.3),
        azim_cull=True,
        az_center_rad=az_center,
        az_halfwidth_rad=math.pi / 8,
        az_pad_rad=0.01,
    )
    keep_cuda = lr._lidar_cull_keep(**kwargs)
    lr._USE_CUDA_CULL = False
    try:
        keep_ref = lr._lidar_cull_keep(**kwargs)
    finally:
        lr._USE_CUDA_CULL = True
    assert torch.equal(keep_cuda, keep_ref)
    # The wedge must actually cull (π/8 of the circle plus margins).
    assert keep_cuda.sum() < n // 2


@cuda
def test_azimuth_wedge_keeps_far_side_of_seam() -> None:
    """A wedge centred at +π must keep Gaussians just across the -π seam."""
    from splatsim import lidar_renderer as lr

    # Two Gaussians a hair on either side of the seam, one at the wedge's
    # antipode.
    means = torch.tensor(
        [
            [-50.0, 0.5, 0.0],  # az ≈ +π - ε  (inside)
            [-50.0, -0.5, 0.0],  # az ≈ -π + ε (inside via wrap)
            [50.0, 0.0, 0.0],  # az = 0 (antipode, outside)
        ],
        device="cuda",
    )
    scales = torch.full((3, 3), 0.1, device="cuda")
    keep = lr._lidar_cull_keep(
        means=means,
        scales=scales,
        sensor_to_world=torch.eye(4, device="cuda"),
        min_range_m=0.1,
        max_range_m=None,
        cull_scale_sigmas=3.0,
        elev_fov_cull=False,
        sin_min=0.0,
        cos_min=1.0,
        sin_max=0.0,
        cos_max=1.0,
        azim_cull=True,
        az_center_rad=math.pi,
        az_halfwidth_rad=math.pi / 8,
        az_pad_rad=0.01,
    )
    assert keep.tolist() == [True, True, False]


# ── sector geometry (CPU) ────────────────────────────────────────────────


def _spec(n_columns: int = 128, n_rows: int = 8):
    from splatsim.lidar_renderer import LidarSensorSpec

    return LidarSensorSpec(
        name="t",
        sensor_type="",
        n_columns=n_columns,
        s2b=np.eye(4),
        el_lo_rad=-0.35,
        el_hi_rad=0.26,
        n_rows_uniform=n_rows,
        spinning_frequency_hz=10.0,
    )


def test_sector_geometry_slices_the_full_grid() -> None:
    from splatsim.lidar_renderer import _panorama_geometry

    spec = _spec(n_columns=128)
    full = _panorama_geometry(spec, "cpu")
    s_count = 4
    secs = [_panorama_geometry(spec, "cpu", (k, s_count)) for k in range(s_count)]

    assert torch.equal(torch.cat([s.azs_cw for s in secs]), full.azs_cw)
    assert torch.equal(torch.cat([s.dirs for s in secs], dim=1), full.dirs)
    # Sector roll times are relative to the sector's own mid-window pose;
    # shifting each by its window midpoint must reproduce the full timeline.
    sweep = full.window_s
    rt = torch.cat(
        [
            torch.flip(s.roll_time_s, [0]) + ((2 * k + 1) / (2 * s_count) - 0.5) * sweep
            for k, s in enumerate(secs)
        ]
    )
    assert torch.allclose(rt, torch.flip(full.roll_time_s, [0]), atol=1e-7)
    assert secs[0].window_s == pytest.approx(sweep / s_count)
    # Ascending-grid tile offsets: sector S-1 (CW end = ascending start) is 0.
    widths = [s.azs_cw.shape[0] for s in secs]
    assert all(w == 128 // s_count for w in widths)
    assert [s.tile_col_offset for s in reversed(secs)] == [
        i * (128 // s_count) // 16 for i in range(s_count)
    ]


def test_uneven_sector_counts_split_on_tile_boundaries() -> None:
    """3 sectors over 8 tiles: widths differ by one tile but still slice the
    full grid exactly (the real rig has 3600-column sensors, where no sector
    count divides into equal tile-aligned wedges)."""
    from splatsim.lidar_renderer import _panorama_geometry

    spec = _spec(n_columns=128)
    full = _panorama_geometry(spec, "cpu")
    secs = [_panorama_geometry(spec, "cpu", (k, 3)) for k in range(3)]
    assert [s.azs_cw.shape[0] for s in secs] == [32, 48, 48]
    assert all(s.azs_cw.shape[0] % 16 == 0 for s in secs)
    assert torch.equal(torch.cat([s.azs_cw for s in secs]), full.azs_cw)
    # Roll times must still be the full timeline shifted by each POSE window's
    # midpoint (k/3 fractions), even though the columns lean past the window.
    sweep = full.window_s
    rt = torch.cat(
        [
            torch.flip(s.roll_time_s, [0]) + ((2 * k + 1) / 6 - 0.5) * sweep
            for k, s in enumerate(secs)
        ]
    )
    assert torch.allclose(rt, torch.flip(full.roll_time_s, [0]), atol=1e-7)
    # The kernel's extent expansion must cover the actual roll-time span.
    for s in secs:
        assert 2 * s.roll_time_s.abs().max().item() <= s.rs_time_s + 1e-6


def test_sector_rendering_needs_tile_aligned_columns() -> None:
    from splatsim.lidar_renderer import _panorama_geometry

    with pytest.raises(ValueError, match="multiple"):
        _panorama_geometry(_spec(n_columns=100), "cpu", (0, 2))
    with pytest.raises(ValueError, match="exceeds"):
        _panorama_geometry(_spec(n_columns=128), "cpu", (0, 9))


# ── sector render equivalence (GPU) ──────────────────────────────────────


def _gaussians(n: int = 20_000) -> dict[str, Any]:
    g = torch.Generator(device="cuda").manual_seed(0)
    return dict(
        means=(torch.rand(n, 3, device="cuda", generator=g) - 0.5) * 120.0,
        quats=torch.randn(n, 4, device="cuda", generator=g),
        scales=torch.rand(n, 3, device="cuda", generator=g) * 0.3 + 0.02,
        opacities=torch.rand(n, device="cuda", generator=g) * 0.8 + 0.1,
        intensity_sig=torch.rand(n, device="cuda", generator=g),
        raydrop_logit=torch.randn(n, device="cuda", generator=g) - 2.0,
    )


@cuda
@pytest.mark.parametrize("sector_mask", [False, True], ids=["compact", "mask"])
def test_static_sector_concat_matches_full_render_exactly(
    sector_mask: bool, monkeypatch
) -> None:
    """Both cull applications — compaction (default) and the mask-based
    projection (SPLATSIM_LIDAR_SECTOR_MASK=1) — must reproduce the full
    render exactly."""
    from splatsim import lidar_renderer as lr
    from splatsim.lidar_renderer import render_lidar_panorama

    monkeypatch.setattr(lr, "_SECTOR_MASK", sector_mask)
    spec = _spec(n_columns=512, n_rows=32)
    kw: dict[str, Any] = dict(
        **_gaussians(), lidar_spec=spec, min_range_m=0.5, max_range_m=100.0
    )
    s2w = torch.eye(4, device="cuda")
    s2w[2, 3] = 1.5

    full = render_lidar_panorama(sensor_to_world=s2w, **kw)
    s_count = 8
    parts = [
        render_lidar_panorama(sensor_to_world=s2w, sector=(k, s_count), **kw)
        for k in range(s_count)
    ]
    for key in ("alpha", "distance", "points", "intensity", "raydrop_logit"):
        cat = torch.cat([p[key] for p in parts], dim=1)
        assert torch.equal(cat, full[key]), key


@cuda
def test_translating_sector_concat_stays_close_to_full_render() -> None:
    from splatsim.lidar_renderer import render_lidar_panorama

    spec = _spec(n_columns=512, n_rows=32)
    kw: dict[str, Any] = dict(
        **_gaussians(), lidar_spec=spec, min_range_m=0.5, max_range_m=100.0
    )
    sweep = 0.1
    vel = torch.tensor([8.0, 2.0, 0.0], device="cuda")

    def pose_at(f: float) -> torch.Tensor:
        m = torch.eye(4, device="cuda")
        m[:3, 3] = torch.tensor([0.0, 0.0, 1.5], device="cuda") + vel * (f * sweep)
        return m

    s_count = 8
    full = render_lidar_panorama(
        sensor_to_world=pose_at(0.0), sensor_to_world_end=pose_at(1.0), **kw
    )
    parts = [
        render_lidar_panorama(
            sensor_to_world=pose_at(k / s_count),
            sensor_to_world_end=pose_at((k + 1) / s_count),
            sector=(k, s_count),
            **kw,
        )
        for k in range(s_count)
    ]
    d = (torch.cat([p["distance"] for p in parts], dim=1) - full["distance"]).abs()
    q = d.flatten().sort().values
    # Both are approximations of the same swept scan (per-sector poses vs one
    # whole-sweep pair); they must agree everywhere except a handful of cells
    # sitting on depth discontinuities, where a sub-column azimuth shift flips
    # which surface the median lands on.
    assert q[int(0.999 * q.numel())].item() < 0.05
    a = (torch.cat([p["alpha"] for p in parts], dim=1) - full["alpha"]).abs()
    assert a.mean().item() < 5e-3


@cuda
def test_depth_lanes_match_serial_rasterization(monkeypatch) -> None:
    """The depth-parallel rasterizer (16 lanes/pixel, associative composition)
    must reproduce the serial kernel: alpha and median depth measure
    bit-identical (the median is an exact selection; the thresholds are
    resolved by re-walking the crossing lane with the serial logic), features
    to float epsilon (a lane partial is scaled by the prefix transmittance,
    which rounds differently from the serial running product)."""
    from splatsim import lidar_renderer as lr
    from splatsim.lidar_renderer import render_lidar_panorama

    spec = _spec(n_columns=512, n_rows=32)
    kw: dict[str, Any] = dict(
        **_gaussians(), lidar_spec=spec, min_range_m=0.5, max_range_m=100.0
    )
    start = torch.eye(4, device="cuda")
    start[2, 3] = 1.5
    end = start.clone()
    end[:3, 3] += torch.tensor([0.8, 0.2, 0.0], device="cuda")

    outs = {}
    for lanes in (False, True):
        monkeypatch.setattr(lr, "_DEPTH_LANES", lanes)
        outs[lanes] = {
            "static": render_lidar_panorama(sensor_to_world=start, **kw),
            "rolling": render_lidar_panorama(
                sensor_to_world=start, sensor_to_world_end=end, **kw
            ),
        }
    for mode in ("static", "rolling"):
        a, b = outs[False][mode], outs[True][mode]
        assert torch.equal(a["alpha"], b["alpha"]), mode
        assert torch.equal(a["distance"], b["distance"]), mode
        for key in ("intensity", "raydrop_logit"):
            assert torch.allclose(a[key], b[key], atol=1e-5), (mode, key)
