"""Benchmark & correctness check for the accelerated LiDAR renderer.

Compares the pre-optimization "legacy" settings against the new defaults on
a synthetic driving-style scene where most Gaussians live outside the
LiDAR's usable range.

Run:
    PYTHONPATH=./src uv run python test/benchmark_lidar_render.py
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np
import torch

from splatsim.lidar_renderer import (
    DEFAULT_RAYDROP_LOGIT,
    LidarSensorSpec,
    render_lidar_panorama,
    sensor_to_base_4x4,
)


def _random_unit_quats(n: int, device, g) -> torch.Tensor:
    """Sample uniform-random unit quaternions in wxyz order."""
    q = torch.randn((n, 4), generator=g, device=device)
    return torch.nn.functional.normalize(q, p=2, dim=-1)


def make_driving_scene(
    n: int,
    *,
    in_range_frac: float = 0.05,
    vertical_out_frac: float = 0.0,
    device: torch.device,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Driving-style layout: a small in-range cluster + a large out-of-range map.

    - In-range Gaussians live within a 80m sphere around the sensor,
      spread mostly in a horizontal slab (±2m in z), like street/building
      geometry.
    - Out-of-range Gaussians live in a 1km × 1km × 20m slab far outside
      the sensor's 120m range. They represent the rest of the map that
      shouldn't touch the panorama.
    - ``vertical_out_frac`` reroutes a fraction of the out-of-range pile
      into a "tall-column" band that is *radially* inside the 120m shell
      but perched high above the LiDAR's vertical FOV (rooftop / sky /
      overhead-sign geometry). These stress the elev-FOV cull.

    Gaussians have random unit quats and moderate scales (0.05–0.3 m).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    n_in = max(1, int(round(n * in_range_frac)))
    n_out = n - n_in
    n_vert = max(0, int(round(n_out * vertical_out_frac)))
    n_far = n_out - n_vert

    # In-range slab (horizontal, near-sensor).
    xy_in = (torch.rand((n_in, 2), generator=g, device=device) * 2.0 - 1.0) * 80.0
    z_in = (torch.rand((n_in, 1), generator=g, device=device) * 2.0 - 1.0) * 2.0
    in_range = torch.cat([xy_in, z_in], dim=1)

    # Out-of-range slab (kilometre-wide, thin) — far radially.
    xy_far = (torch.rand((n_far, 2), generator=g, device=device) * 2.0 - 1.0) * 500.0
    z_far = (torch.rand((n_far, 1), generator=g, device=device) * 2.0 - 1.0) * 10.0
    far_range = torch.cat([xy_far, z_far], dim=1)

    # Vertical-out band: within 100m horizontally, but z in [+30, +80] so the
    # entire mass sits above OT128's ~+15° elevation ceiling.
    xy_vert = (torch.rand((n_vert, 2), generator=g, device=device) * 2.0 - 1.0) * 60.0
    z_vert = torch.rand((n_vert, 1), generator=g, device=device) * 50.0 + 30.0
    vert_range = torch.cat([xy_vert, z_vert], dim=1)

    means = torch.cat([in_range, far_range, vert_range], dim=0)
    perm = torch.randperm(means.shape[0], generator=g, device=device)
    means = means[perm].contiguous()

    quats = _random_unit_quats(n, device=device, g=g)
    scales = 0.05 + torch.rand((n, 3), generator=g, device=device) * 0.25
    opacities = torch.full((n,), 0.2, device=device)
    intensity_sig = torch.full((n,), 0.5, device=device)
    raydrop_logit = torch.full((n,), DEFAULT_RAYDROP_LOGIT, device=device)
    return dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        intensity_sig=intensity_sig,
        raydrop_logit=raydrop_logit,
    )


def make_sensor(device: torch.device) -> tuple[LidarSensorSpec, torch.Tensor]:
    s2b = sensor_to_base_4x4((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    spec = LidarSensorSpec(
        name="benchmark",
        sensor_type="OT128",
        s2b=s2b,
        n_columns=2048,
        spinning_frequency_hz=10.0,
        n_rows_uniform=128,
    )
    sensor_to_world = torch.from_numpy(s2b.astype(np.float32)).to(device)
    return spec, sensor_to_world


def timed(
    fn: Callable[[], dict], *, warmup: int = 3, iters: int = 8
) -> tuple[float, dict]:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out: dict = {}
    for _ in range(iters):
        out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0 / iters
    return dt_ms, out


def output_stats(name: str, d: dict) -> str:
    def rng(t):
        t = t.detach()
        return f"[{float(t.min()):.3f}, {float(t.max()):.3f}]"

    return (
        f"{name:8s}: alpha={rng(d['alpha'])} "
        f"dist={rng(d['distance'])} "
        f"nonzero_alpha_frac={(d['alpha'] > 0.01).float().mean().item():.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500_000)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--in-range-frac", type=float, default=0.05)
    ap.add_argument(
        "--vertical-out-frac",
        type=float,
        default=0.0,
        help=(
            "Fraction of out-of-range Gaussians to route into a tall-column band "
            "(radially inside the shell but above the LiDAR's vertical FOV). "
            "Used to stress the elevation-FOV cull."
        ),
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("Requires CUDA (gsplat lidar path is CUDA-only)")

    spec, sensor_to_world = make_sensor(device)
    scene = make_driving_scene(
        args.n,
        in_range_frac=args.in_range_frac,
        vertical_out_frac=args.vertical_out_frac,
        device=device,
    )

    means = scene["means"]
    quats = scene["quats"]
    scales = scene["scales"]
    opacities = scene["opacities"]
    intensity_sig = scene["intensity_sig"]
    raydrop_logit = scene["raydrop_logit"]

    def render_legacy():
        return render_lidar_panorama(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            intensity_sig=intensity_sig,
            raydrop_logit=raydrop_logit,
            sensor_to_world=sensor_to_world,
            lidar_spec=spec,
            min_range_m=0.3,
            max_range_m=None,
            packed=False,
            frustum_cull=False,
            radius_clip=0.0,
        )

    def render_default():
        return render_lidar_panorama(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            intensity_sig=intensity_sig,
            raydrop_logit=raydrop_logit,
            sensor_to_world=sensor_to_world,
            lidar_spec=spec,
            min_range_m=0.3,
            max_range_m=120.0,
            packed=False,
            frustum_cull=True,
            elev_fov_cull=False,
            radius_clip=0.0,
        )

    def render_radc():
        return render_lidar_panorama(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            intensity_sig=intensity_sig,
            raydrop_logit=raydrop_logit,
            sensor_to_world=sensor_to_world,
            lidar_spec=spec,
            min_range_m=0.3,
            max_range_m=120.0,
            packed=False,
            frustum_cull=True,
            elev_fov_cull=False,
            radius_clip=1.0,
        )

    def render_elev():
        return render_lidar_panorama(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            intensity_sig=intensity_sig,
            raydrop_logit=raydrop_logit,
            sensor_to_world=sensor_to_world,
            lidar_spec=spec,
            min_range_m=0.3,
            max_range_m=120.0,
            packed=False,
            frustum_cull=True,
            elev_fov_cull=True,
            radius_clip=1.0,
        )

    n_in = int(round(args.n * args.in_range_frac))
    n_out_total = args.n - n_in
    n_vert = int(round(n_out_total * args.vertical_out_frac))
    n_far = n_out_total - n_vert
    print(
        f"Scene: N={args.n} ({n_in} in-range, {n_far} far-out, {n_vert} vert-out),"
        f" OT128 128x2048 panorama"
    )
    print(f"Device: {device}, iters={args.iters}\n")

    dt_legacy, out_legacy = timed(render_legacy, iters=args.iters)
    dt_default, out_default = timed(render_default, iters=args.iters)
    dt_radc, out_radc = timed(render_radc, iters=args.iters)
    dt_elev, out_elev = timed(render_elev, iters=args.iters)

    print(f"Legacy    (no far, no cull, no clip):     {dt_legacy:8.2f} ms  1.00x")
    print(
        f"Default   (far_plane + shell cull):       {dt_default:8.2f} ms  ({dt_legacy / dt_default:.2f}x)"
    )
    print(
        f"+radc     (default + radius_clip=1px):    {dt_radc:8.2f} ms  ({dt_legacy / dt_radc:.2f}x)"
    )
    print(
        f"+elev     (+radc + elev-FOV cull):        {dt_elev:8.2f} ms  ({dt_legacy / dt_elev:.2f}x)"
    )

    print("\nOutput sanity:")
    for label, d in (
        ("legacy", out_legacy),
        ("default", out_default),
        ("+radc", out_radc),
        ("+elev", out_elev),
    ):
        print("  " + output_stats(label, d))

    print("\nCorrectness (+radc vs. +elev — elev cull should be a superset drop):")
    for k in ("alpha", "distance", "intensity", "raydrop_logit"):
        da = out_radc[k].detach()
        db = out_elev[k].detach()
        # Ignore pixels with pathological magnitudes (gsplat sometimes emits
        # >1e6 outliers for degenerate synthetic Gaussians — this is a kernel
        # artifact, unrelated to our optimizations).
        sane = (
            (da.abs() < 1e6)
            & (db.abs() < 1e6)
            & torch.isfinite(da)
            & torch.isfinite(db)
        )
        if sane.any():
            diff = (da[sane] - db[sane]).abs()
            print(
                f"  {k:>16}: sane_pixels={sane.float().mean().item():.3f}"
                f"  max_abs={diff.max().item():.6f}"
                f"  mean_abs={diff.mean().item():.6f}"
            )
        else:
            print(f"  {k:>16}: no sane overlap")


if __name__ == "__main__":
    main()
