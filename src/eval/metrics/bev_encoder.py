"""Representation agreement: OnePlanner BEV-encoder feature similarity.

Both the rendered and the GT point clouds (at the same ego pose, dynamic objects
masked out) are pushed through the OnePlanner BEV encoder, producing two
``[512, 180, 180]`` bird's-eye-view feature maps. This metric asks *how close the
learned representations are* -- i.e. whether a downstream planner would "see" the
reconstructed scene the same way it sees the real one -- rather than raw
geometric distance.

Per frame it logs to Rerun:

* scalars ``metrics/bev/cosine`` (mean per-cell cosine over occupied cells),
  ``metrics/bev/global_cosine`` (cosine of the globally mean-pooled descriptors),
  and ``metrics/bev/rel_l2`` (Frobenius ``||gt-rd|| / ||gt||``);
* images ``bev/gt`` / ``bev/rendered`` -- a shared-basis PCA(512->3) RGB view so
  the two feature maps are directly comparable by eye;
* image ``bev/cosine_map`` -- the per-cell cosine similarity as a heatmap.

The feature maps stay on the GPU the encoder produced them on; only the three
scalars and the small ``(180, 180, 3)`` uint8 images cross back to the host.
"""

from __future__ import annotations

import numpy as np
import torch

from ..context import EvalContext
from ..frame import FrameData
from ..rerun_io import log_scalar
from .base import LidarEvalMetric, summarize_series

# (rerun entity path, legend label, RGB colour). The stats key for each series is
# the last path segment, so logging can zip against the metric's stats dict.
_BEV_SERIES: tuple[tuple[str, str, tuple[int, int, int]], ...] = (
    ("metrics/bev/cosine", "bev per-cell cosine", (120, 220, 160)),
    ("metrics/bev/global_cosine", "bev global cosine", (250, 190, 90)),
    ("metrics/bev/rel_l2", "bev relative L2", (230, 120, 160)),
)


def _to_points(xyz: np.ndarray, intensity: np.ndarray, cfg) -> np.ndarray:
    """Assemble the (N, F) encoder input: (x, y, z, intensity, time_lag)."""
    n = xyz.shape[0]
    cols = [xyz[:, 0], xyz[:, 1], xyz[:, 2]]
    if cfg.num_point_features >= 4:
        cols.append(intensity if cfg.use_intensity else np.zeros(n, np.float32))
    if cfg.num_point_features >= 5:
        cols.append(np.zeros(n, np.float32))  # time_lag: single sweep -> 0
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.float32)


def _colorize(scalar01: np.ndarray) -> np.ndarray:
    """Map an (H, W) array in [0, 1] to an (H, W, 3) uint8 heatmap (blue->red)."""
    s = np.clip(scalar01, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * s - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * s - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * s - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _compare_features(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """Compare two (C, H, W) BEV maps on-device.

    Returns ``(stats, cosine_map, active_mask)`` where ``stats`` holds the scalar
    similarities, ``cosine_map`` is the (H, W) per-cell cosine, and
    ``active_mask`` marks cells occupied in either map. Keys in ``stats`` match
    the last segment of each :data:`_BEV_SERIES` path.
    """
    c, h, w = a.shape
    fa = a.reshape(c, -1)
    fb = b.reshape(c, -1)
    na = fa.norm(dim=0)
    nb = fb.norm(dim=0)
    active = (na > 1e-6) | (nb > 1e-6)

    denom = torch.clamp(na * nb, min=1e-12)
    cell_cos = torch.where(active, (fa * fb).sum(0) / denom, torch.zeros_like(na))
    cosine = float(cell_cos[active].mean()) if bool(active.any()) else float("nan")

    ga = fa.mean(1)
    gb = fb.mean(1)
    global_cosine = float(ga @ gb / torch.clamp(ga.norm() * gb.norm(), min=1e-12))
    rel_l2 = float((fa - fb).norm() / torch.clamp(fa.norm(), min=1e-12))

    stats = {"cosine": cosine, "global_cosine": global_cosine, "rel_l2": rel_l2}
    return stats, cell_cos.reshape(h, w), active.reshape(h, w)


def _pca_rgb(
    a: torch.Tensor, b: torch.Tensor, active: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Shared-basis PCA(C->3) of two (C, H, W) maps -> two (H, W, 3) uint8 images.

    The basis is the top-3 eigenvectors of the pooled feature covariance over the
    occupied cells of *both* maps, so the colours are directly comparable; empty
    cells render black. Using ``eigh`` on the CxC covariance (fixed ~C^3) avoids a
    full SVD of the tall (occupied-cell x C) matrix every frame.
    """
    c, h, w = a.shape
    fa = a.reshape(c, -1).T  # (HW, C)
    fb = b.reshape(c, -1).T
    act = active.reshape(-1)
    pool = torch.cat([fa[act], fb[act]], dim=0)
    if pool.shape[0] < 3:
        z = np.zeros((h, w, 3), np.uint8)
        return z, z
    mean = pool.mean(0, keepdim=True)
    centered = pool - mean
    cov = (centered.T @ centered) / (centered.shape[0] - 1)
    _, evecs = torch.linalg.eigh(cov)  # ascending eigenvalues
    basis = evecs[:, -3:]  # top-3 principal directions, (C, 3)

    def project(f: torch.Tensor) -> np.ndarray:
        proj = (f - mean) @ basis  # (HW, 3)
        lo = torch.quantile(proj[act], 0.02, dim=0)
        hi = torch.quantile(proj[act], 0.98, dim=0)
        proj = (proj - lo) / torch.clamp(hi - lo, min=1e-6)
        img = proj.clamp(0, 1).reshape(h, w, 3)
        img[~active] = 0.0
        return (img * 255).to(torch.uint8).cpu().numpy()

    return project(fa), project(fb)


class BEVEncoderMetric(LidarEvalMetric):
    """Cosine / L2 agreement of the BEV encoder features of both clouds."""

    name = "bev"

    def __init__(self, args, ctx: EvalContext) -> None:
        # Import lazily so the (heavy, optional) backend is only required when the
        # BEV metric is actually selected.
        from ..bev import BEVConfig, build_bev_encoder

        cfg = BEVConfig(use_intensity=not getattr(args, "bev_no_intensity", False))
        self.cfg = cfg
        self.encoder = build_bev_encoder(args, cfg)
        self._cosine: list[float] = []
        self._global_cosine: list[float] = []
        self._rel_l2: list[float] = []
        mx, my = cfg.meters_per_pixel
        print(
            f"[bev] encoder ready: {cfg.feature_channels}ch "
            f"{cfg.bev_size[0]}x{cfg.bev_size[1]} BEV, "
            f"{mx:.2f}x{my:.2f} m/px, intensity={cfg.use_intensity}"
        )

    def setup_rerun(self, rr, ctx: EvalContext) -> None:
        for path, name, color in _BEV_SERIES:
            rr.log(path, rr.SeriesLines(names=[name], colors=[color]), static=True)

    def update(self, rr, frame: FrameData, ctx: EvalContext) -> dict[str, float]:
        gt_pts = _to_points(
            frame.gt_static_xyz, frame.gt_intensity[~frame.gt_dynamic], self.cfg
        )
        rd_pts = _to_points(
            frame.rd_static_xyz, frame.rd_intensity[~frame.rd_dynamic], self.cfg
        )
        feat_gt = self.encoder.encode(gt_pts)  # (C, H, W) torch on device
        feat_rd = self.encoder.encode(rd_pts)

        stats, cosine_map, active = _compare_features(feat_gt, feat_rd)
        for path, _, _ in _BEV_SERIES:
            log_scalar(rr, path, stats[path.rsplit("/", 1)[-1]])

        gt_rgb, rd_rgb = _pca_rgb(feat_gt, feat_rd, active)
        rr.log("bev/gt", rr.Image(gt_rgb))
        rr.log("bev/rendered", rr.Image(rd_rgb))
        # Per-cell cosine in [-1, 1] -> [0, 1] heatmap; empty cells black.
        active_np = active.cpu().numpy()
        heat = _colorize(((cosine_map + 1.0) * 0.5).cpu().numpy())
        heat[~active_np] = 0
        rr.log("bev/cosine_map", rr.Image(heat))

        self._cosine.append(stats["cosine"])
        self._global_cosine.append(stats["global_cosine"])
        self._rel_l2.append(stats["rel_l2"])
        return {"bev_cos": stats["cosine"], "bev_relL2": stats["rel_l2"]}

    def summarize(self) -> None:
        summarize_series("bev per-cell cosine", self._cosine)
        summarize_series("bev global cosine  ", self._global_cosine)
        summarize_series("bev relative L2    ", self._rel_l2)
