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


def bev_occupancy(xyz: np.ndarray, cfg) -> np.ndarray:
    """(H, W) bool mask of BEV cells that contain at least one point.

    Aligned to the encoder's BEV output orientation (verified empirically):
    ``row`` is the x axis, ``col`` is the y axis, ego at the grid centre. Used to
    restrict the loss to cells the reconstruction actually populates -- a cell is
    "occupied" where the rendered LiDAR (i.e. the background Gaussians) returns a
    point, so far regions the scene has not built yet are excluded.
    """
    h, w = cfg.bev_size
    lo = cfg.point_cloud_range
    occ = np.zeros((h, w), dtype=bool)
    if xyz.shape[0] == 0:
        return occ
    row = np.floor((xyz[:, 0] - lo[0]) / (lo[3] - lo[0]) * h).astype(np.int64)
    col = np.floor((xyz[:, 1] - lo[1]) / (lo[4] - lo[1]) * w).astype(np.int64)
    keep = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    occ[row[keep], col[keep]] = True
    return occ


def _colorize(scalar01: np.ndarray) -> np.ndarray:
    """Map an (H, W) array in [0, 1] to an (H, W, 3) uint8 heatmap (blue->red)."""
    s = np.clip(scalar01, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * s - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * s - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * s - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _compare_features(
    a: torch.Tensor, b: torch.Tensor, active: torch.Tensor
) -> tuple[dict[str, float], torch.Tensor]:
    """Compare two (C, H, W) BEV maps over the ``active`` cells only.

    ``active`` is an (H, W) bool mask selecting which cells to aggregate -- the
    caller restricts this to cells the reconstruction actually populates (see
    :func:`bev_occupancy`), so regions the scene has not reconstructed yet (e.g.
    far background with no Gaussians) do not inflate the loss. Because the 2D
    backbone makes the *output* feature map dense (every cell non-zero), the mask
    must come from the input occupancy, not the feature norm.

    Returns ``(stats, cosine_map)``: ``stats`` holds the scalar similarities
    (keys match the last segment of each :data:`_BEV_SERIES` path); ``cosine_map``
    is the (H, W) per-cell cosine, zeroed outside ``active`` for the heatmap.
    """
    c, h, w = a.shape
    fa = a.reshape(c, -1)
    fb = b.reshape(c, -1)
    act = active.reshape(-1)
    nan = float("nan")
    if not bool(act.any()):
        stats = {"cosine": nan, "global_cosine": nan, "rel_l2": nan}
        return stats, torch.zeros(h, w, device=a.device)

    fa_a = fa[:, act]  # (C, n_active)
    fb_a = fb[:, act]
    na = fa_a.norm(dim=0)
    nb = fb_a.norm(dim=0)
    denom = torch.clamp(na * nb, min=1e-12)
    cell_cos_active = (fa_a * fb_a).sum(0) / denom
    cosine = float(cell_cos_active.mean())

    # Global descriptor + relative L2, also over active cells only.
    ga = fa_a.mean(1)
    gb = fb_a.mean(1)
    global_cosine = float(ga @ gb / torch.clamp(ga.norm() * gb.norm(), min=1e-12))
    rel_l2 = float((fa_a - fb_a).norm() / torch.clamp(fa_a.norm(), min=1e-12))

    cell_cos = torch.zeros(h * w, device=a.device)
    cell_cos[act] = cell_cos_active
    stats = {"cosine": cosine, "global_cosine": global_cosine, "rel_l2": rel_l2}
    return stats, cell_cos.reshape(h, w)


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
        # Which BEV cells the loss aggregates over. Default 'rendered' = cells the
        # reconstruction (background Gaussians) actually populates, so far regions
        # the scene has not built yet don't inflate the error.
        self.active_cells = getattr(args, "bev_active_cells", "rendered")
        self.encoder = build_bev_encoder(args, cfg)
        self._cosine: list[float] = []
        self._global_cosine: list[float] = []
        self._rel_l2: list[float] = []
        mx, my = cfg.meters_per_pixel
        print(
            f"[bev] encoder ready: {cfg.feature_channels}ch "
            f"{cfg.bev_size[0]}x{cfg.bev_size[1]} BEV, "
            f"{mx:.2f}x{my:.2f} m/px, intensity={cfg.use_intensity}, "
            f"active_cells={self.active_cells}"
        )

    def _active_mask(self, frame: FrameData, device) -> torch.Tensor:
        """(H, W) bool mask of BEV cells to score, per ``self.active_cells``."""
        occ_rd = bev_occupancy(frame.rd_static_xyz, self.cfg)
        occ_gt = bev_occupancy(frame.gt_static_xyz, self.cfg)
        mask = {
            "rendered": occ_rd,
            "gt": occ_gt,
            "intersection": occ_rd & occ_gt,
            "union": occ_rd | occ_gt,
        }[self.active_cells]
        return torch.from_numpy(mask).to(device)

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

        active = self._active_mask(frame, feat_gt.device)
        stats, cosine_map = _compare_features(feat_gt, feat_rd, active)
        for path, _, _ in _BEV_SERIES:
            log_scalar(rr, path, stats[path.rsplit("/", 1)[-1]])

        gt_rgb, rd_rgb = _pca_rgb(feat_gt, feat_rd, active)
        rr.log("bev/gt", rr.Image(gt_rgb))
        rr.log("bev/rendered", rr.Image(rd_rgb))
        # Per-cell cosine in [-1, 1] -> [0, 1] heatmap; non-scored cells black.
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
