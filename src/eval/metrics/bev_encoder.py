"""Representation agreement: OnePlanner BEV-encoder feature similarity.

Both the rendered and the GT point clouds (at the same ego pose, dynamic objects
masked out) are pushed through the OnePlanner BEV encoder, producing two
``[512, 180, 180]`` bird's-eye-view feature maps. This metric asks *how close the
learned representations are* -- i.e. whether a downstream planner would "see" the
reconstructed scene the same way it sees the real one -- rather than raw
geometric distance.

Each similarity (per-cell cosine, globally mean-pooled cosine, and Frobenius
relative L2 ``||gt-rd|| / ||gt||``) is logged for **two aggregations** so they can
be compared in Rerun:

* ``metrics/bev/cosine`` … -- over **occupied** cells only: the cells the
  reconstruction populates (rendered LiDAR = background Gaussians; see
  :func:`bev_occupancy`). The honest number for the region actually built.
* ``metrics/bev/cosine_all`` … -- over **every** 180x180 cell. Matches the naive
  dense comparison but is optimistic: most cells are empty on both sides and
  trivially agree (identical "no input" response), inflating cosine.

Per frame it also logs images: ``bev/gt`` / ``bev/rendered`` (shared-basis
PCA(512->3) RGB, directly comparable by eye) and per-cell cosine heatmaps
``bev/cosine_map`` (occupied) / ``bev/cosine_map_all`` (all cells).

The per-cell cosine map is computed once (dense) and both aggregations are
derived from it; the feature maps stay on the GPU, only the scalars and the small
``(180, 180, 3)`` uint8 images cross back to the host.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..context import EvalContext
from ..frame import FrameData
from ..rerun_io import log_scalar
from .base import LidarEvalMetric, summarize_series

# Each BEV metric is logged for TWO aggregations so they can be compared in
# Rerun: "occupied" (only cells the reconstruction populates, per bev_occupancy)
# and "all" (every 180x180 cell). Columns: path, legend label, RGB colour,
# stats key, variant.
_BEV_SERIES: tuple[tuple[str, str, tuple[int, int, int], str, str], ...] = (
    (
        "metrics/bev/cosine",
        "per-cell cosine (occupied)",
        (120, 220, 160),
        "cosine",
        "occupied",
    ),
    (
        "metrics/bev/global_cosine",
        "global cosine (occupied)",
        (250, 190, 90),
        "global_cosine",
        "occupied",
    ),
    (
        "metrics/bev/rel_l2",
        "relative L2 (occupied)",
        (230, 120, 160),
        "rel_l2",
        "occupied",
    ),
    (
        "metrics/bev/cosine_all",
        "per-cell cosine (all cells)",
        (90, 150, 110),
        "cosine",
        "all",
    ),
    (
        "metrics/bev/global_cosine_all",
        "global cosine (all cells)",
        (190, 145, 70),
        "global_cosine",
        "all",
    ),
    (
        "metrics/bev/rel_l2_all",
        "relative L2 (all cells)",
        (175, 90, 120),
        "rel_l2",
        "all",
    ),
)

_METRIC_KEYS = ("cosine", "global_cosine", "rel_l2")
_VARIANTS = ("occupied", "all")


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


def _per_cell_cosine(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense per-cell cosine of two (C, H, W) maps.

    Returns ``(fa, fb, cos)`` where ``fa``/``fb`` are the (C, H*W) views and
    ``cos`` is the (H*W,) per-cell cosine -- computed once so both the occupied
    and all-cell aggregations can be derived from it without recomputing.
    """
    c = a.shape[0]
    fa = a.reshape(c, -1)
    fb = b.reshape(c, -1)
    cos = F.cosine_similarity(fa, fb, dim=0, eps=1e-12)  # (H*W,)
    return fa, fb, cos


def _variant_stats(
    fa: torch.Tensor, fb: torch.Tensor, cos: torch.Tensor, active: torch.Tensor | None
) -> dict[str, float]:
    """Similarity scalars over ``active`` cells (``None`` = every cell).

    ``cosine`` slices the precomputed per-cell ``cos``; ``global_cosine`` and
    ``rel_l2`` need the masked feature columns (not sliceable from ``cos``).
    """
    if active is None:
        fa_a, fb_a, cell = fa, fb, cos
    elif bool(active.any()):
        fa_a, fb_a, cell = fa[:, active], fb[:, active], cos[active]
    else:
        nan = float("nan")
        return {"cosine": nan, "global_cosine": nan, "rel_l2": nan}
    ga, gb = fa_a.mean(1), fb_a.mean(1)
    return {
        "cosine": float(cell.mean()),
        "global_cosine": float(F.cosine_similarity(ga, gb, dim=0, eps=1e-12)),
        "rel_l2": float((fa_a - fb_a).norm() / torch.clamp(fa_a.norm(), min=1e-12)),
    }


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
        # Which cells the "occupied" variant aggregates over (rendered/gt/…).
        self.active_cells = getattr(args, "bev_active_cells", "rendered")
        self.encoder = build_bev_encoder(args, cfg)
        self._history: dict[tuple[str, str], list[float]] = {
            (variant, key): [] for variant in _VARIANTS for key in _METRIC_KEYS
        }
        mx, my = cfg.meters_per_pixel
        print(
            f"[bev] encoder ready: {cfg.feature_channels}ch "
            f"{cfg.bev_size[0]}x{cfg.bev_size[1]} BEV, "
            f"{mx:.2f}x{my:.2f} m/px, intensity={cfg.use_intensity}, "
            f"occupied={self.active_cells}"
        )

    def _occupied_mask(self, frame: FrameData) -> np.ndarray:
        """(H, W) numpy bool mask of reconstruction-occupied cells.

        Only the occupancy operand(s) the selected ``active_cells`` mode needs are
        computed (single-cloud modes do one occupancy pass, not two).
        """
        mode = self.active_cells
        if mode == "rendered":
            return bev_occupancy(frame.rd_static_xyz, self.cfg)
        if mode == "gt":
            return bev_occupancy(frame.gt_static_xyz, self.cfg)
        occ_rd = bev_occupancy(frame.rd_static_xyz, self.cfg)
        occ_gt = bev_occupancy(frame.gt_static_xyz, self.cfg)
        return occ_rd & occ_gt if mode == "intersection" else occ_rd | occ_gt

    def setup_rerun(self, rr, ctx: EvalContext) -> None:
        for path, name, color, _, variant in _BEV_SERIES:
            # Reflect the actual occupancy region in the legend, not a fixed word.
            label = (
                name.replace("(occupied)", f"({self.active_cells})")
                if variant == "occupied"
                else name
            )
            rr.log(path, rr.SeriesLines(names=[label], colors=[color]), static=True)

    def update(self, rr, frame: FrameData, ctx: EvalContext) -> dict[str, float]:
        gt_pts = _to_points(frame.gt_static_xyz, frame.gt_static_intensity, self.cfg)
        rd_pts = _to_points(frame.rd_static_xyz, frame.rd_static_intensity, self.cfg)
        feat_gt = self.encoder.encode(gt_pts)  # (C, H, W) torch on device
        feat_rd = self.encoder.encode(rd_pts)
        h, w = feat_gt.shape[1:]

        # Per-cell cosine once (dense); both aggregations derive from it.
        fa, fb, cos = _per_cell_cosine(feat_gt, feat_rd)
        occ_np = self._occupied_mask(frame)  # (H, W) numpy bool
        occ_flat = torch.from_numpy(occ_np.reshape(-1)).to(feat_gt.device)
        stats = {
            "occupied": _variant_stats(fa, fb, cos, occ_flat),
            "all": _variant_stats(fa, fb, cos, None),
        }
        for path, _, _, key, variant in _BEV_SERIES:
            value = stats[variant][key]
            log_scalar(rr, path, value)
            self._history[variant, key].append(value)

        # Images: PCA-RGB over occupied cells + a cosine heatmap per variant.
        gt_rgb, rd_rgb = _pca_rgb(feat_gt, feat_rd, occ_flat.reshape(h, w))
        rr.log("bev/gt", rr.Image(gt_rgb))
        rr.log("bev/rendered", rr.Image(rd_rgb))
        heat_all = _colorize(((cos.reshape(h, w) + 1.0) * 0.5).cpu().numpy())
        heat_occ = heat_all.copy()
        heat_occ[~occ_np] = 0
        rr.log("bev/cosine_map", rr.Image(heat_occ))
        rr.log("bev/cosine_map_all", rr.Image(heat_all))

        return {
            "cos_occ": stats["occupied"]["cosine"],
            "cos_all": stats["all"]["cosine"],
        }

    def summarize(self) -> None:
        labels = {
            "cosine": "per-cell cosine",
            "global_cosine": "global cosine",
            "rel_l2": "relative L2",
        }
        for variant in _VARIANTS:
            for key in _METRIC_KEYS:
                summarize_series(
                    f"bev {labels[key]:15s} [{variant:8s}]", self._history[variant, key]
                )
