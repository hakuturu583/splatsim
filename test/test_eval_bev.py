"""CPU-only tests for the LiDAR-eval metric framework + BEV preprocessing.

Covers the pieces that do not need CUDA / TensorRT / a T4 dataset: hard
voxelisation, the BEV feature-comparison maths, the geometry config, and metric
selection. The TensorRT backend and the full per-frame pipeline are exercised
separately (they require a GPU + the autoware plugins + a dataset).
"""

from __future__ import annotations

import argparse
import types
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from eval.context import EvalContext
    from eval.frame import FrameData


# ── voxelisation ─────────────────────────────────────────────────────────────


def _cfg(coors_order: str = "zyx"):
    from eval.bev.config import BEVConfig

    return BEVConfig(
        point_cloud_range=(-10.0, -10.0, -2.0, 10.0, 10.0, 2.0),
        voxel_size=(1.0, 1.0, 1.0),
        max_num_points=3,
        max_voxels=100,
        num_point_features=5,
        coors_order=coors_order,
    )


def test_hard_voxelize_caps_points_and_range():
    from eval.bev.voxelize import hard_voxelize

    cfg = _cfg()
    pts = np.array(
        [
            [0.1, 0.1, 0.1, 0.5, 0.0],
            [0.2, 0.2, 0.2, 0.6, 0.0],
            [0.3, 0.3, 0.3, 0.7, 0.0],
            [0.4, 0.4, 0.4, 0.8, 0.0],  # 4th in same voxel -> dropped (cap 3)
            [3.5, 3.5, 0.5, 0.1, 0.0],  # different voxel
            [3.6, 3.6, 0.6, 0.2, 0.0],
            [999.0, 0.0, 0.0, 0.0, 0.0],  # out of range -> dropped
        ],
        dtype=np.float32,
    )
    v, n, c = hard_voxelize(torch.from_numpy(pts), cfg)
    assert v.shape == (2, 3, 5)
    assert sorted(n.tolist()) == [2, 3]
    assert c.dtype == torch.int32 and c.shape == (2, 3)
    # populated slots per voxel match the (capped) point count
    for i in range(2):
        nz = int((v[i].abs().sum(dim=1) > 0).sum())
        assert nz == int(n[i])


def test_hard_voxelize_empty_input():
    from eval.bev.voxelize import hard_voxelize

    cfg = _cfg()
    far = np.full((5, 5), 1e6, dtype=np.float32)
    v, n, c = hard_voxelize(torch.from_numpy(far), cfg)
    assert v.shape[0] == 0 and n.shape[0] == 0 and c.shape[0] == 0


def test_coors_order_zyx_vs_xyz():
    from eval.bev.voxelize import hard_voxelize

    pt = np.array([[3.5, 3.5, 0.5, 0.0, 0.0]], dtype=np.float32)
    _, _, c_zyx = hard_voxelize(torch.from_numpy(pt), _cfg(coors_order="zyx"))
    _, _, c_xyz = hard_voxelize(torch.from_numpy(pt), _cfg(coors_order="xyz"))
    # same voxel, reversed axis order
    assert c_zyx[0].tolist() == c_xyz[0].tolist()[::-1]


# ── config geometry ──────────────────────────────────────────────────────────


def test_bev_config_grid_and_pitch():
    from eval.bev.config import BEVConfig

    cfg = BEVConfig()  # defaults: +/-122.4 m, 0.17 m voxels, 180x180 BEV
    assert cfg.grid_size[:2] == (1440, 1440)
    mx, my = cfg.meters_per_pixel
    assert mx == pytest.approx(1.36, abs=1e-3)
    assert my == pytest.approx(1.36, abs=1e-3)


# ── feature comparison ───────────────────────────────────────────────────────


def test_compare_identical_features():
    from eval.metrics.bev_encoder import _per_cell_cosine, _variant_stats

    rng = np.random.default_rng(0)
    a = torch.from_numpy(rng.standard_normal((8, 6, 6)).astype(np.float32))
    fa, fb, cos = _per_cell_cosine(a, a.clone())
    assert tuple(cos.shape) == (36,)  # H*W flat
    active = torch.ones(36, dtype=torch.bool)
    stats = _variant_stats(fa, fb, cos, active)
    assert stats["cosine"] == pytest.approx(1.0, abs=1e-5)
    assert stats["global_cosine"] == pytest.approx(1.0, abs=1e-5)
    assert stats["rel_l2"] == pytest.approx(0.0, abs=1e-6)
    # active=None (all cells) matches the all-True mask.
    assert _variant_stats(fa, fb, cos, None)["cosine"] == pytest.approx(1.0, abs=1e-5)


def test_compare_orthogonal_and_scaled_features():
    from eval.metrics.bev_encoder import _per_cell_cosine, _variant_stats

    c, h, w = 4, 3, 3
    a = torch.zeros((c, h, w))
    b = torch.zeros((c, h, w))
    a[0] = 1.0  # channel-0 unit vectors
    b[1] = 1.0  # orthogonal channel-1 unit vectors
    fa, fb, cos = _per_cell_cosine(a, b)
    assert _variant_stats(fa, fb, cos, None)["cosine"] == pytest.approx(0.0, abs=1e-5)

    # cosine is scale-invariant: scaling one map leaves per-cell cosine at 1.
    fa2, fb2, cos2 = _per_cell_cosine(a, a * 5.0)
    stats2 = _variant_stats(fa2, fb2, cos2, None)
    assert stats2["cosine"] == pytest.approx(1.0, abs=1e-5)
    assert stats2["rel_l2"] > 0.0  # but L2 grows


def test_bev_occupancy_alignment():
    from eval.bev.config import BEVConfig
    from eval.metrics.bev_encoder import bev_occupancy

    cfg = BEVConfig()
    h, w = cfg.bev_size
    # A point at +x should land at row > centre, +y at col > centre.
    occ = bev_occupancy(np.array([[60.0, 0.0, 0.0], [0.0, 60.0, 0.0]], np.float32), cfg)
    rows, cols = np.where(occ)
    assert occ.sum() == 2
    # +x point: row above centre, col near centre; +y point: the reverse.
    assert rows.max() > h // 2  # +x pushed the row past centre
    assert cols.max() > w // 2  # +y pushed the col past centre
    # empty cloud -> empty mask
    assert bev_occupancy(np.empty((0, 3), np.float32), cfg).sum() == 0


def test_pca_rgb_and_colorize_shapes():
    from eval.metrics.bev_encoder import _colorize, _pca_rgb

    rng = np.random.default_rng(1)
    a = torch.from_numpy(rng.standard_normal((16, 5, 5)).astype(np.float32))
    b = torch.from_numpy(rng.standard_normal((16, 5, 5)).astype(np.float32))
    active = torch.ones((5, 5), dtype=torch.bool)
    ga, rb = _pca_rgb(a, b, active)
    assert ga.shape == (5, 5, 3) and rb.shape == (5, 5, 3)
    assert ga.dtype == np.uint8

    heat = _colorize(np.linspace(0, 1, 25).reshape(5, 5))
    assert heat.shape == (5, 5, 3) and heat.dtype == np.uint8


# ── metric selection ─────────────────────────────────────────────────────────


def test_build_metrics_selection_and_unknown():
    from eval.metrics import ChamferMetric, build_metrics

    args = argparse.Namespace(metrics="chamfer")
    # chamfer ctor does not touch ctx; a bare stand-in is enough.
    ctx = cast("EvalContext", types.SimpleNamespace())
    metrics = build_metrics(args, ctx)
    assert len(metrics) == 1 and isinstance(metrics[0], ChamferMetric)

    with pytest.raises(SystemExit):
        build_metrics(argparse.Namespace(metrics="nope"), ctx)


def test_chamfer_metric_update_cpu():
    from eval.metrics.chamfer import ChamferMetric

    class FakeRR:
        def __init__(self):
            self.scalars = {}

        def Scalars(self, v):  # noqa: N802 - mimic rerun API
            return v

        def SeriesLines(self, **kw):  # noqa: N802
            return kw

        def log(self, path, payload, **kw):
            if not isinstance(payload, dict):
                self.scalars[path] = payload

    n = 200
    rng = np.random.default_rng(0)
    xyz = rng.standard_normal((n, 3)).astype(np.float32) * 5.0
    # Identical clouds, all static + in-range -> the subset views are the full
    # clouds (mirrors FrameData's cached_property gathers).
    frame = cast(
        "FrameData",
        types.SimpleNamespace(
            gt_static_xyz=xyz,
            rd_static_xyz=xyz.copy(),
            gt_ranged_xyz=xyz,
        ),
    )
    ctx = cast(
        "EvalContext",
        types.SimpleNamespace(
            args=argparse.Namespace(max_points=0),
            device=torch.device("cpu"),
            rng=rng,
        ),
    )
    m = ChamferMetric(argparse.Namespace(), ctx)
    out = m.update(FakeRR(), frame, ctx)
    # Identical clouds -> ~0 Chamfer (a few 1e-4 m from torch.cdist's matmul path).
    assert out["raw"] == pytest.approx(0.0, abs=1e-2)
    assert out["ranged"] == pytest.approx(0.0, abs=1e-2)
