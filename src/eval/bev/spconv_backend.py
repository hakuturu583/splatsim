"""spconv backend for the OnePlanner BEV encoder ONNX.

Runs the encoder ONNX *without* TensorRT by converting it to a ``torch.nn.Module``
with `onnx2torch` and supplying the only two operators onnx2torch cannot handle --
the ``autoware`` custom ops ``GetIndicePairsImplicitGemm`` and ``ImplicitGemm`` --
via `spconv`, which is exactly the library the ``autoware_tensorrt_plugins`` wrap
(``SpconvOps::get_indice_pairs_implicit_gemm`` / ``ConvGemmOps::implicit_gemm``).

Why this over the TensorRT backend: it is pure-pip (``onnx2torch`` + ``spconv``),
CUDA-aligned with the project's torch, and decoupled from a prebuilt plugin ``.so``
whose TensorRT/CUDA version must be matched exactly. onnx2torch converts every
standard op (VFE, the sparse->dense scatter, and the 2D SECOND/SECONDFPN backbone)
and loads all weights from the ONNX initializers automatically; only the two
sparse-conv ops are implemented here.

The op<->spconv mapping (verified against spconv 2.3.8):

* ``GetIndicePairsImplicitGemm(indices) -> (out_inds, pair_fwd, pair_mask,
  mask_argsort, num_act)`` == ``ops.get_indice_pairs_implicit_gemm(...)`` returning
  ``res`` with ``out_inds=res[0], pair_fwd=res[2], pair_mask=res[4][0],
  mask_argsort=res[6][0]``.
* ``ImplicitGemm(features, filters, pair_fwd, pair_mask, mask_argsort) ->
  out_features`` == ``ops.implicit_gemm(features, filters, pair_fwd, [pair_mask],
  [mask_argsort], num_act, masks, is_train=False, is_subm)[0]`` with the
  single-split ``masks=[0xffffffff]`` the plugin hardcodes for MaskImplicitGemm.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import BEVConfig
from .voxelize import hard_voxelize

_CONVERTERS_REGISTERED = False


def _require_deps():
    try:
        import onnx  # noqa: F401
        import onnx2torch  # noqa: F401
        import spconv.pytorch.ops  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SystemExit(
            "The spconv BEV backend needs onnx, onnx2torch and spconv.\n"
            "Install the optional 'bev' extra:  uv sync --extra bev"
        ) from exc


def _attr_map(node) -> dict:
    """Parse an OnnxNode's attributes into a plain name->value dict."""
    import onnx

    out = {}
    for a in node.proto.attribute:
        out[a.name] = onnx.helper.get_attribute_value(a)
    return out


def _register_converters() -> None:
    """Register the two autoware sparse-conv ops as onnx2torch converters (once)."""
    global _CONVERTERS_REGISTERED
    if _CONVERTERS_REGISTERED:
        return
    import spconv.pytorch.ops as spops
    from onnx2torch.node_converters.registry import add_converter
    from onnx2torch.utils.common import OnnxMapping, OperationConverterResult
    from spconv.core import ConvAlgo

    algo_of = {
        0: ConvAlgo.Native,
        1: ConvAlgo.MaskImplicitGemm,
        2: ConvAlgo.MaskSplitImplicitGemm,
    }

    class _GetIndicePairs(nn.Module):
        def __init__(self, a: dict) -> None:
            super().__init__()
            self.batch_size = int(a["batch_size"])
            self.spatial_shape = [int(v) for v in a["spatial_shape"]]
            self.algo = algo_of[int(a["algo"])]
            self.ksize = [int(v) for v in a["ksize"]]
            self.stride = [int(v) for v in a["stride"]]
            self.padding = [int(v) for v in a["padding"]]
            self.dilation = [int(v) for v in a["dilation"]]
            self.out_padding = [int(v) for v in a["out_padding"]]
            self.subm = bool(int(a["subm"]))
            self.transpose = bool(int(a["transpose"]))

        def forward(self, indices):
            indices = indices.to(torch.int32).contiguous()
            res = spops.get_indice_pairs_implicit_gemm(
                indices,
                self.batch_size,
                self.spatial_shape,
                self.algo,
                self.ksize,
                self.stride,
                self.padding,
                self.dilation,
                self.out_padding,
                self.subm,
                self.transpose,
                is_train=False,
                direct_table=not self.subm,
            )
            out_inds = res[0]
            pair_fwd = res[2]
            pair_mask = res[4][0]  # single split (MaskImplicitGemm)
            mask_argsort = res[6][0]
            num_act = torch.tensor(
                out_inds.shape[0], dtype=torch.int32, device=indices.device
            )
            return out_inds, pair_fwd, pair_mask, mask_argsort, num_act

    class _ImplicitGemm(nn.Module):
        def __init__(self, a: dict) -> None:
            super().__init__()
            self.is_subm = bool(int(a["is_subm"]))

        def forward(self, features, filters, pair_fwd, pair_mask, mask_argsort):
            features = features.contiguous()
            filters = filters.contiguous()
            num_act = int(pair_mask.shape[0])
            # Single-split mask the autoware plugin hardcodes for MaskImplicitGemm.
            masks = [np.array([0xFFFFFFFF], dtype=np.uint32)]
            out = spops.implicit_gemm(
                features,
                filters,
                pair_fwd,
                [pair_mask.contiguous()],
                [mask_argsort.contiguous()],
                num_act,
                masks,
                False,
                self.is_subm,
            )
            return out[0]

    def _mapping(node) -> OnnxMapping:
        return OnnxMapping(
            inputs=tuple(node.input_values), outputs=tuple(node.output_values)
        )

    @add_converter(
        operation_type="GetIndicePairsImplicitGemm", version=1, domain="autoware"
    )
    def _convert_get_indice_pairs(node, graph):  # noqa: ANN001, ARG001
        return OperationConverterResult(
            torch_module=_GetIndicePairs(_attr_map(node)),
            onnx_mapping=_mapping(node),
        )

    @add_converter(operation_type="ImplicitGemm", version=1, domain="autoware")
    def _convert_implicit_gemm(node, graph):  # noqa: ANN001, ARG001
        return OperationConverterResult(
            torch_module=_ImplicitGemm(_attr_map(node)),
            onnx_mapping=_mapping(node),
        )

    _CONVERTERS_REGISTERED = True


class SpconvBEVEncoder:
    """Runs ``oneplanner_bev_encoder.onnx`` via onnx2torch + spconv (no TensorRT)."""

    def __init__(self, onnx_path: str, cfg: BEVConfig, *, device: str = "cuda") -> None:
        if not torch.cuda.is_available():
            raise SystemExit("The spconv BEV backend requires a CUDA device.")
        _require_deps()
        import onnx
        from onnx2torch import convert

        # The graph's sparse ops use spatial_shape = [.., .., z], so voxel coords
        # must be z-last; override the config's coors order for this backend.
        self.cfg = cfg if cfg.coors_order == "xyz" else _with_coors_order(cfg, "xyz")
        self.device = torch.device(device)
        _register_converters()
        print(f"[bev] converting {onnx_path} via onnx2torch + spconv (one-off)...")
        model = onnx.load(onnx_path)
        self.module = convert(model).to(self.device).eval()
        print("[bev] spconv BEV module ready")

    @torch.no_grad()
    def encode(self, points: np.ndarray) -> torch.Tensor:
        """Voxelise + run the encoder. ``points`` (N, F) -> (C, H, W) on device."""
        pts = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        voxels, num_points, coors = hard_voxelize(pts, self.cfg)
        if voxels.shape[0] == 0:
            h, w = self.cfg.bev_size
            return torch.zeros(
                (self.cfg.feature_channels, h, w),
                dtype=torch.float32,
                device=self.device,
            )
        out = self.module(voxels, num_points.to(torch.int32), coors.to(torch.int32))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out[0]  # drop batch -> (C, H, W)


def _with_coors_order(cfg: BEVConfig, order: str) -> BEVConfig:
    import dataclasses

    return dataclasses.replace(cfg, coors_order=order)
