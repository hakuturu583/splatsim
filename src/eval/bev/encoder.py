"""The BEV-encoder interface + backend factory.

A backend maps a raw point cloud to a BEV feature map. Only the TensorRT backend
is implemented (the ONNX needs the ``autoware_tensorrt_plugins`` sparse-conv
ops), but the indirection keeps the metric backend-agnostic and leaves room for
an spconv/PyTorch backend later.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .config import BEVConfig

if TYPE_CHECKING:
    import numpy as np
    import torch


@runtime_checkable
class BEVEncoder(Protocol):
    """Maps an (N, F) point cloud to a (C, H, W) BEV feature map."""

    cfg: BEVConfig

    def encode(self, points: "np.ndarray") -> "torch.Tensor":
        """Run the encoder. ``points`` is (N, F); returns a (C, H, W) tensor.

        The feature map is returned on the encoder's CUDA device so downstream
        comparison can stay on the GPU without a host round-trip.
        """
        ...


def build_bev_encoder(args, cfg: BEVConfig | None = None) -> BEVEncoder:
    """Instantiate the BEV encoder backend selected by ``args.bev_backend``.

    Paths fall back to the ``ONEPLANNER_BEV_ONNX`` / ``ONEPLANNER_TRT_PLUGINS``
    environment variables so the personal artifact locations need not be baked
    into the repo. Raises :class:`SystemExit` with an actionable message when a
    required artifact or dependency is missing.
    """
    cfg = cfg or BEVConfig()
    backend = getattr(args, "bev_backend", "tensorrt")
    if backend != "tensorrt":
        raise SystemExit(
            f"Unknown --bev-backend {backend!r}; only 'tensorrt' is implemented."
        )

    onnx_path = getattr(args, "bev_onnx", None) or os.environ.get("ONEPLANNER_BEV_ONNX")
    plugin_path = getattr(args, "bev_plugins", None) or os.environ.get(
        "ONEPLANNER_TRT_PLUGINS"
    )
    if not onnx_path:
        raise SystemExit(
            "BEV metric needs the encoder ONNX. Pass --bev-onnx PATH "
            "(oneplanner_bev_encoder.onnx) or set $ONEPLANNER_BEV_ONNX."
        )
    if not plugin_path:
        raise SystemExit(
            "BEV metric needs the autoware TensorRT plugins. Pass --bev-plugins "
            "PATH (libautoware_tensorrt_plugins.so) or set $ONEPLANNER_TRT_PLUGINS."
        )

    from .tensorrt_backend import TensorRTBEVEncoder

    return TensorRTBEVEncoder(
        onnx_path=onnx_path,
        plugin_path=plugin_path,
        cfg=cfg,
        device=str(getattr(args, "device", "cuda")),
        engine_cache=getattr(args, "bev_engine_cache", None),
        fp16=bool(getattr(args, "bev_fp16", False)),
    )
