"""The BEV-encoder base class + backend factory.

A backend maps a raw point cloud to a BEV feature map. Two run the same ONNX:
``spconv`` (onnx2torch + spconv) and ``tensorrt`` (autoware plugins). The shared
:class:`BaseBEVEncoder` owns the scaffolding common to both -- CUDA guard, the
voxelise + empty-cloud preamble of :meth:`~BaseBEVEncoder.encode` -- so each
backend implements only its engine/module build and :meth:`_run`.
"""

from __future__ import annotations

import abc
import os

import numpy as np
import torch

from .config import BEVConfig
from .voxelize import hard_voxelize


class BaseBEVEncoder(abc.ABC):
    """Maps an (N, F) point cloud to a (C, H, W) BEV feature map on the GPU."""

    def __init__(self, cfg: BEVConfig, device: str = "cuda") -> None:
        if not torch.cuda.is_available():
            raise SystemExit("The BEV-encoder metric requires a CUDA device.")
        self.cfg = cfg
        self.device = torch.device(device)

    @torch.no_grad()
    def encode(self, points: np.ndarray) -> torch.Tensor:
        """Voxelise ``points`` (N, F) and run the encoder -> (C, H, W) on device."""
        pts = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        voxels, num_points, coors = hard_voxelize(pts, self.cfg)
        if voxels.shape[0] == 0:
            return self._empty_bev()
        return self._run(voxels, num_points, coors)

    def _empty_bev(self) -> torch.Tensor:
        """Zero BEV map for an empty cloud (no points survived voxelisation)."""
        h, w = self.cfg.bev_size
        return torch.zeros(
            (self.cfg.feature_channels, h, w), dtype=torch.float32, device=self.device
        )

    @abc.abstractmethod
    def _run(
        self, voxels: torch.Tensor, num_points: torch.Tensor, coors: torch.Tensor
    ) -> torch.Tensor:
        """Run the backend on the voxelised inputs -> (C, H, W) feature map."""


def build_bev_encoder(args, cfg: BEVConfig | None = None) -> BaseBEVEncoder:
    """Instantiate the BEV encoder backend selected by ``args.bev_backend``.

    Two backends run the same encoder ONNX:

    * ``spconv`` (default) -- onnx2torch + spconv, pure-pip and CUDA-aligned with
      the project's torch; needs only the ONNX.
    * ``tensorrt`` -- TensorRT + the ``autoware_tensorrt_plugins`` ``.so``; the
      production path, but its TensorRT/CUDA must match the prebuilt plugin.

    The ONNX / plugin paths fall back to the ``ONEPLANNER_BEV_ONNX`` /
    ``ONEPLANNER_TRT_PLUGINS`` environment variables so the personal artifact
    locations need not be baked into the repo. Raises :class:`SystemExit` with an
    actionable message when a required artifact or dependency is missing.
    """
    cfg = cfg or BEVConfig()
    backend = getattr(args, "bev_backend", "spconv")
    device = str(getattr(args, "device", "cuda"))

    onnx_path = getattr(args, "bev_onnx", None) or os.environ.get("ONEPLANNER_BEV_ONNX")
    if not onnx_path:
        raise SystemExit(
            "BEV metric needs the encoder ONNX. Pass --bev-onnx PATH "
            "(oneplanner_bev_encoder.onnx) or set $ONEPLANNER_BEV_ONNX."
        )

    if backend == "spconv":
        from .spconv_backend import SpconvBEVEncoder

        return SpconvBEVEncoder(onnx_path, cfg, device=device)

    if backend == "tensorrt":
        plugin_path = getattr(args, "bev_plugins", None) or os.environ.get(
            "ONEPLANNER_TRT_PLUGINS"
        )
        if not plugin_path:
            raise SystemExit(
                "The tensorrt BEV backend needs the autoware TensorRT plugins. "
                "Pass --bev-plugins PATH (libautoware_tensorrt_plugins.so) or set "
                "$ONEPLANNER_TRT_PLUGINS -- or use the default --bev-backend spconv."
            )
        from .tensorrt_backend import TensorRTBEVEncoder

        return TensorRTBEVEncoder(
            onnx_path=onnx_path,
            plugin_path=plugin_path,
            cfg=cfg,
            device=device,
            engine_cache=getattr(args, "bev_engine_cache", None),
            fp16=bool(getattr(args, "bev_fp16", False)),
        )

    raise SystemExit(
        f"Unknown --bev-backend {backend!r}; choose 'spconv' or 'tensorrt'."
    )
