"""OnePlanner BEV encoder backend for the LiDAR evaluation.

The BEV encoder is the LiDAR branch of OnePlanner's BEVFusion perception stack,
shipped as ``oneplanner_bev_encoder.onnx``: it maps a voxelised point cloud to a
bird's-eye-view feature map ``[1, 512, 180, 180]``. The ONNX embeds custom
sparse-convolution operators (``GetIndicePairsImplicitGemm`` / ``ImplicitGemm``,
domain ``autoware``) that only run under **TensorRT** with the
``autoware_tensorrt_plugins`` shared library loaded -- plain ``onnxruntime``
cannot execute it.

The backend is split so the pieces are independently testable / swappable:

* :mod:`eval.bev.config` -- :class:`BEVConfig` (range, voxel size, output shape).
* :mod:`eval.bev.voxelize` -- pure-torch hard voxelisation (CPU or CUDA).
* :mod:`eval.bev.encoder` -- the :class:`BEVEncoder` protocol + factory.
* :mod:`eval.bev.tensorrt_backend` -- the TensorRT implementation.
"""

from __future__ import annotations

from .config import BEVConfig
from .encoder import BaseBEVEncoder, build_bev_encoder

__all__ = ["BEVConfig", "BaseBEVEncoder", "build_bev_encoder"]
