"""TensorRT backend for the OnePlanner BEV encoder ONNX.

The encoder ONNX embeds the ``autoware`` sparse-conv custom ops, so it can only
run under TensorRT with ``autoware_tensorrt_plugins`` loaded. This backend:

1. ``ctypes``-loads the plugin ``.so`` (RTLD_GLOBAL) and registers its creators.
2. Builds (and disk-caches) a TensorRT engine from the ONNX, with a dynamic
   optimisation profile over the variable voxel count.
3. Runs inference using **torch** tensors as the CUDA I/O buffers -- their
   ``data_ptr()`` are handed to ``execute_async_v3`` -- so no ``pycuda`` /
   ``cuda-python`` dependency is needed.

Only ``tensorrt`` (matching the system ``libnvinfer.so`` the plugins were built
against -- TensorRT 10 / CUDA 12 here) and ``torch`` are required.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from .config import BEVConfig
from .voxelize import hard_voxelize

_INPUT_NAMES = ("voxels", "num_points_per_voxel", "coors")
_OUTPUT_NAME = "bev_feature_map"

# Loaded plugin libraries, keyed by realpath, so repeated backend construction in
# one process does not re-open (and re-register) the same creators.
_LOADED_PLUGINS: set[str] = set()


def _require_tensorrt():
    try:
        import tensorrt as trt
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SystemExit(
            "tensorrt is required for the BEV-encoder metric but is not "
            "installed.\nInstall the optional 'bev' extra:  uv sync --extra bev\n"
            "(the wheel must match the system libnvinfer the autoware plugins "
            "were built against -- TensorRT 10 / CUDA 12 here)."
        ) from exc
    return trt


def _load_plugins(plugin_path: str, trt, logger) -> None:
    """ctypes-load the plugin .so once and register its TensorRT creators."""
    real = os.path.realpath(plugin_path)
    if real in _LOADED_PLUGINS:
        return
    if not os.path.exists(real):
        raise SystemExit(f"TensorRT plugin library not found: {plugin_path}")
    ctypes.CDLL(real, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(logger, "")
    _LOADED_PLUGINS.add(real)


class TensorRTBEVEncoder:
    """Runs ``oneplanner_bev_encoder.onnx`` via TensorRT + autoware plugins."""

    def __init__(
        self,
        onnx_path: str,
        plugin_path: str,
        cfg: BEVConfig,
        *,
        device: str = "cuda",
        engine_cache: str | None = None,
        fp16: bool = False,
        opt_voxels: int = 90000,
    ) -> None:
        if not torch.cuda.is_available():
            raise SystemExit("The BEV-encoder metric requires a CUDA device.")
        self.cfg = cfg
        self.device = torch.device(device)
        self._onnx_path = onnx_path
        self._fp16 = fp16
        self._opt_voxels = opt_voxels

        self._trt = _require_tensorrt()
        self._logger = self._trt.Logger(self._trt.Logger.WARNING)
        _load_plugins(plugin_path, self._trt, self._logger)

        engine_bytes = self._build_or_load_engine(engine_cache)
        runtime = self._trt.Runtime(self._logger)
        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise SystemExit("Failed to deserialize the BEV TensorRT engine.")
        self._context = self._engine.create_execution_context()
        self._validate_io()

    # -- engine build / cache ------------------------------------------------

    def _cache_path(self, engine_cache: str | None) -> Path:
        if engine_cache:
            return Path(engine_cache)
        gpu = torch.cuda.get_device_name(self.device).replace(" ", "_")
        tag = hashlib.sha1(
            f"{os.path.realpath(self._onnx_path)}|{os.path.getmtime(self._onnx_path)}"
            f"|{gpu}|fp16={self._fp16}|trt={self._trt.__version__}".encode()
        ).hexdigest()[:16]
        return Path(self._onnx_path).with_suffix(f".{gpu}.{tag}.engine")

    def _build_or_load_engine(self, engine_cache: str | None) -> bytes:
        cache = self._cache_path(engine_cache)
        if cache.exists():
            print(f"[bev] loading cached TensorRT engine {cache}")
            return cache.read_bytes()
        print(f"[bev] building TensorRT engine from {self._onnx_path} (one-off)...")
        engine_bytes = self._build_engine()
        try:
            cache.write_bytes(engine_bytes)
            print(f"[bev] cached engine to {cache}")
        except OSError as exc:  # pragma: no cover - fs dependent
            print(f"[bev] WARNING: could not cache engine ({exc}); rebuilding next run")
        return engine_bytes

    def _build_engine(self) -> bytes:
        trt = self._trt
        builder = trt.Builder(self._logger)
        flags = 0
        if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, self._logger)
        with open(self._onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errs = "\n".join(
                    str(parser.get_error(i)) for i in range(parser.num_errors)
                )
                raise SystemExit(f"Failed to parse BEV ONNX:\n{errs}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
        if self._fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Dynamic profile over the voxel count (dim 0 of every input).
        lo, opt, hi = 1, self._opt_voxels, self.cfg.max_voxels
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "voxels",
            (lo, self.cfg.max_num_points, self.cfg.num_point_features),
            (opt, self.cfg.max_num_points, self.cfg.num_point_features),
            (hi, self.cfg.max_num_points, self.cfg.num_point_features),
        )
        profile.set_shape("num_points_per_voxel", (lo,), (opt,), (hi,))
        profile.set_shape("coors", (lo, 3), (opt, 3), (hi, 3))
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise SystemExit("TensorRT failed to build the BEV engine.")
        return bytes(serialized)

    # -- inference -----------------------------------------------------------

    def _validate_io(self) -> None:
        names = {
            self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)
        }
        missing = (set(_INPUT_NAMES) | {_OUTPUT_NAME}) - names
        if missing:
            raise SystemExit(
                f"BEV ONNX is missing expected tensors {sorted(missing)}; "
                f"engine has {sorted(names)}"
            )

    @torch.no_grad()
    def encode(self, points: np.ndarray) -> torch.Tensor:
        """Voxelise + run the encoder.

        ``points`` (N, F) -> (C, H, W) float32 tensor on ``self.device`` (kept
        on-GPU so the metric's comparison avoids a host round-trip).
        """
        pts = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        voxels, num_points, coors = hard_voxelize(pts, self.cfg)
        m = int(voxels.shape[0])
        if m == 0:
            h, w = self.cfg.bev_size
            return torch.zeros(
                (self.cfg.feature_channels, h, w),
                dtype=torch.float32,
                device=self.device,
            )

        buffers = {
            "voxels": voxels.contiguous(),
            "num_points_per_voxel": num_points.contiguous(),
            "coors": coors.contiguous(),
        }
        for name, tensor in buffers.items():
            self._context.set_input_shape(name, tuple(tensor.shape))
            self._context.set_tensor_address(name, tensor.data_ptr())

        out_shape = tuple(self._context.get_tensor_shape(_OUTPUT_NAME))
        out = torch.empty(out_shape, dtype=torch.float32, device=self.device)
        self._context.set_tensor_address(_OUTPUT_NAME, out.data_ptr())

        stream = torch.cuda.current_stream(self.device)
        ok = self._context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed for the BEV encoder.")
        stream.synchronize()
        return out[0]  # drop batch -> (C, H, W), still on device
