"""JIT-compiled CUDA extension for the fused LiDAR cull kernel.

The extension exposes a single function ``lidar_cull_mask`` that produces
the boolean keep mask combining the range shell test and (optionally) the
elevation-FOV band test in a single CUDA kernel launch. It replaces the
~10-op PyTorch chain in :func:`splatsim.lidar_renderer.render_lidar_panorama`.

Compilation is performed lazily on first use via ``torch.utils.cpp_extension.load``.
The Docker builder stage runs this once so the runtime image (no nvcc) just
imports the ``.so`` from ``TORCH_EXTENSIONS_DIR`` — see :func:`_load_prebuilt`.
Failures fall back to the pure-PyTorch path in the renderer.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

import torch

# `Any` rather than a concrete Protocol: the loaded object is a Python
# module produced by pybind11 whose function signatures aren't visible to
# static analysers, and we call exactly one attribute (``lidar_cull_mask``).
_EXT: Any = None
_EXT_LOCK = threading.Lock()
_EXT_LOAD_FAILED = False
_EXT_LOAD_ERROR: Optional[BaseException] = None

_SRC_PATH = Path(__file__).resolve().parent / "csrc" / "lidar_cull_kernel.cu"


def _try_load() -> Any:
    """Compile and load the CUDA extension, or return ``None`` on failure."""
    global _EXT, _EXT_LOAD_FAILED, _EXT_LOAD_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_LOAD_FAILED:
        return None
    with _EXT_LOCK:
        if _EXT is not None:
            return _EXT
        if _EXT_LOAD_FAILED:
            return None
        if not torch.cuda.is_available():
            _EXT_LOAD_FAILED = True
            _EXT_LOAD_ERROR = RuntimeError("CUDA not available")
            return None
        try:
            _EXT = _load_prebuilt() or _load_via_jit()
            if _EXT is None:
                _EXT_LOAD_FAILED = True
                _EXT_LOAD_ERROR = RuntimeError(
                    "no pre-built .so found and JIT path unavailable"
                )
            return _EXT
        except Exception as exc:  # noqa: BLE001 — we want any failure to fall back
            _EXT_LOAD_FAILED = True
            _EXT_LOAD_ERROR = exc
            return None


_EXT_NAME = "splatsim_lidar_cull_ext"


def _build_dir() -> Path:
    """Where the compiled ``.so`` lives — same convention torch itself uses.

    ``torch.utils.cpp_extension.load(name=X)`` writes the artifact to
    ``$TORCH_EXTENSIONS_DIR/X/X.so`` (or a per-python-and-cuda-version
    subdir of ``~/.cache`` if the env var is unset). We compute the same
    path so the "load pre-built .so" path picks up what the Docker
    builder stage produced.
    """
    from torch.utils.cpp_extension import _get_build_directory  # type: ignore[attr-defined]

    return Path(_get_build_directory(_EXT_NAME, verbose=False))


def _load_prebuilt() -> Any:
    """Try to load a previously-compiled ``.so`` without invoking nvcc.

    This is the path the runtime Docker image uses: the builder image
    JITs the extension into ``$TORCH_EXTENSIONS_DIR`` and the runtime
    image (which has no CUDA toolkit) imports the ``.so`` directly.
    """
    try:
        build_dir = _build_dir()
        so_path = build_dir / f"{_EXT_NAME}.so"
        if not so_path.exists():
            return None
        from torch.utils.cpp_extension import _import_module_from_library  # type: ignore[attr-defined]

        return _import_module_from_library(
            _EXT_NAME, str(build_dir), is_python_module=True
        )
    except Exception:
        return None


def _load_via_jit() -> Any:
    from torch.utils.cpp_extension import load

    # `load(name=X)` uses ``$TORCH_EXTENSIONS_DIR/X/`` automatically.
    return load(
        name=_EXT_NAME,
        sources=[str(_SRC_PATH)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def is_available() -> bool:
    """Return ``True`` iff the extension is loaded and usable."""
    return _try_load() is not None


def last_error() -> Optional[BaseException]:
    """Return the exception raised on the last failed load attempt, if any."""
    return _EXT_LOAD_ERROR


def lidar_cull_mask(
    means: torch.Tensor,
    scales: torch.Tensor,
    sensor_pos: torch.Tensor,
    up_world: torch.Tensor,
    *,
    min_range: float,
    max_range: Optional[float],
    cull_scale_sigmas: float,
    use_elev: bool,
    sin_min: float,
    cos_min: float,
    sin_max: float,
    cos_max: float,
) -> torch.Tensor:
    """Fused frustum + elevation-FOV cull. Returns a (N,) bool tensor.

    ``max_range=None`` disables the upper bound.
    """
    ext = _try_load()
    if ext is None:
        raise RuntimeError(
            "splatsim CUDA cull extension is not available; "
            f"last load error: {_EXT_LOAD_ERROR!r}"
        )
    return ext.lidar_cull_mask(
        means,
        scales,
        sensor_pos,
        up_world,
        float(min_range),
        float(max_range) if max_range is not None else -1.0,
        float(cull_scale_sigmas),
        bool(use_elev),
        float(sin_min),
        float(cos_min),
        float(sin_max),
        float(cos_max),
    )
