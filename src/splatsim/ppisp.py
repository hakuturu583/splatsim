"""PPISP (Physically-Plausible ISP) integration for splatsim.

Loads the ``ppisp.json`` payload embedded in a splatsim USDZ (schema
``splatsim.ppisp/v1``, see ``3dgs_io.ppisp``) and applies the four-stage
exposure→vignetting→color→CRF correction to rendered RGB frames via the
CUDA op shipped by :mod:`ppisp` (``ppisp.ppisp_apply``).

PPISP was trained per-frame during scene reconstruction, so there is no
built-in parameter set for a *novel view*. This module estimates the
per-frame parameters (``exposure`` and 8-D ``color`` latent) for a novel
viewpoint via inverse-distance-weighted k-NN over the training rig-pose
positions. Static per-camera parameters (``vignetting`` and ``crf``) are
looked up directly by camera name.

The PPISP payload is optional. When the USDZ does not contain
``ppisp.json`` (or parsing fails), :func:`load_ppisp_tables` returns
``None`` and callers should fall back to the pre-PPISP behavior.
"""

from __future__ import annotations

import importlib as _importlib
import json as _json
import logging as _logging
import zipfile as _zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_3dgs_io = _importlib.import_module("3dgs_io")
_parse_ppisp = _3dgs_io.parse_ppisp

_PPISP_ARCHIVE_PATH = "ppisp.json"

_logger = _logging.getLogger(__name__)


@dataclass
class PpispTables:
    """Device-resident PPISP parameters ready to feed :func:`apply_ppisp`.

    ``vignetting`` / ``crf`` are static per camera and indexed by
    ``camera_index[name]``. ``frame_positions`` / ``frame_exposure`` /
    ``frame_color`` hold one entry per training frame, ordered
    consistently, and drive the k-NN interpolation for novel views.

    ``color_pinv_block_diag`` is not stored here because the CUDA op
    already applies the chromaticity homography internally when given
    the 8-D ``color`` latent.
    """

    camera_names: tuple[str, ...]
    camera_index: dict[str, int]
    vignetting: torch.Tensor  # [N, 3, 5]
    crf: torch.Tensor  # [N, 3, 4]
    frame_positions: torch.Tensor  # [M, 3]
    frame_exposure: torch.Tensor  # [M]
    frame_color: torch.Tensor  # [M, 8]

    @property
    def num_cameras(self) -> int:
        return len(self.camera_names)

    @property
    def num_frames(self) -> int:
        return int(self.frame_positions.shape[0])


def read_ppisp(usdz_path: str | Path) -> Any | None:
    """Return the parsed ``3dgs_io.Ppisp`` from ``usdz_path`` or ``None``.

    Returns ``None`` when the archive does not contain ``ppisp.json`` or
    parsing fails; the caller should fall back to non-PPISP behavior.
    """
    try:
        with _zipfile.ZipFile(usdz_path) as zf:
            with zf.open(_PPISP_ARCHIVE_PATH) as fp:
                doc = _json.load(fp)
    except KeyError:
        return None
    except (OSError, _zipfile.BadZipFile, _json.JSONDecodeError) as exc:
        _logger.info("PPISP not loaded from %s: %s", usdz_path, exc)
        return None
    try:
        return _parse_ppisp(doc)
    except Exception as exc:  # 3dgs_io raises ValueError on schema issues
        _logger.warning("PPISP present but not parseable in %s: %s", usdz_path, exc)
        return None


def _rig_positions_by_timestamp(rigs: list[Any]) -> dict[int, np.ndarray]:
    """Collect rig-center translations keyed by ``timestamp_us``.

    PPISP frames are keyed by timestamp only; the per-frame parameters
    describe scene state shared across all cameras of the rig. Using the
    rig-center translation (not any specific camera's) matches that
    invariant. Duplicate timestamps across rigs are kept as first-seen.
    """
    out: dict[int, np.ndarray] = {}
    for rig in rigs:
        for pose in rig.poses or []:
            ts = int(pose.timestamp_us)
            if ts in out:
                continue
            out[ts] = np.asarray(pose.translation, dtype=np.float64)
    return out


def build_ppisp_tables(
    ppisp: Any,
    rigs: list[Any],
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    centroid: np.ndarray | torch.Tensor | None = None,
) -> PpispTables | None:
    """Materialize :class:`PpispTables` from a parsed ``Ppisp`` + rigs.

    ``centroid`` (root-local → centered offset applied by
    :class:`splatsim.background.Background`) is subtracted from the
    frame positions so k-NN comparisons happen in the same frame as the
    ``viewmat`` fed to :class:`splatsim.renderer.Renderer`. Pass
    ``None`` when the caller has not re-centered the scene.

    Returns ``None`` if ``ppisp`` has no cameras or no frames.
    """
    cameras = ppisp.cameras or {}
    frames = ppisp.frames or []
    if not cameras or not frames:
        return None

    camera_names = tuple(sorted(cameras.keys()))
    camera_index = {name: i for i, name in enumerate(camera_names)}

    vignetting = np.stack(
        [np.asarray(cameras[n].vignetting, dtype=np.float32) for n in camera_names]
    )
    crf = np.stack([np.asarray(cameras[n].crf, dtype=np.float32) for n in camera_names])
    if vignetting.shape[1:] != (3, 5) or crf.shape[1:] != (3, 4):
        _logger.warning(
            "PPISP camera parameter shapes unexpected: vignetting=%s crf=%s",
            vignetting.shape,
            crf.shape,
        )
        return None

    ts_to_pos = _rig_positions_by_timestamp(rigs)
    frame_positions: list[np.ndarray] = []
    frame_exposure: list[float] = []
    frame_color: list[np.ndarray] = []
    missing: list[int] = []
    for frame in frames:
        ts = int(frame.timestamp_us)
        pos = ts_to_pos.get(ts)
        if pos is None:
            missing.append(ts)
            continue
        frame_positions.append(pos.astype(np.float32))
        frame_exposure.append(float(frame.exposure))
        color_vec = frame.color if frame.color is not None else (0.0,) * 8
        frame_color.append(np.asarray(color_vec, dtype=np.float32))

    if missing:
        _logger.info(
            "PPISP: %d/%d frames dropped (no matching rig pose timestamp).",
            len(missing),
            len(frames),
        )
    if not frame_positions:
        return None

    positions_np = np.stack(frame_positions)
    if centroid is not None:
        c = (
            centroid.detach().cpu().numpy()
            if isinstance(centroid, torch.Tensor)
            else np.asarray(centroid)
        ).astype(np.float32)
        positions_np = positions_np - c[None, :]

    device_t = torch.device(device)
    return PpispTables(
        camera_names=camera_names,
        camera_index=camera_index,
        vignetting=torch.from_numpy(vignetting).to(device_t, dtype=dtype),
        crf=torch.from_numpy(crf).to(device_t, dtype=dtype),
        frame_positions=torch.from_numpy(positions_np).to(device_t, dtype=dtype),
        frame_exposure=torch.tensor(frame_exposure, device=device_t, dtype=dtype),
        frame_color=torch.from_numpy(np.stack(frame_color)).to(device_t, dtype=dtype),
    )


def load_ppisp_tables(
    usdz_path: str | Path,
    rigs: list[Any],
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    centroid: np.ndarray | torch.Tensor | None = None,
) -> PpispTables | None:
    """One-shot: read ``ppisp.json`` from a USDZ and build the tables.

    Returns ``None`` if the USDZ has no PPISP payload, if parsing
    fails, or if the payload has no usable frames.
    """
    ppisp = read_ppisp(usdz_path)
    if ppisp is None:
        return None
    return build_ppisp_tables(
        ppisp, rigs, device=device, dtype=dtype, centroid=centroid
    )


def _knn_weights(
    positions: torch.Tensor,
    query: torch.Tensor,
    k: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(indices, weights)`` for inverse-distance k-NN on rows.

    ``positions`` is ``[M, 3]``, ``query`` is ``[3]``. Weights sum to 1.
    ``k`` is capped at ``M``.
    """
    d = (positions - query).norm(dim=-1)
    k_eff = int(min(k, d.shape[0]))
    topk_d, topk_i = torch.topk(d, k_eff, largest=False)
    w = 1.0 / (topk_d + eps)
    w = w / w.sum()
    return topk_i, w


def apply_ppisp(
    tables: PpispTables,
    rgb: torch.Tensor,
    camera_name: str,
    c2w_position: torch.Tensor | np.ndarray,
    *,
    k: int = 4,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply PPISP correction to a rendered RGB frame.

    ``rgb`` is ``[H, W, 3]`` in linear color space (the raw output of
    :mod:`gsplat` rasterization before any tone mapping). ``camera_name``
    picks the static vignetting/CRF; ``c2w_position`` is the camera
    center in the same world frame used for the renderer's ``viewmat``
    (i.e. tile-local, re-centered if the background centroid was
    subtracted at table-build time). The per-frame ``exposure`` and
    ``color`` latent are interpolated across the ``k`` closest training
    rig positions with inverse-distance weights.

    If ``camera_name`` is not in the PPISP payload, ``rgb`` is returned
    unchanged (the caller may still apply its own exposure scalar).
    """
    cam_idx = tables.camera_index.get(camera_name)
    if cam_idx is None:
        return rgb

    if not isinstance(c2w_position, torch.Tensor):
        query = torch.as_tensor(
            c2w_position,
            device=tables.frame_positions.device,
            dtype=tables.frame_positions.dtype,
        )
    else:
        query = c2w_position.to(
            device=tables.frame_positions.device, dtype=tables.frame_positions.dtype
        )
    query = query.reshape(-1)[:3]

    idx, w = _knn_weights(tables.frame_positions, query, k=k, eps=eps)
    exposure_interp = (tables.frame_exposure[idx] * w).sum().reshape(1)
    color_interp = (tables.frame_color[idx] * w.unsqueeze(-1)).sum(dim=0, keepdim=True)

    # ppisp CUDA op wants float32 contiguous tensors on the same device.
    ppisp_mod = _importlib.import_module("ppisp")
    ppisp_apply = ppisp_mod.ppisp_apply

    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    return ppisp_apply(
        exposure_interp,
        tables.vignetting,
        color_interp,
        tables.crf,
        rgb.to(torch.float32).contiguous(),
        None,
        width,
        height,
        int(cam_idx),
        0,
    )
