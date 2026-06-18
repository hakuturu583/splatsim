"""Reader helpers for the scene USDZ format emitted by ``3dgs_io.save_scene_usdz``.

A scene USDZ archive bundles:

* ``default.usda`` — USD stage referencing ``tileset.json`` and ``scene.json``.
* ``scene.json`` — splatsim.scene/v1 metadata (world transform, render defaults, ...).
* ``tileset.json`` — Cesium 3D Tiles document declaring ``EXT_3dgs_spz``.
* ``chunks/chunk_NNNNNN.spz`` — Niantic SPZ binaries, one per tile.

3dgs_io only ships a writer (``save_scene_usdz``); this module is splatsim's
reader.
"""

from __future__ import annotations

import importlib as _importlib
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from splatsim._conversions import GaussianTensors, cloud_to_tensors

_3dgs_io = _importlib.import_module("3dgs_io")
_load_spz = _3dgs_io.load_spz


def read_scene_json(usdz_path: str | Path) -> dict[str, Any]:
    """Read ``scene.json`` out of a scene USDZ without extracting the whole archive."""
    with zipfile.ZipFile(usdz_path) as zf:
        if "scene.json" not in zf.namelist():
            raise ValueError(
                f"{usdz_path}: missing scene.json (not a 3dgs_io scene USDZ)"
            )
        return json.loads(zf.read("scene.json"))


def extract_scene_usdz(usdz_path: str | Path) -> Path:
    """Extract a scene USDZ to a fresh temp directory and return its path."""
    out_dir = Path(tempfile.mkdtemp(prefix="splatsim_usdz_"))
    with zipfile.ZipFile(usdz_path) as zf:
        zf.extractall(out_dir)
    return out_dir


def load_spz_tileset(
    tileset_path: str | Path,
    device: torch.device,
    *,
    use_sh: bool = False,
) -> tuple[GaussianTensors, np.ndarray]:
    """Load a Cesium 3D Tiles document whose children are SPZ files.

    Returns the concatenated :class:`GaussianTensors` together with the
    root tile's transform (row-major 4x4, ECEF→tile-local convention).
    """
    tileset_path = Path(tileset_path)
    base_dir = tileset_path.parent
    with tileset_path.open() as f:
        tileset = json.load(f)

    root = tileset["root"]
    # 3D Tiles stores transforms column-major; flip to row-major.
    root_transform = (
        np.asarray(
            root.get("transform", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
            dtype=np.float64,
        )
        .reshape(4, 4)
        .T
    )

    tensor_list: list[GaussianTensors] = []
    for child in root.get("children", []):
        chunk_uri = child["content"]["uri"]
        cloud = _load_spz(str(base_dir / chunk_uri))
        if cloud.num_points == 0:
            continue
        tensor_list.append(cloud_to_tensors(cloud, device, use_sh=use_sh))

    if not tensor_list:
        raise ValueError(f"{tileset_path}: no SPZ chunks found")

    merged = _concat_tensors(tensor_list)
    return merged, root_transform


def _concat_tensors(tensors: list[GaussianTensors]) -> GaussianTensors:
    sh_degrees = {t.sh_degree for t in tensors}
    if len(sh_degrees) != 1:
        raise ValueError(f"Mixed SH degrees across chunks: {sh_degrees}")
    sh_degree = sh_degrees.pop()
    return GaussianTensors(
        means=torch.cat([t.means for t in tensors], dim=0),
        quats=torch.cat([t.quats for t in tensors], dim=0),
        scales=torch.cat([t.scales for t in tensors], dim=0),
        opacities=torch.cat([t.opacities for t in tensors], dim=0),
        colors=torch.cat([t.colors for t in tensors], dim=0),
        sh_degree=sh_degree,
    )
