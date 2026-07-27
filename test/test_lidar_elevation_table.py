"""Elevation-table consistency between LiDAR beam emission and reconstruction.

Regression guard for tier4/splatsim#75: the point-cloud reconstruction table
(``_sensor_row_elevations`` → ``LidarRenderer._elevs``) must use the *same*
per-row elevations as the beam-emission side (``_build_lidar_coeffs`` →
``LidarSensorSpec.coeffs(...).row_elevations_rad``).

Before the fix, ``_sensor_row_elevations`` ignored
``LidarSensorSpec.row_elevations_rad`` and fell back to a uniform elevation fan
whenever ``sensor_type`` was unknown/empty, so a scene carrying an explicit
calibrated per-beam table (e.g. a USDZ rig exported with ``--rig-lidar``) fired
beams at the calibrated angles but reconstructed points on a uniform fan — the
per-row error projected ground returns ~1.3 m too low.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from splatsim.lidar_renderer import (
    _TABLES_RAD,
    LidarSensorSpec,
    _sensor_row_elevations,
)

# A hand-made, strictly-descending (top→bottom), NON-uniform beam table. The
# gaps between rows deliberately vary so it cannot coincide with any uniform
# linspace fan.
_NONUNIFORM_ELEV_DEG = (12.0, 3.0, -1.0, -5.0, -15.0, -24.0)
_NONUNIFORM_ELEV_RAD = tuple(math.radians(v) for v in _NONUNIFORM_ELEV_DEG)


def _calibrated_spec() -> LidarSensorSpec:
    """Spec with a calibrated per-beam table and an unknown/empty sensor_type."""
    return LidarSensorSpec(
        name="rig_lidar",
        sensor_type="",  # not in _TABLES_RAD → old code fell back to uniform
        s2b=np.eye(4),
        n_columns=64,
        n_rows_uniform=len(_NONUNIFORM_ELEV_RAD),
        row_elevations_rad=_NONUNIFORM_ELEV_RAD,
    )


def _uniform_fan(spec: LidarSensorSpec) -> torch.Tensor:
    """The uniform linspace fan the old code would have produced for ``spec``."""
    elevs = torch.linspace(
        spec.el_hi_rad,
        spec.el_lo_rad,
        spec.n_rows_uniform,
        dtype=torch.float32,
    )
    elevs[0] = elevs[0] - 1e-6
    elevs[-1] = elevs[-1] - 1e-6
    return elevs


def test_reconstruction_uses_calibrated_table() -> None:
    """Calibrated table must win over the uniform fallback (core invariant)."""
    spec = _calibrated_spec()
    recon = _sensor_row_elevations(spec)
    expected = torch.tensor(_NONUNIFORM_ELEV_RAD, dtype=torch.float32)
    assert torch.allclose(recon, expected), (
        "reconstruction ignored the calibrated row_elevations_rad table"
    )


def test_reconstruction_is_not_uniform_fan() -> None:
    """Guard against a silent fallback to the uniform fan for a calibrated spec."""
    spec = _calibrated_spec()
    recon = _sensor_row_elevations(spec)
    fan = _uniform_fan(spec)
    # Same shape here (n_rows_uniform == len(table)), so a real value diff.
    assert recon.shape == fan.shape
    assert not torch.allclose(recon, fan), (
        "reconstruction fell back to the uniform elevation fan"
    )


def test_known_sensor_type_unchanged() -> None:
    """A spec with a known sensor_type still returns its _TABLES_RAD table."""
    sensor_type = next(iter(_TABLES_RAD))
    spec = LidarSensorSpec(
        name="known",
        sensor_type=sensor_type,
        s2b=np.eye(4),
        n_columns=64,
    )
    recon = _sensor_row_elevations(spec)
    expected = torch.tensor(_TABLES_RAD[sensor_type], dtype=torch.float32)
    assert torch.allclose(recon, expected)


def test_no_calibrated_table_falls_back_to_uniform() -> None:
    """No row_elevations_rad + unknown sensor_type → uniform fan (unchanged)."""
    spec = LidarSensorSpec(
        name="uniform",
        sensor_type="",
        s2b=np.eye(4),
        n_columns=64,
        n_rows_uniform=16,
        row_elevations_rad=(),
    )
    recon = _sensor_row_elevations(spec)
    assert torch.allclose(recon, _uniform_fan(spec))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_emission_and_reconstruction_share_table() -> None:
    """End-to-end invariant: emission and reconstruction tables must be equal.

    ``spec.coeffs(device).row_elevations_rad`` is the exact table gsplat fires
    beams with; ``_sensor_row_elevations`` is the table used to unproject the
    range panorama. They must match row-for-row, or points land at the wrong
    elevation. This fails on the pre-fix code (uniform fan on the recon side).
    """
    spec = _calibrated_spec()
    device = torch.device("cuda")
    emission = spec.coeffs(device).row_elevations_rad.detach().cpu()
    recon = _sensor_row_elevations(spec)
    assert emission.shape == recon.shape
    assert torch.allclose(emission, recon, atol=1e-6), (
        "beam-emission and point-cloud-reconstruction elevation tables diverge"
    )
