"""splatsim LiDAR evaluation package.

The evaluation compares a LiDAR scan **rendered** from a reconstructed 3DGS
``.usdz`` scene against the **ground-truth** LiDAR scan recorded in the matching
WebAuto / T4 dataset, at the *same* ego pose. It is organised as a small
metric-plugin framework so that each evaluation item lives in its own file and
exposes a common :class:`~eval.metrics.base.LidarEvalMetric` interface:

* :class:`~eval.metrics.chamfer.ChamferMetric` -- geometric agreement of the two
  point clouds (symmetric Chamfer distance, raw and range-aware).
* :class:`~eval.metrics.bev_encoder.BEVEncoderMetric` -- semantic/representation
  agreement: both clouds are pushed through the OnePlanner BEV encoder and the
  resulting bird's-eye-view feature maps are compared.

The per-frame driver (:mod:`eval.runner`) renders + masks each frame once
(:mod:`eval.frame`) and hands the shared :class:`~eval.frame.FrameData` to every
registered metric, which scores it and logs its own scalars / imagery to a
single shared Rerun recording.
"""

from __future__ import annotations
