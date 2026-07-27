from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

import spz

SH_C0 = 0.2820947917738781  # 1 / (2 * sqrt(pi))


@dataclass
class GaussianTensors:
    """GPU-ready Gaussian parameters in gsplat convention."""

    means: Tensor  # [N, 3]
    quats: Tensor  # [N, 4] wxyz (gsplat convention)
    scales: Tensor  # [N, 3] actual scale (exp applied)
    opacities: Tensor  # [N] range [0, 1] (sigmoid applied)
    colors: Tensor  # [N, 3] RGB or [N, K, 3] SH coefficients
    sh_degree: int  # 0 = RGB only, 1-3 = SH degree
    # Optional per-Gaussian LiDAR attributes. Populated only for scenes trained
    # with tier4/gaussian_factory (see PLY properties `lidar_intensity_raw`,
    # `lidar_raydrop_logit`). Standard 3DGS scenes leave these as None and the
    # LiDAR renderer falls back to SH-derived intensity and a fixed low raydrop.
    intensity_raw: Tensor | None = None  # [N] unbounded, sigmoid at render
    raydrop_logit: Tensor | None = None  # [N] unbounded logit
    # Optional per-Gaussian LiDAR participation mask. Emitted by 3dgs_io >=
    # v1.1.0 as the sidecar's `lidar_mask` channel: True (1) = the Gaussian
    # participates in LiDAR / near-field geometry, False (0) = appearance-only
    # / far-field (excluded from the LiDAR geometry pass). ``None`` means the
    # channel is absent, i.e. all Gaussians participate (backward compatible
    # with old 2-channel sidecars). Consumed only by the LiDAR renderer; the
    # RGB/camera path ignores it.
    lidar_mask: Tensor | None = None  # [N] bool; True = participates in LiDAR

    def __getitem__(self, idx: Tensor | slice) -> GaussianTensors:
        """Return a new instance with all tensor fields indexed/sliced by *idx*."""
        return GaussianTensors(
            means=self.means[idx],
            quats=self.quats[idx],
            scales=self.scales[idx],
            opacities=self.opacities[idx],
            colors=self.colors[idx],
            sh_degree=self.sh_degree,
            intensity_raw=None
            if self.intensity_raw is None
            else self.intensity_raw[idx],
            raydrop_logit=None
            if self.raydrop_logit is None
            else self.raydrop_logit[idx],
            lidar_mask=None if self.lidar_mask is None else self.lidar_mask[idx],
        )


def cloud_to_tensors(
    cloud: spz.GaussianCloud,  # ty: ignore[unresolved-attribute]
    device: torch.device,
    *,
    use_sh: bool = False,
) -> GaussianTensors:
    """Convert an spz GaussianCloud to gsplat-ready GPU tensors.

    Handles all convention differences:
    - Quaternion order: spz (x,y,z,w) -> gsplat (w,x,y,z)
    - Scales: log-scale -> actual scale via exp()
    - Opacities: logit -> [0,1] via sigmoid()
    - Colors: SH DC pre-activation -> RGB via dc * SH_C0 + 0.5
    """
    n = cloud.num_points

    # Positions: flat (N*3,) -> (N, 3)
    positions = np.array(cloud.positions, dtype=np.float32).reshape(n, 3)
    means = torch.from_numpy(positions).to(device)

    # Quaternions: spz (x,y,z,w) -> gsplat (w,x,y,z)
    quats_xyzw = np.array(cloud.rotations, dtype=np.float32).reshape(n, 4)
    quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]]
    quats = torch.from_numpy(quats_wxyz).to(device)

    # Scales: log-scale -> actual scale
    log_scales = np.array(cloud.scales, dtype=np.float32).reshape(n, 3)
    scales = torch.from_numpy(log_scales).to(device).exp()

    # Opacities: logit -> sigmoid
    alpha_logits = np.array(cloud.alphas, dtype=np.float32)
    opacities = torch.from_numpy(alpha_logits).to(device).sigmoid()

    # Colors
    sh_degree = cloud.sh_degree
    raw_colors = np.array(cloud.colors, dtype=np.float32).reshape(n, 3)

    if use_sh and sh_degree > 0:
        # Build full SH coefficient tensor [N, K, 3]
        # DC component: raw spz colors value (gsplat applies SH basis internally)
        dc = torch.from_numpy(raw_colors).to(device).unsqueeze(1)  # [N, 1, 3]

        # Higher-order SH coefficients
        raw_sh = np.array(cloud.sh, dtype=np.float32)
        num_higher_coeffs = len(raw_sh) // (n * 3)
        higher_sh = torch.from_numpy(raw_sh.reshape(n, num_higher_coeffs, 3)).to(
            device
        )  # [N, K-1, 3]

        colors = torch.cat([dc, higher_sh], dim=1)  # [N, K, 3]
    else:
        # RGB mode: convert SH DC to actual RGB
        colors_rgb = raw_colors * SH_C0 + 0.5
        colors_rgb = np.clip(colors_rgb, 0.0, 1.0)
        colors = torch.from_numpy(colors_rgb).to(device)  # [N, 3]
        sh_degree = 0

    return GaussianTensors(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        sh_degree=sh_degree,
    )


def quat_to_rotation_matrix(q: Tensor) -> Tensor:
    """Convert a (w,x,y,z) quaternion to a 3x3 rotation matrix.

    Differentiable torch counterpart of
    :func:`splatsim._geometry.quat_to_matrix` (same wxyz convention); kept here
    because it stays on-device and preserves gradients for the render path.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack(
        [
            torch.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]
            ),
            torch.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]
            ),
            torch.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
            ),
        ]
    )  # [3, 3]


def quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Quaternion multiplication in (w,x,y,z) convention.

    q1: [4] single quaternion
    q2: [N, 4] batch of quaternions
    Returns: [N, 4]
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )  # [N, 4]


def apply_rigid_transform(
    tensors: GaussianTensors,
    position: Tensor,
    rotation: Tensor,
) -> GaussianTensors:
    """Apply a rigid body transform (translation + rotation) to Gaussian tensors.

    Args:
        tensors: Base Gaussian tensors to transform.
        position: [3] world-space translation.
        rotation: [4] wxyz quaternion for orientation.

    Returns:
        New GaussianTensors with transformed means and quats.
    """
    rot_mat = quat_to_rotation_matrix(rotation)  # [3, 3]

    # Transform positions: R @ p + t
    new_means = tensors.means @ rot_mat.T + position.unsqueeze(0)

    # Compose quaternions
    new_quats = quat_multiply(rotation, tensors.quats)

    return GaussianTensors(
        means=new_means,
        quats=new_quats,
        scales=tensors.scales,
        opacities=tensors.opacities,
        colors=tensors.colors,
        sh_degree=tensors.sh_degree,
        intensity_raw=tensors.intensity_raw,
        raydrop_logit=tensors.raydrop_logit,
        lidar_mask=tensors.lidar_mask,
    )
