"""Preprocessing + geometry configuration for the BEV encoder.

Defaults match the deployed ``oneplanner_bev_encoder.onnx`` variant
(``use_intensity: true``): 5-dim points ``(x, y, z, intensity, time_lag)``,
voxel grid ``1440 x 1440 x 41`` over a ``+/-122.4 m`` horizontal range, producing
a ``[1, 512, 180, 180]`` BEV feature map (``1.36 m`` per pixel, ego at the
grid centre).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class BEVConfig:
    """Voxelisation + output geometry for the BEV encoder.

    ``coors_order`` selects the axis order of the ``coors`` tensor fed to the
    ONNX. The graph's ``spatial_shape`` attribute is ``[1440, 1440, 41]`` (z last),
    so the coordinates must be z-last -> ``"xyz"``. This is a property of the ONNX,
    not of the backend, so both backends use it; exposed as a knob only because it
    is the single convention most likely to differ between exports.
    """

    point_cloud_range: tuple[float, float, float, float, float, float] = (
        -122.4,
        -122.4,
        -3.0,
        122.4,
        122.4,
        5.0,
    )
    voxel_size: tuple[float, float, float] = (0.17, 0.17, 0.2)
    max_num_points: int = 10
    max_voxels: int = 160000
    num_point_features: int = 5  # x, y, z, intensity, time_lag
    use_intensity: bool = True
    coors_order: str = "xyz"
    feature_channels: int = 512
    bev_size: tuple[int, int] = (180, 180)

    @property
    def grid_size(self) -> tuple[int, int, int]:
        """(gx, gy, gz) voxel counts implied by range / voxel size."""
        lo = self.point_cloud_range[:3]
        hi = self.point_cloud_range[3:]
        vs = self.voxel_size
        return (
            int(round((hi[0] - lo[0]) / vs[0])),
            int(round((hi[1] - lo[1]) / vs[1])),
            int(round((hi[2] - lo[2]) / vs[2])),
        )

    @property
    def meters_per_pixel(self) -> tuple[float, float]:
        """(x, y) BEV pixel pitch in metres (voxel size x downsample factor)."""
        bh, bw = self.bev_size
        span_x = self.point_cloud_range[3] - self.point_cloud_range[0]
        span_y = self.point_cloud_range[4] - self.point_cloud_range[1]
        return span_x / bw, span_y / bh
