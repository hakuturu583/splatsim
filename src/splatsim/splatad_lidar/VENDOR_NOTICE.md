# Vendored: SplatAD spherical LiDAR rasterizer

This directory is a **vendored copy of the gsplat fork from
[carlinds/splatad](https://github.com/carlinds/splatad)** (SplatAD, CVPR 2025),
licensed **Apache-2.0**. It is included in-tree so gaussian_factory's LiDAR sim can
use SplatAD's purpose-built **`lidar_rasterization`** (spherical projection, median
range, per-Gaussian `lidar_features` for MLP intensity/ray-drop decoding,
rolling-shutter, beam-divergence antialiasing, non-uniform elevation) — which renders
sharp LiDAR rings where the upstream gsplat expected-depth panorama smears them.

## Why in-tree (not a gsplat pin swap)
The camera/RGB path stays on the pinned upstream gsplat (nerfstudio 4e52698). SplatAD's
gsplat is v1.0.0 and swapping the pin would regress the camera path + 2026 features. So
this is a SEPARATE, self-contained CUDA extension that coexists with the pinned gsplat:

- CUDA extension renamed `gsplat_cuda` -> **`splatad_lidar_cuda`** (no build/.so collision).
- Python package imports rewritten `gsplat.*` -> `gaussian_factory.splatad_lidar.*`.
- No `torch.ops` namespace collision (this build calls the ext module `_C` directly).
- Import as: `from gaussian_factory.splatad_lidar.rendering import lidar_rasterization`.

## Build (Blackwell / sm_120)
JIT-compiles on first import. `third_party/glm` is vendored (headers only). Set
`TORCH_CUDA_ARCH_LIST="9.0+PTX"` (nvcc 12.4 has no native sm_120; PTX forward-compat).
Verified to build + run alongside pip gsplat 1.5.3.

## Attribution
- SplatAD / gsplat fork: carlinds/splatad, Apache-2.0 (see LICENSE in upstream repo).
- GLM (`third_party/glm`): g-truc/glm, MIT / Happy Bunny (see `third_party/glm/copying.txt`).
