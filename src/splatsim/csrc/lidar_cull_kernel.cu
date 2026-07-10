// Fused LiDAR cull kernel: computes a single boolean keep mask combining
// the shell/frustum test (min_range, max_range) and the elevation-FOV test
// against a spherical band around the sensor's up-axis.
//
// All arithmetic is done in fp32; a single pass over means (Nx3) and
// scales (Nx3) reads 24 bytes per gaussian and writes 1 byte to `keep`,
// dwarfing the ~130 bytes/gaussian traffic of the PyTorch chain.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

namespace {

template <bool USE_ELEV>
__global__ void lidar_cull_kernel(
    const float* __restrict__ means,       // (N, 3) row-major
    const float* __restrict__ scales,      // (N, 3) row-major
    const float* __restrict__ sensor_pos,  // (3,) device pointer, read once per block
    const float* __restrict__ up_world,    // (3,) device pointer, ignored when !USE_ELEV
    const float min_range,
    const float max_range,        // <0 means "no upper bound"
    const float cull_scale_sigmas,
    const float sin_min,
    const float cos_min,
    const float sin_max,
    const float cos_max,
    const int32_t N,
    bool* __restrict__ keep) {
  // Cache the sensor pose in shared memory so all N threads amortize a
  // single global fetch. sensor_pos and up_world are tiny (3 floats each)
  // and passing them via device pointers avoids the launch-side sync we
  // used to pay to read them as scalar kernel args.
  __shared__ float s_sensor[3];
  __shared__ float s_up[3];
  if (threadIdx.x < 3) {
    s_sensor[threadIdx.x] = sensor_pos[threadIdx.x];
    if (USE_ELEV) {
      s_up[threadIdx.x] = up_world[threadIdx.x];
    }
  }
  __syncthreads();

  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  const int64_t idx3 = i * 3;
  // Load means + scales as (N, 3) row-major. Not 16B-aligned, so use
  // scalar loads and rely on the compiler to coalesce. On sm_89 this is
  // essentially the same throughput as float4.
  const float mx = means[idx3 + 0];
  const float my = means[idx3 + 1];
  const float mz = means[idx3 + 2];

  const float sx = scales[idx3 + 0];
  const float sy = scales[idx3 + 1];
  const float sz = scales[idx3 + 2];

  // delta = means - sensor_pos
  const float dx = mx - s_sensor[0];
  const float dy = my - s_sensor[1];
  const float dz = mz - s_sensor[2];

  // dist = |delta|
  const float dist = sqrtf(dx * dx + dy * dy + dz * dz);

  // max_scale (nan → 0).
  // Match torch.nan_to_num(scales.amax(...), nan=0.0): if *any* scale
  // component is NaN, amax returns NaN and we then map it to zero.
  // CUDA's fmaxf skips NaN inputs (per IEEE 754), so we can't just
  // rely on it — check explicitly first.
  float ms;
  if (isnan(sx) || isnan(sy) || isnan(sz)) {
    ms = 0.0f;
  } else {
    ms = fmaxf(fmaxf(sx, sy), sz);
  }
  const float margin = cull_scale_sigmas * ms;

  // Shell test: overlap [dist - margin, dist + margin] with [min_range, max_range]
  bool k = (dist + margin) >= min_range;
  if (max_range >= 0.0f) {
    k = k && ((dist - margin) <= max_range);
  }

  // Elevation-FOV test (linear-in-margin bound):
  //   z_s = dot(delta, up_world)
  //   sin(elev) = z_s / dist  (guarded: dist>0 handled below)
  //   keep if
  //     z_s + margin·cos(elev_min) >= dist·sin(elev_min)   (above lower plane)
  //     z_s - margin·cos(elev_max) <= dist·sin(elev_max)   (below upper plane)
  if (USE_ELEV && k) {
    const float z_s = dx * s_up[0] + dy * s_up[1] + dz * s_up[2];
    k = k && ((z_s + margin * cos_min) >= (dist * sin_min));
    k = k && ((z_s - margin * cos_max) <= (dist * sin_max));
  }

  keep[i] = k;
}

}  // anonymous namespace

// Public entry point: bool mask (torch::kBool) of length N.
torch::Tensor lidar_cull_mask(
    const torch::Tensor& means,          // (N, 3), float32, CUDA
    const torch::Tensor& scales,         // (N, 3), float32, CUDA
    const torch::Tensor& sensor_pos,     // (3,), float32, CUDA — same device
    const torch::Tensor& up_world,       // (3,), float32, CUDA — ignored if !use_elev
    double min_range,
    double max_range,                    // pass negative to disable upper bound
    double cull_scale_sigmas,
    bool use_elev,
    double sin_min,
    double cos_min,
    double sin_max,
    double cos_max) {
  TORCH_CHECK(means.is_cuda(), "means must be CUDA");
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA");
  TORCH_CHECK(means.dtype() == torch::kFloat32, "means must be float32");
  TORCH_CHECK(scales.dtype() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(means.is_contiguous(), "means must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(means.dim() == 2 && means.size(1) == 3, "means must be (N, 3)");
  TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3, "scales must be (N, 3)");
  TORCH_CHECK(means.size(0) == scales.size(0), "means and scales must have same N");

  const int64_t N64 = means.size(0);
  TORCH_CHECK(N64 <= std::numeric_limits<int32_t>::max(),
              "cull kernel supports up to INT32_MAX gaussians");
  const int32_t N = static_cast<int32_t>(N64);

  auto keep = torch::empty({N64}, means.options().dtype(torch::kBool));
  if (N == 0) {
    return keep;
  }

  // sensor_pos / up_world stay on device: the kernel loads them once per
  // block via shared memory. We coerce dtype/device to match `means` so
  // the caller can pass a slice of the 4×4 pose matrix directly.
  auto sensor_dev = sensor_pos.detach().to(means.device(), torch::kFloat32).contiguous();
  auto up_dev = up_world.detach().to(means.device(), torch::kFloat32).contiguous();
  TORCH_CHECK(sensor_dev.numel() == 3, "sensor_pos must have exactly 3 elements");
  TORCH_CHECK(up_dev.numel() == 3, "up_world must have exactly 3 elements");

  const int block = 256;
  const int grid = static_cast<int>((N + block - 1) / block);

  auto stream = at::cuda::getCurrentCUDAStream();

  if (use_elev) {
    lidar_cull_kernel<true><<<grid, block, 0, stream>>>(
        means.data_ptr<float>(),
        scales.data_ptr<float>(),
        sensor_dev.data_ptr<float>(),
        up_dev.data_ptr<float>(),
        static_cast<float>(min_range),
        static_cast<float>(max_range),
        static_cast<float>(cull_scale_sigmas),
        static_cast<float>(sin_min),
        static_cast<float>(cos_min),
        static_cast<float>(sin_max),
        static_cast<float>(cos_max),
        N,
        keep.data_ptr<bool>());
  } else {
    lidar_cull_kernel<false><<<grid, block, 0, stream>>>(
        means.data_ptr<float>(),
        scales.data_ptr<float>(),
        sensor_dev.data_ptr<float>(),
        up_dev.data_ptr<float>(),
        static_cast<float>(min_range),
        static_cast<float>(max_range),
        static_cast<float>(cull_scale_sigmas),
        0.f, 1.f, 0.f, 1.f,   // unused
        N,
        keep.data_ptr<bool>());
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return keep;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "lidar_cull_mask",
      &lidar_cull_mask,
      "Fused frustum + elevation-FOV cull for LiDAR (CUDA)",
      pybind11::arg("means"),
      pybind11::arg("scales"),
      pybind11::arg("sensor_pos"),
      pybind11::arg("up_world"),
      pybind11::arg("min_range"),
      pybind11::arg("max_range"),
      pybind11::arg("cull_scale_sigmas"),
      pybind11::arg("use_elev"),
      pybind11::arg("sin_min"),
      pybind11::arg("cos_min"),
      pybind11::arg("sin_max"),
      pybind11::arg("cos_max"));
}
