# OmniDreams rendering backend

A **second rendering backend** for SplatSim that speaks the *same* gRPC contract
as the 3D Gaussian Splatting (3DGS) backend, but synthesises camera frames with
NVIDIA's [OmniDreams / Cosmos-Dreams](https://github.com/nv-tlabs/omni-dreams)
autoregressive video world model (served through the
[FlashDreams](https://github.com/NVIDIA/flashdreams) runtime) instead of
rasterising Gaussians.

## Why this exists / the abstraction

Both backends implement `splatsim.v1.RenderingService`
(`proto/splatsim/v1/rendering_service.proto`) and listen on port `50051`:
`Initialize` → `StreamCameraData` (poses in) → frames published over CycloneDDS
as `sensor_msgs/Image`. A client therefore talks to either backend identically —
the choice is which image you run.

The **only** protocol extension is one optional field:

```proto
message InitializeRequest {
  ...
  optional bytes initial_image = 16;  // PNG/JPEG anchor frame
}
```

OmniDreams needs one real RGB frame to anchor scene appearance before it can
generate; the 3DGS backend renders from geometry alone and ignores this field,
so it stays **optional** and fully backward compatible.

## Mapping onto the shared contract

| RenderingService RPC   | OmniDreams backend                                             |
| ---------------------- | -------------------------------------------------------------- |
| `Initialize`           | Build FlashDreams pipeline, **seed** rollout from `initial_image`, open DDS publishers. Requires `initial_image`. |
| `StreamCameraData`     | Each streamed ego pose = one autoregressive world-model step → publish frame. The pose stream is the model's trajectory conditioning. |
| `InitializeLidar` / `StreamLidarData` / `InitializeCameraRig` / `StreamRigData` | `UNIMPLEMENTED` — OmniDreams is a monocular camera world model. |
| `SetLod`               | No-op (`enabled=false`); LoD is not applicable. |

## Layout

- `renderer.py` — `OmniDreamsRenderer`, a stateful `Renderer`-shaped adapter
  (`seed()` then `render(cam_to_world)`) that is also the single **integration
  seam** to FlashDreams. Every model-specific call
  (`OMNIDREAMS_PIPELINE_CONFIG.setup()`, `reset`, `step`) is isolated here and
  marked `# INTEGRATION SEAM`, resolved lazily at `Initialize` time so importing
  the package (or running the 3DGS backend) never pulls the multi-GB Cosmos
  stack.
- `server.py` — `OmniDreamsServicer`, the gRPC servicer.
- `cli.py` — `splatsim-omnidreams-server` entry point (shares the serve/CLI
  boilerplate in `splatsim.grpc_service._serve` with the 3DGS backend).

## Running

Build and run the dedicated image (separate from the 3DGS `docker/Dockerfile`):

```bash
docker buildx build -f docker/Dockerfile.omnidreams -t splatsim-omnidreams .
docker run --rm -it --gpus all -e HF_TOKEN=hf_xxx -p 50051:50051 splatsim-omnidreams
```

`HF_TOKEN` must have read access to the gated `nvidia/omni-dreams-models`
weights. OmniDreams needs ~48 GB VRAM. Optional: `OMNIDREAMS_TEXT_PROMPT` sets
the driving-context prompt (a backend config detail, not a protocol field).

## Integration-seam status

The FlashDreams call sites in `renderer.py` (`_build_pipeline`, and the `reset`
/ `step` calls) follow the published FlashDreams `integrations_v2/omnidreams`
API and must be pinned against the `FLASHDREAMS_REF` used in
`docker/Dockerfile.omnidreams`. They raise a clear error if FlashDreams is
absent, so the transport layer is exercisable independently of the (gated,
48 GB-VRAM) model.
