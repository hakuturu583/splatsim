"""OmniDreams renderer.

Parallels :class:`splatsim.renderer.Renderer` (the gsplat backend) so the gRPC
servicer treats "render a frame at this camera pose" identically regardless of
backend. Where the gsplat ``Renderer`` rasterises a static Gaussian scene from a
``viewmat``, this one steps a stateful autoregressive world model from a
``cam_to_world`` pose, after being seeded once with an anchor frame.

This class is also the single **integration seam** to NVIDIA's OmniDreams /
Cosmos-Dreams model via the FlashDreams runtime: every model-specific call is
isolated here and marked ``# INTEGRATION SEAM``, and resolved lazily at
:meth:`seed` time (never at import) so importing the package — or running the
3DGS backend — never pulls the multi-GB Cosmos stack.
"""

from __future__ import annotations

import io
import logging
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class OmniDreamsRenderer:
    """Camera renderer backed by the OmniDreams world model.

    Unlike the gsplat renderer this one is **stateful**: :meth:`seed` primes the
    autoregressive rollout from the anchor image and must be called (once, at
    Initialize) before :meth:`render`.
    """

    def __init__(self, width: int, height: int, *, device: str = "cuda") -> None:
        self.width = int(width)
        self.height = int(height)
        self.device = device
        self._pipeline = None  # FlashDreams pipeline, built lazily on first seed
        self._text_prompt = os.environ.get(
            "OMNIDREAMS_TEXT_PROMPT",
            "A photorealistic urban driving scene, clear daytime weather.",
        )

    @property
    def seeded(self) -> bool:
        return self._pipeline is not None

    def seed(self, initial_image: bytes, *, scene_path: str | None = None) -> None:
        """Prime the rollout from PNG/JPEG anchor bytes (the optional
        ``InitializeRequest.initial_image``). Builds the FlashDreams pipeline on
        first call and seeds its autoregressive context with the anchor frame,
        the scene / HD-map conditioning, and the text prompt."""
        anchor = self._fit_to_output(self._decode(initial_image))
        if self._pipeline is None:
            self._pipeline = self._build_pipeline()

        # INTEGRATION SEAM: prime the autoregressive context. Method name /
        # signature to be pinned against the target FlashDreams release, e.g.
        #   self._pipeline.reset(init_frame=anchor, scene=scene_path,
        #                        prompt=self._text_prompt)
        self._pipeline.reset(
            init_frame=anchor,
            scene=scene_path,
            prompt=self._text_prompt,
        )
        logger.info(
            "OmniDreams rollout seeded (%dx%d, scene=%s) from %d-byte anchor",
            self.width,
            self.height,
            scene_path,
            len(initial_image),
        )

    def render(self, cam_to_world: "NDArray[np.float64]") -> "NDArray[np.uint8]":
        """Generate the next frame at ``cam_to_world`` (4x4). Returns H x W x 3
        uint8 **RGB** (the servicer flips to BGR for DDS, as the gsplat path
        does). The pose is the model's trajectory conditioning for this step."""
        if self._pipeline is None:
            raise RuntimeError(
                "OmniDreamsRenderer.render called before seed(); the OmniDreams "
                "backend requires InitializeRequest.initial_image."
            )
        # INTEGRATION SEAM: one autoregressive world-model step. Returns the next
        # generated frame; normalise it to H x W x 3 uint8 RGB.
        frame = self._pipeline.step(pose=cam_to_world)
        return self._as_rgb_uint8(frame)

    # -- FlashDreams pipeline construction --------------------------------

    def _build_pipeline(self):
        """Instantiate the FlashDreams OmniDreams pipeline (heavy, GPU, gated)."""
        try:
            # INTEGRATION SEAM: FlashDreams OmniDreams pipeline construction, per
            # the flashdreams docs (integrations_v2/omnidreams):
            #   from omnidreams.config import OMNIDREAMS_PIPELINE_CONFIG
            #   pipeline = OMNIDREAMS_PIPELINE_CONFIG.setup().to("cuda").eval()
            from omnidreams.config import OMNIDREAMS_PIPELINE_CONFIG  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - env-specific
            raise RuntimeError(
                "The OmniDreams backend requires the FlashDreams runtime "
                "(NVIDIA/flashdreams, integrations_v2/omnidreams) and its gated "
                "Hugging Face weights. Install it in this image and set HF_TOKEN. "
                f"Import failed: {exc}"
            ) from exc

        logger.info("Building OmniDreams pipeline on %s ...", self.device)
        pipeline = OMNIDREAMS_PIPELINE_CONFIG.setup().to(self.device).eval()
        logger.info("OmniDreams pipeline ready")
        return pipeline

    # -- image helpers -----------------------------------------------------

    @staticmethod
    def _decode(data: bytes) -> "NDArray[np.uint8]":
        """Decode PNG/JPEG anchor bytes to an ``H x W x 3`` uint8 RGB array.

        Raises ``ValueError`` on empty or undecodable input so ``Initialize``
        fails loudly rather than seeding the model with garbage."""
        if not data:
            raise ValueError("initial_image is empty")
        try:
            from PIL import Image as PILImage  # noqa: PLC0415

            with PILImage.open(io.BytesIO(data)) as img:
                return np.asarray(img.convert("RGB"), dtype=np.uint8)
        except Exception as exc:  # pragma: no cover - depends on input data
            raise ValueError(f"failed to decode initial_image: {exc}") from exc

    def _fit_to_output(self, image: "NDArray[np.uint8]") -> "NDArray[np.uint8]":
        h, w = image.shape[:2]
        if (w, h) == (self.width, self.height):
            return np.ascontiguousarray(image)
        from PIL import Image as PILImage  # noqa: PLC0415

        resized = PILImage.fromarray(image).resize(
            (self.width, self.height), PILImage.BILINEAR
        )
        return np.asarray(resized, dtype=np.uint8)

    @staticmethod
    def _as_rgb_uint8(frame) -> "NDArray[np.uint8]":
        """Normalise a model output (torch tensor or ndarray, CHW/HWC, float or
        uint8) to an ``H x W x 3`` uint8 RGB array. Left non-contiguous — the
        servicer's single BGR copy compacts it, avoiding a redundant copy here.
        Trimmed to what the FlashDreams output actually needs once the seam is
        pinned; the broad shape/dtype handling covers the unpinned contract."""
        arr = frame.detach().cpu().numpy() if hasattr(frame, "detach") else frame
        arr = np.squeeze(np.asarray(arr))
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.moveaxis(arr, 0, -1)  # CHW -> HWC
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        arr = arr[:, :, :3]
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        return arr.astype(np.uint8, copy=False)
