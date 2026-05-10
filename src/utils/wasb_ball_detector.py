"""WASB-SBDT ball detector wrapper.

Plugs the vendored HRNet model from
``third_party/wasb_sbdt/src/models/hrnet.py`` into the
:class:`src.utils.ball_detector.BallDetector` interface.

The upstream :class:`TracknetV2Detector` is hard-coded to CUDA and
expects a Hydra-style config, so this module bypasses it: we build
HRNet directly with an OmegaConf-wrapped config dict that mirrors
``third_party/wasb_sbdt/src/configs/model/wasb.yaml``, load the
checkpoint, and run our own postprocessor.

Detection flow per frame:

1. **Buffer** — keep a 3-frame ring of raw BGR frames. Early frames are
   front-padded with the first frame so detection is available from
   frame 0 (with reduced accuracy on the first two).
2. **Warp** — letterbox each buffered frame into ``input_size`` via the
   same ``get_affine_transform`` matrix WASB uses, plus ImageNet
   normalisation on RGB pixels.
3. **Forward** — stack as ``(1, 9, H, W)``, run HRNet, sigmoid the
   output ``(1, 3, H, W)`` heatmaps.
4. **Postprocess** — connected-component blobs above
   ``confidence`` threshold on the last (current-frame) heatmap; pick
   the blob with the highest sum-of-weights and return its
   heatmap-weighted centroid mapped back to original image space, with
   the peak heatmap value as confidence in ``[0, 1]``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.utils.ball_detector import BallDetector

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_WASB_SRC = _REPO_ROOT / "third_party" / "wasb_sbdt" / "src"


# Mirrors third_party/wasb_sbdt/src/configs/model/wasb.yaml. HRNet reads
# this through OmegaConf; both dict-style and attribute access are used
# inside the upstream module.
_WASB_MODEL_CFG: dict = {
    "name": "hrnet",
    "frames_in": 3,
    "frames_out": 3,
    "inp_height": 288,
    "inp_width": 512,
    "out_height": 288,
    "out_width": 512,
    "rgb_diff": False,
    "out_scales": [0],
    "MODEL": {
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "PRETRAINED_LAYERS": ["*"],
            "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
            "STAGE1": {
                "NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS": [1], "NUM_CHANNELS": [32], "FUSE_METHOD": "SUM",
            },
            "STAGE2": {
                "NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2], "NUM_CHANNELS": [16, 32], "FUSE_METHOD": "SUM",
            },
            "STAGE3": {
                "NUM_MODULES": 1, "NUM_BRANCHES": 3, "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2], "NUM_CHANNELS": [16, 32, 64], "FUSE_METHOD": "SUM",
            },
            "STAGE4": {
                "NUM_MODULES": 1, "NUM_BRANCHES": 4, "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2, 2], "NUM_CHANNELS": [16, 32, 64, 128], "FUSE_METHOD": "SUM",
            },
            "DECONV": {
                "NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2,
            },
        },
        "INIT_WEIGHTS": True,
    },
}


# RGB ImageNet normalisation used by ``build_img_transforms`` in
# ``third_party/wasb_sbdt/src/dataloaders/__init__.py``.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _get_affine_transform(
    center: tuple[float, float],
    scale: float,
    output_size: tuple[int, int],
    inv: bool = False,
) -> np.ndarray:
    """Compute the same letterbox affine matrix WASB's data loader uses.

    Equivalent to ``utils.image.get_affine_transform(center, scale, 0,
    output_size, inv=inv)`` upstream, simplified for the
    no-rotation/no-shift inference path.  Returns a ``(2, 3)`` matrix
    suitable for ``cv2.warpAffine``.
    """
    cx, cy = float(center[0]), float(center[1])
    src_w = float(scale)
    dst_w = float(output_size[0])
    dst_h = float(output_size[1])

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0] = [cx, cy]
    src[1] = [cx, cy - 0.5 * src_w]
    # 3rd anchor lies 90° counter-clockwise from (src[1] - src[0]).
    d0 = src[0] - src[1]
    src[2] = src[1] + np.array([-d0[1], d0[0]], dtype=np.float32)

    dst[0] = [0.5 * dst_w, 0.5 * dst_h]
    dst[1] = [0.5 * dst_w, 0.5 * dst_h - 0.5 * dst_w]
    d0d = dst[0] - dst[1]
    dst[2] = dst[1] + np.array([-d0d[1], d0d[0]], dtype=np.float32)

    if inv:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


def _affine_apply(pt: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Apply a ``(2, 3)`` affine to a single ``(2,)`` point."""
    homogeneous = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
    return mat @ homogeneous


def _pick_device(requested: str | None) -> str:
    """Resolve ``'auto' | 'cpu' | 'cuda' | 'mps'`` against runtime support."""
    import torch

    want = (requested or "auto").strip().lower()
    if want == "auto":
        if torch.cuda.is_available():
            return "cuda"
        # MPS works for WASB (no funky attention ops), but stay on CPU
        # by default to mirror the conservative choice we use for
        # ``hmr_world`` on macOS.
        return "cpu"
    if want.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("WASB requested cuda but CUDA is not available — falling back to CPU")
        return "cpu"
    if want == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            logger.warning("WASB requested mps but MPS is not available — falling back to CPU")
            return "cpu"
    return want


class WASBBallDetector(BallDetector):
    """Single-frame interface around the vendored WASB HRNet model."""

    _frames_in: int = 3

    def __init__(
        self,
        checkpoint: str | Path,
        confidence: float = 0.3,
        input_size: tuple[int, int] = (512, 288),
        device: str | None = None,
    ) -> None:
        import sys
        import torch
        from omegaconf import OmegaConf

        ckpt_path = Path(checkpoint).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"WASB checkpoint not found at {ckpt_path}. Run "
                "third_party/wasb_sbdt/src/setup_scripts/setup_weights.sh "
                "or download from MODEL_ZOO.md."
            )

        # Make upstream HRNet importable without dragging in the rest of
        # the WASB src tree (`models/__init__.py` imports several model
        # files that aren't needed for inference).
        wasb_src = str(_WASB_SRC)
        added = False
        if wasb_src not in sys.path:
            sys.path.insert(0, wasb_src)
            added = True
        try:
            from models.hrnet import HRNet  # type: ignore
        finally:
            if added:
                # Leave it in place: subsequent imports of upstream
                # submodules (if anyone needs them) should keep working.
                # The path entry is harmless.
                pass

        cfg = OmegaConf.create(_WASB_MODEL_CFG)
        self._inp_w, self._inp_h = int(input_size[0]), int(input_size[1])
        if (self._inp_w, self._inp_h) != (_WASB_MODEL_CFG["inp_width"], _WASB_MODEL_CFG["inp_height"]):
            # The checkpoint was trained at 512x288; running at a
            # different resolution silently degrades accuracy. Allow
            # the override but warn.
            logger.warning(
                "WASB input_size=%s differs from checkpoint training "
                "size (512, 288); detection accuracy will degrade.",
                input_size,
            )
            cfg.inp_width = self._inp_w
            cfg.inp_height = self._inp_h
            cfg.out_width = self._inp_w
            cfg.out_height = self._inp_h

        self._device = _pick_device(device)
        model = HRNet(cfg)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        else:
            state_dict = state
        # Checkpoints from upstream were saved through DataParallel and
        # carry a "module." prefix on every key.
        stripped = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(stripped, strict=False)
        if missing:
            logger.warning("WASB load_state_dict: %d missing keys", len(missing))
        if unexpected:
            logger.warning("WASB load_state_dict: %d unexpected keys", len(unexpected))
        model.to(self._device).eval()
        self._model = model
        self._torch = torch
        self._confidence = float(confidence)
        self._buffer: list[np.ndarray] = []

    def _preprocess_buffer(
        self, frame_size: tuple[int, int]
    ) -> tuple["np.ndarray", np.ndarray]:
        h, w = frame_size
        center = (w / 2.0, h / 2.0)
        scale = float(max(h, w))
        trans = _get_affine_transform(center, scale, (self._inp_w, self._inp_h), inv=False)
        trans_inv = _get_affine_transform(center, scale, (self._inp_w, self._inp_h), inv=True)

        channels = []
        for f in self._buffer:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            warped = cv2.warpAffine(rgb, trans, (self._inp_w, self._inp_h), flags=cv2.INTER_LINEAR)
            x = warped.astype(np.float32) / 255.0
            x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
            x = x.transpose(2, 0, 1)  # (3, H, W)
            channels.append(x)
        stacked = np.concatenate(channels, axis=0)  # (9, H, W)
        return stacked, trans_inv

    def detect(self, frame: np.ndarray) -> tuple[float, float, float] | None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame must be (H, W, 3); got {frame.shape}")

        # Maintain a 3-frame ring; front-pad on the first frame so the
        # first call already runs the network instead of returning None.
        if not self._buffer:
            self._buffer = [frame.copy(), frame.copy(), frame.copy()]
        else:
            self._buffer.append(frame.copy())
            if len(self._buffer) > self._frames_in:
                self._buffer.pop(0)

        h, w = frame.shape[:2]
        stacked, trans_inv = self._preprocess_buffer((h, w))
        inp = self._torch.from_numpy(stacked).unsqueeze(0).to(self._device)

        with self._torch.no_grad():
            out = self._model(inp)
        # HRNet returns ``dict[scale -> tensor]``; WASB uses scale=0 only.
        logits = out[0]
        hms = self._torch.sigmoid(logits).cpu().numpy()  # (1, 3, H, W)
        # Channel index 2 corresponds to the most recent input frame.
        hm = hms[0, -1]

        peak = float(hm.max())
        if peak < self._confidence:
            return None

        _, binary = cv2.threshold(hm, self._confidence, 1, cv2.THRESH_BINARY)
        n_labels, labels = cv2.connectedComponents(binary.astype(np.uint8))

        best: tuple[float, float, float] | None = None
        best_blob_score = -1.0
        for label in range(1, n_labels):
            ys, xs = np.where(labels == label)
            ws = hm[ys, xs]
            blob_score = float(ws.sum())
            if blob_score <= best_blob_score:
                continue
            x = float(np.sum(xs * ws) / np.sum(ws))
            y = float(np.sum(ys * ws) / np.sum(ws))
            peak_blob = float(ws.max())
            best = (x, y, peak_blob)
            best_blob_score = blob_score

        if best is None:
            return None

        cx, cy, conf = best
        uv = _affine_apply(np.array([cx, cy], dtype=np.float32), trans_inv)
        return float(uv[0]), float(uv[1]), conf
