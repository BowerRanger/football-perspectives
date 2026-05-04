"""Per-frame camera calibration via TVCalib.

TVCalib (Theiner & Ewerth, WACV 2023, MIT,
https://github.com/MM4SPA/tvcalib) is included as a git submodule under
``third_party/tvcalib``.  Unlike PnLCalib's discrete PnP solve, TVCalib
runs a differentiable per-frame AdamW optimisation: a DeepLab segmenter
extracts pitch-line pixels, then point-to-line / point-to-pointcloud NDC
distances are minimised against a SoccerNet 3D pitch model with
prior-regularised camera parameters.

Per-segment masking lets the loss ignore segments with zero detected
points, so the solver degrades gracefully on frames where only a handful
of pitch features are visible — exactly the failure mode that defeats
PnLCalib's PnP-from-keypoints path.

Coordinate frame
----------------
TVCalib returns camera parameters in the SoccerNet convention: pitch
centre origin, ``X ∈ [-52.5, 52.5]``, ``Y ∈ [-34, 34]``, ``Z`` pointing
into the ground (main-tribune camera at ``c_y > 0``, ``c_z < 0``).  The
transform to our corner-origin, z-up frame is identical to the one used
for PnLCalib — both libraries share the same SN pitch convention — so we
reuse :func:`src.utils.neural_calibrator.convert_pnlcalib_to_ours`.
"""

from __future__ import annotations

import logging
import sys
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.neural_calibrator import (
    NeuralCalibration,
    _normalize_device,
    convert_pnlcalib_to_ours,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TVCALIB_ROOT = _REPO_ROOT / "third_party" / "tvcalib"
_SN_SEG_ROOT = _TVCALIB_ROOT / "sn_segmentation"
_WEIGHTS_DIR = _REPO_ROOT / "data" / "models" / "tvcalib"

# TIB Cloud mirror of the DeepLabV3 pitch-line segmenter trained by the
# TVCalib authors.  See third_party/tvcalib/README.md.
_SEGMENTER_URL = "https://tib.eu/cloud/s/x68XnTcZmsY4Jpg/download/train_59.pt"
_SEGMENTER_FILENAME = "train_59.pt"

# Hard-coded segmenter input dims (the DeepLab checkpoint was trained at
# 256 × 455).  Also used by the skeletonise + extremities pipeline.
_SEG_H = 256
_SEG_W = 455

# Supported camera priors.  Each maps to one of the cam_distr modules
# shipped with TVCalib; see ``third_party/tvcalib/tvcalib/cam_distr/``.
_CAM_DISTR_BY_PRIOR = {
    "center": "tv_main_center",
    "left": "tv_main_left",
    "right": "tv_main_right",
    "behind": "tv_main_behind",
    "tribune": "tv_main_tribune",
}


def _download_if_missing(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` if the file is missing (idempotent).

    Writes to a ``.partial`` sibling and renames on completion to avoid
    leaving half-written weight files behind on Ctrl-C.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    logger.info("Downloading TVCalib segmenter weights: %s -> %s", url, dest)
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 — trusted URL
    tmp.replace(dest)


def _install_torch_six_shim() -> None:
    """Shim ``torch._six`` for TVCalib on torch >= 2.

    TVCalib's ``sncalib_dataset.py`` imports ``string_classes`` from
    ``torch._six``, which was removed in torch 2.0.  The attribute used
    to be ``(str, bytes)`` — on Python 3 ``str`` alone covers the check.
    """
    if "torch._six" in sys.modules:
        return
    try:
        import torch._six  # noqa: F401
        return
    except ImportError:
        pass
    import types

    shim = types.ModuleType("torch._six")
    shim.string_classes = (str, bytes)  # type: ignore[attr-defined]
    sys.modules["torch._six"] = shim


def _install_pytorch_lightning_shim() -> None:
    """Provide a minimal ``pytorch_lightning`` shim when absent.

    TVCalib's ``cam_modules.py`` uses ``from pytorch_lightning import
    LightningModule`` only as a base class — no Trainer, no Callbacks.
    When the real package isn't installed we substitute a stub whose
    ``LightningModule`` is just ``torch.nn.Module``.  If the real package
    (either 1.x or 2.x) is already importable, we leave it alone.
    """
    if "pytorch_lightning" in sys.modules:
        return
    try:
        import pytorch_lightning  # noqa: F401
        return
    except ImportError:
        pass

    import types

    import torch.nn as _nn

    shim = types.ModuleType("pytorch_lightning")
    shim.LightningModule = _nn.Module  # type: ignore[attr-defined]

    def _seed_everything(seed: int, workers: bool = False) -> int:
        import random as _r

        import torch as _t

        _r.seed(seed)
        np.random.seed(seed)
        _t.manual_seed(seed)
        return int(seed)

    shim.seed_everything = _seed_everything  # type: ignore[attr-defined]
    sys.modules["pytorch_lightning"] = shim


def _import_tvcalib_modules() -> dict[str, Any]:
    """Import TVCalib with sys.path munging + optional PL shim.

    Mirrors :func:`src.utils.neural_calibrator._import_pnlcalib_modules`:
    TVCalib uses bare ``from tvcalib...`` imports, so we prepend the
    submodule root to ``sys.path`` before importing and leave it there
    (subsequent calls are cheap — imports are cached in ``sys.modules``).
    """
    if not _TVCALIB_ROOT.exists():
        raise RuntimeError(
            f"TVCalib submodule missing at {_TVCALIB_ROOT}. "
            "Run `git submodule update --init --recursive`."
        )

    for extra in (_TVCALIB_ROOT, _SN_SEG_ROOT):
        p = str(extra)
        if p not in sys.path:
            sys.path.insert(0, p)

    _install_pytorch_lightning_shim()
    _install_torch_six_shim()

    try:
        from tvcalib.cam_modules import SNProjectiveCamera  # type: ignore  # noqa: F401
        from tvcalib.inference import (  # type: ignore
            InferenceDatasetCalibration,
            InferenceSegmentationModel,
        )
        from tvcalib.module import TVCalibModule  # type: ignore
        from tvcalib.utils.objects_3d import (  # type: ignore
            SoccerPitchLineCircleSegments,
            SoccerPitchSNCircleCentralSplit,
        )
    except ImportError as exc:
        # Surface the underlying import failure so config/env problems
        # are diagnosable.  The top-level hint still points at the
        # extras install for the common case.
        raise RuntimeError(
            f"TVCalib import failed: {exc}. "
            "If deps are missing, install extras with "
            "`pip install -e '.[tvcalib]'` (kornia, SoccerNet)."
        ) from exc

    try:
        from sn_segmentation.src.custom_extremities import (  # type: ignore
            generate_class_synthesis,
            get_line_extremities,
        )
    except ImportError as exc:
        raise RuntimeError(
            "sn_segmentation submodule missing under third_party/tvcalib. "
            "Run `git submodule update --init --recursive`."
        ) from exc

    return {
        "SNProjectiveCamera": SNProjectiveCamera,
        "TVCalibModule": TVCalibModule,
        "InferenceSegmentationModel": InferenceSegmentationModel,
        "InferenceDatasetCalibration": InferenceDatasetCalibration,
        "SoccerPitchLineCircleSegments": SoccerPitchLineCircleSegments,
        "SoccerPitchSNCircleCentralSplit": SoccerPitchSNCircleCentralSplit,
        "generate_class_synthesis": generate_class_synthesis,
        "get_line_extremities": get_line_extremities,
    }


def _load_cam_distr(prior: str, sigma_scale: float) -> dict:
    """Import and instantiate the cam_distr prior for ``prior``.

    ``prior`` must be one of :data:`_CAM_DISTR_BY_PRIOR`.
    """
    mod_name = _CAM_DISTR_BY_PRIOR[prior]
    mod = __import__(f"tvcalib.cam_distr.{mod_name}", fromlist=["get_cam_distr"])
    return mod.get_cam_distr(sigma_scale, 1, 1)


class TVCalibCalibrator:
    """Per-frame TVCalib camera calibration.

    Exposes the same :meth:`calibrate` signature as
    :class:`src.utils.neural_calibrator.PnLCalibrator`, so the calibration
    stage can dispatch on either backend through
    :class:`src.utils.neural_calibrator.CalibratorProtocol`.

    The segmenter and solver are constructed lazily on the first
    :meth:`calibrate` call; subsequent calls reuse them.  The solver's
    internal AdamW state is reset at the start of every
    ``self_optim_batch`` call (TVCalib's own behaviour), so frames are
    independent — there is no implicit warm-start across calls.
    """

    def __init__(
        self,
        weights_dir: Path | None = None,
        device: str = "auto",
        prior: str = "center",
        optim_steps: int = 800,
        ndc_loss_threshold: float = 0.017,
        sigma_scale: float = 1.96,
        image_width: int = 1920,
        image_height: int = 1080,
        skeleton_radius: int = 4,
        join_maxdist: int = 30,
    ) -> None:
        self._weights_dir = (
            Path(weights_dir) if weights_dir is not None else _WEIGHTS_DIR
        )
        self._device = _normalize_device(device)

        prior_key = (prior or "center").strip().lower()
        if prior_key not in _CAM_DISTR_BY_PRIOR:
            raise ValueError(
                f"Unknown TVCalib prior {prior!r}; "
                f"choose from {sorted(_CAM_DISTR_BY_PRIOR)}."
            )
        self._prior = prior_key
        self._optim_steps = int(optim_steps)
        self._ndc_loss_threshold = float(ndc_loss_threshold)
        self._sigma_scale = float(sigma_scale)
        self._image_width = int(image_width)
        self._image_height = int(image_height)
        self._skeleton_radius = int(skeleton_radius)
        self._join_maxdist = int(join_maxdist)

        self._loaded = False
        self._modules: dict[str, Any] = {}
        self._segmenter: Any = None
        self._solver: Any = None
        self._object3d: Any = None

    # ------------------------------------------------------------------ setup

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        self._modules = _import_tvcalib_modules()
        import torch
        from SoccerNet.Evaluation.utils_calibration import SoccerPitch
        from torchvision.models.segmentation import deeplabv3_resnet101

        weights_path = self._weights_dir / _SEGMENTER_FILENAME
        _download_if_missing(_SEGMENTER_URL, weights_path)

        # TVCalib's InferenceSegmentationModel.__init__ calls
        # ``torch.load(checkpoint)`` without ``map_location``, which
        # blows up on CPU-only machines when the checkpoint was saved
        # with CUDA storage (``train_59.pt`` is).  Build the segmenter
        # manually with the correct map_location and then install it
        # into a bare InferenceSegmentationModel instance so the rest
        # of TVCalib sees the expected attributes.
        InferenceSegmentationModel = self._modules["InferenceSegmentationModel"]
        model = deeplabv3_resnet101(
            num_classes=len(SoccerPitch.lines_classes) + 1, aux_loss=True,
        )
        state = torch.load(str(weights_path), map_location=self._device)
        model.load_state_dict(state["model"], strict=False)
        model.to(self._device).eval()

        seg = InferenceSegmentationModel.__new__(InferenceSegmentationModel)
        seg.device = self._device
        seg.model = model
        self._segmenter = seg

        SoccerPitchLineCircleSegments = self._modules[
            "SoccerPitchLineCircleSegments"
        ]
        SoccerPitchSNCircleCentralSplit = self._modules[
            "SoccerPitchSNCircleCentralSplit"
        ]
        self._object3d = SoccerPitchLineCircleSegments(
            device=self._device,
            base_field=SoccerPitchSNCircleCentralSplit(),
        )

        cam_distr = _load_cam_distr(self._prior, self._sigma_scale)

        TVCalibModule = self._modules["TVCalibModule"]
        self._solver = TVCalibModule(
            self._object3d,
            cam_distr,
            None,
            (self._image_height, self._image_width),
            optim_steps=self._optim_steps,
            device=self._device,
            tqdm_kwqargs={"disable": True, "leave": False},
        )
        self._loaded = True
        logger.info(
            "TVCalib loaded on device=%s, prior=%s, optim_steps=%d",
            self._device,
            self._prior,
            self._optim_steps,
        )

    # --------------------------------------------------------------- inference

    def _preprocess_frame(self, frame_bgr: np.ndarray) -> Any:
        """BGR → normalised (1, 3, 256, 455) tensor for the segmenter."""
        import torch

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (_SEG_W, _SEG_H), interpolation=cv2.INTER_LINEAR,
        )
        tensor = (
            torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(self._device)

    def _build_x_dict(
        self,
        keypoints_raw: dict[str, list[dict[str, float]]],
        source_width: int,
        source_height: int,
    ) -> dict[str, Any]:
        """Convert the segmenter's per-class point dict into the per-sample
        tensor dict expected by :class:`TVCalibModule`.

        Delegates to TVCalib's own
        :meth:`InferenceDatasetCalibration.prepare_per_sample` and then
        adds the leading ``(batch, temporal)`` dims the solver requires.
        """
        InferenceDatasetCalibration = self._modules["InferenceDatasetCalibration"]

        # prepare_per_sample expects every segment to have an entry (even
        # empty); TVCalib's InferenceDatasetCalibration.__getitem__ does
        # this fill-in before calling.  We replicate that here.
        filled: dict[str, list[dict[str, float]]] = {}
        for name in self._object3d.segment_names:
            filled[name] = list(keypoints_raw.get(name, []))

        per_sample = InferenceDatasetCalibration.prepare_per_sample(
            filled,
            self._object3d,
            num_points_on_line_segments=4,
            num_points_on_circle_segments=8,
            image_width_source=source_width,
            image_height_source=source_height,
            pad_pixel_position_xy=0.0,
        )
        x_dict: dict[str, Any] = {}
        for k, v in per_sample.items():
            x_dict[k] = v.unsqueeze(0).unsqueeze(0).to(self._device)
        return x_dict

    def calibrate(self, frame_bgr: np.ndarray) -> NeuralCalibration | None:
        """Run TVCalib on a single BGR frame.

        Returns ``None`` when the segmenter picks up no usable pitch
        features or when the solver's NDC loss exceeds a generous
        upper bound (five times the configured ``ndc_loss_threshold``).
        """
        self._ensure_loaded()
        import torch

        src_h, src_w = frame_bgr.shape[:2]

        with torch.no_grad():
            img = self._preprocess_frame(frame_bgr)
            sem_classes = self._segmenter.inference(img)  # [1, H, W]
        sem_np = sem_classes[0].detach().cpu().numpy().astype(np.uint8)

        buckets = self._modules["generate_class_synthesis"](
            sem_np, radius=self._skeleton_radius,
        )
        if not buckets:
            logger.debug("TVCalib: segmenter produced no line classes")
            return None

        keypoints_raw = self._modules["get_line_extremities"](
            buckets,
            maxdist=self._join_maxdist,
            width=_SEG_W,
            height=_SEG_H,
            num_points_lines=4,
            num_points_circles=8,
        )
        if not keypoints_raw:
            logger.debug("TVCalib: no extremities extracted from buckets")
            return None

        x_dict = self._build_x_dict(keypoints_raw, src_w, src_h)

        n_line_pts = int(x_dict["lines__is_keypoint_mask"].sum().item())
        n_circle_pts = int(x_dict["circles__is_keypoint_mask"].sum().item())
        total_pts = n_line_pts + n_circle_pts
        if total_pts < 4:
            logger.debug(
                "TVCalib: only %d valid points in x_dict — skipping solve",
                total_pts,
            )
            return None

        per_sample_loss, cam, _ = self._solver.self_optim_batch(x_dict)
        loss_ndc = float(
            per_sample_loss["loss_ndc_total"].detach().cpu().item(),
        )
        if not np.isfinite(loss_ndc):
            logger.debug("TVCalib: non-finite NDC loss")
            return None
        # Generous upper gate (5× threshold).  Strict acceptance happens
        # downstream in the plausibility + MAD-consensus filter in
        # src/stages/calibration.py.
        if loss_ndc > self._ndc_loss_threshold * 5.0:
            logger.debug(
                "TVCalib: NDC loss %.4f above gate %.4f — rejecting",
                loss_ndc,
                self._ndc_loss_threshold * 5.0,
            )
            return None

        R_sn = cam.rotation[0].detach().cpu().numpy().astype(np.float64)
        phi = cam.phi_dict_flat
        C_sn = np.array(
            [
                float(phi["c_x"].detach().cpu().numpy().reshape(-1)[0]),
                float(phi["c_y"].detach().cpu().numpy().reshape(-1)[0]),
                float(phi["c_z"].detach().cpu().numpy().reshape(-1)[0]),
            ],
            dtype=np.float64,
        )

        rvec, tvec, world_position = convert_pnlcalib_to_ours(R_sn, C_sn)

        K = cam.intrinsics_raster[0].detach().cpu().numpy().astype(np.float64)

        return NeuralCalibration(
            K=K,
            rvec=rvec,
            tvec=tvec,
            world_position=world_position,
        )
