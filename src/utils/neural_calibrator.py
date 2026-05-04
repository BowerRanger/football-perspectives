"""Neural pitch calibration via PnLCalib.

PnLCalib (https://github.com/mguti97/PnLCalib, GPL-2.0) is included as a git
submodule under ``third_party/PnLCalib``.  This module wraps its ``inference.py``
so the rest of the pipeline can call a single ``calibrate(frame)`` method.

Weights are downloaded on first use to ``data/models/pnlcalib/`` and cached.

Coordinate convention
---------------------
PnLCalib uses a pitch-centred world frame with z pointing *down* (crossbar at
z = -2.44).  Our pipeline uses the FIFA near-left-corner convention with z
pointing *up* (pitch ``x ∈ [0, 105]``, ``y ∈ [0, 68]``, near = y = 0, far =
y = 68).  The transformation is:

    x_ours = x_pnl + 52.5
    y_ours = 34 - y_pnl
    z_ours = -z_pnl

For a full camera pose ``[R_pnl | t_pnl]`` in PnLCalib world, the equivalent
pose in ours is::

    R_ours = R_pnl @ diag(1, -1, -1)
    t_ours = R_pnl @ [-52.5, 34, 0]^T + t_pnl

Both produce identical pixel projections for a given pitch point.
"""

from __future__ import annotations

import logging
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PNLCALIB_ROOT = _REPO_ROOT / "third_party" / "PnLCalib"
_WEIGHTS_DIR = _REPO_ROOT / "data" / "models" / "pnlcalib"
_SV_KP_URL = "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp"
_SV_LINES_URL = "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines"

# Pitch dimensions (FIFA) — must match src/utils/pitch.py.
_PITCH_LENGTH = 105.0
_PITCH_WIDTH = 68.0

# Coordinate-conversion constants derived from the pitch dimensions.
_HALF_LENGTH = _PITCH_LENGTH / 2.0  # 52.5
_HALF_WIDTH = _PITCH_WIDTH / 2.0    # 34.0
# Diagonal sign-flip for converting PnLCalib world basis → ours.
_BASIS_FLIP = np.diag([1.0, -1.0, -1.0])
# Translation of PnLCalib origin (pitch centre, z-up-negative) expressed in
# our frame (near-left corner, z-up-positive).  See module docstring.
_ORIGIN_IN_OURS = np.array([_HALF_LENGTH, _HALF_WIDTH, 0.0])


@dataclass(frozen=True)
class NeuralCalibration:
    """Single-frame neural calibration result in OUR pitch coordinate frame.

    Attributes
    ----------
    K:
        3x3 intrinsic matrix (pixels).
    rvec:
        3-vector, Rodrigues rotation.  ``X_camera = R @ X_world + t``
        where ``R = cv2.Rodrigues(rvec)``.
    tvec:
        3-vector translation.
    world_position:
        Camera position in our pitch world coordinates
        (``x ∈ [0, 105]``, ``y ∈ [0, 68]``, ``z`` above pitch).
    """

    K: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    world_position: np.ndarray


class CalibratorProtocol(Protocol):
    """Interface implemented by per-frame neural calibrators.

    Concrete implementations: :class:`PnLCalibrator` (this module) and
    :class:`src.utils.tvcalib_calibrator.TVCalibCalibrator`.  The stage
    :class:`src.stages.calibration.CameraCalibrationStage` dispatches on
    this protocol so either backend plugs in without further changes.
    """

    def calibrate(self, frame_bgr: np.ndarray) -> NeuralCalibration | None:
        ...


def _normalize_device(device: str) -> str:
    """Mirror of src/utils/pose_estimator._normalize_device."""
    requested = (device or "auto").strip().lower()
    if requested == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda:0"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda":
        return "cuda:0"
    return requested


def _download_if_missing(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` if the file is missing.

    Writes to a ``.partial`` file and renames on completion so interrupted
    downloads don't leave a half-written file on disk.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    logger.info("Downloading PnLCalib weights: %s -> %s", url, dest)
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 — trusted GitHub URL
    tmp.replace(dest)


def _import_pnlcalib_modules() -> dict[str, Any]:
    """Import PnLCalib's inference helpers, keeping sys.path clean afterwards.

    PnLCalib uses bare ``import utils.*`` / ``import model.*``.  Our project
    installs ``src/`` on sys.path via an editable install, so ``utils`` would
    normally resolve to ``src/utils/``.  We temporarily remove ``src/`` and put
    the PnLCalib root first.  The loaded modules remain in ``sys.modules`` as
    ``utils``/``model``/``sn_calibration``, but our own code never does bare
    ``import utils`` — it always uses ``from src.utils.X import Y`` — so there
    is no runtime collision.
    """
    if not _PNLCALIB_ROOT.exists():
        raise RuntimeError(
            f"PnLCalib submodule missing at {_PNLCALIB_ROOT}. "
            "Run `git submodule update --init --recursive`."
        )

    src_path = str(_REPO_ROOT / "src")
    removed = [p for p in sys.path if p == src_path or p.rstrip("/") == src_path]
    for p in removed:
        sys.path.remove(p)
    sys.path.insert(0, str(_PNLCALIB_ROOT))

    try:
        from utils.utils_calib import FramebyFrameCalib  # type: ignore  # noqa: F401
        from utils.utils_heatmap import (  # type: ignore
            complete_keypoints,
            coords_to_dict,
            get_keypoints_from_heatmap_batch_maxpool,
            get_keypoints_from_heatmap_batch_maxpool_l,
        )
        from model.cls_hrnet import get_cls_net  # type: ignore
        from model.cls_hrnet_l import get_cls_net as get_cls_net_l  # type: ignore
    finally:
        # Remove our temporary insertion but leave ``src`` path restored.
        try:
            sys.path.remove(str(_PNLCALIB_ROOT))
        except ValueError:
            pass
        for p in removed:
            sys.path.append(p)

    return {
        "FramebyFrameCalib": FramebyFrameCalib,
        "get_cls_net": get_cls_net,
        "get_cls_net_l": get_cls_net_l,
        "get_keypoints_from_heatmap_batch_maxpool": get_keypoints_from_heatmap_batch_maxpool,
        "get_keypoints_from_heatmap_batch_maxpool_l": get_keypoints_from_heatmap_batch_maxpool_l,
        "complete_keypoints": complete_keypoints,
        "coords_to_dict": coords_to_dict,
    }


def convert_pnlcalib_to_ours(
    rotation_matrix: np.ndarray,
    position_meters_pnl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a PnLCalib camera pose to our pitch coordinate frame.

    Returns ``(rvec_ours, tvec_ours, world_position_ours)``.
    """
    R_pnl = np.asarray(rotation_matrix, dtype=np.float64)
    C_pnl = np.asarray(position_meters_pnl, dtype=np.float64).reshape(3)

    # World position in our frame: x+=52.5, y=34-y, z=-z.
    C_ours = np.array(
        [
            C_pnl[0] + _HALF_LENGTH,
            _HALF_WIDTH - C_pnl[1],
            -C_pnl[2],
        ],
        dtype=np.float64,
    )

    # Rotation: R_ours = R_pnl @ diag(1, -1, -1).
    R_ours = R_pnl @ _BASIS_FLIP

    # Translation: X_cam = R_ours @ X_ours + t_ours, so t_ours = -R_ours @ C_ours.
    t_ours = -R_ours @ C_ours

    rvec_ours, _ = cv2.Rodrigues(R_ours)
    return rvec_ours.reshape(3), t_ours.reshape(3), C_ours


class PnLCalibrator:
    """Thin wrapper around PnLCalib's single-frame inference pipeline.

    The model + weights are loaded lazily on the first call to
    :meth:`calibrate`.  The same instance can be reused across frames and
    shots.
    """

    def __init__(
        self,
        weights_dir: Path | None = None,
        device: str = "auto",
        kp_threshold: float = 0.3434,
        line_threshold: float = 0.7867,
        pnl_refine: bool = True,
    ) -> None:
        self._weights_dir = Path(weights_dir) if weights_dir is not None else _WEIGHTS_DIR
        self._device = _normalize_device(device)
        self._kp_threshold = float(kp_threshold)
        self._line_threshold = float(line_threshold)
        self._pnl_refine = bool(pnl_refine)
        self._loaded = False
        self._model_kp = None
        self._model_line = None
        self._pnlcalib_modules: dict[str, Any] = {}

    # ------------------------------------------------------------------ setup

    def _ensure_weights(self) -> tuple[Path, Path]:
        kp_path = self._weights_dir / "SV_kp"
        lines_path = self._weights_dir / "SV_lines"
        _download_if_missing(_SV_KP_URL, kp_path)
        _download_if_missing(_SV_LINES_URL, lines_path)
        return kp_path, lines_path

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import torch
        import yaml

        modules = _import_pnlcalib_modules()
        self._pnlcalib_modules = modules

        kp_path, lines_path = self._ensure_weights()

        cfg_kp_path = _PNLCALIB_ROOT / "config" / "hrnetv2_w48.yaml"
        cfg_line_path = _PNLCALIB_ROOT / "config" / "hrnetv2_w48_l.yaml"
        with cfg_kp_path.open("r") as fh:
            cfg_kp = yaml.safe_load(fh)
        with cfg_line_path.open("r") as fh:
            cfg_line = yaml.safe_load(fh)

        state_kp = torch.load(str(kp_path), map_location=self._device)
        model_kp = modules["get_cls_net"](cfg_kp)
        model_kp.load_state_dict(state_kp)
        model_kp.to(self._device).eval()

        state_line = torch.load(str(lines_path), map_location=self._device)
        model_line = modules["get_cls_net_l"](cfg_line)
        model_line.load_state_dict(state_line)
        model_line.to(self._device).eval()

        self._model_kp = model_kp
        self._model_line = model_line
        self._loaded = True
        logger.info(
            "PnLCalib loaded on device=%s, weights from %s",
            self._device,
            self._weights_dir,
        )

    # ------------------------------------------------------------- inference

    def _run_heatmaps(
        self, frame_bgr: np.ndarray,
    ) -> tuple[dict[int, dict[str, float]], dict, tuple[int, int]]:
        """Run the HRNet keypoint + line heads and return normalised dicts.

        Shared between :meth:`calibrate` (which also runs PnLCalib's
        calibration solver) and :meth:`extract_keypoints_pixels` (which
        just exposes the keypoints for our own fixed-position solver).

        Returns ``(kp_dict, line_dict, (image_width, image_height))`` where
        the dicts use PnLCalib's own keypoint/line IDs and ``x``/``y`` in
        the [0, 1] normalised range from ``complete_keypoints``.
        """
        import torch
        import torchvision.transforms as T
        import torchvision.transforms.functional as F
        from PIL import Image

        self._ensure_loaded()
        modules = self._pnlcalib_modules

        height, width = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(Image.fromarray(rgb)).float().unsqueeze(0)
        if tensor.size()[-1] != 960:
            tensor = T.Resize((540, 960), antialias=True)(tensor)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            heatmaps_kp = self._model_kp(tensor)
            heatmaps_line = self._model_line(tensor)

        kp_coords = modules["get_keypoints_from_heatmap_batch_maxpool"](
            heatmaps_kp[:, :-1, :, :]
        )
        line_coords = modules["get_keypoints_from_heatmap_batch_maxpool_l"](
            heatmaps_line[:, :-1, :, :]
        )
        kp_dict = modules["coords_to_dict"](kp_coords, threshold=self._kp_threshold)
        line_dict = modules["coords_to_dict"](line_coords, threshold=self._line_threshold)

        tensor_h = tensor.size()[-2]
        tensor_w = tensor.size()[-1]
        kp_dict, line_dict = modules["complete_keypoints"](
            kp_dict[0], line_dict[0], w=tensor_w, h=tensor_h, normalize=True,
        )
        return kp_dict, line_dict, (width, height)

    def extract_keypoints_pixels(
        self, frame_bgr: np.ndarray,
    ) -> dict[int, tuple[float, float]]:
        """Return PnLCalib's raw 2D keypoint detections in pixel coords.

        This exposes just the HRNet heatmap → keypoint-dict pipeline
        WITHOUT running PnLCalib's full calibration solver.  Our own
        fixed-position PnP solver (`src/utils/fixed_position_solver.py`)
        consumes this output to compute rotation + focal length at
        frames where PnLCalib's full solver fails.

        The returned dict maps PnLCalib keypoint IDs (1-57 plus aux
        58-73) to ``(x_pixel, y_pixel)`` in the original frame's
        resolution.  Only keypoints above the configured confidence
        threshold appear.
        """
        kp_dict, _, (width, height) = self._run_heatmaps(frame_bgr)
        pixels: dict[int, tuple[float, float]] = {}
        for kp_id, entry in kp_dict.items():
            # PnLCalib returns normalised [0, 1] coords when normalize=True.
            x = float(entry["x"]) * float(width)
            y = float(entry["y"]) * float(height)
            pixels[int(kp_id)] = (x, y)
        return pixels

    def calibrate(self, frame_bgr: np.ndarray) -> NeuralCalibration | None:
        """Run PnLCalib on a single BGR frame.

        Returns ``None`` when PnLCalib's heuristic voting does not converge,
        which happens for extreme camera angles (e.g., behind-goal replays).
        """
        self._ensure_loaded()
        modules = self._pnlcalib_modules

        height, width = frame_bgr.shape[:2]

        # FramebyFrameCalib is a per-instance accumulator.  We create a fresh
        # one per frame so residual state never leaks.
        cam = modules["FramebyFrameCalib"](iwidth=width, iheight=height, denormalize=True)

        kp_dict, line_dict, _ = self._run_heatmaps(frame_bgr)

        cam.update(kp_dict, line_dict)
        try:
            final_params_dict = cam.heuristic_voting(refine_lines=self._pnl_refine)
        except Exception as exc:  # noqa: BLE001 — PnLCalib's optimiser can raise
            logger.debug("PnLCalib heuristic_voting raised: %s", exc)
            return None

        if final_params_dict is None:
            return None

        cam_params = final_params_dict.get("cam_params")
        if cam_params is None:
            return None

        fx = float(cam_params["x_focal_length"])
        fy = float(cam_params["y_focal_length"])
        principal_point = np.asarray(cam_params["principal_point"], dtype=np.float64)
        position_meters = np.asarray(cam_params["position_meters"], dtype=np.float64)
        rotation_matrix = np.asarray(cam_params["rotation_matrix"], dtype=np.float64)

        K = np.array(
            [
                [fx, 0.0, float(principal_point[0])],
                [0.0, fy, float(principal_point[1])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        rvec, tvec, world_position = convert_pnlcalib_to_ours(
            rotation_matrix, position_meters,
        )

        return NeuralCalibration(
            K=K,
            rvec=rvec,
            tvec=tvec,
            world_position=world_position,
        )
