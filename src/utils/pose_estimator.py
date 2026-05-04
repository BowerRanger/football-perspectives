import glob
import logging
import os
from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any

import numpy as np

from src.schemas.poses import COCO_KEYPOINT_NAMES, Keypoint


class PoseEstimator(ABC):
    """Estimates 2D COCO keypoints for a single player crop."""

    @abstractmethod
    def estimate(
        self, crop: np.ndarray, bbox_offset: tuple[float, float]
    ) -> list[Keypoint]:
        """
        Args:
            crop: BGR image of the player (with padding).
            bbox_offset: (x, y) pixel coordinates of the crop's top-left corner
                         in the original video frame. Used to return keypoints in
                         frame-absolute coordinates.
        Returns:
            List of 17 Keypoints in COCO order, in original-frame pixel coordinates.
        """
        ...


class FakePoseEstimator(PoseEstimator):
    """Returns deterministic keypoints spread evenly down the crop for tests."""

    def __init__(self, conf: float = 0.9) -> None:
        self._conf = conf

    def estimate(
        self, crop: np.ndarray, bbox_offset: tuple[float, float]
    ) -> list[Keypoint]:
        h, w = crop.shape[:2]
        ox, oy = bbox_offset
        return [
            Keypoint(
                name=name,
                x=ox + w / 2.0,
                y=oy + h * (i + 1) / (len(COCO_KEYPOINT_NAMES) + 1),
                conf=self._conf,
            )
            for i, name in enumerate(COCO_KEYPOINT_NAMES)
        ]


def _zero_confidence_keypoints() -> list[Keypoint]:
    return [Keypoint(name=name, x=0.0, y=0.0, conf=0.0) for name in COCO_KEYPOINT_NAMES]


def _normalize_device(device: str) -> str:
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


_CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints")


def _resolve_mmpose_config(
    model_config: str,
    checkpoint: str | None,
    checkpoints_dir: str = _CHECKPOINTS_DIR,
) -> tuple[str, str | None]:
    """Resolve a config alias or path and its paired checkpoint.

    If *model_config* points to an existing file, return it unchanged.
    Otherwise treat it as an mmpose config alias and download via ``mim``.
    The paired checkpoint is discovered from *checkpoints_dir* when not
    provided explicitly.
    """
    if os.path.isfile(model_config):
        return model_config, checkpoint

    # Not a file path — treat as an alias and resolve via openmim.
    try:
        from mim import download as mim_download  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            f"Config '{model_config}' is not a file path and openmim is not installed. "
            "Install openmim to allow automatic config/checkpoint download: "
            "pip install openmim"
        ) from exc

    os.makedirs(checkpoints_dir, exist_ok=True)

    config_path = os.path.join(checkpoints_dir, model_config + ".py")

    # If already downloaded, skip the network fetch.
    if not os.path.isfile(config_path):
        logging.info(
            "MMPose config '%s' not found locally — downloading via mim …", model_config
        )
        mim_download("mmpose", configs=[model_config], dest_root=checkpoints_dir)

    if not os.path.isfile(config_path):
        raise RuntimeError(
            f"mim download succeeded but config file not found at '{config_path}'. "
            "Check that the alias is a valid mmpose config name."
        )

    if checkpoint is None:
        pth_files = sorted(glob.glob(os.path.join(checkpoints_dir, model_config + "*.pth")))
        checkpoint = pth_files[0] if pth_files else None

    return config_path, checkpoint


class MMPoseEstimator(PoseEstimator):
    """Pose estimator using MMPose top-down inference on tracked player crops.

    The estimator expects an MMPose top-down config or alias that yields COCO-17
    keypoints. Predictions are normalized into the repository's fixed COCO order
    and remapped from crop-local coordinates into original-frame coordinates.
    """

    def __init__(
        self,
        model_config: str,
        checkpoint: str | None = None,
        device: str = "auto",
    ) -> None:
        self._device = _normalize_device(device)

        try:
            apis = import_module("mmpose.apis")
            self._init_model = getattr(apis, "init_model")
            self._inference_topdown = getattr(apis, "inference_topdown")
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Failed to import MMPoseEstimator dependencies. Install mmpose, mmcv, "
                "mmengine, and a compatible torch build before running pose estimation."
            ) from exc

        self._model_config, self._checkpoint = _resolve_mmpose_config(model_config, checkpoint)

        try:
            self._model = self._init_model(
                self._model_config,
                self._checkpoint,
                device=self._device,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize MMPoseEstimator for config "
                f"'{self._model_config}'. Check that the config/checkpoint pair is valid "
                f"and that device '{self._device}' is supported."
            ) from exc

        self._warmup()

    def _warmup(self) -> None:
        """Run one dummy forward pass to trigger Metal/CUDA shader compilation up front.

        On MPS this can take 1-3 minutes on first run. Doing it here makes the
        wait visible rather than silent inside the first real frame.
        """
        if self._device == "cpu":
            return
        logging.info(
            "  [pose] warming up %s shaders (first run compiles kernels, may take 1-3 min) …",
            self._device.upper(),
        )
        dummy = np.zeros((192, 192, 3), dtype=np.uint8)
        try:
            self._inference_topdown(self._model, dummy)
            logging.info("  [pose] %s warmup done", self._device.upper())
        except Exception as exc:
            logging.warning(
                "  [pose] %s warmup failed (%s) — falling back to CPU", self._device.upper(), exc
            )
            self._device = "cpu"
            self._model = self._init_model(
                self._model_config,
                self._checkpoint,
                device="cpu",
            )

    def estimate(
        self, crop: np.ndarray, bbox_offset: tuple[float, float]
    ) -> list[Keypoint]:
        if crop.size == 0:
            return _zero_confidence_keypoints()

        predictions = self._run_inference(crop)
        sample = self._select_prediction(predictions)
        if sample is None:
            return _zero_confidence_keypoints()

        keypoints, scores = self._extract_keypoints(sample)
        if keypoints.shape[0] != len(COCO_KEYPOINT_NAMES):
            raise RuntimeError(
                "MMPoseEstimator returned an unsupported keypoint layout: "
                f"expected {len(COCO_KEYPOINT_NAMES)} COCO keypoints, got {keypoints.shape[0]}."
            )

        ox, oy = bbox_offset
        return [
            Keypoint(
                name=name,
                x=float(coords[0]) + ox,
                y=float(coords[1]) + oy,
                conf=float(scores[index]),
            )
            for index, (name, coords) in enumerate(zip(COCO_KEYPOINT_NAMES, keypoints))
        ]

    def _run_inference(self, crop: np.ndarray) -> Any:
        try:
            return self._inference_topdown(self._model, crop)
        except TypeError as exc:
            logging.debug(
                "MMPose crop-only inference failed, retrying with an explicit bbox: %s",
                exc,
            )
            height, width = crop.shape[:2]
            bbox = np.array([[0.0, 0.0, float(width), float(height)]], dtype=np.float32)
            return self._inference_topdown(
                self._model,
                crop,
                bboxes=bbox,
                bbox_format="xyxy",
            )

    def _select_prediction(self, predictions: Any) -> Any | None:
        if predictions is None:
            return None
        if isinstance(predictions, (list, tuple)):
            return predictions[0] if predictions else None
        return predictions

    def _extract_keypoints(self, sample) -> tuple[np.ndarray, np.ndarray]:
        pred_instances = getattr(sample, "pred_instances", sample)
        keypoints = getattr(pred_instances, "keypoints", None)
        scores = getattr(pred_instances, "keypoint_scores", None)

        if keypoints is None and isinstance(pred_instances, dict):
            keypoints = pred_instances.get("keypoints")
            scores = pred_instances.get("keypoint_scores")

        if keypoints is None:
            raise RuntimeError(
                "MMPoseEstimator received predictions without keypoints. "
                "Check the selected config/checkpoint pair."
            )

        keypoints_array = np.asarray(keypoints, dtype=float)
        if keypoints_array.ndim == 3:
            keypoints_array = keypoints_array[0]
        if keypoints_array.ndim != 2 or keypoints_array.shape[1] < 2:
            raise RuntimeError(
                "MMPoseEstimator received malformed keypoint coordinates from MMPose."
            )

        if scores is None and keypoints_array.shape[1] >= 3:
            scores_array = keypoints_array[:, 2]
            keypoints_array = keypoints_array[:, :2]
        else:
            keypoints_array = keypoints_array[:, :2]
            if scores is None:
                scores_array = np.ones(keypoints_array.shape[0], dtype=float)
            else:
                scores_array = np.asarray(scores, dtype=float)
                if scores_array.ndim == 2:
                    scores_array = scores_array[0]

        if scores_array.ndim != 1:
            raise RuntimeError(
                "MMPoseEstimator received malformed keypoint confidence scores from MMPose."
            )

        return keypoints_array, scores_array
