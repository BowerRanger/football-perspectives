from abc import ABC, abstractmethod

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
    """Returns deterministic keypoints spread evenly down the crop — used in tests."""

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


class ViTPoseEstimator(PoseEstimator):
    """
    Pose estimator using ViTPose via HuggingFace Transformers.

    The model outputs heatmaps (B, num_joints, H_out, W_out). Each heatmap's
    argmax gives the predicted joint location in heatmap space; we rescale to
    frame-absolute coordinates using the crop dimensions and offset.
    """

    def __init__(self, model_name: str = "nielsr/vitpose-base-simple") -> None:
        from transformers import AutoImageProcessor, AutoModel  # lazy import

        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()

    def estimate(
        self, crop: np.ndarray, bbox_offset: tuple[float, float]
    ) -> list[Keypoint]:
        import torch
        from PIL import Image

        ox, oy = bbox_offset
        pil_img = Image.fromarray(crop[:, :, ::-1])  # BGR → RGB
        inputs = self._processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)

        # ViTPose outputs heatmaps on outputs.heatmaps: (B, num_joints, H_out, W_out)
        heatmaps = getattr(outputs, "heatmaps", None)
        if heatmaps is not None and heatmaps.dim() == 4:
            _, J, H_out, W_out = heatmaps.shape
            crop_h, crop_w = crop.shape[:2]
            kps = []
            for j in range(min(J, 17)):
                hm = heatmaps[0, j].numpy()
                flat_idx = int(np.argmax(hm))
                ky, kx = divmod(flat_idx, W_out)
                px = ox + (kx / W_out) * crop_w
                py = oy + (ky / H_out) * crop_h
                conf = float(hm.max())
                kps.append(Keypoint(name=COCO_KEYPOINT_NAMES[j], x=px, y=py, conf=conf))
            return kps

        import logging
        logging.warning(
            "ViTPoseEstimator: unexpected model output shape — returning zero-confidence keypoints. "
            "Check model name and output attributes."
        )
        return [Keypoint(name=name, x=0.0, y=0.0, conf=0.0) for name in COCO_KEYPOINT_NAMES]
