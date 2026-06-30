"""Single-frame ball detectors.

All implementations return ``(u, v, confidence)`` in pixel coordinates,
or ``None`` when no ball is found.  Confidence is in ``[0, 1]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BallDetector(ABC):
    """Detects the ball in a single frame and returns its pixel position."""

    # True for detectors that re-detect meaningfully on an arbitrary crop
    # (WASB/YOLO). The foot-guided zoom pass re-queries the detector on
    # crops around player feet, so it only runs for such detectors — never
    # the scripted FakeBallDetector (re-querying would desync its cycle).
    SUPPORTS_REDETECT: bool = True

    @abstractmethod
    def detect(self, frame: np.ndarray) -> tuple[float, float, float] | None:
        """Returns ``(u, v, confidence)`` or ``None`` if no detection."""
        ...

    def detect_candidates(
        self, frame: np.ndarray, min_score: float, top_k: int = 5,
    ) -> list[tuple[float, float, float]]:
        """Low-threshold candidate detections ``[(u, v, score), ...]``.

        Default adapter wraps :meth:`detect` (a single candidate, if its
        confidence clears ``min_score``). Detectors with heatmap access
        override this with true top-k extraction.
        """
        det = self.detect(frame)
        if det is None or det[2] < min_score:
            return []
        return [det]

    def reset(self) -> None:
        """Clear any temporal state (frame buffers). Default: no-op."""
        return None


class YOLOBallDetector(BallDetector):
    """Ball detector using a YOLOv8 model (COCO class 32 or fine-tuned).

    Fallback option when WASB is unavailable. Stock YOLO weights miss a
    large fraction of small/blurry/occluded broadcast balls — prefer
    :class:`WASBBallDetector` (when vendored).
    """

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.3) -> None:
        from ultralytics import YOLO  # lazy import — model download on first use
        self._model = YOLO(model_name)
        self._confidence = confidence
        self._ball_class_id = 32  # COCO 'sports ball'

    def detect(self, frame: np.ndarray) -> tuple[float, float, float] | None:
        results = self._model(frame, verbose=False)[0]
        best: tuple[float, float, float] | None = None
        for box in results.boxes:
            if int(box.cls) != self._ball_class_id:
                continue
            conf = float(box.conf)
            if conf < self._confidence:
                continue
            if best is not None and conf <= best[2]:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            best = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, conf)
        return best

    def detect_candidates(
        self, frame: np.ndarray, min_score: float, top_k: int = 5,
    ) -> list[tuple[float, float, float]]:
        results = self._model(frame, verbose=False)[0]
        out: list[tuple[float, float, float]] = []
        for box in results.boxes:
            if int(box.cls) != self._ball_class_id:
                continue
            conf = float(box.conf)
            if conf < min_score:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, conf))
        out.sort(key=lambda c: -c[2])
        return out[:top_k]


def _wasb_module() -> "WASBBallDetector":
    """Lazy import to avoid pulling torch + the WASB submodule into the
    namespace whenever this file is imported."""
    from src.utils.wasb_ball_detector import WASBBallDetector as _Cls
    return _Cls


class WASBBallDetector(BallDetector):
    """Detector backed by the vendored WASB-SBDT HRNet model.

    Implementation lives in :mod:`src.utils.wasb_ball_detector`; this
    class is a thin shim so callers can write
    ``from src.utils.ball_detector import WASBBallDetector`` like they
    do for the other detector classes.
    """

    def __new__(cls, *args, **kwargs):  # type: ignore[override]
        impl_cls = _wasb_module()
        return impl_cls(*args, **kwargs)

    def detect(self, frame: np.ndarray) -> tuple[float, float, float] | None:  # pragma: no cover
        raise NotImplementedError  # handled by the implementation class


class FakeBallDetector(BallDetector):
    """Deterministic detector for tests — cycles through pre-supplied detections.

    Each entry is either ``(u, v, confidence)`` or ``None``. Optional
    ``candidates`` (a parallel cycle of candidate lists) scripts
    :meth:`detect_candidates`; ``reset_count`` records :meth:`reset` calls.
    """

    SUPPORTS_REDETECT: bool = False  # scripted cycle — never re-query on crops

    def __init__(
        self,
        detections: list[tuple[float, float, float] | None],
        candidates: list[list[tuple[float, float, float]]] | None = None,
    ) -> None:
        self._detections = detections
        self._candidates = candidates
        self._idx = 0
        self._cand_idx = 0
        self.reset_count = 0

    def detect(self, frame: np.ndarray) -> tuple[float, float, float] | None:
        d = self._detections[self._idx % len(self._detections)]
        self._idx += 1
        return d

    def detect_candidates(
        self, frame: np.ndarray, min_score: float, top_k: int = 5,
    ) -> list[tuple[float, float, float]]:
        if self._candidates is None:
            return super().detect_candidates(frame, min_score, top_k)
        cands = self._candidates[self._cand_idx % len(self._candidates)]
        self._cand_idx += 1
        kept = [c for c in cands if c[2] >= min_score]
        kept.sort(key=lambda c: -c[2])
        return kept[:top_k]

    def reset(self) -> None:
        self.reset_count += 1
