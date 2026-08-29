"""Persistent content-hash cache around a :class:`BallDetector`.

Promoted from ``scripts/eval_ball_accuracy.py``'s ``CachingDetector`` (the
sub-20cm accuracy campaign's ``--det-cache`` flag) so the ball *stage*
can opt into the same trick: WASB inference dominates ball-stage
runtime, and video frames decode deterministically, so caching on a
frame-content hash lets a first run pay the cost and every later run —
including a second fold of the same eval run, or a stage re-run after
editing anchors — replay stored results instead of re-invoking the
detector.

Two independent caches per detector instance:

* ``detect`` — keyed off the full frame passed to :meth:`detect`.
* ``candidates`` — keyed off whatever frame is passed to
  :meth:`detect_candidates` (a full frame in the second-pass corridor
  gate, or a crop in the second-pass zoom / foot-guided zoom). Content
  hashing already tells a crop apart from a full frame (different shape
  and pixel bytes), so the two passes never collide; splitting them into
  separate dicts additionally means a full-frame ``detect`` and a
  same-content ``detect_candidates`` call can't collide either.

Cache-file compatibility: when constructed with ``fingerprint=None``
(the default, and what ``scripts/eval_ball_accuracy.py`` still passes),
this class behaves exactly like the original ``CachingDetector`` — any
on-disk cache is trusted verbatim, matching the pre-existing
``--det-cache`` behaviour and the already-committed caches under
``docs/superpowers/notes/ball-accuracy/det_cache/``. Passing an explicit
``fingerprint`` (as the ball stage's config-driven wiring does via
:func:`build_detector_fingerprint`) additionally validates the cache
against the detector identity that produced it — checkpoint path +
content hash/size, detector class, and threshold/input-size config — so
swapping a checkpoint or editing ``ball.wasb.*`` invalidates stale
entries instead of silently replaying results from a different model.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.ball_detector import BallDetector

logger = logging.getLogger(__name__)

# Default location of the ball stage's opt-in cache, relative to the
# pipeline output dir (see config/default.yaml ball.detection_cache.path).
DEFAULT_CACHE_RELPATH = "ball/detection_cache.json"


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def build_detector_fingerprint(cfg: dict, detector: BallDetector) -> dict[str, Any]:
    """Fingerprint identifying the detector ``_build_detector`` constructs
    from a ``ball.*`` config dict.

    Includes whatever is behaviour-relevant for the configured backend:
    detector class name, checkpoint path + sha256 + file size for WASB
    (so a checkpoint swap invalidates), confidence threshold and
    letterbox input size. Two constructions that would produce different
    detections for the same frame should get different fingerprints.
    """
    backend = str(cfg.get("detector", "yolo")).strip().lower()
    fp: dict[str, Any] = {"class": type(detector).__name__, "backend": backend}
    if backend == "wasb":
        wasb_cfg = cfg.get("wasb", {}) or {}
        checkpoint = wasb_cfg.get("checkpoint")
        if checkpoint:
            ckpt_path = Path(checkpoint).expanduser().resolve()
            fp["checkpoint_path"] = str(ckpt_path)
            if ckpt_path.exists():
                st = ckpt_path.stat()
                fp["checkpoint_size"] = st.st_size
                fp["checkpoint_sha256"] = _sha256_file(ckpt_path)
            else:
                fp["checkpoint_missing"] = True
        fp["confidence"] = float(wasb_cfg.get("confidence", 0.3))
        fp["input_size"] = list(wasb_cfg.get("input_size", (512, 288)))
    elif backend == "yolo":
        fp["yolo_model"] = cfg.get("yolo_model", "yolov8n.pt")
        fp["confidence"] = float(cfg.get("confidence_threshold", 0.3))
    return fp


class CachingBallDetector(BallDetector):
    """Content-hash cache around a real detector.

    ``fingerprint``, when given, is persisted alongside the cached
    detections and compared on load; a mismatch (or a fingerprint being
    supplied where the on-disk cache has none, or vice versa) is treated
    as a stale cache and discarded — the wrapped detector runs fresh and
    overwrites the file with the new fingerprint on the next
    :meth:`save`. ``fingerprint=None`` (the default) skips this check
    entirely and always trusts whatever is on disk, matching the
    original eval-only ``CachingDetector`` this class replaces.
    """

    def __init__(
        self,
        inner: BallDetector,
        cache_path: str | Path,
        fingerprint: dict[str, Any] | None = None,
        *,
        autosave_every: int = 200,
    ) -> None:
        self._inner = inner
        self._path = Path(cache_path)
        self._fingerprint = fingerprint
        self._detect: dict[str, tuple | None] = {}
        self._cands: dict[str, list] = {}
        self._dirty = 0
        self._autosave_every = max(1, int(autosave_every))
        self.SUPPORTS_REDETECT = getattr(inner, "SUPPORTS_REDETECT", True)
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "ball detection cache: failed to read %s (%s) — starting empty",
                self._path, exc,
            )
            return
        if self._fingerprint is not None and data.get("fingerprint") != self._fingerprint:
            logger.info(
                "ball detection cache: fingerprint mismatch at %s — "
                "stale entries discarded, detector will run fresh",
                self._path,
            )
            return
        self._detect = {k: (tuple(v) if v is not None else None)
                        for k, v in data.get("detect", {}).items()}
        self._cands = {k: [tuple(c) for c in v]
                       for k, v in data.get("candidates", {}).items()}

    @staticmethod
    def _key(frame: np.ndarray) -> str:
        h = hashlib.md5(frame[::4, ::4].tobytes())
        h.update(str(frame.shape).encode())
        return h.hexdigest()

    def detect(self, frame: np.ndarray) -> tuple[float, float, float] | None:
        k = self._key(frame)
        if k in self._detect:
            return self._detect[k]
        det = self._inner.detect(frame)
        self._detect[k] = tuple(det) if det is not None else None
        self._dirty += 1
        if self._dirty >= self._autosave_every:
            self.save()
        return det

    def detect_candidates(
        self, frame: np.ndarray, min_score: float, top_k: int = 5,
    ) -> list[tuple[float, float, float]]:
        k = f"{self._key(frame)}:{min_score}:{top_k}"
        if k in self._cands:
            return list(self._cands[k])
        out = self._inner.detect_candidates(frame, min_score, top_k)
        self._cands[k] = [tuple(c) for c in out]
        self._dirty += 1
        if self._dirty >= self._autosave_every:
            self.save()
        return out

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "detect": {k: (list(v) if v is not None else None)
                       for k, v in self._detect.items()},
            "candidates": {k: [list(c) for c in v]
                           for k, v in self._cands.items()},
        }
        if self._fingerprint is not None:
            payload["fingerprint"] = self._fingerprint
        self._path.write_text(json.dumps(payload))
        self._dirty = 0


def wrap_if_enabled(
    detector: BallDetector, cfg: dict, output_dir: str | Path,
) -> BallDetector:
    """Wrap ``detector`` in :class:`CachingBallDetector` when
    ``ball.detection_cache.enabled`` is true; otherwise return it
    unchanged.

    Default is opt-out (``enabled: false``) so first-run/default
    behaviour is identical to before this cache existed. ``path``
    defaults to :data:`DEFAULT_CACHE_RELPATH` under ``output_dir``;
    a relative path in config is always resolved against the output
    dir, never the cwd.
    """
    cache_cfg = cfg.get("detection_cache", {}) or {}
    if not bool(cache_cfg.get("enabled", False)):
        return detector
    raw_path = cache_cfg.get("path") or DEFAULT_CACHE_RELPATH
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(output_dir) / path
    fingerprint = build_detector_fingerprint(cfg, detector)
    return CachingBallDetector(detector, path, fingerprint=fingerprint)
