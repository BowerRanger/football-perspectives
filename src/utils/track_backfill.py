"""Backward track-extension pass for the tracking stage (ball-stage
campaign, Workstream 4).

Motivation: YOLOv8x + ByteTrack/BoT-SORT sometimes mints a track several
frames after a player is actually visible in the shot — a warm-up
artifact (partial visibility / low confidence on the very first frames),
not evidence the player wasn't there. Downstream code that needs an
exact frame (e.g. ball touch-attribution locating an FK joint at a
manual touch anchor) then has nothing to anchor to if the anchor frame
predates the track's first frame. Concrete case: on gberch, P008
(Hato)'s track starts at frame 62 though visible earlier; a manual
ball-touch at frame 56 is unattributable.

HARD CONSTRAINT — operator data: ``tracks.json`` carries operator-
assigned ``player_id``/``player_name`` per track, and downstream stages
(ball touch attribution, hmr_world) key off ``track_id``. A naive
tracking re-run mints new track ids and destroys those annotations.
This module never re-runs tracking or renumbers anything — it only
PREPENDS frames to the FRONT of an existing track's ``frames`` list,
walking backward from that track's original first frame toward the
shot start (frame 0 — ``Track.frames[i].frame`` is shot-clip-relative,
per ``src.stages.tracking.PlayerTrackingStage._track_shot``). Tracks
that aren't selected, or whose first frame is already close to the shot
start, come back byte-identical (same object, not a copy) — see
``backfill_track``'s ``not_eligible`` path and ``backfill_tracks_result``'s
``select`` filter.

Algorithm (per eligible track):
  1. Skip tracks whose first frame is already <= ``min_late_start_frames``
     into the shot — nothing meaningful to backfill.
  2. Seed the walk from the track's first (frame, bbox).
  3. Step backward one frame at a time. The tracking stage doesn't
     persist raw per-frame detections anywhere (only the tracked
     output survives), so each step re-runs ``detector.detect()`` on
     that single frame — cheap for the handful of frames a warm-up gap
     typically spans, and self-contained (no dependency on the
     original stage run's ordering/state, unlike a tracker's temporal
     association).
  4. Conservative acceptance gate: the best same-class candidate must
     clear ``min_iou`` against the current reference bbox, AND must
     beat the second-best candidate by at least ``ambiguity_margin``
     (raw IoU) — a crowded frame where two candidates are both
     plausible stops the walk rather than guessing which is which.
     When ``use_appearance_gate`` is on, the best candidate must ALSO
     clear ``max_appearance_distance`` (HSV histogram distance) against
     the reference crop — an extra rejection gate, never a tie-breaker,
     so it can only make acceptance MORE conservative.
  5. A frame with no qualifying candidate counts as a miss; misses are
     tolerated up to ``patience`` in a row (the reference bbox/crop is
     held across misses). A miss beyond patience stops the walk. An
     AMBIGUOUS frame stops the walk immediately, not patience-gated —
     picking the wrong one here would silently graft another player's
     bbox onto this track's identity, so there is no "try again".
  6. Every accepted frame is tagged ``source="backfill"`` and prepended;
     the walk also stops cleanly at the shot boundary (frame 0) or a
     configured ``max_backfill_frames`` safety cap.

The detector sits behind the existing ``PlayerDetector`` interface so
unit tests inject ``FakePlayerDetector`` (per-frame scripted output,
keyed by call order — see ``tests/test_track_backfill.py``); frame
lookup sits behind ``FrameSource`` so the same walk logic drives both
the in-stage path (``VideoFrameSource`` over the shot's own clip) and
``scripts/backfill_tracks.py`` (same clip, read-only re-open).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from src.schemas.tracks import Track, TrackFrame, TracksResult
from src.utils.player_detector import Detection, PlayerDetector

BACKFILL_SOURCE = "backfill"


class FrameSource(Protocol):
    """Random-access frame lookup, shot-clip-relative frame index."""

    def get(self, frame_idx: int) -> np.ndarray | None:
        """Return the BGR frame at ``frame_idx``, or ``None`` if unavailable."""
        ...


class VideoFrameSource:
    """``FrameSource`` backed by a video file via OpenCV, seeking per call.

    The backward walk's access pattern is a short, monotonically
    decreasing run of frame indices (a handful of frames per track in
    the common case), so per-call seeking is simpler and cheap enough —
    no need to decode the whole clip forward just to reach the early
    frames a late track-birth left behind.
    """

    def __init__(self, clip_path: Path) -> None:
        import cv2

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(str(clip_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open clip: {clip_path}")

    def get(self, frame_idx: int) -> np.ndarray | None:
        if frame_idx < 0:
            return None
        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()


@dataclass(frozen=True)
class BackfillConfig:
    """Tuning knobs. Mirrors ``tracking.backfill.*`` in ``config/default.yaml``."""

    enabled: bool = False
    min_late_start_frames: int = 2
    max_backfill_frames: int = 60
    min_iou: float = 0.3
    ambiguity_margin: float = 0.1
    patience: int = 2
    use_appearance_gate: bool = False
    max_appearance_distance: float = 0.6

    @classmethod
    def from_dict(cls, raw: dict | None) -> "BackfillConfig":
        raw = raw or {}
        defaults = cls()
        return cls(
            enabled=bool(raw.get("enabled", defaults.enabled)),
            min_late_start_frames=int(raw.get("min_late_start_frames", defaults.min_late_start_frames)),
            max_backfill_frames=int(raw.get("max_backfill_frames", defaults.max_backfill_frames)),
            min_iou=float(raw.get("min_iou", defaults.min_iou)),
            ambiguity_margin=float(raw.get("ambiguity_margin", defaults.ambiguity_margin)),
            patience=int(raw.get("patience", defaults.patience)),
            use_appearance_gate=bool(raw.get("use_appearance_gate", defaults.use_appearance_gate)),
            max_appearance_distance=float(
                raw.get("max_appearance_distance", defaults.max_appearance_distance)
            ),
        )


@dataclass(frozen=True)
class TrackBackfillReport:
    track_id: str
    player_id: str
    original_start_frame: int
    new_start_frame: int
    frames_added: int
    # "shot_start" | "miss_patience" | "ambiguous" | "max_frames" |
    # "frame_unavailable" | "not_eligible"
    stop_reason: str


def _iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def _best_candidates(
    reference_bbox: tuple[float, float, float, float],
    detections: list[Detection],
    class_name: str,
) -> list[tuple[float, Detection]]:
    """IoU-scored same-class candidates against ``reference_bbox``, best first."""
    scored = [
        (_iou(reference_bbox, d.bbox), d) for d in detections if d.class_name == class_name
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


def _crop(frame_img: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    h, w = frame_img.shape[:2]
    xi1, yi1 = max(0, int(x1)), max(0, int(y1))
    xi2, yi2 = min(w, int(x2)), min(h, int(y2))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    return frame_img[yi1:yi2, xi1:xi2]


def _hsv_hist(crop: np.ndarray | None) -> np.ndarray | None:
    if crop is None or crop.size == 0:
        return None
    import cv2

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _appearance_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """1 - histogram correlation (0 = identical, up to 2 = opposite).

    Either input missing (empty/unavailable crop) returns 0.0 — no
    penalty — since this is an OPTIONAL extra gate layered on top of
    the IoU gate, never a substitute for it.
    """
    if a is None or b is None:
        return 0.0
    import cv2

    corr = cv2.compareHist(a, b, cv2.HISTCMP_CORREL)
    return max(0.0, 1.0 - float(corr))


def backfill_track(
    track: Track,
    frame_source: FrameSource,
    detector: PlayerDetector,
    cfg: BackfillConfig,
) -> tuple[Track, TrackBackfillReport]:
    """Walk backward from ``track``'s first frame.

    Returns a NEW ``Track`` with any accepted frames prepended — the
    input ``track`` is never mutated. When nothing is added (not
    eligible, or the very first backward step is a miss/ambiguous),
    the SAME object is returned so untouched tracks serialise byte-
    identically.
    """
    if not track.frames:
        return track, TrackBackfillReport(
            track_id=track.track_id,
            player_id=track.player_id,
            original_start_frame=-1,
            new_start_frame=-1,
            frames_added=0,
            stop_reason="not_eligible",
        )

    first = track.frames[0]
    if first.frame <= cfg.min_late_start_frames:
        return track, TrackBackfillReport(
            track_id=track.track_id,
            player_id=track.player_id,
            original_start_frame=first.frame,
            new_start_frame=first.frame,
            frames_added=0,
            stop_reason="not_eligible",
        )

    reference_bbox: tuple[float, float, float, float] = tuple(first.bbox)
    reference_hist = None
    if cfg.use_appearance_gate:
        seed_frame = frame_source.get(first.frame)
        if seed_frame is not None:
            reference_hist = _hsv_hist(_crop(seed_frame, reference_bbox))

    added: list[TrackFrame] = []
    misses = 0
    stop_reason = "shot_start"
    cursor = first.frame - 1
    steps = 0

    while cursor >= 0:
        if steps >= cfg.max_backfill_frames:
            stop_reason = "max_frames"
            break
        steps += 1

        frame_img = frame_source.get(cursor)
        if frame_img is None:
            stop_reason = "frame_unavailable"
            break

        detections = detector.detect(frame_img)
        scored = _best_candidates(reference_bbox, detections, track.class_name)
        best_iou = scored[0][0] if scored else 0.0
        second_iou = scored[1][0] if len(scored) > 1 else 0.0

        if best_iou < cfg.min_iou:
            misses += 1
            if misses > cfg.patience:
                stop_reason = "miss_patience"
                break
            cursor -= 1
            continue

        if (best_iou - second_iou) < cfg.ambiguity_margin:
            stop_reason = "ambiguous"
            break

        best_det = scored[0][1]
        candidate_hist = None
        if cfg.use_appearance_gate and reference_hist is not None:
            candidate_hist = _hsv_hist(_crop(frame_img, best_det.bbox))
            dist = _appearance_distance(reference_hist, candidate_hist)
            if dist > cfg.max_appearance_distance:
                misses += 1
                if misses > cfg.patience:
                    stop_reason = "miss_patience"
                    break
                cursor -= 1
                continue

        added.append(TrackFrame(
            frame=cursor,
            bbox=list(best_det.bbox),
            confidence=best_det.confidence,
            pitch_position=None,
            interpolated=False,
            source=BACKFILL_SOURCE,
        ))
        reference_bbox = best_det.bbox
        if cfg.use_appearance_gate:
            reference_hist = candidate_hist if candidate_hist is not None else reference_hist
        misses = 0
        cursor -= 1

    if not added:
        return track, TrackBackfillReport(
            track_id=track.track_id,
            player_id=track.player_id,
            original_start_frame=first.frame,
            new_start_frame=first.frame,
            frames_added=0,
            stop_reason=stop_reason,
        )

    added.reverse()
    new_track = replace(track, frames=[*added, *track.frames])
    return new_track, TrackBackfillReport(
        track_id=track.track_id,
        player_id=track.player_id,
        original_start_frame=first.frame,
        new_start_frame=new_track.frames[0].frame,
        frames_added=len(added),
        stop_reason=stop_reason,
    )


def backfill_tracks_result(
    tracks_result: TracksResult,
    frame_source: FrameSource,
    detector: PlayerDetector,
    cfg: BackfillConfig,
    select: Callable[[Track], bool] | None = None,
) -> tuple[TracksResult, list[TrackBackfillReport]]:
    """Apply ``backfill_track`` to every track in ``tracks_result`` for
    which ``select(track)`` is true (default: all tracks).

    Track order and every field on non-selected/non-eligible tracks are
    preserved exactly (same objects) — only selected, eligible tracks'
    ``frames`` lists change. Reports are returned only for SELECTED
    tracks (including ``not_eligible`` ones), so a caller can print a
    per-track summary of what was attempted.
    """
    select_fn = select or (lambda _t: True)
    new_tracks: list[Track] = []
    reports: list[TrackBackfillReport] = []
    for track in tracks_result.tracks:
        if not select_fn(track):
            new_tracks.append(track)
            continue
        new_track, report = backfill_track(track, frame_source, detector, cfg)
        new_tracks.append(new_track)
        reports.append(report)
    return replace(tracks_result, tracks=new_tracks), reports
