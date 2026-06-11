"""Shot-boundary detection for highlights reels (prepare_shots split mode).

Thin wrapper around PySceneDetect plus span hygiene. Resurrected from
the legacy segmentation stage (deleted in 262d08a) with the same
detector defaults; ``AdaptiveDetector`` is the default because plain
content thresholds misfire on fast broadcast pans/zooms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video


@dataclass(frozen=True)
class ShotSpan:
    """One detected shot, in source-video frames/seconds."""

    start_frame: int
    end_frame: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def detect_spans(
    video_path: Path,
    *,
    detector: str = "adaptive",
    threshold: float = 27.0,
    adaptive_threshold: float = 3.0,
    min_scene_len_frames: int = 13,
    adaptive_min_content_val: float = 15.0,
    min_shot_duration_s: float = 0.0,
) -> list[ShotSpan]:
    """Detect hard cuts in ``video_path`` and return shot spans.

    Spans shorter than ``min_shot_duration_s`` are dropped (sub-second
    flashes are useless for reconstruction). If the detector finds no
    cuts at all, the whole video is returned as a single span so the
    caller always has something to work with.
    """
    video = open_video(str(video_path))
    manager = SceneManager()
    if detector == "adaptive":
        manager.add_detector(AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            min_scene_len=min_scene_len_frames,
            min_content_val=adaptive_min_content_val,
        ))
    elif detector == "content":
        manager.add_detector(ContentDetector(
            threshold=threshold, min_scene_len=min_scene_len_frames,
        ))
    else:
        raise ValueError(
            "prepare_shots.split.detector must be 'adaptive' or 'content'"
        )
    manager.detect_scenes(video)
    scenes = manager.get_scene_list()

    spans = [
        ShotSpan(
            start_frame=s[0].get_frames(),
            end_frame=s[1].get_frames() - 1,
            start_s=s[0].get_seconds(),
            end_s=s[1].get_seconds(),
        )
        for s in scenes
    ]
    if not spans:
        spans = [_whole_video_span(video_path)]
    if min_shot_duration_s > 0:
        spans = [s for s in spans if s.duration_s >= min_shot_duration_s]
    return spans


def _whole_video_span(video_path: Path) -> ShotSpan:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    end_frame = max(0, frames - 1)
    return ShotSpan(
        start_frame=0,
        end_frame=end_frame,
        start_s=0.0,
        end_s=frames / fps,
    )


def merge_short_spans(
    spans: list[ShotSpan],
    *,
    max_short_duration_s: float,
    max_gap_s: float,
) -> list[ShotSpan]:
    """Merge likely false cuts: contiguous spans where either side is
    very short (camera whip-pans read as cuts to content detectors)."""
    if not spans:
        return []
    merged: list[ShotSpan] = [spans[0]]
    for current in spans[1:]:
        prev = merged[-1]
        gap_s = current.start_s - prev.end_s
        should_merge = gap_s <= max_gap_s and (
            prev.duration_s <= max_short_duration_s
            or current.duration_s <= max_short_duration_s
        )
        if should_merge:
            merged[-1] = ShotSpan(
                start_frame=prev.start_frame,
                end_frame=current.end_frame,
                start_s=prev.start_s,
                end_s=current.end_s,
            )
        else:
            merged.append(current)
    return merged
