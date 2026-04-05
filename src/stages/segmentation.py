from dataclasses import dataclass
from pathlib import Path
import logging
import subprocess

import cv2
import numpy as np
from scenedetect import open_video, SceneManager, ContentDetector, AdaptiveDetector

from src.pipeline.base import BaseStage
from src.schemas.shots import Shot, ShotsManifest
from src.utils.ball_detector import BallDetector, YOLOBallDetector
from src.utils.ffmpeg import extract_clip


@dataclass
class _ShotSpan:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float


def _span_duration(span: _ShotSpan) -> float:
    return max(0.0, span.end_time - span.start_time)


def _estimate_pitch_ratio(frame_bgr: np.ndarray) -> float:
    """Estimate how much of a frame is green pitch in [0, 1]."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Broad green range for broadcast football turf in varying lighting.
    mask = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
    return float(mask.mean() / 255.0)


def _estimate_frame_brightness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean() / 255.0)


def _read_frame_at_time(
    cap: cv2.VideoCapture,
    fps: float,
    total_frames: int,
    time_s: float,
) -> np.ndarray | None:
    if fps <= 0 or total_frames <= 0:
        return None
    frame_idx = int(max(0, min(total_frames - 1, round(time_s * fps))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def _sample_pitch_ratios(
    cap: cv2.VideoCapture,
    fps: float,
    total_frames: int,
    span: _ShotSpan,
    sample_points: list[float],
) -> list[float]:
    ratios: list[float] = []
    duration = _span_duration(span)
    for sample_point in sample_points:
        clamped_point = max(0.0, min(1.0, sample_point))
        sample_time = span.start_time + (duration * clamped_point)
        frame = _read_frame_at_time(cap, fps, total_frames, sample_time)
        if frame is None:
            continue
        ratios.append(_estimate_pitch_ratio(frame))
    return ratios


def _sample_brightness_values(
    cap: cv2.VideoCapture,
    fps: float,
    total_frames: int,
    span: _ShotSpan,
    sample_points: list[float],
) -> list[float]:
    values: list[float] = []
    duration = _span_duration(span)
    for sample_point in sample_points:
        clamped_point = max(0.0, min(1.0, sample_point))
        sample_time = span.start_time + (duration * clamped_point)
        frame = _read_frame_at_time(cap, fps, total_frames, sample_time)
        if frame is None:
            continue
        values.append(_estimate_frame_brightness(frame))
    return values


def _shot_contains_ball(
    cap: cv2.VideoCapture,
    span: _ShotSpan,
    ball_detector: BallDetector,
) -> bool:
    if span.end_frame < span.start_frame:
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, span.start_frame)
    frame_count = span.end_frame - span.start_frame + 1
    for _ in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break
        if ball_detector.detect(frame) is not None:
            return True
    return False


def _is_reaction_shot(
    duration: float,
    pitch_ratios: list[float],
    reaction_max_duration_s: float,
    min_pitch_ratio: float,
    reaction_max_peak_pitch_ratio: float,
) -> bool:
    if reaction_max_duration_s <= 0 or duration > reaction_max_duration_s:
        return False
    if not pitch_ratios:
        return False

    sorted_ratios = sorted(pitch_ratios)
    median_pitch_ratio = sorted_ratios[len(sorted_ratios) // 2]
    max_pitch_ratio = max(pitch_ratios)
    return (
        median_pitch_ratio < min_pitch_ratio
        and max_pitch_ratio < reaction_max_peak_pitch_ratio
    )


def _is_fade_transition_shot(
    duration: float,
    brightness_values: list[float],
    max_duration_s: float,
    black_frame_threshold: float,
    min_brightness_range: float,
) -> bool:
    if max_duration_s <= 0 or duration > max_duration_s:
        return False
    if not brightness_values:
        return False

    min_brightness = min(brightness_values)
    brightness_range = max(brightness_values) - min_brightness
    return (
        min_brightness <= black_frame_threshold
        and brightness_range >= min_brightness_range
    )


def _merge_adjacent_short_spans(
    spans: list[_ShotSpan],
    max_short_duration_s: float,
    max_gap_s: float,
) -> list[_ShotSpan]:
    """Merge likely false cuts from camera motion when shots are very short and contiguous."""
    if not spans:
        return []

    merged: list[_ShotSpan] = [spans[0]]
    for current in spans[1:]:
        prev = merged[-1]
        prev_dur = _span_duration(prev)
        curr_dur = _span_duration(current)
        gap_s = current.start_time - prev.end_time
        should_merge = gap_s <= max_gap_s and (
            prev_dur <= max_short_duration_s or curr_dur <= max_short_duration_s
        )
        if should_merge:
            merged[-1] = _ShotSpan(
                start_frame=prev.start_frame,
                end_frame=current.end_frame,
                start_time=prev.start_time,
                end_time=current.end_time,
            )
            continue
        merged.append(current)
    return merged


def detect_shots(
    video_path: Path,
    threshold: float = 30.0,
    detector: str = "content",
    min_scene_len_frames: int = 15,
    adaptive_threshold: float = 3.0,
    adaptive_min_content_val: float = 15.0,
) -> list[_ShotSpan]:
    video = open_video(str(video_path))
    manager = SceneManager()
    if detector == "adaptive":
        manager.add_detector(
            AdaptiveDetector(
                adaptive_threshold=adaptive_threshold,
                min_scene_len=min_scene_len_frames,
                min_content_val=adaptive_min_content_val,
            )
        )
    elif detector == "content":
        manager.add_detector(
            ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames)
        )
    else:
        raise ValueError("shot_segmentation.detector must be 'content' or 'adaptive'")
    manager.detect_scenes(video)
    scenes = manager.get_scene_list()
    return [
        _ShotSpan(
            start_frame=s[0].get_frames(),
            end_frame=s[1].get_frames() - 1,
            start_time=s[0].get_seconds(),
            end_time=s[1].get_seconds(),
        )
        for s in scenes
    ]


class ShotSegmentationStage(BaseStage):
    name = "segmentation"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        video_path: Path,
        ball_detector: BallDetector | None = None,
        **_,
    ) -> None:
        super().__init__(config, output_dir)
        self.video_path = video_path
        self.ball_detector = ball_detector

    def is_complete(self) -> bool:
        return (self.output_dir / "shots" / "shots_manifest.json").exists()

    def run(self) -> None:
        shots_dir = self.output_dir / "shots"
        shots_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config.get("shot_segmentation", {})
        threshold = cfg.get("threshold", 28.0)
        min_dur = cfg.get("min_shot_duration_s", 0.5)
        detector = cfg.get("detector", "content")
        min_scene_len_frames = cfg.get("min_scene_len_frames", 15)
        adaptive_threshold = cfg.get("adaptive_threshold", 3.5)
        adaptive_min_content_val = cfg.get("adaptive_min_content_val", 18.0)
        # Reaction filtering is opt-in unless explicitly configured.
        reaction_max_duration_s = cfg.get("reaction_max_duration_s", 3.2)
        min_pitch_ratio = cfg.get("min_pitch_ratio", 0.12)
        reaction_max_peak_pitch_ratio = cfg.get("reaction_max_peak_pitch_ratio", 0.2)
        reaction_sample_points = cfg.get("reaction_sample_points", [0.2, 0.5, 0.8])
        merge_short_shots_max_duration_s = cfg.get("merge_short_shots_max_duration_s", 1.2)
        merge_max_gap_s = cfg.get("merge_max_gap_s", 0.08)
        require_ball_in_shot = bool(cfg.get("require_ball_in_shot", True))
        exclude_fade_transitions = bool(cfg.get("exclude_fade_transitions", True))
        fade_transition_max_duration_s = float(cfg.get("fade_transition_max_duration_s", 1.0))
        fade_black_frame_threshold = float(cfg.get("fade_black_frame_threshold", 0.18))
        fade_min_brightness_range = float(cfg.get("fade_min_brightness_range", 0.25))
        fade_sample_points = cfg.get("fade_sample_points", [0.15, 0.5, 0.85])

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            cap.release()
            raise RuntimeError(
                f"Invalid video properties for {self.video_path}: fps={fps}, total_frames={total_frames}"
            )

        spans = detect_shots(
            self.video_path,
            threshold=threshold,
            detector=detector,
            min_scene_len_frames=min_scene_len_frames,
            adaptive_threshold=adaptive_threshold,
            adaptive_min_content_val=adaptive_min_content_val,
        )

        filtered_spans: list[_ShotSpan] = []
        try:
            for span in spans:
                duration = _span_duration(span)
                if duration < min_dur:
                    continue

                if reaction_max_duration_s > 0:
                    pitch_ratios = _sample_pitch_ratios(
                        cap,
                        fps,
                        total_frames,
                        span,
                        list(reaction_sample_points),
                    )
                    is_short_reaction = _is_reaction_shot(
                        duration,
                        pitch_ratios,
                        reaction_max_duration_s,
                        min_pitch_ratio,
                        reaction_max_peak_pitch_ratio,
                    )
                    if is_short_reaction:
                        continue

                if exclude_fade_transitions:
                    brightness_values = _sample_brightness_values(
                        cap,
                        fps,
                        total_frames,
                        span,
                        list(fade_sample_points),
                    )
                    is_fade_transition = _is_fade_transition_shot(
                        duration,
                        brightness_values,
                        fade_transition_max_duration_s,
                        fade_black_frame_threshold,
                        fade_min_brightness_range,
                    )
                    if is_fade_transition:
                        continue

                filtered_spans.append(span)

            spans = _merge_adjacent_short_spans(
                filtered_spans,
                max_short_duration_s=merge_short_shots_max_duration_s,
                max_gap_s=merge_max_gap_s,
            )

            if require_ball_in_shot:
                if self.ball_detector is not None:
                    ball_detector = self.ball_detector
                else:
                    ball_cfg = self.config.get("detection", {})
                    ball_model = str(ball_cfg.get("ball_model", "")).strip()
                    if not ball_model:
                        raise ValueError(
                            "detection.ball_model is required when shot_segmentation.require_ball_in_shot=true"
                        )
                    ball_confidence = float(ball_cfg.get("confidence_threshold", 0.3))
                    ball_detector = YOLOBallDetector(
                        model_name=ball_model,
                        confidence=ball_confidence,
                    )
                spans = [
                    span
                    for span in spans
                    if _shot_contains_ball(cap, span, ball_detector)
                ]
        finally:
            cap.release()

        shots: list[Shot] = []
        for span in spans:
            shot_id = f"shot_{len(shots) + 1:03d}"
            clip_path = shots_dir / f"{shot_id}.mp4"

            try:
                extract_clip(self.video_path, clip_path, span.start_time, span.end_time)
            except subprocess.CalledProcessError as exc:
                logging.warning("failed to extract %s: %s", shot_id, exc)
                continue

            shots.append(Shot(
                id=shot_id,
                start_frame=span.start_frame,
                end_frame=span.end_frame,
                start_time=span.start_time,
                end_time=span.end_time,
                clip_file=str(clip_path.relative_to(self.output_dir)),
            ))

        manifest = ShotsManifest(
            source_file=str(self.video_path),
            fps=fps,
            total_frames=total_frames,
            shots=shots,
        )
        manifest.save(shots_dir / "shots_manifest.json")
        print(f"  → {len(shots)} shots written to {shots_dir}")
