from dataclasses import dataclass
from pathlib import Path

import cv2
from scenedetect import open_video, SceneManager, ContentDetector

from src.pipeline.base import BaseStage
from src.schemas.shots import Shot, ShotsManifest
from src.utils.ffmpeg import extract_clip, extract_thumbnail


@dataclass
class _ShotSpan:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float


def detect_shots(video_path: Path, threshold: float = 30.0) -> list[_ShotSpan]:
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
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

    def __init__(self, config: dict, output_dir: Path, video_path: Path, **_) -> None:
        super().__init__(config, output_dir)
        self.video_path = video_path

    def is_complete(self) -> bool:
        return (self.output_dir / "shots" / "shots_manifest.json").exists()

    def run(self) -> None:
        shots_dir = self.output_dir / "shots"
        shots_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config.get("shot_segmentation", {})
        threshold = cfg.get("threshold", 30.0)
        min_dur = cfg.get("min_shot_duration_s", 0.5)

        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        spans = detect_shots(self.video_path, threshold=threshold)
        spans = [s for s in spans if (s.end_time - s.start_time) >= min_dur]

        shots: list[Shot] = []
        for i, span in enumerate(spans):
            shot_id = f"shot_{i+1:03d}"
            clip_path = shots_dir / f"{shot_id}.mp4"
            thumb_path = shots_dir / f"{shot_id}_thumb.jpg"
            mid_s = (span.start_time + span.end_time) / 2

            extract_clip(self.video_path, clip_path, span.start_time, span.end_time)
            extract_thumbnail(self.video_path, thumb_path, mid_s)

            shots.append(Shot(
                id=shot_id,
                start_frame=span.start_frame,
                end_frame=span.end_frame,
                start_time=span.start_time,
                end_time=span.end_time,
                clip_file=str(clip_path.relative_to(self.output_dir)),
                thumbnail=str(thumb_path.relative_to(self.output_dir)),
            ))

        manifest = ShotsManifest(
            source_file=str(self.video_path),
            fps=fps,
            total_frames=total_frames,
            shots=shots,
        )
        manifest.save(shots_dir / "shots_manifest.json")
        print(f"  → {len(shots)} shots written to {shots_dir}")
