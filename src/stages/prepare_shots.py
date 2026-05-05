"""Stage 1: Prepare shots — treat the input clip as one already-trimmed shot.

The user manually trims clips in CapCut (or similar) and provides them as
``--input clip.mp4``. This stage simply copies the clip into ``shots/`` and
writes a flat single-shot manifest. No automatic scene segmentation is
performed.

See spec section 5.1.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2

from src.pipeline.base import BaseStage
from src.schemas.shots import Shot, ShotsManifest

logger = logging.getLogger(__name__)


def _video_metadata(clip_path: Path) -> tuple[float, int]:
    """Return ``(fps, frame_count)`` for the clip; (0.0, 0) on failure."""
    cap = cv2.VideoCapture(str(clip_path))
    try:
        if not cap.isOpened():
            return 0.0, 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return fps, frames
    finally:
        cap.release()


class PrepareShotsStage(BaseStage):
    name = "prepare_shots"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        video_path: Path | None = None,
        **_: object,
    ) -> None:
        super().__init__(config, output_dir)
        self.video_path = video_path

    def is_complete(self) -> bool:
        return (self.output_dir / "shots" / "shots_manifest.json").exists()

    def run(self) -> None:
        if self.video_path is None:
            raise ValueError(
                "PrepareShotsStage requires video_path; pass --input <clip.mp4>"
            )
        clip_src = Path(self.video_path).resolve()
        if not clip_src.exists():
            raise FileNotFoundError(f"Input clip not found: {clip_src}")

        shots_dir = self.output_dir / "shots"
        shots_dir.mkdir(parents=True, exist_ok=True)

        shot_id = clip_src.stem
        clip_dest = shots_dir / f"{shot_id}.mp4"

        # Copy the clip into shots/ unless it already lives there.
        try:
            same_file = clip_dest.exists() and clip_dest.samefile(clip_src)
        except FileNotFoundError:
            same_file = False
        if not same_file:
            shutil.copy2(clip_src, clip_dest)

        fps, frame_count = _video_metadata(clip_dest)
        if frame_count <= 0:
            logger.warning(
                "prepare_shots: cv2 reported 0 frames for %s — manifest will "
                "still be written but downstream stages may fail.",
                clip_dest,
            )
        effective_fps = fps if fps > 0 else 25.0
        end_frame = max(0, frame_count - 1)

        shot = Shot(
            id=shot_id,
            start_frame=0,
            end_frame=end_frame,
            start_time=0.0,
            end_time=(end_frame + 1) / effective_fps if frame_count > 0 else 0.0,
            clip_file=str(clip_dest.relative_to(self.output_dir)),
        )
        manifest = ShotsManifest(
            source_file=str(clip_src),
            fps=effective_fps,
            total_frames=frame_count,
            shots=[shot],
        )
        manifest.save(shots_dir / "shots_manifest.json")
        logger.info(
            "prepare_shots: wrote 1 shot (%s, %d frames @ %.2f fps)",
            shot_id, frame_count, effective_fps,
        )
