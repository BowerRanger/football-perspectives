from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import logging
import re

import cv2


_SHOT_ID_PATTERN = re.compile(r"[^A-Za-z0-9_-]")


def _sanitise_shot_id(raw: str) -> str:
    """Reduce a clip filename stem to a filesystem-safe shot id.

    Strips characters outside ``[A-Za-z0-9_-]`` and truncates to 64
    chars. Raises ``ValueError`` if the result is empty (e.g. input was
    ``"   "``), since downstream code uses shot_id as a routing key
    and an empty key collides across shots.
    """
    cleaned = _SHOT_ID_PATTERN.sub("", raw)
    cleaned = cleaned[:64]
    if not cleaned:
        raise ValueError(f"shot_id sanitised to empty string from {raw!r}")
    return cleaned


@dataclass
class Shot:
    id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    clip_file: str
    speed_factor: float = 1.0


@dataclass
class ShotsManifest:
    source_file: str
    fps: float
    total_frames: int
    shots: list[Shot] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ShotsManifest":
        data = json.loads(path.read_text())
        fields = {
            "id",
            "start_frame",
            "end_frame",
            "start_time",
            "end_time",
            "clip_file",
            "speed_factor",
        }
        shots = [Shot(**{k: v for k, v in s.items() if k in fields}) for s in data.pop("shots")]
        return cls(shots=shots, **data)

    @classmethod
    def infer_from_clips(
        cls,
        shots_dir: Path,
        output_dir: Path | None = None,
    ) -> "ShotsManifest":
        resolved_output_dir = output_dir or shots_dir.parent
        video_extensions = {".mp4", ".mov", ".avi", ".mkv"}

        def sort_key(path: Path) -> tuple[int, int | str, str]:
            match = re.fullmatch(r"shot_(\d+)", path.stem)
            if match:
                return (0, int(match.group(1)), path.name.lower())
            return (1, path.stem.lower(), path.name.lower())

        clip_paths = sorted(
            [
                path
                for path in shots_dir.iterdir()
                if path.is_file()
                and not path.name.startswith(".")
                and path.suffix.lower() in video_extensions
            ],
            key=sort_key,
        ) if shots_dir.exists() else []

        clip_metadata: list[tuple[Path, float, int]] = []

        for clip_path in clip_paths:
            cap = cv2.VideoCapture(str(clip_path))
            try:
                if not cap.isOpened():
                    logging.warning("skipping unreadable prepared clip: %s", clip_path.name)
                    continue

                clip_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                clip_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if clip_frames <= 0:
                    logging.warning("skipping prepared clip with no frames: %s", clip_path.name)
                    continue
                clip_metadata.append((clip_path, clip_fps, clip_frames))
            finally:
                cap.release()

        valid_fps_values = [clip_fps for _, clip_fps, _ in clip_metadata if clip_fps > 0]
        inferred_fps = valid_fps_values[0] if valid_fps_values else 25.0
        if any(abs(clip_fps - inferred_fps) > 0.01 for clip_fps in valid_fps_values[1:]):
            raise ValueError("Prepared clips must share a common FPS to infer a shots manifest")

        total_frames = 0
        shots: list[Shot] = []
        start_frame = 0
        effective_fps = inferred_fps if inferred_fps > 0 else 25.0

        for clip_path, _, clip_frames in clip_metadata:
            end_frame = start_frame + clip_frames - 1
            shots.append(Shot(
                id=clip_path.stem,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_frame / effective_fps,
                end_time=(end_frame + 1) / effective_fps,
                clip_file=str(clip_path.relative_to(resolved_output_dir)),
            ))
            start_frame = end_frame + 1
            total_frames += clip_frames

        return cls(
            source_file="unknown",
            fps=inferred_fps,
            total_frames=total_frames,
            shots=shots,
        )

    @classmethod
    def load_or_infer(
        cls,
        shots_dir: Path,
        persist: bool = False,
    ) -> "ShotsManifest":
        manifest_path = shots_dir / "shots_manifest.json"
        if manifest_path.exists():
            return cls.load(manifest_path)

        manifest = cls.infer_from_clips(shots_dir, output_dir=shots_dir.parent)
        if persist:
            manifest.save(manifest_path)
        return manifest
