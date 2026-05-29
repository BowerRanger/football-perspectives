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
class MomentInfo:
    minute: int
    added_time: int = 0
    event_type: str = "goal"
    description: str = ""


@dataclass
class KitColors:
    home_primary: str
    away_primary: str
    home_goalkeeper: str = ""
    away_goalkeeper: str = ""
    referee: str = "#000000"


@dataclass
class RosterEntry:
    name: str
    team: str
    position: str = ""
    shirt_number: int | None = None


@dataclass
class MatchInfo:
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    venue: str
    competition: str = ""
    date: str = ""
    moment: MomentInfo | None = None
    kits: KitColors | None = None
    roster: list[RosterEntry] = field(default_factory=list)


def _filter_kwargs(cls: type, data: dict) -> dict:
    """Drop keys not declared on ``cls`` so unknown fields written by a
    newer writer don't break the loader."""
    known = {f.name for f in cls.__dataclass_fields__.values()}
    return {k: v for k, v in data.items() if k in known}


def _load_match(data: dict | None) -> MatchInfo | None:
    if not data:
        return None
    moment_raw = data.get("moment")
    kits_raw = data.get("kits")
    roster_raw = data.get("roster") or []
    moment = MomentInfo(**_filter_kwargs(MomentInfo, moment_raw)) if moment_raw else None
    kits = KitColors(**_filter_kwargs(KitColors, kits_raw)) if kits_raw else None
    roster = [RosterEntry(**_filter_kwargs(RosterEntry, e)) for e in roster_raw]
    match_kwargs = _filter_kwargs(
        MatchInfo,
        {k: v for k, v in data.items() if k not in {"moment", "kits", "roster"}},
    )
    return MatchInfo(moment=moment, kits=kits, roster=roster, **match_kwargs)


@dataclass
class ShotsManifest:
    source_file: str
    fps: float
    total_frames: int
    shots: list[Shot] = field(default_factory=list)
    match: MatchInfo | None = None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ShotsManifest":
        data = json.loads(path.read_text())
        shots = [
            Shot(**_filter_kwargs(Shot, s)) for s in data.pop("shots", [])
        ]
        match = _load_match(data.pop("match", None))
        manifest_kwargs = _filter_kwargs(cls, data)
        return cls(shots=shots, match=match, **manifest_kwargs)

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
