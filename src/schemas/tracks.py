from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class TrackFrame:
    frame: int
    bbox: list[float]           # [x1, y1, x2, y2] in pixel space
    confidence: float
    pitch_position: list[float] | None  # [x, y] in pitch metres, or None


@dataclass
class Track:
    track_id: str
    class_name: str             # "player" | "goalkeeper" | "referee" | "ball"
    team: str                   # "A" | "B" | "referee" | "unknown"
    player_id: str = ""         # global ID across shots (e.g., "P001")
    player_name: str = ""       # user-friendly name (e.g., "Salah")
    frames: list[TrackFrame] = field(default_factory=list)


@dataclass
class TracksResult:
    shot_id: str
    tracks: list[Track] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "TracksResult":
        data = json.loads(path.read_text())
        tracks = []
        for t in data.pop("tracks"):
            frames = [TrackFrame(**f) for f in t.pop("frames")]
            tracks.append(Track(frames=frames, **t))
        return cls(tracks=tracks, **data)
