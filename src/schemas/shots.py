from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class Shot:
    id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    clip_file: str
    thumbnail: str


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
        shots = [Shot(**s) for s in data.pop("shots")]
        return cls(shots=shots, **data)
