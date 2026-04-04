from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class Alignment:
    shot_id: str
    frame_offset: int
    confidence: float
    method: str  # "ball_trajectory" | "player_formation" | "manual"
    overlap_frames: list[int]  # [start_frame, end_frame] in reference shot


@dataclass
class SyncMap:
    reference_shot: str
    alignments: list[Alignment] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "SyncMap":
        data = json.loads(path.read_text())
        alignments = [Alignment(**a) for a in data.pop("alignments")]
        return cls(alignments=alignments, **data)
