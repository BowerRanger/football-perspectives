from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class CameraFrame:
    frame: int
    intrinsic_matrix: list[list[float]]
    rotation_vector: list[float]
    translation_vector: list[float]
    reprojection_error: float
    num_correspondences: int
    confidence: float
    tracked_landmark_types: list[str] = field(default_factory=list)


@dataclass
class CalibrationResult:
    shot_id: str
    camera_type: str  # "static" | "tracking"
    frames: list[CameraFrame] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "CalibrationResult":
        data = json.loads(path.read_text())
        frames = [CameraFrame(**f) for f in data.pop("frames")]
        return cls(frames=frames, **data)
