from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

COCO_KEYPOINT_NAMES: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@dataclass
class Keypoint:
    name: str
    x: float
    y: float
    conf: float


@dataclass
class PlayerPoseFrame:
    frame: int
    keypoints: list[Keypoint] = field(default_factory=list)


@dataclass
class PlayerPoses:
    track_id: str
    player_id: str = ""
    player_name: str = ""
    frames: list[PlayerPoseFrame] = field(default_factory=list)


@dataclass
class PosesResult:
    shot_id: str
    players: list[PlayerPoses] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "PosesResult":
        data = json.loads(path.read_text())
        players = []
        for p in data.pop("players"):
            frames = []
            for f in p.pop("frames"):
                kps = [Keypoint(**k) for k in f.pop("keypoints")]
                frames.append(PlayerPoseFrame(keypoints=kps, **f))
            players.append(PlayerPoses(frames=frames, **p))
        return cls(players=players, **data)
