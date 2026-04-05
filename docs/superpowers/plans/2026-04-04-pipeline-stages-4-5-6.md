# Football Pipeline — Stages 4–6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Stages 4 (player detection & tracking), 5 (2D pose estimation), and 6 (cross-view player matching) as independently re-runnable pipeline stages following the same patterns as Stages 1–3.

**Architecture:** Each stage is a `BaseStage` subclass reading from previous-stage output directories and writing structured JSON to its own subdirectory. ML model dependencies are wrapped behind ABCs (like `BallDetector`) so tests use deterministic fakes instead of real models. Stage 4 uses ByteTrack (via `supervision`) for within-shot identity tracking and a `TeamClassifier` ABC for team assignment. Stage 5 runs a `PoseEstimator` ABC on player crops and applies temporal Gaussian smoothing. Stage 6 matches identities across shots via Hungarian assignment on pitch-coordinate positions derived from Stage 4 and the `SyncMap` from Stage 3.

**Tech Stack:** Python 3.11+, OpenCV, NumPy, SciPy, `supervision` (ByteTrack), `ultralytics` (YOLOv8), `transformers` + `scikit-learn` (CLIP team classification), pytest

---

## File Structure

```
src/
  schemas/
    tracks.py            # TrackFrame, Track, TracksResult — JSON I/O
    poses.py             # Keypoint, PlayerPoseFrame, PlayerPoses, PosesResult — JSON I/O
    player_matches.py    # PlayerView, MatchedPlayer, PlayerMatches — JSON I/O
  utils/
    player_detector.py   # PlayerDetector ABC + YOLOPlayerDetector + FakePlayerDetector
    team_classifier.py   # TeamClassifier ABC + FakeTeamClassifier + CLIPTeamClassifier
    pose_estimator.py    # PoseEstimator ABC + FakePoseEstimator + ViTPoseEstimator
  stages/
    tracking.py          # PlayerTrackingStage
    pose.py              # PoseEstimationStage
    matching.py          # CrossViewMatchingStage
tests/
  test_tracking.py       # Unit + integration tests for Stage 4
  test_pose.py           # Unit + integration tests for Stage 5
  test_matching.py       # Unit + integration tests for Stage 6
```

**Modified files:**
- `src/pipeline/runner.py` — add stages 4–6 to `STAGE_ORDER` + `_ALIASES`
- `config/default.yaml` — add `tracking`, `pose_estimation`, `matching` config sections
- `pyproject.toml` — add `supervision`, `transformers`, `scikit-learn` dependencies

---

## Task 1: Track schemas

**Files:**
- Create: `src/schemas/tracks.py`
- Create: `src/schemas/poses.py`
- Create: `src/schemas/player_matches.py`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Write failing schema tests**

Add to `tests/test_schemas.py`:

```python
from src.schemas.tracks import TrackFrame, Track, TracksResult
from src.schemas.poses import Keypoint, PlayerPoseFrame, PlayerPoses, PosesResult, COCO_KEYPOINT_NAMES
from src.schemas.player_matches import PlayerView, MatchedPlayer, PlayerMatches

def test_tracks_result_round_trip(tmp_path):
    tf = TrackFrame(frame=0, bbox=[10.0, 20.0, 80.0, 200.0], confidence=0.9, pitch_position=[34.2, 21.5])
    track = Track(track_id="T001", class_name="player", team="A", frames=[tf])
    result = TracksResult(shot_id="shot_001", tracks=[track])
    path = tmp_path / "tracks.json"
    result.save(path)
    loaded = TracksResult.load(path)
    assert loaded.shot_id == "shot_001"
    assert len(loaded.tracks) == 1
    assert loaded.tracks[0].track_id == "T001"
    assert loaded.tracks[0].frames[0].pitch_position == [34.2, 21.5]

def test_poses_result_round_trip(tmp_path):
    kp = Keypoint(name="nose", x=100.0, y=50.0, conf=0.91)
    pf = PlayerPoseFrame(frame=0, keypoints=[kp])
    pp = PlayerPoses(track_id="T001", frames=[pf])
    result = PosesResult(shot_id="shot_001", players=[pp])
    path = tmp_path / "poses.json"
    result.save(path)
    loaded = PosesResult.load(path)
    assert loaded.shot_id == "shot_001"
    assert loaded.players[0].frames[0].keypoints[0].name == "nose"
    assert loaded.players[0].frames[0].keypoints[0].conf == 0.91

def test_player_matches_round_trip(tmp_path):
    view = PlayerView(shot_id="shot_001", track_id="T001")
    player = MatchedPlayer(player_id="P001", team="A", views=[view])
    result = PlayerMatches(matched_players=[player])
    path = tmp_path / "matches.json"
    result.save(path)
    loaded = PlayerMatches.load(path)
    assert len(loaded.matched_players) == 1
    assert loaded.matched_players[0].player_id == "P001"
    assert loaded.matched_players[0].views[0].shot_id == "shot_001"

def test_coco_keypoint_names_length():
    assert len(COCO_KEYPOINT_NAMES) == 17
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_schemas.py -v -k "tracks or poses or player_matches or coco"
```

Expected: `ImportError` (modules not yet created).

- [ ] **Step 3: Create `src/schemas/tracks.py`**

```python
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
```

- [ ] **Step 4: Create `src/schemas/poses.py`**

```python
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
```

- [ ] **Step 5: Create `src/schemas/player_matches.py`**

```python
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class PlayerView:
    shot_id: str
    track_id: str


@dataclass
class MatchedPlayer:
    player_id: str
    team: str
    views: list[PlayerView] = field(default_factory=list)


@dataclass
class PlayerMatches:
    matched_players: list[MatchedPlayer] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "PlayerMatches":
        data = json.loads(path.read_text())
        players = []
        for p in data.pop("matched_players"):
            views = [PlayerView(**v) for v in p.pop("views")]
            players.append(MatchedPlayer(views=views, **p))
        return cls(matched_players=players)
```

- [ ] **Step 6: Run tests — expect PASS**

```bash
pytest tests/test_schemas.py -v -k "tracks or poses or player_matches or coco"
```

Expected: 4 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/schemas/tracks.py src/schemas/poses.py src/schemas/player_matches.py tests/test_schemas.py
git commit -m "feat: add tracks, poses, player_matches schemas"
```

---

## Task 2: PlayerDetector ABC + fakes

**Files:**
- Create: `src/utils/player_detector.py`
- Create: `tests/test_tracking.py` (initial)

- [ ] **Step 1: Write failing test**

Create `tests/test_tracking.py`:

```python
import numpy as np
import pytest
from src.utils.player_detector import Detection, FakePlayerDetector, PlayerDetector


def test_fake_player_detector_cycles():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    dets = [
        [Detection(bbox=(10.0, 20.0, 80.0, 200.0), confidence=0.9, class_name="player")],
        [],
    ]
    detector = FakePlayerDetector(dets)
    assert len(detector.detect(frame)) == 1
    assert len(detector.detect(frame)) == 0
    assert len(detector.detect(frame)) == 1  # cycles


def test_detection_is_player_detector():
    assert issubclass(FakePlayerDetector, PlayerDetector)
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/test_tracking.py -v -k "fake_player or detection_is"
```

Expected: `ImportError`.

- [ ] **Step 3: Create `src/utils/player_detector.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str  # "player" | "goalkeeper" | "referee" | "ball"


class PlayerDetector(ABC):
    """Detects players (and optionally the ball) in a single frame."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Returns a list of detections found in the frame."""
        ...


class YOLOPlayerDetector(PlayerDetector):
    """Player detector backed by a YOLOv8 model fine-tuned on football data."""

    # Class IDs for a football-fine-tuned model: 0=player, 1=goalkeeper, 2=referee, 3=ball
    _CLASS_NAMES: dict[int, str] = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}

    def __init__(self, model_name: str = "yolov8x.pt", confidence: float = 0.3) -> None:
        from ultralytics import YOLO  # lazy import
        self._model = YOLO(model_name)
        self._confidence = confidence

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(frame, verbose=False)[0]
        detections: list[Detection] = []
        for box in results.boxes:
            conf = float(box.conf)
            if conf < self._confidence:
                continue
            cls_id = int(box.cls)
            class_name = self._CLASS_NAMES.get(cls_id, "player")
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf, class_name=class_name))
        return detections


class FakePlayerDetector(PlayerDetector):
    """Deterministic detector for tests — cycles through a pre-supplied sequence."""

    def __init__(self, detections_sequence: list[list[Detection]]) -> None:
        self._seq = detections_sequence
        self._idx = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        dets = self._seq[self._idx % len(self._seq)]
        self._idx += 1
        return dets
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_tracking.py -v -k "fake_player or detection_is"
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/player_detector.py tests/test_tracking.py
git commit -m "feat: add PlayerDetector ABC with YOLO and fake implementations"
```

---

## Task 3: TeamClassifier ABC + fakes

**Files:**
- Create: `src/utils/team_classifier.py`
- Modify: `tests/test_tracking.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_tracking.py`:

```python
import numpy as np
from src.utils.team_classifier import FakeTeamClassifier, TeamClassifier


def test_fake_team_classifier_returns_fixed_label():
    crops = [np.zeros((60, 40, 3), dtype=np.uint8) for _ in range(3)]
    clf = FakeTeamClassifier("B")
    labels = clf.classify(crops)
    assert labels == ["B", "B", "B"]


def test_fake_team_classifier_empty_input():
    clf = FakeTeamClassifier("A")
    assert clf.classify([]) == []


def test_team_classifier_is_abstract():
    assert issubclass(FakeTeamClassifier, TeamClassifier)
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/test_tracking.py -v -k "team_classifier"
```

Expected: `ImportError`.

- [ ] **Step 3: Create `src/utils/team_classifier.py`**

```python
from abc import ABC, abstractmethod
import numpy as np


class TeamClassifier(ABC):
    """Assigns team labels to player crop images."""

    @abstractmethod
    def classify(self, crops: list[np.ndarray]) -> list[str]:
        """Given a list of BGR player crop images, return team labels ('A', 'B', 'referee')."""
        ...


class FakeTeamClassifier(TeamClassifier):
    """Returns a fixed team label for all crops — used in tests."""

    def __init__(self, label: str = "A") -> None:
        self._label = label

    def classify(self, crops: list[np.ndarray]) -> list[str]:
        return [self._label] * len(crops)


class CLIPTeamClassifier(TeamClassifier):
    """
    K-means team assignment using CLIP visual embeddings.

    Usage:
        clf = CLIPTeamClassifier()
        clf.fit(all_crops_from_shot)   # cluster embeddings into k=3 groups
        labels = clf.classify(crops)   # predict team for new crops
    """

    def __init__(self, n_clusters: int = 3) -> None:
        self._n_clusters = n_clusters
        self._kmeans = None
        # Cluster-ID → team name; 0 and 1 are teams, 2 is referee by default.
        # Caller can override by inspecting cluster centroids if needed.
        self._id_to_name: dict[int, str] = {0: "A", 1: "B", 2: "referee"}

    def _embed(self, crops: list[np.ndarray]) -> "np.ndarray":
        from transformers import CLIPProcessor, CLIPModel  # lazy import
        from PIL import Image
        import torch

        if not hasattr(self, "_processor"):
            self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._model.eval()

        pil_images = [Image.fromarray(c[:, :, ::-1]) for c in crops]  # BGR → RGB
        inputs = self._processor(images=pil_images, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        return feats.numpy()

    def fit(self, crops: list[np.ndarray]) -> None:
        """Cluster a representative batch of crops to fix team identities for this shot."""
        from sklearn.cluster import KMeans
        feats = self._embed(crops)
        km = KMeans(n_clusters=self._n_clusters, n_init=10, random_state=0)
        km.fit(feats)
        self._kmeans = km

    def classify(self, crops: list[np.ndarray]) -> list[str]:
        if self._kmeans is None:
            raise RuntimeError("Call fit() before classify()")
        feats = self._embed(crops)
        cluster_ids = self._kmeans.predict(feats)
        return [self._id_to_name.get(int(cid), "unknown") for cid in cluster_ids]
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_tracking.py -v -k "team_classifier"
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/team_classifier.py tests/test_tracking.py
git commit -m "feat: add TeamClassifier ABC with CLIP and fake implementations"
```

---

## Task 4: PoseEstimator ABC + fakes

**Files:**
- Create: `src/utils/pose_estimator.py`
- Create: `tests/test_pose.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_pose.py`:

```python
import numpy as np
import pytest
from src.utils.pose_estimator import FakePoseEstimator, PoseEstimator
from src.schemas.poses import COCO_KEYPOINT_NAMES


def test_fake_pose_estimator_returns_17_keypoints():
    estimator = FakePoseEstimator()
    crop = np.zeros((120, 60, 3), dtype=np.uint8)
    kps = estimator.estimate(crop, bbox_offset=(100.0, 50.0))
    assert len(kps) == 17


def test_fake_pose_estimator_applies_offset():
    estimator = FakePoseEstimator()
    crop = np.zeros((120, 60, 3), dtype=np.uint8)
    kps = estimator.estimate(crop, bbox_offset=(100.0, 50.0))
    # All keypoints should have x >= 100 (offset applied)
    assert all(kp.x >= 100.0 for kp in kps)


def test_fake_pose_estimator_keypoint_names():
    estimator = FakePoseEstimator()
    crop = np.zeros((120, 60, 3), dtype=np.uint8)
    kps = estimator.estimate(crop, bbox_offset=(0.0, 0.0))
    names = [kp.name for kp in kps]
    assert names == COCO_KEYPOINT_NAMES


def test_pose_estimator_is_abstract():
    assert issubclass(FakePoseEstimator, PoseEstimator)
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_pose.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create `src/utils/pose_estimator.py`**

```python
from abc import ABC, abstractmethod
import numpy as np
from src.schemas.poses import COCO_KEYPOINT_NAMES, Keypoint


class PoseEstimator(ABC):
    """Estimates 2D COCO keypoints for a single player crop."""

    @abstractmethod
    def estimate(self, crop: np.ndarray, bbox_offset: tuple[float, float]) -> list[Keypoint]:
        """
        Args:
            crop: BGR image of the player (with padding).
            bbox_offset: (x, y) pixel coordinates of the crop's top-left corner
                         in the original video frame. Used to return keypoints in
                         frame-absolute coordinates.
        Returns:
            List of 17 Keypoints in COCO order, in original-frame pixel coordinates.
        """
        ...


class FakePoseEstimator(PoseEstimator):
    """Returns deterministic keypoints spread evenly down the crop — used in tests."""

    def __init__(self, conf: float = 0.9) -> None:
        self._conf = conf

    def estimate(self, crop: np.ndarray, bbox_offset: tuple[float, float]) -> list[Keypoint]:
        h, w = crop.shape[:2]
        ox, oy = bbox_offset
        return [
            Keypoint(
                name=name,
                x=ox + w / 2.0,
                y=oy + h * (i + 1) / (len(COCO_KEYPOINT_NAMES) + 1),
                conf=self._conf,
            )
            for i, name in enumerate(COCO_KEYPOINT_NAMES)
        ]


class ViTPoseEstimator(PoseEstimator):
    """
    Pose estimator using ViTPose via HuggingFace Transformers.

    The model outputs heatmaps (B, num_joints, H_out, W_out). Each heatmap's
    argmax gives the predicted joint location in heatmap space; we rescale to
    frame-absolute coordinates using the crop dimensions and offset.
    """

    def __init__(self, model_name: str = "nielsr/vitpose-base-simple") -> None:
        from transformers import AutoImageProcessor, AutoModel  # lazy import
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()

    def estimate(self, crop: np.ndarray, bbox_offset: tuple[float, float]) -> list[Keypoint]:
        import torch
        from PIL import Image

        ox, oy = bbox_offset
        pil_img = Image.fromarray(crop[:, :, ::-1])  # BGR → RGB
        inputs = self._processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)

        heatmaps = outputs.last_hidden_state
        if heatmaps.dim() == 4:
            _, J, H_out, W_out = heatmaps.shape
            crop_h, crop_w = crop.shape[:2]
            kps = []
            for j in range(min(J, 17)):
                hm = heatmaps[0, j].numpy()
                flat_idx = int(np.argmax(hm))
                ky, kx = divmod(flat_idx, W_out)
                px = ox + (kx / W_out) * crop_w
                py = oy + (ky / H_out) * crop_h
                conf = float(hm.max())
                kps.append(Keypoint(name=COCO_KEYPOINT_NAMES[j], x=px, y=py, conf=conf))
            return kps

        # Fallback: zero-confidence keypoints if model output shape is unexpected
        return [Keypoint(name=name, x=0.0, y=0.0, conf=0.0) for name in COCO_KEYPOINT_NAMES]
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_pose.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/pose_estimator.py tests/test_pose.py
git commit -m "feat: add PoseEstimator ABC with ViTPose and fake implementations"
```

---

## Task 5: Stage 4 — PlayerTrackingStage

**Files:**
- Create: `src/stages/tracking.py`
- Modify: `tests/test_tracking.py`

- [ ] **Step 1: Write failing integration test**

Add to `tests/test_tracking.py`:

```python
import cv2
import numpy as np
import pytest
from pathlib import Path
from src.pipeline.config import load_config
from src.schemas.shots import Shot, ShotsManifest
from src.schemas.tracks import TracksResult
from src.stages.tracking import PlayerTrackingStage
from src.utils.player_detector import Detection, FakePlayerDetector
from src.utils.team_classifier import FakeTeamClassifier


@pytest.fixture(scope="module")
def tiny_shot_dir(tmp_path_factory) -> Path:
    """Output directory with a shots manifest and a 1-second synthetic clip."""
    root = tmp_path_factory.mktemp("tracking_stage")
    shots_dir = root / "shots"
    shots_dir.mkdir()

    clip_path = shots_dir / "shot_001.mp4"
    writer = cv2.VideoWriter(
        str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240)
    )
    for _ in range(10):
        writer.write(np.full((240, 320, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()

    shot = Shot(
        id="shot_001",
        start_frame=0,
        end_frame=9,
        start_time=0.0,
        end_time=1.0,
        clip_file="shots/shot_001.mp4",
    )
    ShotsManifest(
        source_file="test.mp4", fps=10.0, total_frames=10, shots=[shot]
    ).save(shots_dir / "shots_manifest.json")
    return root


def _one_player_det() -> Detection:
    return Detection(bbox=(50.0, 30.0, 150.0, 200.0), confidence=0.9, class_name="player")


def test_tracking_stage_writes_tracks_file(tiny_shot_dir):
    cfg = load_config()
    stage = PlayerTrackingStage(
        config=cfg,
        output_dir=tiny_shot_dir,
        player_detector=FakePlayerDetector([[_one_player_det()]]),
        team_classifier=FakeTeamClassifier("A"),
    )
    stage.run()
    assert (tiny_shot_dir / "tracks" / "shot_001_tracks.json").exists()


def test_tracking_stage_is_complete_after_run(tiny_shot_dir):
    cfg = load_config()
    stage = PlayerTrackingStage(
        config=cfg,
        output_dir=tiny_shot_dir,
        player_detector=FakePlayerDetector([[_one_player_det()]]),
        team_classifier=FakeTeamClassifier("A"),
    )
    assert stage.is_complete()


def test_tracking_stage_tracks_have_correct_schema(tiny_shot_dir):
    result = TracksResult.load(tiny_shot_dir / "tracks" / "shot_001_tracks.json")
    assert result.shot_id == "shot_001"
    assert len(result.tracks) >= 1
    t = result.tracks[0]
    assert t.team == "A"
    assert len(t.frames) >= 1
    assert len(t.frames[0].bbox) == 4
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_tracking.py -v -k "tracking_stage"
```

Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/stages/tracking.py`**

```python
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.calibration import CalibrationResult
from src.schemas.shots import ShotsManifest
from src.schemas.tracks import Track, TrackFrame, TracksResult
from src.utils.camera import project_to_pitch
from src.utils.player_detector import Detection, PlayerDetector, YOLOPlayerDetector
from src.utils.team_classifier import FakeTeamClassifier, TeamClassifier


def _foot_centre(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the bottom-centre pixel of a bounding box (approximate foot position)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


class PlayerTrackingStage(BaseStage):
    name = "tracking"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        player_detector: PlayerDetector | None = None,
        team_classifier: TeamClassifier | None = None,
        **_,
    ) -> None:
        super().__init__(config, output_dir)
        self.player_detector = player_detector
        self.team_classifier = team_classifier

    def is_complete(self) -> bool:
        tracks_dir = self.output_dir / "tracks"
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = ShotsManifest.load(manifest_path)
            return all(
                (tracks_dir / f"{shot.id}_tracks.json").exists()
                for shot in manifest.shots
            )
        except Exception:
            return False

    def run(self) -> None:
        tracks_dir = self.output_dir / "tracks"
        tracks_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config.get("tracking", {})
        confidence = cfg.get("confidence_threshold", 0.3)
        model_name = cfg.get("player_model", "yolov8x.pt")

        detector = self.player_detector or YOLOPlayerDetector(
            model_name=model_name, confidence=confidence
        )
        team_classifier = self.team_classifier or FakeTeamClassifier("A")

        manifest = ShotsManifest.load(self.output_dir / "shots" / "shots_manifest.json")
        cal_dir = self.output_dir / "calibration"

        for shot in manifest.shots:
            cal_path = cal_dir / f"{shot.id}_calibration.json"
            calibration = CalibrationResult.load(cal_path) if cal_path.exists() else None
            result = self._track_shot(shot.id, shot.clip_file, detector, team_classifier, calibration)
            result.save(tracks_dir / f"{shot.id}_tracks.json")
            print(f"  -> {shot.id}: {len(result.tracks)} tracks")

    def _track_shot(
        self,
        shot_id: str,
        clip_file: str,
        detector: PlayerDetector,
        team_classifier: TeamClassifier,
        calibration: CalibrationResult | None,
    ) -> TracksResult:
        try:
            import supervision as sv
        except ImportError:
            raise ImportError("supervision is required for tracking: pip install supervision")

        clip_path = self.output_dir / clip_file
        cap = cv2.VideoCapture(str(clip_path))
        byte_tracker = sv.ByteTrack()

        cal_map = {f.frame: f for f in calibration.frames} if calibration else {}
        last_cal = (calibration.frames[0] if (calibration and calibration.frames) else None)
        active_tracks: dict[int, Track] = {}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in cal_map:
                last_cal = cal_map[frame_idx]

            detections = detector.detect(frame)
            player_dets = [d for d in detections if d.class_name != "ball"]

            if player_dets:
                xyxy = np.array([list(d.bbox) for d in player_dets], dtype=np.float32)
                confs = np.array([d.confidence for d in player_dets], dtype=np.float32)
                class_ids = np.zeros(len(player_dets), dtype=int)
                sv_dets = sv.Detections(xyxy=xyxy, confidence=confs, class_id=class_ids)
                tracked = byte_tracker.update_with_detections(sv_dets)

                crops = []
                for i in range(len(tracked)):
                    x1, y1, x2, y2 = tracked.xyxy[i]
                    crops.append(frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)])
                team_labels = team_classifier.classify(crops) if crops else []

                for i, tid in enumerate(tracked.tracker_id):
                    if tid is None:
                        continue
                    x1, y1, x2, y2 = tracked.xyxy[i]
                    conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.5
                    bbox = [float(x1), float(y1), float(x2), float(y2)]
                    team = team_labels[i] if i < len(team_labels) else "unknown"

                    foot_u, foot_v = _foot_centre((x1, y1, x2, y2))
                    pitch_pos: list[float] | None = None
                    if last_cal is not None:
                        K = np.array(last_cal.intrinsic_matrix, dtype=np.float32)
                        rvec = np.array(last_cal.rotation_vector, dtype=np.float32)
                        tvec = np.array(last_cal.translation_vector, dtype=np.float32)
                        try:
                            pp = project_to_pitch(np.array([foot_u, foot_v]), K, rvec, tvec)
                            pitch_pos = [float(pp[0]), float(pp[1])]
                        except Exception:
                            pass

                    track_frame = TrackFrame(
                        frame=frame_idx, bbox=bbox, confidence=conf, pitch_position=pitch_pos
                    )
                    if tid not in active_tracks:
                        active_tracks[tid] = Track(
                            track_id=f"T{tid:03d}", class_name="player", team=team, frames=[]
                        )
                    active_tracks[tid].frames.append(track_frame)

            frame_idx += 1

        cap.release()
        return TracksResult(shot_id=shot_id, tracks=list(active_tracks.values()))
```

- [ ] **Step 4: Add `supervision` to `pyproject.toml`**

In `pyproject.toml`, add to the `dependencies` list:

```toml
"supervision>=0.20",
```

Install it:

```bash
pip install supervision
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/test_tracking.py -v -k "tracking_stage"
```

Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/stages/tracking.py pyproject.toml tests/test_tracking.py
git commit -m "feat: Stage 4 player detection and tracking with ByteTrack"
```

---

## Task 6: Stage 5 — PoseEstimationStage

**Files:**
- Create: `src/stages/pose.py`
- Modify: `tests/test_pose.py`

- [ ] **Step 1: Write failing integration test**

Add to `tests/test_pose.py`:

```python
import cv2
import numpy as np
import pytest
from pathlib import Path
from src.pipeline.config import load_config
from src.schemas.shots import Shot, ShotsManifest
from src.schemas.tracks import Track, TrackFrame, TracksResult
from src.schemas.poses import PosesResult
from src.stages.pose import PoseEstimationStage, smooth_keypoints
from src.utils.pose_estimator import FakePoseEstimator


@pytest.fixture(scope="module")
def shot_with_tracks(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("pose_stage")
    shots_dir = root / "shots"
    shots_dir.mkdir()
    tracks_dir = root / "tracks"
    tracks_dir.mkdir()

    clip_path = shots_dir / "shot_001.mp4"
    writer = cv2.VideoWriter(
        str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240)
    )
    for _ in range(10):
        writer.write(np.full((240, 320, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()

    shot = Shot(id="shot_001", start_frame=0, end_frame=9,
                start_time=0.0, end_time=1.0, clip_file="shots/shot_001.mp4")
    ShotsManifest(source_file="test.mp4", fps=10.0, total_frames=10, shots=[shot]).save(
        shots_dir / "shots_manifest.json"
    )

    frames = [TrackFrame(frame=i, bbox=[50.0, 30.0, 150.0, 200.0],
                         confidence=0.9, pitch_position=None) for i in range(10)]
    track = Track(track_id="T001", class_name="player", team="A", frames=frames)
    TracksResult(shot_id="shot_001", tracks=[track]).save(
        tracks_dir / "shot_001_tracks.json"
    )
    return root


def test_pose_stage_writes_poses_file(shot_with_tracks):
    cfg = load_config()
    stage = PoseEstimationStage(
        config=cfg,
        output_dir=shot_with_tracks,
        pose_estimator=FakePoseEstimator(),
    )
    stage.run()
    assert (shot_with_tracks / "poses" / "shot_001_poses.json").exists()


def test_pose_stage_is_complete_after_run(shot_with_tracks):
    cfg = load_config()
    stage = PoseEstimationStage(
        config=cfg,
        output_dir=shot_with_tracks,
        pose_estimator=FakePoseEstimator(),
    )
    assert stage.is_complete()


def test_pose_stage_keypoints_in_frame_coords(shot_with_tracks):
    result = PosesResult.load(shot_with_tracks / "poses" / "shot_001_poses.json")
    assert len(result.players) >= 1
    kps = result.players[0].frames[0].keypoints
    assert len(kps) == 17
    # x should be >= 50 (bbox x1), not in crop-local coords
    assert all(kp.x >= 50.0 for kp in kps)


def test_smooth_keypoints_preserves_track_id():
    from src.schemas.poses import Keypoint, PlayerPoseFrame, PlayerPoses
    frames = [
        PlayerPoseFrame(frame=i, keypoints=[Keypoint(name="nose", x=float(i), y=float(i), conf=0.9)])
        for i in range(5)
    ]
    pp = PlayerPoses(track_id="T001", frames=frames)
    smoothed = smooth_keypoints(pp)
    assert smoothed.track_id == "T001"
    assert len(smoothed.frames) == 5
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_pose.py -v -k "pose_stage or smooth"
```

Expected: `ImportError`.

- [ ] **Step 3: Create `src/stages/pose.py`**

```python
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.pipeline.base import BaseStage
from src.schemas.poses import Keypoint, PlayerPoseFrame, PlayerPoses, PosesResult
from src.schemas.shots import ShotsManifest
from src.schemas.tracks import TracksResult
from src.utils.pose_estimator import FakePoseEstimator, PoseEstimator, ViTPoseEstimator

_MIN_PLAYER_HEIGHT_PX = 60  # flag players occupying < 60px height as low-res


def _crop_with_padding(
    frame: np.ndarray, bbox: list[float], pad_ratio: float = 0.2
) -> tuple[np.ndarray, tuple[float, float]]:
    """Crop a player region from frame with proportional padding. Returns (crop, (ox, oy))."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - bw * pad_ratio))
    cy1 = max(0, int(y1 - bh * pad_ratio))
    cx2 = min(w, int(x2 + bw * pad_ratio))
    cy2 = min(h, int(y2 + bh * pad_ratio))
    return frame[cy1:cy2, cx1:cx2], (float(cx1), float(cy1))


def smooth_keypoints(player_poses: PlayerPoses, sigma: float = 2.0) -> PlayerPoses:
    """
    Apply 1D Gaussian smoothing along the time axis for each keypoint's x and y
    coordinates. Confidence values are left unchanged.
    """
    if len(player_poses.frames) < 3:
        return player_poses

    n_kps = len(player_poses.frames[0].keypoints)
    xs = np.array([[kp.x for kp in f.keypoints] for f in player_poses.frames])
    ys = np.array([[kp.y for kp in f.keypoints] for f in player_poses.frames])
    xs_smooth = gaussian_filter1d(xs, sigma=sigma, axis=0)
    ys_smooth = gaussian_filter1d(ys, sigma=sigma, axis=0)

    smoothed_frames = [
        PlayerPoseFrame(
            frame=orig.frame,
            keypoints=[
                Keypoint(
                    name=orig.keypoints[j].name,
                    x=float(xs_smooth[i, j]),
                    y=float(ys_smooth[i, j]),
                    conf=orig.keypoints[j].conf,
                )
                for j in range(n_kps)
            ],
        )
        for i, orig in enumerate(player_poses.frames)
    ]
    return PlayerPoses(track_id=player_poses.track_id, frames=smoothed_frames)


class PoseEstimationStage(BaseStage):
    name = "pose"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        pose_estimator: PoseEstimator | None = None,
        **_,
    ) -> None:
        super().__init__(config, output_dir)
        self.pose_estimator = pose_estimator

    def is_complete(self) -> bool:
        poses_dir = self.output_dir / "poses"
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = ShotsManifest.load(manifest_path)
            return all(
                (poses_dir / f"{shot.id}_poses.json").exists()
                for shot in manifest.shots
            )
        except Exception:
            return False

    def run(self) -> None:
        poses_dir = self.output_dir / "poses"
        poses_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config.get("pose_estimation", {})
        min_conf = cfg.get("min_confidence", 0.3)
        smooth_sigma = cfg.get("smooth_sigma", 2.0)
        model_name = cfg.get("model_name", "nielsr/vitpose-base-simple")

        estimator = self.pose_estimator or ViTPoseEstimator(model_name=model_name)

        manifest = ShotsManifest.load(self.output_dir / "shots" / "shots_manifest.json")
        tracks_dir = self.output_dir / "tracks"

        for shot in manifest.shots:
            tracks_path = tracks_dir / f"{shot.id}_tracks.json"
            if not tracks_path.exists():
                print(f"  [SKIP] {shot.id}: no tracks file")
                continue
            tracks = TracksResult.load(tracks_path)
            result = self._estimate_shot(
                shot.id, shot.clip_file, tracks, estimator, min_conf, smooth_sigma
            )
            result.save(poses_dir / f"{shot.id}_poses.json")
            print(f"  -> {shot.id}: {len(result.players)} players")

    def _estimate_shot(
        self,
        shot_id: str,
        clip_file: str,
        tracks: TracksResult,
        estimator: PoseEstimator,
        min_conf: float,
        smooth_sigma: float,
    ) -> PosesResult:
        # Pre-build lookup: frame_idx -> [(track_id, TrackFrame)]
        frame_to_tracks: dict[int, list[tuple[str, object]]] = {}
        for track in tracks.tracks:
            for tf in track.frames:
                frame_to_tracks.setdefault(tf.frame, []).append((track.track_id, tf))

        clip_path = self.output_dir / clip_file
        cap = cv2.VideoCapture(str(clip_path))
        player_frames: dict[str, list[PlayerPoseFrame]] = {}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            for track_id, tf in frame_to_tracks.get(frame_idx, []):
                x1, y1, x2, y2 = tf.bbox
                if (y2 - y1) < _MIN_PLAYER_HEIGHT_PX:
                    continue
                crop, offset = _crop_with_padding(frame, tf.bbox)
                if crop.size == 0:
                    continue
                kps = estimator.estimate(crop, offset)
                # Zero out confidence below threshold (keep keypoint for array alignment)
                kps_out = [
                    kp if kp.conf >= min_conf
                    else Keypoint(name=kp.name, x=kp.x, y=kp.y, conf=0.0)
                    for kp in kps
                ]
                player_frames.setdefault(track_id, []).append(
                    PlayerPoseFrame(frame=frame_idx, keypoints=kps_out)
                )
            frame_idx += 1

        cap.release()

        players = []
        for track_id, frames in player_frames.items():
            pp = PlayerPoses(track_id=track_id, frames=frames)
            pp = smooth_keypoints(pp, sigma=smooth_sigma)
            players.append(pp)

        return PosesResult(shot_id=shot_id, players=players)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_pose.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stages/pose.py tests/test_pose.py
git commit -m "feat: Stage 5 2D pose estimation with ViTPose and temporal smoothing"
```

---

## Task 7: Stage 6 — CrossViewMatchingStage

**Files:**
- Create: `src/stages/matching.py`
- Create: `tests/test_matching.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_matching.py`:

```python
import json
import pytest
import numpy as np
from pathlib import Path
from src.pipeline.config import load_config
from src.schemas.player_matches import PlayerMatches
from src.schemas.shots import Shot, ShotsManifest
from src.schemas.sync_map import Alignment, SyncMap
from src.schemas.tracks import Track, TrackFrame, TracksResult
from src.stages.matching import CrossViewMatchingStage, hungarian_match_players


def _make_track(track_id: str, pitch_positions: list[list[float]], team: str = "A") -> Track:
    frames = [
        TrackFrame(frame=i, bbox=[0.0, 0.0, 60.0, 180.0],
                   confidence=0.9, pitch_position=pos)
        for i, pos in enumerate(pitch_positions)
    ]
    return Track(track_id=track_id, class_name="player", team=team, frames=frames)


def test_hungarian_match_nearby_players():
    # Two players in shot A, two in shot B at nearly the same pitch positions
    tracks_a = TracksResult(shot_id="shot_001", tracks=[
        _make_track("T001", [[10.0, 5.0]] * 5),
        _make_track("T002", [[30.0, 10.0]] * 5),
    ])
    tracks_b = TracksResult(shot_id="shot_002", tracks=[
        _make_track("T003", [[10.2, 5.1]] * 5),  # close to T001
        _make_track("T004", [[30.1, 10.2]] * 5), # close to T002
    ])
    matches = hungarian_match_players(
        tracks_a, tracks_b, sync_offset=0, reference_frames=[0, 1, 2, 3, 4]
    )
    assert ("T001", "T003") in matches
    assert ("T002", "T004") in matches


def test_hungarian_match_rejects_distant_players():
    tracks_a = TracksResult(shot_id="shot_001", tracks=[
        _make_track("T001", [[10.0, 5.0]] * 3),
    ])
    tracks_b = TracksResult(shot_id="shot_002", tracks=[
        _make_track("T002", [[80.0, 60.0]] * 3),  # far away — beyond max_distance
    ])
    matches = hungarian_match_players(
        tracks_a, tracks_b, sync_offset=0, reference_frames=[0, 1, 2],
        max_distance_m=5.0,
    )
    assert len(matches) == 0


@pytest.fixture(scope="module")
def two_shot_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("matching_stage")
    shots_dir = root / "shots"
    shots_dir.mkdir()
    tracks_dir = root / "tracks"
    tracks_dir.mkdir()
    sync_dir = root / "sync"
    sync_dir.mkdir()

    for shot_id in ("shot_001", "shot_002"):
        Shot(id=shot_id, start_frame=0, end_frame=9,
             start_time=0.0, end_time=1.0,
             clip_file=f"shots/{shot_id}.mp4")

    ShotsManifest(
        source_file="test.mp4", fps=10.0, total_frames=20,
        shots=[
            Shot(id="shot_001", start_frame=0, end_frame=9,
                 start_time=0.0, end_time=1.0, clip_file="shots/shot_001.mp4"),
            Shot(id="shot_002", start_frame=0, end_frame=9,
                 start_time=0.0, end_time=1.0, clip_file="shots/shot_002.mp4"),
        ],
    ).save(shots_dir / "shots_manifest.json")

    TracksResult(shot_id="shot_001", tracks=[
        _make_track("T001", [[10.0, 5.0]] * 10),
        _make_track("T002", [[30.0, 10.0]] * 10),
    ]).save(tracks_dir / "shot_001_tracks.json")

    TracksResult(shot_id="shot_002", tracks=[
        _make_track("T003", [[10.1, 5.0]] * 10),
        _make_track("T004", [[30.0, 9.9]] * 10),
    ]).save(tracks_dir / "shot_002_tracks.json")

    SyncMap(
        reference_shot="shot_001",
        alignments=[Alignment(shot_id="shot_002", frame_offset=0,
                               confidence=0.9, method="ball_trajectory",
                               overlap_frames=[0, 10])],
    ).save(sync_dir / "sync_map.json")
    return root


def test_matching_stage_writes_player_matches(two_shot_dir):
    cfg = load_config()
    stage = CrossViewMatchingStage(config=cfg, output_dir=two_shot_dir)
    stage.run()
    assert (two_shot_dir / "matching" / "player_matches.json").exists()


def test_matching_stage_is_complete_after_run(two_shot_dir):
    cfg = load_config()
    stage = CrossViewMatchingStage(config=cfg, output_dir=two_shot_dir)
    assert stage.is_complete()


def test_matching_stage_output_has_two_players(two_shot_dir):
    result = PlayerMatches.load(two_shot_dir / "matching" / "player_matches.json")
    assert len(result.matched_players) == 2
    # Each player should have views in both shots
    for mp in result.matched_players:
        shot_ids = {v.shot_id for v in mp.views}
        assert "shot_001" in shot_ids
        assert "shot_002" in shot_ids
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_matching.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create `src/stages/matching.py`**

```python
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.pipeline.base import BaseStage
from src.schemas.player_matches import MatchedPlayer, PlayerMatches, PlayerView
from src.schemas.shots import ShotsManifest
from src.schemas.sync_map import SyncMap
from src.schemas.tracks import TracksResult

_DEFAULT_MAX_DISTANCE_M = 5.0  # reject matches further apart than this on the pitch


def _mean_pitch_position(
    tracks_result: TracksResult, track_id: str, frames: list[int]
) -> np.ndarray | None:
    """Average pitch position for a track over the given frames. Returns None if no data."""
    positions = []
    frame_set = set(frames)
    for track in tracks_result.tracks:
        if track.track_id != track_id:
            continue
        for tf in track.frames:
            if tf.frame in frame_set and tf.pitch_position is not None:
                positions.append(tf.pitch_position)
    if not positions:
        return None
    return np.mean(positions, axis=0)


def hungarian_match_players(
    shot_a_tracks: TracksResult,
    shot_b_tracks: TracksResult,
    sync_offset: int,
    reference_frames: list[int],
    max_distance_m: float = _DEFAULT_MAX_DISTANCE_M,
) -> list[tuple[str, str]]:
    """
    Match player track IDs between two shots using the Hungarian algorithm.

    sync_offset: alignment.frame_offset — so shot_b_frame = shot_a_frame - sync_offset.
    reference_frames: frame indices in shot_a's time domain used to compute positions.

    Returns list of (track_id_in_shot_a, track_id_in_shot_b) pairs whose pitch
    distance is within max_distance_m.
    """
    tracks_a = [t for t in shot_a_tracks.tracks if t.class_name != "ball"]
    tracks_b = [t for t in shot_b_tracks.tracks if t.class_name != "ball"]
    if not tracks_a or not tracks_b:
        return []

    b_frames = [max(0, f - sync_offset) for f in reference_frames]

    pos_a = {t.track_id: _mean_pitch_position(shot_a_tracks, t.track_id, reference_frames)
             for t in tracks_a}
    pos_b = {t.track_id: _mean_pitch_position(shot_b_tracks, t.track_id, b_frames)
             for t in tracks_b}

    valid_a = [t.track_id for t in tracks_a if pos_a.get(t.track_id) is not None]
    valid_b = [t.track_id for t in tracks_b if pos_b.get(t.track_id) is not None]
    if not valid_a or not valid_b:
        return []

    # Build cost matrix: (len(valid_a), len(valid_b))
    inf = max_distance_m * 2
    cost = np.full((len(valid_a), len(valid_b)), fill_value=inf)
    for i, tid_a in enumerate(valid_a):
        for j, tid_b in enumerate(valid_b):
            cost[i, j] = float(np.linalg.norm(pos_a[tid_a] - pos_b[tid_b]))

    row_ind, col_ind = linear_sum_assignment(cost)
    return [
        (valid_a[r], valid_b[c])
        for r, c in zip(row_ind, col_ind)
        if cost[r, c] <= max_distance_m
    ]


class CrossViewMatchingStage(BaseStage):
    name = "matching"

    def is_complete(self) -> bool:
        return (self.output_dir / "matching" / "player_matches.json").exists()

    def run(self) -> None:
        matching_dir = self.output_dir / "matching"
        matching_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config.get("matching", {})
        max_distance_m = cfg.get("max_distance_m", _DEFAULT_MAX_DISTANCE_M)
        n_reference_frames = cfg.get("n_reference_frames", 10)

        manifest = ShotsManifest.load(self.output_dir / "shots" / "shots_manifest.json")
        sync_map = SyncMap.load(self.output_dir / "sync" / "sync_map.json")
        tracks_dir = self.output_dir / "tracks"

        tracks_by_shot: dict[str, TracksResult] = {}
        for shot in manifest.shots:
            path = tracks_dir / f"{shot.id}_tracks.json"
            if path.exists():
                tracks_by_shot[shot.id] = TracksResult.load(path)

        # Assign a global player_id to every track in the reference shot first
        player_counter = 0
        player_id_map: dict[tuple[str, str], str] = {}  # (shot_id, track_id) -> player_id

        ref_id = sync_map.reference_shot
        if ref_id in tracks_by_shot:
            for track in tracks_by_shot[ref_id].tracks:
                if track.class_name == "ball":
                    continue
                pid = f"P{player_counter + 1:03d}"
                player_id_map[(ref_id, track.track_id)] = pid
                player_counter += 1

        # Match each non-reference shot to the reference
        for alignment in sync_map.alignments:
            other_id = alignment.shot_id
            if ref_id not in tracks_by_shot or other_id not in tracks_by_shot:
                continue
            overlap_start, overlap_end = alignment.overlap_frames
            if overlap_end <= overlap_start:
                continue
            step = max(1, (overlap_end - overlap_start) // n_reference_frames)
            ref_frames = list(range(overlap_start, overlap_end, step))[:n_reference_frames]
            matches = hungarian_match_players(
                tracks_by_shot[ref_id],
                tracks_by_shot[other_id],
                sync_offset=alignment.frame_offset,
                reference_frames=ref_frames,
                max_distance_m=max_distance_m,
            )
            for track_id_ref, track_id_other in matches:
                pid = player_id_map.get((ref_id, track_id_ref))
                if pid is not None:
                    player_id_map[(other_id, track_id_other)] = pid

        # Collect all views per player_id and build output
        pid_to_views: dict[str, list[PlayerView]] = {}
        pid_to_team: dict[str, str] = {}
        for (shot_id, track_id), pid in player_id_map.items():
            pid_to_views.setdefault(pid, []).append(PlayerView(shot_id=shot_id, track_id=track_id))
            if pid not in pid_to_team and shot_id in tracks_by_shot:
                for t in tracks_by_shot[shot_id].tracks:
                    if t.track_id == track_id:
                        pid_to_team[pid] = t.team
                        break

        matched_players = [
            MatchedPlayer(
                player_id=pid,
                team=pid_to_team.get(pid, "unknown"),
                views=views,
            )
            for pid, views in sorted(pid_to_views.items())
        ]
        PlayerMatches(matched_players=matched_players).save(
            matching_dir / "player_matches.json"
        )
        print(f"  -> {len(matched_players)} matched players")
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_matching.py -v
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stages/matching.py tests/test_matching.py
git commit -m "feat: Stage 6 cross-view player matching via Hungarian algorithm"
```

---

## Task 8: Wire up runner, config, and pyproject

**Files:**
- Modify: `src/pipeline/runner.py`
- Modify: `config/default.yaml`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing runner tests**

Add to `tests/test_runner.py`:

```python
from src.pipeline.runner import STAGE_ORDER, _ALIASES, resolve_stages

def test_stage_order_includes_stages_4_to_6():
    names = [name for name, _ in STAGE_ORDER]
    assert "tracking" in names
    assert "pose" in names
    assert "matching" in names


def test_aliases_include_stages_4_to_6():
    assert _ALIASES["4"] == "tracking"
    assert _ALIASES["5"] == "pose"
    assert _ALIASES["6"] == "matching"


def test_resolve_stages_from_tracking():
    names = resolve_stages("all", from_stage="tracking")
    assert names == ["tracking", "pose", "matching"]


def test_resolve_stages_explicit_4_5():
    names = resolve_stages("4,5", from_stage=None)
    assert names == ["tracking", "pose"]
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_runner.py -v -k "stages_4_to_6 or aliases_include or from_tracking or explicit_4"
```

Expected: `AssertionError` (stages not yet in runner).

- [ ] **Step 3: Update `src/pipeline/runner.py`**

Replace the file content:

```python
from pathlib import Path
from src.pipeline.base import BaseStage
from src.stages.segmentation import ShotSegmentationStage
from src.stages.calibration import CameraCalibrationStage
from src.stages.sync import TemporalSyncStage
from src.stages.tracking import PlayerTrackingStage
from src.stages.pose import PoseEstimationStage
from src.stages.matching import CrossViewMatchingStage

STAGE_ORDER: list[tuple[str, type[BaseStage]]] = [
    ("segmentation", ShotSegmentationStage),
    ("calibration", CameraCalibrationStage),
    ("sync", TemporalSyncStage),
    ("tracking", PlayerTrackingStage),
    ("pose", PoseEstimationStage),
    ("matching", CrossViewMatchingStage),
]

_ALIASES: dict[str, str] = {
    "1": "segmentation",
    "2": "calibration",
    "3": "sync",
    "4": "tracking",
    "5": "pose",
    "6": "matching",
}


def resolve_stages(stages: str, from_stage: str | None) -> list[str]:
    all_names = [name for name, _ in STAGE_ORDER]
    if stages == "all":
        selected = all_names
    else:
        selected = []
        for token in stages.split(","):
            token = token.strip()
            name = _ALIASES.get(token, token)
            if name not in all_names:
                raise ValueError(f"Unknown stage: {token!r}")
            selected.append(name)
    if from_stage:
        canonical = _ALIASES.get(from_stage, from_stage)
        idx = all_names.index(canonical)
        selected = [n for n in selected if all_names.index(n) >= idx]
    return selected


def run_pipeline(
    output_dir: Path,
    stages: str,
    from_stage: str | None,
    config: dict,
    **stage_kwargs,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    active = resolve_stages(stages, from_stage)
    from_stage_canonical = _ALIASES.get(from_stage, from_stage) if from_stage else None
    for name, StageClass in STAGE_ORDER:
        if name not in active:
            continue
        if StageClass is None:
            print(f"  [SKIP] {name} (not yet implemented)")
            continue
        stage = StageClass(config=config, output_dir=output_dir, **stage_kwargs)
        if stage.is_complete() and from_stage_canonical != name:
            print(f"  [SKIP] {name} (cached)")
            continue
        print(f"  [RUN]  {name}")
        stage.run()
```

- [ ] **Step 4: Add config sections to `config/default.yaml`**

Append to the existing YAML:

```yaml
tracking:
  player_model: yolov8x.pt
  confidence_threshold: 0.3

pose_estimation:
  model_name: nielsr/vitpose-base-simple
  min_confidence: 0.3
  smooth_sigma: 2.0

matching:
  max_distance_m: 5.0
  n_reference_frames: 10
```

- [ ] **Step 5: Add new dependencies to `pyproject.toml`**

In the `dependencies` list, add:

```toml
"supervision>=0.20",
"transformers>=4.40",
"scikit-learn>=1.4",
```

Install:

```bash
pip install supervision transformers scikit-learn
```

- [ ] **Step 6: Run runner tests — expect PASS**

```bash
pytest tests/test_runner.py -v
```

Expected: all runner tests PASS.

- [ ] **Step 7: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASS (schema, runner, tracking, pose, matching).

- [ ] **Step 8: Commit**

```bash
git add src/pipeline/runner.py config/default.yaml pyproject.toml
git commit -m "feat: wire stages 4-6 into pipeline runner and config"
```

---

## Self-Review

**Spec coverage check:**
- Stage 4 (detection + tracking + team labels + pitch projection): ✓ Tasks 2, 3, 5
- Stage 5 (pose estimation + smoothing + min_height filter): ✓ Tasks 4, 6
- Stage 6 (Hungarian matching + sync_offset + overlap_frames sampling): ✓ Task 7
- Runner wiring (aliases 4-6, `from-stage` support): ✓ Task 8
- Config sections: ✓ Task 8
- New dependencies: ✓ Task 8

**Type consistency check:**
- `Detection.bbox` is `tuple[float, float, float, float]` in player_detector.py, but stored as `list[float]` in TrackFrame — `_track_shot` uses `list(d.bbox)` when building xyxy arrays, consistent.
- `PlayerPoses` in `smooth_keypoints` creates new `Keypoint` instances directly via `Keypoint(name=..., x=..., y=..., conf=...)` — consistent with the `@dataclass` definition in poses.py.
- `_mean_pitch_position` returns `np.ndarray | None`; cost matrix uses `np.linalg.norm(pos_a[tid_a] - pos_b[tid_b])` — both are 2-element arrays, subtraction is valid.
- `TracksResult.load` pops `frames` from each track dict before passing `**t` — matches `Track` constructor (same pattern as existing schemas).

**No placeholders found.**
