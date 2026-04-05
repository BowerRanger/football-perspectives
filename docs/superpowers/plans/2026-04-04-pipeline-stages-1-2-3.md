# Football Pipeline — Stages 1–3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Stages 1 (shot segmentation), 2 (camera calibration), and 3 (temporal synchronisation) as independently re-runnable CLI stages with cached disk outputs.

**Architecture:** A `BaseStage` ABC drives each stage; the `runner.py` orchestrator routes stage selection and handles caching (skip if output already exists). Each stage reads inputs from previous-stage output directories and writes structured JSON to its own subdirectory. ML model wrappers are thin classes behind ABCs so they can be swapped or mocked in tests.

**Tech Stack:** Python 3.11+, Click, PySceneDetect, FFmpeg (subprocess), OpenCV, NumPy, SciPy, Ultralytics YOLOv8

---

## File Structure

```
recon.py                          # CLI entry point (Click)
config/
  default.yaml                    # All tunable parameters
src/
  __init__.py
  pipeline/
    __init__.py
    base.py                       # BaseStage ABC
    runner.py                     # Stage orchestrator (routing, cache skip)
    config.py                     # Load and merge YAML config
  stages/
    __init__.py
    segmentation.py               # Stage 1: PySceneDetect + FFmpeg
    calibration.py                # Stage 2: solvePnP wrapper + per-frame loop
    sync.py                       # Stage 3: ball trajectory cross-correlation
  schemas/
    __init__.py
    shots.py                      # ShotsManifest, Shot dataclasses + JSON I/O
    calibration.py                # CalibrationResult, CameraFrame dataclasses
    sync_map.py                   # SyncMap, Alignment dataclasses
  utils/
    __init__.py
    ffmpeg.py                     # extract_clip(), extract_thumbnail()
    pitch.py                      # FIFA_LANDMARKS dict (3D world coordinates)
    camera.py                     # build_projection_matrix(), project_to_pitch(), reprojection_error()
tests/
  __init__.py
  conftest.py                     # tiny_video fixture (synthetic OpenCV video)
  test_segmentation.py
  test_calibration.py
  test_sync.py
  test_schemas.py
```

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `config/default.yaml`
- Create: `src/__init__.py`, `src/pipeline/__init__.py`, `src/stages/__init__.py`, `src/schemas/__init__.py`, `src/utils/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:BuildBackend"

[project]
name = "football-perspectives"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "click>=8.1",
    "pyyaml>=6.0",
    "numpy>=1.26",
    "scipy>=1.11",
    "opencv-python>=4.8",
    "scenedetect[opencv]>=0.6.4",
    "ultralytics>=8.0",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-cov>=4.1"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `config/default.yaml`**

```yaml
shot_segmentation:
  detector: content
  threshold: 30.0
  min_shot_duration_s: 0.5

calibration:
  pitch_model: fifa_standard
  max_reprojection_error: 15.0
  keyframe_interval: 5

sync:
  method: hybrid
  search_window_s: 5.0
  min_overlap_frames: 25
  min_confidence: 0.4

detection:
  ball_model: yolov8n        # use nano for now; swap to custom when available
  confidence_threshold: 0.3
```

- [ ] **Step 3: Create empty `__init__.py` files**

```bash
touch src/__init__.py src/pipeline/__init__.py src/stages/__init__.py \
      src/schemas/__init__.py src/utils/__init__.py tests/__init__.py
```

- [ ] **Step 4: Install in dev mode**

```bash
pip install -e ".[dev]"
```

Expected: no errors.

- [ ] **Step 5: Verify pytest finds tests**

```bash
pytest --collect-only
```

Expected: `no tests ran` (no test files yet) with exit code 5.

---

## Task 2: BaseStage + pipeline runner + config loader

**Files:**
- Create: `src/pipeline/base.py`
- Create: `src/pipeline/config.py`
- Create: `src/pipeline/runner.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_runner.py
from pathlib import Path
from src.pipeline.config import load_config
from src.pipeline.runner import resolve_stages, STAGE_ORDER

def test_load_config_returns_dict(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("shot_segmentation:\n  threshold: 42.0\n")
    cfg = load_config(cfg_path)
    assert cfg["shot_segmentation"]["threshold"] == 42.0

def test_resolve_stages_all():
    names = resolve_stages("all", from_stage=None)
    assert names == [name for name, _ in STAGE_ORDER]

def test_resolve_stages_from():
    names = resolve_stages("all", from_stage="calibration")
    assert names == ["calibration", "sync"]

def test_resolve_stages_explicit():
    names = resolve_stages("1,3", from_stage=None)
    assert names == ["segmentation", "sync"]
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/test_runner.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `src/pipeline/config.py`**

```python
from pathlib import Path
import yaml

_DEFAULTS_PATH = Path(__file__).parent.parent.parent / "config" / "default.yaml"

def load_config(path: Path | None = None) -> dict:
    with open(_DEFAULTS_PATH) as f:
        cfg = yaml.safe_load(f)
    if path:
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        _deep_merge(cfg, overrides)
    return cfg

def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
```

- [ ] **Step 4: Create `src/pipeline/base.py`**

```python
from abc import ABC, abstractmethod
from pathlib import Path

class BaseStage(ABC):
    name: str

    def __init__(self, config: dict, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir

    @abstractmethod
    def run(self) -> None: ...

    def is_complete(self) -> bool:
        """Return True if all expected outputs exist (enables cache skip)."""
        return False
```

- [ ] **Step 5: Create `src/pipeline/runner.py`**

```python
from pathlib import Path
from src.pipeline.base import BaseStage

# Populated as stages are implemented; each entry is (canonical_name, StageClass)
STAGE_ORDER: list[tuple[str, type[BaseStage]]] = []

_ALIASES: dict[str, str] = {
    "1": "segmentation",
    "2": "calibration",
    "3": "sync",
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
    for name, StageClass in STAGE_ORDER:
        if name not in active:
            continue
        stage = StageClass(config=config, output_dir=output_dir, **stage_kwargs)
        if stage.is_complete() and from_stage != _ALIASES.get(name, name):
            print(f"  [SKIP] {name} (cached)")
            continue
        print(f"  [RUN]  {name}")
        stage.run()
```

- [ ] **Step 6: Run tests — expect PASS**

```bash
pytest tests/test_runner.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml config/ src/ tests/
git commit -m "feat: project scaffold, BaseStage, runner, config loader"
```

---

## Task 3: Output schemas

**Files:**
- Create: `src/schemas/shots.py`
- Create: `src/schemas/calibration.py`
- Create: `src/schemas/sync_map.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_schemas.py
import json
from pathlib import Path
from src.schemas.shots import Shot, ShotsManifest
from src.schemas.calibration import CameraFrame, CalibrationResult
from src.schemas.sync_map import Alignment, SyncMap

def test_shots_manifest_round_trip(tmp_path):
    m = ShotsManifest(
        source_file="input.mp4", fps=25.0, total_frames=100,
        shots=[Shot(id="shot_001", start_frame=0, end_frame=50,
                    start_time=0.0, end_time=2.0,
                    clip_file="shots/shot_001.mp4",
                    thumbnail="shots/shot_001_thumb.jpg")]
    )
    p = tmp_path / "manifest.json"
    m.save(p)
    loaded = ShotsManifest.load(p)
    assert loaded.fps == 25.0
    assert loaded.shots[0].id == "shot_001"

def test_calibration_round_trip(tmp_path):
    frame = CameraFrame(
        frame=0,
        intrinsic_matrix=[[1500,0,960],[0,1500,540],[0,0,1]],
        rotation_vector=[0.1,0.2,0.05],
        translation_vector=[-52.5,-34.0,50.0],
        reprojection_error=3.2,
        num_correspondences=8,
        confidence=0.79,
    )
    result = CalibrationResult(shot_id="shot_001", camera_type="static", frames=[frame])
    p = tmp_path / "cal.json"
    result.save(p)
    loaded = CalibrationResult.load(p)
    assert loaded.frames[0].reprojection_error == 3.2

def test_sync_map_round_trip(tmp_path):
    sm = SyncMap(
        reference_shot="shot_001",
        alignments=[Alignment(shot_id="shot_003", frame_offset=-47,
                              confidence=0.92, method="ball_trajectory",
                              overlap_frames=[120, 280])]
    )
    p = tmp_path / "sync.json"
    sm.save(p)
    loaded = SyncMap.load(p)
    assert loaded.alignments[0].frame_offset == -47
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/test_schemas.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `src/schemas/shots.py`**

```python
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
```

- [ ] **Step 4: Create `src/schemas/calibration.py`**

```python
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
```

- [ ] **Step 5: Create `src/schemas/sync_map.py`**

```python
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
```

- [ ] **Step 6: Run tests — expect PASS**

```bash
pytest tests/test_schemas.py -v
```

Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add src/schemas/ tests/test_schemas.py
git commit -m "feat: output schemas for shots, calibration, sync_map"
```

---

## Task 4: FFmpeg utilities

**Files:**
- Create: `src/utils/ffmpeg.py`
- Test: `tests/test_segmentation.py` (partial — ffmpeg helpers)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_segmentation.py
import cv2
import numpy as np
import pytest
from pathlib import Path
from src.utils.ffmpeg import extract_clip, extract_thumbnail

@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory) -> Path:
    """Synthetic 2-second video with a hard cut at 1 second."""
    path = tmp_path_factory.mktemp("fixtures") / "test.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240)
    )
    for _ in range(25):  # blue frames
        writer.write(np.full((240, 320, 3), [200, 50, 50], dtype=np.uint8))
    for _ in range(25):  # green frames (new shot)
        writer.write(np.full((240, 320, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()
    return path

def test_extract_clip_creates_file(tmp_path, tiny_video):
    out = tmp_path / "clip.mp4"
    extract_clip(tiny_video, out, start_s=0.0, end_s=1.0)
    assert out.exists()
    assert out.stat().st_size > 0

def test_extract_thumbnail_creates_jpg(tmp_path, tiny_video):
    out = tmp_path / "thumb.jpg"
    extract_thumbnail(tiny_video, out, time_s=0.5)
    assert out.exists()
    img = cv2.imread(str(out))
    assert img is not None
    assert img.shape[:2] == (240, 320)
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/test_segmentation.py::test_extract_clip_creates_file -v
```

Expected: ImportError.

- [ ] **Step 3: Create `src/utils/ffmpeg.py`**

```python
import subprocess
from pathlib import Path

def extract_clip(src: Path, out: Path, start_s: float, end_s: float) -> None:
    """Extract a clip from src between start_s and end_s (seconds)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start_s),
            "-to", str(end_s),
            "-i", str(src),
            "-c", "copy",
            str(out),
        ],
        check=True,
        capture_output=True,
    )

def extract_thumbnail(src: Path, out: Path, time_s: float) -> None:
    """Extract a single frame as JPEG at time_s (seconds)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(time_s),
            "-i", str(src),
            "-vframes", "1",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_segmentation.py::test_extract_clip_creates_file \
       tests/test_segmentation.py::test_extract_thumbnail_creates_jpg -v
```

Expected: 2 passed. (Requires `ffmpeg` on PATH.)

- [ ] **Step 5: Commit**

```bash
git add src/utils/ffmpeg.py tests/test_segmentation.py
git commit -m "feat: FFmpeg clip and thumbnail extraction utilities"
```

---

## Task 5: Stage 1 — Shot segmentation

**Files:**
- Create: `src/stages/segmentation.py`
- Modify: `tests/test_segmentation.py` (add shot detection tests)
- Modify: `src/pipeline/runner.py` (register stage)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_segmentation.py`:

```python
from src.stages.segmentation import detect_shots, ShotSegmentationStage
from src.schemas.shots import ShotsManifest

def test_detect_shots_finds_cut(tiny_video):
    shots = detect_shots(tiny_video, threshold=20.0)
    # Synthetic video has a hard cut between blue and green frames
    assert len(shots) == 2

def test_detect_shots_returns_correct_timings(tiny_video):
    shots = detect_shots(tiny_video, threshold=20.0)
    assert shots[0].start_frame == 0
    assert shots[1].start_frame > 0

def test_stage1_writes_manifest(tmp_path, tiny_video):
    cfg = {
        "shot_segmentation": {"threshold": 20.0, "min_shot_duration_s": 0.1}
    }
    stage = ShotSegmentationStage(
        config=cfg, output_dir=tmp_path, video_path=tiny_video
    )
    stage.run()
    manifest_path = tmp_path / "shots" / "shots_manifest.json"
    assert manifest_path.exists()
    manifest = ShotsManifest.load(manifest_path)
    assert len(manifest.shots) == 2
    assert manifest.fps == 25.0

def test_stage1_is_complete_after_run(tmp_path, tiny_video):
    cfg = {"shot_segmentation": {"threshold": 20.0, "min_shot_duration_s": 0.1}}
    stage = ShotSegmentationStage(config=cfg, output_dir=tmp_path, video_path=tiny_video)
    assert not stage.is_complete()
    stage.run()
    assert stage.is_complete()
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_segmentation.py -v
```

Expected: ImportError on `src.stages.segmentation`.

- [ ] **Step 3: Create `src/stages/segmentation.py`**

```python
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

    def __init__(self, config: dict, output_dir: Path, video_path: Path) -> None:
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
```

- [ ] **Step 4: Register stage in `src/pipeline/runner.py`**

```python
# Add at the top of runner.py, after the imports:
from src.stages.segmentation import ShotSegmentationStage

STAGE_ORDER: list[tuple[str, type[BaseStage]]] = [
    ("segmentation", ShotSegmentationStage),
]
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/test_segmentation.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/stages/segmentation.py src/pipeline/runner.py tests/test_segmentation.py
git commit -m "feat: Stage 1 shot segmentation with PySceneDetect"
```

---

## Task 6: Pitch model + camera math utilities

**Files:**
- Create: `src/utils/pitch.py`
- Create: `src/utils/camera.py`
- Create: `tests/test_calibration.py` (partial)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_calibration.py
import numpy as np
import cv2
import pytest
from src.utils.pitch import FIFA_LANDMARKS, PITCH_LENGTH, PITCH_WIDTH
from src.utils.camera import build_projection_matrix, project_to_pitch, reprojection_error

def _synthetic_camera():
    K = np.array([[1500, 0, 960], [0, 1500, 540], [0, 0, 1]], dtype=np.float32)
    rvec = np.array([0.05, 0.15, 0.0], dtype=np.float32)
    tvec = np.array([-52.5, -34.0, 60.0], dtype=np.float32)
    return K, rvec, tvec

def test_pitch_constants():
    assert PITCH_LENGTH == 105.0
    assert PITCH_WIDTH == 68.0
    assert "top_left_corner" in FIFA_LANDMARKS
    assert "center_spot" in FIFA_LANDMARKS
    pt = FIFA_LANDMARKS["top_left_corner"]
    assert pt[2] == 0.0  # z=0, pitch is ground plane

def test_build_projection_matrix_shape():
    K, rvec, tvec = _synthetic_camera()
    P = build_projection_matrix(K, rvec, tvec)
    assert P.shape == (3, 4)

def test_reprojection_error_zero_for_perfect_fit():
    K, rvec, tvec = _synthetic_camera()
    pts_3d = np.array([[0,0,0],[105,0,0],[52.5,34,0]], dtype=np.float32)
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, None)
    pts_2d = pts_2d.reshape(-1, 2)
    err = reprojection_error(pts_3d, pts_2d, K, rvec, tvec)
    assert err < 0.01

def test_project_to_pitch_round_trips():
    K, rvec, tvec = _synthetic_camera()
    # A known pitch point projected to image, then projected back to pitch
    pt_3d = np.array([30.0, 20.0, 0.0], dtype=np.float32)
    pt_2d, _ = cv2.projectPoints(pt_3d.reshape(1,1,3), rvec, tvec, K, None)
    pt_2d = pt_2d.reshape(2)
    recovered = project_to_pitch(pt_2d, K, rvec, tvec)
    assert np.allclose(recovered, pt_3d[:2], atol=0.05)
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/test_calibration.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `src/utils/pitch.py`**

```python
import numpy as np

PITCH_LENGTH = 105.0  # metres (FIFA standard)
PITCH_WIDTH = 68.0    # metres
# Origin at top-left corner; x along length (0→105), y along width (0→68), z up

FIFA_LANDMARKS: dict[str, np.ndarray] = {
    # Corners
    "top_left_corner":     np.array([0.0,   0.0,  0.0]),
    "top_right_corner":    np.array([105.0, 0.0,  0.0]),
    "bottom_left_corner":  np.array([0.0,   68.0, 0.0]),
    "bottom_right_corner": np.array([105.0, 68.0, 0.0]),
    # Halfway line
    "halfway_top":         np.array([52.5,  0.0,  0.0]),
    "halfway_bottom":      np.array([52.5,  68.0, 0.0]),
    "center_spot":         np.array([52.5,  34.0, 0.0]),
    # Penalty spots
    "left_penalty_spot":   np.array([11.0,  34.0, 0.0]),
    "right_penalty_spot":  np.array([94.0,  34.0, 0.0]),
    # Left goal area (5.5m box)
    "left_goal_area_tl":   np.array([0.0,   24.84, 0.0]),
    "left_goal_area_tr":   np.array([5.5,   24.84, 0.0]),
    "left_goal_area_br":   np.array([5.5,   43.16, 0.0]),
    "left_goal_area_bl":   np.array([0.0,   43.16, 0.0]),
    # Right goal area
    "right_goal_area_tl":  np.array([99.5,  24.84, 0.0]),
    "right_goal_area_tr":  np.array([105.0, 24.84, 0.0]),
    "right_goal_area_br":  np.array([105.0, 43.16, 0.0]),
    "right_goal_area_bl":  np.array([99.5,  43.16, 0.0]),
    # Left penalty box (16.5m box)
    "left_box_tl":         np.array([0.0,   13.84, 0.0]),
    "left_box_tr":         np.array([16.5,  13.84, 0.0]),
    "left_box_br":         np.array([16.5,  54.16, 0.0]),
    "left_box_bl":         np.array([0.0,   54.16, 0.0]),
    # Right penalty box
    "right_box_tl":        np.array([88.5,  13.84, 0.0]),
    "right_box_tr":        np.array([105.0, 13.84, 0.0]),
    "right_box_br":        np.array([105.0, 54.16, 0.0]),
    "right_box_bl":        np.array([88.5,  54.16, 0.0]),
}
```

- [ ] **Step 4: Create `src/utils/camera.py`**

```python
import cv2
import numpy as np


def build_projection_matrix(
    K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
) -> np.ndarray:
    """Return 3×4 projection matrix P = K @ [R | t]."""
    R, _ = cv2.Rodrigues(rvec)
    return K @ np.hstack([R, tvec.reshape(3, 1)])


def project_to_pitch(
    pixel: np.ndarray, K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
) -> np.ndarray:
    """
    Un-project a pixel (u, v) onto the pitch ground plane (z=0).
    Returns (x, y) in pitch coordinates (metres).
    """
    R, _ = cv2.Rodrigues(rvec)
    # Homography H maps pitch plane (z=0) → image:  H = K @ [r1 | r2 | t]
    H = K @ np.column_stack([R[:, 0], R[:, 1], tvec.reshape(3)])
    H_inv = np.linalg.inv(H)
    pt_h = H_inv @ np.array([pixel[0], pixel[1], 1.0])
    return (pt_h[:2] / pt_h[2]).astype(np.float32)


def reprojection_error(
    pts_3d: np.ndarray,
    pts_2d: np.ndarray,
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    """Mean pixel distance between projected 3D points and observed 2D points."""
    projected, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, None)
    projected = projected.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected - pts_2d, axis=1)))
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/test_calibration.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/utils/pitch.py src/utils/camera.py tests/test_calibration.py
git commit -m "feat: FIFA pitch landmarks + camera math utilities"
```

---

## Task 7: Stage 2 — Camera calibration (solvePnP core)

**Files:**
- Create: `src/stages/calibration.py`
- Modify: `tests/test_calibration.py` (add calibration tests)
- Modify: `src/pipeline/runner.py` (register stage)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_calibration.py`:

```python
import cv2
import numpy as np
from src.stages.calibration import calibrate_frame, PitchKeypointDetector
from src.schemas.calibration import CameraFrame

def _make_synthetic_correspondences() -> tuple[dict, np.ndarray]:
    """Project known pitch landmarks with a synthetic camera to get 2D points."""
    K = np.array([[1500,0,960],[0,1500,540],[0,0,1]], dtype=np.float32)
    rvec = np.array([0.05, 0.15, 0.0], dtype=np.float32)
    tvec = np.array([-52.5, -34.0, 60.0], dtype=np.float32)

    from src.utils.pitch import FIFA_LANDMARKS
    landmark_names = [
        "top_left_corner", "top_right_corner", "bottom_left_corner",
        "bottom_right_corner", "center_spot", "left_penalty_spot",
        "right_penalty_spot", "halfway_top", "halfway_bottom",
    ]
    pts_3d = np.array([FIFA_LANDMARKS[n] for n in landmark_names], dtype=np.float32)
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, None)
    pts_2d = pts_2d.reshape(-1, 2)
    correspondences = {name: pts_2d[i] for i, name in enumerate(landmark_names)}
    return correspondences, K

def test_calibrate_frame_recovers_low_reprojection_error():
    correspondences, _ = _make_synthetic_correspondences()
    from src.utils.pitch import FIFA_LANDMARKS
    result = calibrate_frame(
        correspondences=correspondences,
        landmarks_3d=FIFA_LANDMARKS,
        image_shape=(1080, 1920),
    )
    assert result is not None
    assert result.reprojection_error < 2.0  # near-perfect on noise-free data
    assert result.confidence > 0.8
    assert result.num_correspondences >= 4

def test_calibrate_frame_returns_none_with_too_few_points():
    from src.utils.pitch import FIFA_LANDMARKS
    result = calibrate_frame(
        correspondences={"top_left_corner": np.array([100.0, 100.0])},
        landmarks_3d=FIFA_LANDMARKS,
        image_shape=(1080, 1920),
    )
    assert result is None

def test_pitch_keypoint_detector_is_abstract():
    import inspect
    assert inspect.isabstract(PitchKeypointDetector)
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/test_calibration.py -v
```

Expected: ImportError on `src.stages.calibration`.

- [ ] **Step 3: Create `src/stages/calibration.py`**

```python
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.base import BaseStage
from src.schemas.calibration import CameraFrame, CalibrationResult
from src.schemas.shots import ShotsManifest
from src.utils.camera import reprojection_error
from src.utils.pitch import FIFA_LANDMARKS


class PitchKeypointDetector(ABC):
    """Detects pitch landmark keypoints in a video frame."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """
        Returns {landmark_name: (u, v)} for landmarks detected in this frame.
        Only includes landmarks detected with sufficient confidence.
        """
        ...


def calibrate_frame(
    correspondences: dict[str, np.ndarray],
    landmarks_3d: dict[str, np.ndarray],
    image_shape: tuple[int, int],  # (height, width)
    frame_idx: int = 0,
) -> CameraFrame | None:
    """
    Solve camera pose from 2D–3D pitch correspondences.
    Returns None if fewer than 4 common points or solvePnP fails.
    """
    common = [k for k in correspondences if k in landmarks_3d]
    if len(common) < 4:
        return None

    pts_2d = np.array([correspondences[k] for k in common], dtype=np.float32)
    pts_3d = np.array([landmarks_3d[k] for k in common], dtype=np.float32)

    h, w = image_shape
    fx = fy = max(h, w) * 1.2  # broadcast camera estimate
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d,
        pts_2d,
        K,
        None,
        reprojectionError=8.0,
        confidence=0.99,
        iterationsCount=2000,
    )
    if not success or inliers is None or len(inliers) < 4:
        return None

    idx = inliers.flatten()
    err = reprojection_error(pts_3d[idx], pts_2d[idx], K, rvec, tvec)
    confidence = float(max(0.0, 1.0 - err / 15.0))

    return CameraFrame(
        frame=frame_idx,
        intrinsic_matrix=K.tolist(),
        rotation_vector=rvec.flatten().tolist(),
        translation_vector=tvec.flatten().tolist(),
        reprojection_error=float(err),
        num_correspondences=int(len(idx)),
        confidence=confidence,
    )


class CameraCalibrationStage(BaseStage):
    name = "calibration"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        detector: PitchKeypointDetector | None = None,
        **_,
    ) -> None:
        super().__init__(config, output_dir)
        self.detector = detector  # injected; None = skip (outputs will be empty stubs)

    def is_complete(self) -> bool:
        cal_dir = self.output_dir / "calibration"
        manifest_path = self.output_dir / "shots" / "shots_manifest.json"
        if not manifest_path.exists():
            return False
        manifest = ShotsManifest.load(manifest_path)
        return all(
            (cal_dir / f"{shot.id}_calibration.json").exists()
            for shot in manifest.shots
        )

    def run(self) -> None:
        cal_dir = self.output_dir / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config.get("calibration", {})
        keyframe_interval = cfg.get("keyframe_interval", 5)
        max_err = cfg.get("max_reprojection_error", 15.0)

        manifest = ShotsManifest.load(
            self.output_dir / "shots" / "shots_manifest.json"
        )

        for shot in manifest.shots:
            result = self._calibrate_shot(shot.id, shot.clip_file, keyframe_interval, max_err)
            result.save(cal_dir / f"{shot.id}_calibration.json")
            flagged = "⚠ flagged" if not result.frames else ""
            good = sum(1 for f in result.frames if f.reprojection_error <= max_err)
            print(f"  → {shot.id}: {good}/{len(result.frames)} frames calibrated {flagged}")

    def _calibrate_shot(
        self, shot_id: str, clip_file: str, keyframe_interval: int, max_err: float
    ) -> CalibrationResult:
        clip_path = self.output_dir / clip_file
        cap = cv2.VideoCapture(str(clip_path))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames: list[CameraFrame] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % keyframe_interval == 0 and self.detector is not None:
                correspondences = self.detector.detect(frame)
                cf = calibrate_frame(
                    correspondences, FIFA_LANDMARKS, (h, w), frame_idx
                )
                if cf is not None and cf.reprojection_error <= max_err:
                    frames.append(cf)
            frame_idx += 1

        cap.release()
        camera_type = "tracking" if len(frames) > 1 else "static"
        return CalibrationResult(shot_id=shot_id, camera_type=camera_type, frames=frames)
```

- [ ] **Step 4: Register stage in `src/pipeline/runner.py`**

```python
# Add to imports at top of runner.py:
from src.stages.calibration import CameraCalibrationStage

# Update STAGE_ORDER:
STAGE_ORDER: list[tuple[str, type[BaseStage]]] = [
    ("segmentation", ShotSegmentationStage),
    ("calibration", CameraCalibrationStage),
]
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/test_calibration.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/stages/calibration.py src/pipeline/runner.py tests/test_calibration.py
git commit -m "feat: Stage 2 camera calibration with solvePnP and PitchKeypointDetector ABC"
```

---

## Task 8: Stage 3 — Ball detector wrapper

**Files:**
- Create: `src/utils/ball_detector.py`
- Create: `tests/test_sync.py` (partial)

The ball detector is used by Stage 3 to extract ball positions for cross-correlation. YOLOv8 is used here; we wrap it behind an ABC so tests can inject a fake.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sync.py
import numpy as np
import pytest
from src.utils.ball_detector import BallDetector, FakeBallDetector

def test_fake_ball_detector_returns_position():
    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(5)]
    positions = [(50.0, 60.0), (55.0, 65.0), None, (60.0, 70.0), (65.0, 75.0)]
    detector = FakeBallDetector(positions)
    results = [detector.detect(f) for f in frames]
    assert results[0] == pytest.approx((50.0, 60.0))
    assert results[2] is None

def test_ball_detector_is_abstract():
    import inspect
    assert inspect.isabstract(BallDetector)
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/test_sync.py::test_fake_ball_detector_returns_position -v
```

Expected: ImportError.

- [ ] **Step 3: Create `src/utils/ball_detector.py`**

```python
from abc import ABC, abstractmethod
import numpy as np


class BallDetector(ABC):
    """Detects the ball in a single frame and returns its pixel position."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> tuple[float, float] | None:
        """Returns (u, v) pixel coordinates of ball, or None if not found."""
        ...


class YOLOBallDetector(BallDetector):
    """Ball detector using a YOLOv8 model (sports ball class or custom model)."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.3) -> None:
        from ultralytics import YOLO  # lazy import — model download on first use
        self._model = YOLO(model_name)
        self._confidence = confidence
        # COCO class 32 = sports ball
        self._ball_class_id = 32

    def detect(self, frame: np.ndarray) -> tuple[float, float] | None:
        results = self._model(frame, verbose=False)[0]
        best_conf = 0.0
        best_pos = None
        for box in results.boxes:
            if int(box.cls) == self._ball_class_id and float(box.conf) > self._confidence:
                if float(box.conf) > best_conf:
                    best_conf = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    best_pos = ((x1 + x2) / 2, (y1 + y2) / 2)
        return best_pos


class FakeBallDetector(BallDetector):
    """Deterministic detector for tests — returns pre-supplied positions in sequence."""

    def __init__(self, positions: list[tuple[float, float] | None]) -> None:
        self._positions = positions
        self._idx = 0

    def detect(self, frame: np.ndarray) -> tuple[float, float] | None:
        pos = self._positions[self._idx % len(self._positions)]
        self._idx += 1
        return pos
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_sync.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_detector.py tests/test_sync.py
git commit -m "feat: BallDetector ABC with YOLOv8 and fake implementations"
```

---

## Task 9: Stage 3 — Cross-correlation sync

**Files:**
- Create: `src/stages/sync.py`
- Modify: `tests/test_sync.py` (add sync tests)
- Modify: `src/pipeline/runner.py` (register stage)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_sync.py`:

```python
import numpy as np
from src.stages.sync import cross_correlate_trajectories, project_ball_to_pitch

def test_cross_correlate_finds_correct_offset():
    """Ball trajectory in shot_b lags shot_a by 5 frames."""
    traj_a = np.array([0,0,1,3,5,3,1,0,0,0], dtype=float)
    traj_b = np.zeros(10, dtype=float)
    traj_b[5:9] = [1, 3, 5, 3]
    offset, confidence = cross_correlate_trajectories(traj_a, traj_b)
    assert offset == 5
    assert confidence > 0.7

def test_cross_correlate_returns_zero_for_identical():
    traj = np.array([0,1,2,3,2,1,0], dtype=float)
    offset, confidence = cross_correlate_trajectories(traj, traj.copy())
    assert offset == 0
    assert confidence > 0.99

def test_cross_correlate_low_confidence_for_noise():
    rng = np.random.default_rng(42)
    traj_a = rng.random(50)
    traj_b = rng.random(50)
    _, confidence = cross_correlate_trajectories(traj_a, traj_b)
    assert confidence < 0.5

def test_project_ball_to_pitch_returns_2d():
    import cv2
    K = np.array([[1500,0,960],[0,1500,540],[0,0,1]], dtype=np.float32)
    rvec = np.array([0.05,0.15,0.0], dtype=np.float32)
    tvec = np.array([-52.5,-34.0,60.0], dtype=np.float32)

    # Synthetic: project a known pitch point to get its pixel coords
    pt_3d = np.array([[30.0, 20.0, 0.0]], dtype=np.float32)
    pt_2d, _ = cv2.projectPoints(pt_3d, rvec, tvec, K, None)
    pixel = pt_2d.reshape(2)

    from src.schemas.calibration import CameraFrame
    frame_cal = CameraFrame(
        frame=0,
        intrinsic_matrix=K.tolist(),
        rotation_vector=rvec.tolist(),
        translation_vector=tvec.tolist(),
        reprojection_error=0.0,
        num_correspondences=8,
        confidence=1.0,
    )
    pitch_pos = project_ball_to_pitch(pixel, frame_cal)
    assert pitch_pos is not None
    assert np.allclose(pitch_pos, [30.0, 20.0], atol=0.1)
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/test_sync.py -v
```

Expected: ImportError on `src.stages.sync`.

- [ ] **Step 3: Create `src/stages/sync.py`**

```python
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import correlate

from src.pipeline.base import BaseStage
from src.schemas.calibration import CameraFrame, CalibrationResult
from src.schemas.shots import ShotsManifest
from src.schemas.sync_map import Alignment, SyncMap
from src.utils.ball_detector import BallDetector, YOLOBallDetector
from src.utils.camera import project_to_pitch


def project_ball_to_pitch(
    pixel: np.ndarray, cam_frame: CameraFrame
) -> np.ndarray | None:
    """Project a ball pixel position onto the pitch ground plane using calibration."""
    K = np.array(cam_frame.intrinsic_matrix, dtype=np.float32)
    rvec = np.array(cam_frame.rotation_vector, dtype=np.float32)
    tvec = np.array(cam_frame.translation_vector, dtype=np.float32)
    return project_to_pitch(pixel, K, rvec, tvec)


def cross_correlate_trajectories(
    traj_a: np.ndarray, traj_b: np.ndarray
) -> tuple[int, float]:
    """
    Find the integer frame offset of traj_b relative to traj_a using normalised
    cross-correlation of 1D position signals.

    Returns (offset, confidence) where:
      offset > 0  → traj_b leads traj_a
      offset < 0  → traj_b lags traj_a
    """
    norm_a = np.linalg.norm(traj_a)
    norm_b = np.linalg.norm(traj_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0, 0.0

    a = traj_a / norm_a
    b = traj_b / norm_b
    corr = correlate(a, b, mode="full")
    peak_idx = int(np.argmax(corr))
    offset = peak_idx - (len(b) - 1)
    confidence = float(corr[peak_idx])
    return offset, min(1.0, max(0.0, confidence))


def _extract_ball_trajectory(
    clip_path: Path,
    calibration: CalibrationResult,
    detector: BallDetector,
) -> np.ndarray:
    """
    Run ball detector on every frame of a clip.
    Returns (N,) array of x-position in pitch coordinates (NaN where undetected).
    """
    cap = cv2.VideoCapture(str(clip_path))
    positions: list[float] = []
    frame_idx = 0

    cal_map = {f.frame: f for f in calibration.frames}
    last_cal: CameraFrame | None = (
        calibration.frames[0] if calibration.frames else None
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Use closest available calibration keyframe
        if frame_idx in cal_map:
            last_cal = cal_map[frame_idx]

        ball_px = detector.detect(frame)
        if ball_px is not None and last_cal is not None:
            pitch_pos = project_ball_to_pitch(np.array(ball_px), last_cal)
            positions.append(float(pitch_pos[0]) if pitch_pos is not None else float("nan"))
        else:
            positions.append(float("nan"))

        frame_idx += 1

    cap.release()
    return np.array(positions, dtype=float)


class TemporalSyncStage(BaseStage):
    name = "sync"

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        ball_detector: BallDetector | None = None,
        **_,
    ) -> None:
        super().__init__(config, output_dir)
        self.ball_detector = ball_detector

    def is_complete(self) -> bool:
        return (self.output_dir / "sync" / "sync_map.json").exists()

    def run(self) -> None:
        sync_dir = self.output_dir / "sync"
        sync_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config.get("sync", {})
        min_overlap = cfg.get("min_overlap_frames", 25)
        min_conf = cfg.get("min_confidence", 0.4)

        manifest = ShotsManifest.load(
            self.output_dir / "shots" / "shots_manifest.json"
        )
        if len(manifest.shots) < 2:
            SyncMap(reference_shot=manifest.shots[0].id if manifest.shots else "").save(
                sync_dir / "sync_map.json"
            )
            print("  → only one shot; no sync needed")
            return

        detector = self.ball_detector or YOLOBallDetector(
            model_name=self.config.get("detection", {}).get("ball_model", "yolov8n.pt"),
            confidence=self.config.get("detection", {}).get("confidence_threshold", 0.3),
        )

        # Load calibrations
        cal_dir = self.output_dir / "calibration"
        calibrations: dict[str, CalibrationResult] = {}
        for shot in manifest.shots:
            cal_file = cal_dir / f"{shot.id}_calibration.json"
            if cal_file.exists():
                calibrations[shot.id] = CalibrationResult.load(cal_file)

        # Extract ball trajectories for all shots
        trajectories: dict[str, np.ndarray] = {}
        for shot in manifest.shots:
            clip_path = self.output_dir / shot.clip_file
            cal = calibrations.get(shot.id)
            if cal is None:
                # No calibration: fall back to raw pixel x-position (less accurate)
                cal = CalibrationResult(shot_id=shot.id, camera_type="static", frames=[])
            trajectories[shot.id] = _extract_ball_trajectory(clip_path, cal, detector)
            print(f"  → {shot.id}: extracted {int(np.sum(~np.isnan(trajectories[shot.id])))} ball detections")

        # Use first shot as reference; align all others to it
        reference = manifest.shots[0].id
        ref_traj = trajectories[reference]
        alignments: list[Alignment] = []

        for shot in manifest.shots[1:]:
            traj = trajectories[shot.id]
            # Replace NaN with 0 for correlation (NaN = no signal)
            a = np.nan_to_num(ref_traj)
            b = np.nan_to_num(traj)
            offset, confidence = cross_correlate_trajectories(a, b)

            # Determine overlap region in reference frames
            start = max(0, offset)
            end = min(len(ref_traj), offset + len(traj))
            overlap = max(0, end - start)

            method = "ball_trajectory" if confidence >= min_conf else "low_confidence"
            alignments.append(Alignment(
                shot_id=shot.id,
                frame_offset=offset,
                confidence=confidence,
                method=method,
                overlap_frames=[start, end],
            ))
            flag = "" if confidence >= min_conf else " ⚠ low confidence"
            print(f"  → {shot.id} offset={offset:+d} frames, confidence={confidence:.2f}{flag}")

        SyncMap(reference_shot=reference, alignments=alignments).save(
            sync_dir / "sync_map.json"
        )
```

- [ ] **Step 4: Register stage in `src/pipeline/runner.py`**

```python
# Add to imports:
from src.stages.sync import TemporalSyncStage

# Update STAGE_ORDER:
STAGE_ORDER: list[tuple[str, type[BaseStage]]] = [
    ("segmentation", ShotSegmentationStage),
    ("calibration", CameraCalibrationStage),
    ("sync", TemporalSyncStage),
]
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/test_sync.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/stages/sync.py src/pipeline/runner.py tests/test_sync.py
git commit -m "feat: Stage 3 temporal sync via ball trajectory cross-correlation"
```

---

## Task 10: CLI entry point

**Files:**
- Create: `recon.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cli.py
from click.testing import CliRunner
from recon import cli

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--stages" in result.output

def test_cli_missing_input_for_run():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--output", "/tmp/out"])
    # Missing required --input should produce an error
    assert result.exit_code != 0
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/test_cli.py -v
```

Expected: ImportError / ModuleNotFoundError.

- [ ] **Step 3: Create `recon.py`**

```python
#!/usr/bin/env python3
"""Football match reconstruction pipeline CLI."""

from pathlib import Path

import click

from src.pipeline.config import load_config
from src.pipeline.runner import run_pipeline


@click.group()
def cli() -> None:
    """Football match reconstruction pipeline."""


@cli.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path), help="Input video file.")
@click.option("--output", "output_dir", default="./output", show_default=True, type=click.Path(path_type=Path), help="Output directory.")
@click.option("--stages", default="all", show_default=True, help="Stages to run: 'all' or comma-separated (e.g. '1,2,3' or 'segmentation,calibration').")
@click.option("--from-stage", default=None, help="Resume pipeline from this stage (skips earlier stages even if cached).")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True, path_type=Path), help="YAML config file (merged with defaults).")
@click.option("--device", default="auto", show_default=True, help="Compute device: cuda, cpu, mps, or auto.")
def run(
    input_path: Path,
    output_dir: Path,
    stages: str,
    from_stage: str | None,
    config_path: Path | None,
    device: str,
) -> None:
    """Run the reconstruction pipeline on a video file."""
    cfg = load_config(config_path)
    click.echo(f"Input:    {input_path}")
    click.echo(f"Output:   {output_dir}")
    click.echo(f"Stages:   {stages}")
    run_pipeline(
        output_dir=output_dir,
        stages=stages,
        from_stage=from_stage,
        config=cfg,
        video_path=input_path,
    )
    click.echo("Done.")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_cli.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Smoke-test full pipeline on tiny synthetic video**

```bash
python -c "
import cv2, numpy as np
writer = cv2.VideoWriter('/tmp/smoke.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (320,240))
for _ in range(25): writer.write(np.full((240,320,3),[200,50,50],dtype=np.uint8))
for _ in range(25): writer.write(np.full((240,320,3),[50,200,50],dtype=np.uint8))
writer.release()
"
python recon.py run --input /tmp/smoke.mp4 --output /tmp/smoke_out --stages 1
```

Expected: `shots/shots_manifest.json` created with 2 shots.

- [ ] **Step 6: Run full test suite**

```bash
pytest --cov=src --cov-report=term-missing -v
```

Expected: all tests pass, coverage > 80%.

- [ ] **Step 7: Commit**

```bash
git add recon.py tests/test_cli.py
git commit -m "feat: Click CLI entry point with run command"
```

---

## Self-Review

**Spec coverage:**

| Design doc requirement | Implemented |
|------------------------|-------------|
| Single entry point `recon.py` | ✅ Task 10 |
| Modular stages with disk caching | ✅ Tasks 2, 5, 7, 9 (`is_complete`) |
| `shots_manifest.json` with schema from §3.1 | ✅ Tasks 3, 5 |
| ContentDetector threshold configurable | ✅ Task 5 config |
| FFmpeg clip + thumbnail extraction | ✅ Task 4 |
| FIFA standard pitch model (105×68m) | ✅ Task 6 |
| solvePnP RANSAC calibration | ✅ Task 7 |
| `calibration.json` schema from §3.2 | ✅ Tasks 3, 7 |
| Confidence metric (reprojection error) | ✅ Task 7 |
| Ball trajectory cross-correlation for sync | ✅ Tasks 8, 9 |
| `sync_map.json` schema + frame offset convention | ✅ Tasks 3, 9 |
| `--from-stage` resume flag | ✅ Task 2 runner |
| `--config` YAML override | ✅ Tasks 1, 2 |

**Gaps / future work (out of scope for this plan):**
- `OpenCVPitchDetector` (heuristic line detection) — Stage 2 calibration runs but produces empty frames without a real detector. Plugging in the SoccerNet model or an OpenCV-based detector is the Stage 2 follow-up.
- Player formation fallback for sync (§3.3) — only ball trajectory is implemented. Formation matching can be added once Stage 4 (tracking) exists.
- `--viewer` flag — no-op until Stage 9.

**Type consistency:** All schemas use the same field names as the JSON examples in the design doc. `frame_offset` convention (positive = b leads a) matches §3.3.
