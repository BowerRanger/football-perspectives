# Highlights-Reel Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `prepare_shots` ingests a full highlights reel — splits it into shots, drops reaction shots, groups shots per highlight, auto-aligns each group — and the dashboard gets a groups board with discard/restore and a group-scoped sync timeline.

**Architecture:** Split mode orchestrates four new pure modules (`shot_split`, `shot_features`, `highlight_grouping`, `shot_alignment`) and writes the existing manifest + a group-scoped SyncMap v2. The dashboard panel moves to `static/js/prepare_shots_panel.js` and talks to new bulk-edit/sync endpoints. Spec: `docs/superpowers/specs/2026-06-11-highlights-ingestion-design.md`.

**Tech Stack:** PySceneDetect (AdaptiveDetector), OpenCV (LK flow, Sobel, HSV masks), ffmpeg (re-encode + setpts retime), FastAPI, vanilla JS.

**Test command:** `/Users/joebower/workplace/football-perspectives/.venv/bin/python -m pytest <path> -q` (run from the worktree root). Known pre-existing failure to ignore: `tests/test_blender_export_smpl_skeleton.py` (Blender env issue).

---

### Task 1: Shot schema — kind/excluded/group fields + HighlightGroup + active_shots()

**Files:**
- Modify: `src/schemas/shots.py`
- Test: `tests/test_shots_schema_groups.py` (new)

- [x] **Step 1: Write the failing tests**

```python
"""Schema additions for highlights ingestion: Shot flags + groups."""
import json
from pathlib import Path

from src.schemas.shots import HighlightGroup, Shot, ShotsManifest


def _shot(sid: str, **kw) -> Shot:
    base = dict(id=sid, start_frame=0, end_frame=99, start_time=0.0,
                end_time=4.0, clip_file=f"shots/{sid}.mp4")
    base.update(kw)
    return Shot(**base)


def test_shot_new_fields_default():
    s = _shot("s001")
    assert (s.kind, s.excluded, s.exclude_reason, s.group_id) == ("gameplay", False, "", "")
    assert s.source_start_s == -1.0 and s.source_end_s == -1.0


def test_manifest_groups_round_trip(tmp_path: Path):
    m = ShotsManifest(
        source_file="reel.mp4", fps=25.0, total_frames=200,
        shots=[_shot("s001", group_id="g01"),
               _shot("s002", group_id="g01", kind="reaction", excluded=True,
                     exclude_reason="reaction")],
        groups=[HighlightGroup(id="g01", label="Highlight 1",
                               shot_ids=["s001", "s002"],
                               boundary_rule="start", boundary_confidence=1.0)],
    )
    p = tmp_path / "m.json"
    m.save(p)
    loaded = ShotsManifest.load(p)
    assert loaded.groups[0].id == "g01"
    assert loaded.groups[0].shot_ids == ["s001", "s002"]
    assert loaded.shots[1].excluded is True


def test_active_shots_filters_excluded():
    m = ShotsManifest(source_file="x", fps=25.0, total_frames=0,
                      shots=[_shot("a"), _shot("b", excluded=True)])
    assert [s.id for s in m.active_shots()] == ["a"]


def test_legacy_manifest_without_new_fields_loads(tmp_path: Path):
    legacy = {"source_file": "x", "fps": 25.0, "total_frames": 100,
              "shots": [{"id": "a", "start_frame": 0, "end_frame": 99,
                         "start_time": 0.0, "end_time": 4.0,
                         "clip_file": "shots/a.mp4"}]}
    p = tmp_path / "m.json"
    p.write_text(json.dumps(legacy))
    m = ShotsManifest.load(p)
    assert m.groups == [] and m.shots[0].excluded is False
```

- [x] **Step 2: Run, verify fail** — `pytest tests/test_shots_schema_groups.py -q` → ImportError (`HighlightGroup`).

- [x] **Step 3: Implement** — in `src/schemas/shots.py`: add fields to `Shot` (after `speed_factor`): `kind: str = "gameplay"`, `excluded: bool = False`, `exclude_reason: str = ""`, `group_id: str = ""`, `source_start_s: float = -1.0`, `source_end_s: float = -1.0`. Add dataclass:

```python
@dataclass
class HighlightGroup:
    """One highlight event: an ordered run of shots covering the same
    moment (live + replays). ``shot_ids`` is reel order. ``boundary_rule``
    records which grouping rule opened this group ('start', 'transition',
    'gap', 'live_after_replay', 'manual')."""
    id: str
    label: str
    shot_ids: list[str] = field(default_factory=list)
    boundary_rule: str = "start"
    boundary_confidence: float = 1.0
```

Add `groups: list[HighlightGroup] = field(default_factory=list)` to `ShotsManifest` (before `match`), load it in `ShotsManifest.load` via `_filter_kwargs(HighlightGroup, g)` (pop "groups" like "shots"), and add:

```python
    def active_shots(self) -> list[Shot]:
        """Shots that downstream stages should process (not excluded)."""
        return [s for s in self.shots if not s.excluded]
```

- [x] **Step 4: Run** — `pytest tests/test_shots_schema_groups.py tests/test_prepare_shots.py tests/test_match_schema.py -q` → PASS.
- [x] **Step 5: Commit** — `git commit -m "feat(schema): shot kind/excluded/group fields + HighlightGroup + active_shots()"`

---

### Task 2: SyncMap v2 — group-scoped alignments with v1 migration

**Files:**
- Modify: `src/schemas/sync_map.py`
- Test: `tests/test_sync_map.py` (extend)

- [x] **Step 1: Failing tests** (append to `tests/test_sync_map.py`)

```python
from src.schemas.sync_map import Alignment, GroupSync, SyncMap


def test_v2_round_trip(tmp_path):
    sm = SyncMap(groups=[
        GroupSync(group_id="g01", reference_shot="s001",
                  alignments=[Alignment("s001", 0), Alignment("s002", 37, "motion_profile", 0.8)]),
    ])
    p = tmp_path / "sync_map.json"
    sm.save(p)
    loaded = SyncMap.load(p)
    assert loaded.version == 2
    assert loaded.groups[0].alignments[1].frame_offset == 37


def test_v1_flat_file_migrates_to_ungrouped(tmp_path):
    p = tmp_path / "sync_map.json"
    p.write_text(json.dumps({
        "reference_shot": "a",
        "alignments": [{"shot_id": "a", "frame_offset": 0,
                        "method": "manual", "confidence": 1.0},
                       {"shot_id": "b", "frame_offset": -4,
                        "method": "manual", "confidence": 1.0}],
    }))
    sm = SyncMap.load(p)
    assert sm.groups[0].group_id == ""
    assert sm.groups[0].reference_shot == "a"
    assert sm.offset_for("", "b") == -4


def test_group_helpers():
    sm = SyncMap()
    sm2 = sm.with_group_alignment("g01", "ref", Alignment("x", 5))
    assert sm2.offset_for("g01", "x") == 5
    assert sm.groups == []  # immutability preserved


def test_motion_profile_is_valid_method():
    from src.schemas.sync_map import validate_method
    assert validate_method("motion_profile") == "motion_profile"
```

- [x] **Step 2: Run, verify fail** — ImportError (`GroupSync`).

- [x] **Step 3: Implement** — rewrite `SyncMap` keeping `Alignment` as-is; add `"motion_profile"` to `_VALID_METHODS`. New shape:

```python
@dataclass
class GroupSync:
    """Per-group timeline: reference shot + offsets of member shots."""
    group_id: str
    reference_shot: str
    alignments: list[Alignment] = field(default_factory=list)


@dataclass
class SyncMap:
    version: int = 2
    groups: list[GroupSync] = field(default_factory=list)

    def save(self, path): ...  # asdict + indent=2, mkdir parent (as before)

    @classmethod
    def load(cls, path) -> "SyncMap":
        data = json.loads(path.read_text())
        if "groups" not in data:  # v1 flat file
            alignments = [Alignment(**a) for a in data.get("alignments", [])]
            return cls(groups=[GroupSync(group_id="",
                                         reference_shot=data.get("reference_shot", ""),
                                         alignments=alignments)])
        groups = [GroupSync(group_id=g["group_id"],
                            reference_shot=g.get("reference_shot", ""),
                            alignments=[Alignment(**a) for a in g.get("alignments", [])])
                  for g in data["groups"]]
        return cls(version=2, groups=groups)

    def group(self, group_id: str) -> GroupSync | None: ...
    def offset_for(self, group_id: str, shot_id: str) -> int: ...  # 0 default
    def with_group_alignment(self, group_id, reference_shot, alignment) -> "SyncMap":
        ...  # upsert group (create if absent), upsert alignment by shot_id, return new SyncMap
```

`default_sync_map(reference_shot, shot_ids)` becomes `default_group_sync(group_id, reference_shot, shot_ids) -> GroupSync` (update the one caller in `server.py` later — Task 12; keep a thin `default_sync_map` wrapper returning a v2 SyncMap with `group_id=""` so old tests in the file still pass; update those tests where the old flat shape is asserted).

- [x] **Step 4: Run** — `pytest tests/test_sync_map.py -q` → PASS (revise pre-existing flat-shape tests in the same file to the v2 API as part of this step).
- [x] **Step 5: Commit** — `git commit -m "feat(schema): group-scoped SyncMap v2 with v1 migration"`

---

### Task 3: ffmpeg — frame-accurate re-encode extraction with optional retime

**Files:**
- Modify: `src/utils/ffmpeg.py`
- Test: `tests/test_ffmpeg_reencode.py` (new)

- [x] **Step 1: Failing tests**

```python
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.utils.ffmpeg import extract_clip_reencode

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="ffmpeg not on PATH")
FPS = 25.0


@pytest.fixture()
def source_video(tmp_path: Path) -> Path:
    """4 s of 64x64 frames whose blue channel encodes the frame index."""
    p = tmp_path / "src.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 64))
    for i in range(100):
        frame = np.full((64, 64, 3), (min(255, i * 2), 40, 200), np.uint8)
        w.write(frame)
    w.release()
    return p


def _frame_count(p: Path) -> int:
    cap = cv2.VideoCapture(str(p))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def test_reencode_extracts_requested_span(source_video, tmp_path):
    out = tmp_path / "clip.mp4"
    extract_clip_reencode(source_video, out, start_s=1.0, end_s=3.0, fps=FPS)
    assert out.exists()
    assert abs(_frame_count(out) - 50) <= 2  # 2 s at 25 fps, codec tolerance


def test_reencode_retimes_slow_motion(source_video, tmp_path):
    out = tmp_path / "clip.mp4"
    # speed_factor 2.0 == clip is 2x slow-mo -> retimed result halves duration
    extract_clip_reencode(source_video, out, start_s=0.0, end_s=4.0, fps=FPS,
                          speed_factor=2.0)
    assert abs(_frame_count(out) - 50) <= 3
```

- [x] **Step 2: Run, verify fail** — ImportError.

- [x] **Step 3: Implement** in `src/utils/ffmpeg.py`:

```python
def extract_clip_reencode(
    src: Path, out: Path, start_s: float, end_s: float, fps: float,
    speed_factor: float = 1.0, crf: int = 18,
) -> None:
    """Frame-accurate clip extraction (re-encode, unlike ``extract_clip``).

    ``speed_factor`` > 1 means the span is slow-motion; the output is
    retimed to real time (setpts) and resampled to ``fps``. Audio is
    retimed to match (atempo) so the sync editor keeps sound.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = f"setpts=PTS/{speed_factor:.6f}" if speed_factor != 1.0 else "null"
    cmd = ["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
           "-i", str(src), "-vf", vf, "-r", f"{fps:.6f}",
           "-c:v", "libx264", "-crf", str(crf), "-preset", "fast",
           "-pix_fmt", "yuv420p"]
    # atempo only supports 0.5–100; clamp and skip when ~1.0.
    if abs(speed_factor - 1.0) > 1e-6:
        tempo = min(100.0, max(0.5, speed_factor))
        cmd += ["-af", f"atempo={tempo:.6f}"]
    cmd += ["-c:a", "aac", "-b:a", "96k", str(out)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        # Sources without an audio stream make the -af graph fail; retry video-only.
        cmd_noaudio = [c for c in cmd if not c.startswith("atempo")
                       and c not in ("-af", "-c:a", "aac", "-b:a", "96k")]
        cmd_noaudio.insert(-1, "-an")
        subprocess.run(cmd_noaudio, check=True, capture_output=True)
```

- [x] **Step 4: Run** — `pytest tests/test_ffmpeg_reencode.py -q` → PASS.
- [x] **Step 5: Commit** — `git commit -m "feat(ffmpeg): frame-accurate re-encode extraction with slow-mo retime"`

---

### Task 4: shot_split — PySceneDetect wrapper + span hygiene

**Files:**
- Create: `src/utils/shot_split.py`
- Create: `tests/fixtures/synthetic_reel.py` (fixture builder, importable)
- Test: `tests/test_shot_split.py` (new)

- [x] **Step 1: Fixture builder** — `tests/fixtures/synthetic_reel.py`:

```python
"""Builds tiny synthetic 'highlights reels' for prepare_shots tests.

Segments are visually distinct so PySceneDetect finds the cuts:
- 'green'  : pitch-like (textured green + moving white blob = motion)
- 'crowd'  : low-green noise texture (reaction shot stand-in)
- 'black'  : fade/transition stand-in
- 'green_slow': same as green but blob moves at half speed (slow-mo stand-in)
"""
from pathlib import Path

import cv2
import numpy as np

FPS = 25.0
W, H = 192, 108

def _frame(kind: str, t: int, rng: np.ndarray) -> "np.ndarray":
    if kind == "black":
        return np.zeros((H, W, 3), np.uint8)
    if kind == "crowd":
        return (rng * 0.5 + 64).astype(np.uint8)
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:, :] = (40, 140, 60)  # BGR green
    frame += (rng * 0.08).astype(np.uint8)  # mow-stripe-ish texture
    step = 2 if kind == "green" else 1     # green_slow: half-speed motion
    x = (10 + t * step) % (W - 20)
    cv2.circle(frame, (x + 10, H // 2), 6, (255, 255, 255), -1)
    return frame

def build_reel(path: Path, segments: list[tuple[str, float]]) -> dict:
    """Write segments (kind, duration_s) to ``path``; return span info."""
    rng = np.random.RandomState(7).rand(H, W, 3) * 255
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (W, H))
    spans, frame_idx = [], 0
    for kind, dur_s in segments:
        n = int(round(dur_s * FPS))
        spans.append({"kind": kind, "start_frame": frame_idx,
                      "end_frame": frame_idx + n - 1})
        for t in range(n):
            writer.write(_frame(kind, t, rng))
        frame_idx += n
    writer.release()
    return {"fps": FPS, "total_frames": frame_idx, "spans": spans}
```

- [x] **Step 2: Failing tests** — `tests/test_shot_split.py`:

```python
from pathlib import Path

import pytest

from src.utils.shot_split import ShotSpan, detect_spans, merge_short_spans
from tests.fixtures.synthetic_reel import FPS, build_reel


def test_detect_spans_finds_cuts(tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    build_reel(reel, [("green", 3.0), ("crowd", 2.0), ("green", 3.0)])
    spans = detect_spans(reel, detector="content", threshold=27.0,
                         min_scene_len_frames=8)
    assert len(spans) == 3
    assert spans[0].start_frame == 0
    assert abs(spans[1].start_frame - int(3.0 * FPS)) <= 2


def test_min_duration_filter(tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    build_reel(reel, [("green", 3.0), ("black", 0.4), ("crowd", 3.0)])
    spans = detect_spans(reel, detector="content", threshold=27.0,
                         min_scene_len_frames=8, min_shot_duration_s=1.0)
    assert all(s.end_s - s.start_s >= 1.0 for s in spans)


def test_merge_short_spans_glues_false_cuts():
    a = ShotSpan(0, 10, 0.0, 0.44)
    b = ShotSpan(11, 80, 0.44, 3.24)
    merged = merge_short_spans([a, b], max_short_duration_s=1.2, max_gap_s=0.08)
    assert len(merged) == 1 and merged[0].end_frame == 80
```

- [x] **Step 3: Run, verify fail** — ModuleNotFoundError.

- [x] **Step 4: Implement** `src/utils/shot_split.py` — port the deleted-stage logic (`git show 262d08a~1:src/stages/segmentation.py`): frozen `@dataclass ShotSpan(start_frame, end_frame, start_s, end_s)`; `detect_spans(video_path, *, detector="adaptive", threshold=27.0, adaptive_threshold=3.0, min_scene_len_frames=13, adaptive_min_content_val=15.0, min_shot_duration_s=0.0) -> list[ShotSpan]` using `scenedetect.open_video/SceneManager` with `AdaptiveDetector`/`ContentDetector`, then filter by duration; `merge_short_spans(spans, max_short_duration_s, max_gap_s)` (pure port of `_merge_adjacent_short_spans`). If scenedetect returns zero scenes, return one span covering the whole video (probe frame count via cv2).

- [x] **Step 5: Run** — `pytest tests/test_shot_split.py -q` → PASS.
- [x] **Step 6: Commit** — `git commit -m "feat(prepare): PySceneDetect span detection module (resurrected from 262d08a)"`

---

### Task 5: shot_features — pitch/brightness/motion features + classification

**Files:**
- Create: `src/utils/shot_features.py`
- Test: `tests/test_shot_features.py` (new)

- [x] **Step 1: Failing tests**

```python
from pathlib import Path

import numpy as np
import pytest

from src.utils.shot_features import (
    ShotFeatures, classify_kind, classify_scale, compute_span_features,
    estimate_speed_factors, pitch_ratio,
)
from src.utils.shot_split import ShotSpan
from tests.fixtures.synthetic_reel import FPS, build_reel


def test_pitch_ratio_green_vs_crowd():
    green = np.zeros((40, 40, 3), np.uint8); green[:, :] = (40, 140, 60)
    grey = np.full((40, 40, 3), 110, np.uint8)
    assert pitch_ratio(green) > 0.8
    assert pitch_ratio(grey) < 0.1


def test_compute_span_features_classifies_reaction(tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    info = build_reel(reel, [("green", 2.0), ("crowd", 2.0)])
    spans = [ShotSpan(s["start_frame"], s["end_frame"],
                      s["start_frame"] / FPS, (s["end_frame"] + 1) / FPS)
             for s in info["spans"]]
    feats = compute_span_features(reel, spans, sample_points=[0.2, 0.5, 0.8])
    assert classify_kind(feats[0]) == "gameplay"
    assert classify_kind(feats[1]) == "reaction"


def test_fade_classified_as_transition(tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    info = build_reel(reel, [("black", 1.5)])
    s = info["spans"][0]
    feats = compute_span_features(
        reel, [ShotSpan(s["start_frame"], s["end_frame"], 0.0, 1.5)],
        sample_points=[0.2, 0.5, 0.8])
    assert classify_kind(feats[0]) == "transition"


def test_speed_factor_slow_clip_above_one(tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    info = build_reel(reel, [("green", 3.0), ("green_slow", 3.0)])
    spans = [ShotSpan(s["start_frame"], s["end_frame"],
                      s["start_frame"] / FPS, (s["end_frame"] + 1) / FPS)
             for s in info["spans"]]
    feats = compute_span_features(reel, spans, sample_points=[0.2, 0.5, 0.8])
    feats = estimate_speed_factors(feats)
    assert feats[0].speed_factor == pytest.approx(1.0, abs=0.25)
    assert feats[1].speed_factor > 1.4
```

- [x] **Step 2: Run, verify fail.**

- [x] **Step 3: Implement** `src/utils/shot_features.py`:

```python
@dataclass
class ShotFeatures:
    span: ShotSpan
    pitch_ratio_median: float
    pitch_ratio_peak: float
    brightness_min: float
    brightness_range: float
    motion_rate: float          # mean LK flow magnitude / mean Sobel gradient
    speed_factor: float = 1.0   # filled by estimate_speed_factors
```

- `pitch_ratio(frame_bgr)` / `_brightness(frame_bgr)` — port from the deleted stage (HSV inRange (35,40,40)-(95,255,255); gray mean).
- `compute_span_features(video, spans, sample_points)` — one `cv2.VideoCapture`; per span, seek to `start + frac*(dur)` for each sample point; collect pitch ratios + brightness; for motion rate, at up to 3 of the sample points read two consecutive frames and compute LK-flow mean magnitude ÷ mean Sobel RMS gradient (port `_sample_normalized_motion` from `git show 262d08a~1:src/stages/prepare_shots.py`, reshaped to operate on in-memory frame pairs); guard `_MIN_GRADIENT = 0.1`, zero-flow → rate 0.
- `classify_kind(f, *, reaction_max_median_pitch_ratio=0.12, reaction_max_peak_pitch_ratio=0.20, fade_black_frame_threshold=0.18, fade_min_brightness_range=0.25) -> str` — `transition` when `brightness_min <= fade_black AND brightness_range >= fade_range` **or** `brightness_min + brightness_range < 0.06` (hard black); `reaction` when median < reaction_median and peak < reaction_peak; else `gameplay`.
- `classify_scale(f, *, wide_min_pitch_ratio=0.40, tight_max_pitch_ratio=0.22) -> str` — wide/medium/tight on `pitch_ratio_median`.
- `estimate_speed_factors(feats, *, replay_min_speed_factor=1.25) -> list[ShotFeatures]` — reference rate = median `motion_rate` over feats where kind==gameplay and scale==wide (fallback: all gameplay, then all); each `speed_factor = ref_rate / max(rate, eps)` clamped to `[0.3, 4.0]`; rate < eps → 1.0. Returns new list (immutability).
- `is_replay(f)` helper: `f.speed_factor >= replay_min_speed_factor`.

- [x] **Step 4: Run** — `pytest tests/test_shot_features.py -q` → PASS.
- [x] **Step 5: Commit** — `git commit -m "feat(prepare): per-shot feature extraction + kind/scale/speed classification"`

---

### Task 6: highlight_grouping — boundary rules

**Files:**
- Create: `src/utils/highlight_grouping.py`
- Test: `tests/test_highlight_grouping.py` (new)

- [x] **Step 1: Failing tests** — pure-function table tests, no video IO:

```python
from src.utils.highlight_grouping import GroupingInput, group_shots

def _gi(sid, *, kind="gameplay", scale="wide", speed=1.0, start=0.0, end=4.0):
    return GroupingInput(shot_id=sid, kind=kind, scale=scale,
                         speed_factor=speed, source_start_s=start,
                         source_end_s=end)

def test_single_run_is_one_group():
    groups = group_shots([_gi("a"), _gi("b", scale="medium", start=4, end=8)])
    assert len(groups) == 1 and groups[0].shot_ids == ["a", "b"]

def test_transition_between_shots_starts_new_group():
    shots = [_gi("a"), _gi("t", kind="transition", start=4, end=5),
             _gi("b", start=5, end=9)]
    groups = group_shots(shots)
    assert [g.shot_ids for g in groups] == [["a"], ["b"]]
    assert groups[1].boundary_rule == "transition"

def test_large_source_gap_starts_new_group():
    # an excluded reaction span creates a 6 s hole between kept shots
    groups = group_shots([_gi("a", end=4.0), _gi("b", start=10.0, end=14.0)],
                         gap_boundary_s=5.0)
    assert len(groups) == 2 and groups[1].boundary_rule == "gap"

def test_live_wide_after_replay_starts_new_group():
    shots = [_gi("a"), _gi("r", scale="medium", speed=1.8, start=4, end=8),
             _gi("b", start=8, end=12)]
    groups = group_shots(shots)
    assert [g.shot_ids for g in groups] == [["a", "r"], ["b"]]
    assert groups[1].boundary_rule == "live_after_replay"

def test_reaction_shots_never_grouped():
    groups = group_shots([_gi("a"), _gi("x", kind="reaction", start=4, end=6)])
    assert groups[0].shot_ids == ["a"]

def test_reference_prefers_wide_realtime_member():
    shots = [_gi("r", scale="tight", speed=1.8, end=3),
             _gi("a", start=3, end=9)]
    g = group_shots(shots)[0]
    assert g.reference_shot == "a"
```

- [x] **Step 2: Run, verify fail.**

- [x] **Step 3: Implement** `src/utils/highlight_grouping.py`:

```python
@dataclass(frozen=True)
class GroupingInput:
    shot_id: str
    kind: str
    scale: str
    speed_factor: float
    source_start_s: float
    source_end_s: float

@dataclass
class GroupedHighlight:
    id: str
    label: str
    shot_ids: list[str]
    boundary_rule: str
    boundary_confidence: float
    reference_shot: str

_RULE_CONFIDENCE = {"start": 1.0, "transition": 0.9, "gap": 0.6,
                    "live_after_replay": 0.75}
```

`group_shots(shots, *, gap_boundary_s=5.0, replay_min_speed_factor=1.25) -> list[GroupedHighlight]`: iterate in order; non-gameplay shots only mark a pending `transition_seen` flag (transition kind) and are skipped. For each gameplay shot, boundary checks against the previous *kept* shot: R1 `transition_seen` → "transition"; R2 `source_start_s - prev.source_end_s > gap_boundary_s` → "gap"; R3 shot is `scale=="wide" and speed_factor < replay_min_speed_factor` and current group has any member with `speed_factor >= replay_min_speed_factor` → "live_after_replay". First shot opens group with rule "start". Ids `g01…`, labels `Highlight N`. Reference: first member with `scale=="wide" and speed < replay_min`, else longest member by duration.

- [x] **Step 4: Run** — `pytest tests/test_highlight_grouping.py -q` → PASS.
- [x] **Step 5: Commit** — `git commit -m "feat(prepare): rule-based highlight grouping"`

---

### Task 7: shot_alignment — motion-energy curves + NCC offsets

**Files:**
- Create: `src/utils/shot_alignment.py`
- Test: `tests/test_shot_alignment.py` (new)

- [x] **Step 1: Failing tests**

```python
import numpy as np
from pathlib import Path

from src.utils.shot_alignment import (
    AlignmentResult, align_curves, motion_energy_curve,
)
from tests.fixtures.synthetic_reel import build_reel


def _pulse(n, at, width=6):
    x = np.zeros(n)
    x[max(0, at - width):at + width] = np.hanning(2 * width)[:len(x[max(0, at - width):at + width])]
    return x


def test_align_curves_recovers_known_lag():
    ref = _pulse(200, 60) + 0.05
    shifted = _pulse(200, 100) + 0.05   # event 40 frames later in shot
    r = align_curves(ref, shifted, min_overlap=25)
    assert r.frame_offset == 40         # offset = frame_in_shot - frame_in_ref
    assert r.confidence > 0.9


def test_align_curves_flat_signal_low_confidence():
    r = align_curves(np.ones(100), np.ones(120), min_overlap=25)
    assert r.method == "low_confidence"


def test_motion_energy_curve_peaks_at_motion(tmp_path: Path):
    clip = tmp_path / "c.mp4"
    build_reel(clip, [("green", 2.0)])
    curve = motion_energy_curve(clip, width_px=96)
    assert len(curve) >= 45 and float(np.max(curve)) > 0
```

- [x] **Step 2: Run, verify fail.**

- [x] **Step 3: Implement** `src/utils/shot_alignment.py`:

```python
@dataclass(frozen=True)
class AlignmentResult:
    frame_offset: int
    confidence: float
    method: str   # "motion_profile" | "low_confidence"
```

- `motion_energy_curve(clip_path, *, width_px=192, smooth_sigma=2.0) -> np.ndarray` — decode sequentially with cv2, downscale to `width_px` keeping aspect, gray; per-frame `mean(|diff|)`; `scipy.ndimage.gaussian_filter1d` smoothing.
- `align_curves(ref, other, *, min_overlap, min_confidence=0.5) -> AlignmentResult` — z-normalise both (guard σ≈0 → low_confidence with align-ends offset `len(other) - len(ref)`); slide `other` over `ref` for every lag with overlap ≥ `min_overlap`, NCC per lag (`np.dot` of the overlapping z-scored windows ÷ overlap length); best lag `L` = position of `other[0]` on ref axis → event at ref frame `r` appears at shot frame `r - L`, so **`frame_offset = -L`** (sign convention: `frame_offset = frame_in_shot − frame_in_ref`)… verify against the pulse test: ref pulse at 60, shot pulse at 100 → curves match when other is shifted left by 40 ⇒ `L = -40` ⇒ `frame_offset = 40`. Confidence = `max(0.0, best_ncc)`; method per threshold; on low confidence fall back to align-ends offset.
- `align_group(clip_paths: dict[str, Path], reference_id: str, *, width_px, smooth_sigma, min_overlap_frames, min_confidence) -> dict[str, AlignmentResult]` — computes curves once, aligns every non-reference member; reference maps to `AlignmentResult(0, 1.0, "motion_profile")`.

- [x] **Step 4: Run** — `pytest tests/test_shot_alignment.py -q` → PASS.
- [x] **Step 5: Commit** — `git commit -m "feat(prepare): motion-profile NCC alignment module"`

---

### Task 8: prepare_shots split mode — orchestration, artefacts, idempotency

**Files:**
- Modify: `src/stages/prepare_shots.py`
- Modify: `config/default.yaml` (replace `prepare_shots:` block with spec's block)
- Test: `tests/test_prepare_shots_split.py` (new)

- [x] **Step 1: Failing integration tests**

```python
import json
from pathlib import Path

import pytest

from src.stages.prepare_shots import PrepareShotsStage
from src.schemas.shots import ShotsManifest
from src.schemas.sync_map import SyncMap
from tests.fixtures.synthetic_reel import build_reel

CFG = {"prepare_shots": {
    "mode": "split",
    "split": {"detector": "content", "threshold": 27.0,
              "min_scene_len_frames": 8, "min_shot_duration_s": 1.0,
              "min_input_duration_s": 5,
              "merge_max_gap_s": 0.08, "merge_short_shots_max_duration_s": 0.6},
    "classify": {"sample_points": [0.2, 0.5, 0.8],
                 "replay_min_speed_factor": 1.25,
                 "speed_normalise_threshold": 0.15},
    "group": {"gap_boundary_s": 5.0},
    "align": {"enabled": True, "curve_width_px": 96,
              "smooth_sigma_frames": 2.0, "min_overlap_s": 1.0,
              "min_confidence": 0.5},
}}

SEGMENTS = [("green", 3.0), ("crowd", 2.0), ("green_slow", 3.0),
            ("black", 1.2), ("green", 3.0)]
# Expected: shots s001(green live) s002(crowd→reaction,excluded)
# s003(slow replay) s004(black→transition,excluded) s005(green live)
# Groups: g01=[s001,s003] (replay joins live), g02=[s005] (transition boundary)


@pytest.fixture()
def split_run(tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    build_reel(reel, SEGMENTS)
    out = tmp_path / "out"
    stage = PrepareShotsStage(config=CFG, output_dir=out, video_path=reel)
    stage.run()
    return out


def test_split_writes_clips_and_manifest(split_run):
    m = ShotsManifest.load(split_run / "shots" / "shots_manifest.json")
    assert len(m.shots) == 5
    assert all((split_run / s.clip_file).exists() for s in m.shots)
    kinds = [s.kind for s in m.shots]
    assert kinds.count("reaction") == 1 and kinds.count("transition") == 1


def test_reaction_and_transition_excluded(split_run):
    m = ShotsManifest.load(split_run / "shots" / "shots_manifest.json")
    excluded = {s.id: s.exclude_reason for s in m.shots if s.excluded}
    assert sorted(excluded.values()) == ["reaction", "transition"]
    assert len(m.active_shots()) == 3


def test_grouping_and_sync_map(split_run):
    m = ShotsManifest.load(split_run / "shots" / "shots_manifest.json")
    assert len(m.groups) == 2
    assert len(m.groups[0].shot_ids) == 2 and len(m.groups[1].shot_ids) == 1
    sm = SyncMap.load(split_run / "shots" / "sync_map.json")
    g1 = sm.group(m.groups[0].id)
    assert g1 is not None and len(g1.alignments) == 2


def test_slowmo_retimed_to_realtime(split_run):
    m = ShotsManifest.load(split_run / "shots" / "shots_manifest.json")
    slow = [s for s in m.shots if s.speed_factor > 1.25]
    assert len(slow) == 1
    # retimed clip should be roughly 3.0/speed_factor seconds long
    n = slow[0].end_frame + 1
    assert n < 3.0 * 25.0 * 0.85


def test_split_rerun_is_idempotent(split_run, tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    before = (split_run / "shots" / "shots_manifest.json").read_text()
    PrepareShotsStage(config=CFG, output_dir=split_run, video_path=reel).run()
    assert (split_run / "shots" / "shots_manifest.json").read_text() == before


def test_thumbnails_written(split_run):
    assert len(list((split_run / "shots" / "thumbs").glob("*.jpg"))) == 5


def test_features_sidecar_written(split_run):
    data = json.loads((split_run / "shots" / "shot_features.json").read_text())
    assert set(data) == {f"s{i:03d}" for i in range(1, 6)}
    assert "scale" in data["s001"] and "pitch_ratio_median" in data["s001"]


def test_copy_mode_still_works_for_short_clip(tmp_path: Path):
    clip = tmp_path / "myclip.mp4"
    build_reel(clip, [("green", 2.0)])
    out = tmp_path / "out"
    cfg = {"prepare_shots": {"mode": "auto",
                             "split": {"min_input_duration_s": 90}}}
    PrepareShotsStage(config=cfg, output_dir=out, video_path=clip).run()
    m = ShotsManifest.load(out / "shots" / "shots_manifest.json")
    assert [s.id for s in m.shots] == ["myclip"] and m.groups == []
```

- [x] **Step 2: Run, verify fail** (mode/split config ignored today → wrong shot count).

- [x] **Step 3: Implement** in `src/stages/prepare_shots.py`:
- `run()` dispatches: `mode = cfg.get("mode", "auto")`; probe input duration via `_video_metadata`; `split` when mode=="split", or mode=="auto" and single-file input and `duration_s >= split.min_input_duration_s`; else existing copy path (`_run_copy_mode` — current body extracted verbatim).
- `_run_split_mode(clip_src)`:
  1. idempotency: manifest exists and `source_file == str(resolved input)` → log + return;
  2. `detect_spans` → `merge_short_spans` (config values);
  3. `compute_span_features` + `classify_kind`/`classify_scale` + `estimate_speed_factors`;
  4. shot ids `s001…` in span order; extract each via `extract_clip_reencode` (retime when `abs(sf-1) > speed_normalise_threshold`); `extract_thumbnail` at span midpoint to `shots/thumbs/{sid}.jpg`; probe written clip for fps/frames (`_build_shot` pattern) → `Shot(kind=…, excluded=kind in {"reaction","transition"}, exclude_reason=kind-or-"", group_id=…, source_start_s/end_s, speed_factor)`;
  5. `group_shots` over kept gameplay shots (feed `GroupingInput` from features; transitions included so R1 fires) → manifest `groups` (map `GroupedHighlight` → `HighlightGroup`) + back-fill `shot.group_id`;
  6. `align.enabled` → `align_group` per multi-shot group (curves from extracted clips, `min_overlap_frames = min_overlap_s * fps`) → SyncMap v2 written via `with_group_alignment` (load existing first; skip upsert when an existing alignment for that shot has `method == "manual"`);
  7. write manifest + `shot_features.json` sidecar `{shot_id: {pitch_ratio_median, pitch_ratio_peak, motion_rate, speed_factor, scale, kind, ncc_confidence?, boundary_rule?}}`;
  8. per-step `logger.info` progress lines (job log streaming).
- ffmpeg failure on a span: `logger.warning`, skip shot, continue. Zero spans: raise `ValueError` advising `detector: content` / lower threshold.

- [x] **Step 4: Run** — `pytest tests/test_prepare_shots_split.py tests/test_prepare_shots.py -q` → PASS.
- [x] **Step 5: Commit** — `git commit -m "feat(prepare): highlights split mode — detect/classify/group/align/extract"`

---

### Task 9: Downstream stages consume active_shots()

**Files:**
- Modify: `src/stages/tracking.py`, `src/stages/camera.py`, `src/stages/hmr_world.py`, `src/stages/refined_poses.py`, `src/stages/ball.py`, `src/stages/export.py` (each `for shot in manifest.shots` → `manifest.active_shots()`; locate with `grep -n "manifest.shots" src/stages/*.py src/pipeline/*.py`)
- Test: `tests/test_active_shots_downstream.py` (new)

- [x] **Step 1: Failing test** (tracking only — it's the cheapest stage to drive; the rest are the same one-line change, verified by grep):

```python
from pathlib import Path
from src.schemas.shots import Shot, ShotsManifest


def test_tracking_skips_excluded_shots(tmp_path, monkeypatch):
    out = tmp_path / "out"; (out / "shots").mkdir(parents=True)
    shots = [Shot("a", 0, 9, 0.0, 0.4, "shots/a.mp4"),
             Shot("b", 0, 9, 0.0, 0.4, "shots/b.mp4", excluded=True,
                  exclude_reason="reaction")]
    ShotsManifest("r.mp4", 25.0, 20, shots).save(out / "shots" / "shots_manifest.json")
    from src.stages.tracking import PlayerTrackingStage
    stage = PlayerTrackingStage(config={"tracking": {}}, output_dir=out)
    processed = []
    monkeypatch.setattr(stage, "_track_shot",
                        lambda shot, *a, **k: processed.append(shot.id),
                        raising=False)
    # call the manifest-iteration helper directly if run() needs models;
    # adapt to the actual structure found in Step 2 of implementation.
    ids = [s.id for s in stage._shots_to_process()]
    assert ids == ["a"]
```

- [x] **Step 2: Implement** — read each stage's iteration site; where a stage loads the manifest and loops `manifest.shots`, switch to `manifest.active_shots()`. If a shared helper makes more sense (several stages re-implement "load manifest, apply shot_filter"), add `BaseStage._iter_manifest_shots(output_dir)` — decide while editing; keep diffs minimal. Add `_shots_to_process()` on tracking if its loop is inline (used by the test).
- [x] **Step 3: Run** — `pytest tests/test_active_shots_downstream.py tests/test_tracking.py tests/test_camera_stage.py tests/test_ball_stage.py tests/test_export_stage.py tests/test_hmr_world_stage.py tests/test_refined_poses_stage.py -q` → PASS.
- [x] **Step 4: Commit** — `git commit -m "feat(pipeline): stages skip excluded shots via active_shots()"`

---

### Task 10: Quality report — prepare_shots section

**Files:**
- Modify: `src/pipeline/quality_report.py`
- Test: `tests/test_quality_report.py` (extend)

- [x] **Step 1: Failing test** — build the Task 8 fixture layout in tmp (manifest with groups/exclusions + sync_map v2), call `write_quality_report(out)`, assert:

```python
def test_quality_report_prepare_shots_section(tmp_path):
    # reuse helpers/fixtures from test_prepare_shots_split via a tiny manifest
    ...  # manifest: 5 shots, 2 excluded; groups g01 (2 shots), g02 (1)
    report = json.loads((out / "quality_report.json").read_text())
    sec = report["prepare_shots"]
    assert sec["total_shots"] == 5 and sec["excluded"]["reaction"] == 1
    assert sec["groups"][0]["alignment"]["min_confidence"] is not None
```

- [x] **Step 2: Implement** — follow the existing per-section pattern in `quality_report.py` (each section independent, exception-safe): read manifest + sync map, emit `{total_shots, active_shots, excluded: {reason: n}, group_count, groups: [{id, label, shots, alignment: {min_confidence, methods}}], low_confidence_groups: [ids with any confidence < 0.5]}`.
- [x] **Step 3: Run** — `pytest tests/test_quality_report.py -q` → PASS.
- [x] **Step 4: Commit** — `git commit -m "feat(report): prepare_shots ingestion diagnostics section"`

---

### Task 11: Server — reel upload + RunRequest.input_path

**Files:**
- Modify: `src/web/server.py` (`RunRequest`, `_run_job`, new endpoint near `/api/shots/upload`)
- Test: `tests/test_web_api_reel.py` (new; mirror `tests/test_web_api.py` TestClient setup)

- [x] **Step 1: Failing tests**

```python
def test_upload_reel_saves_and_spawns_job(client, tmp_output, monkeypatch):
    calls = {}
    monkeypatch.setattr("src.web.server.run_pipeline",
                        lambda **kw: calls.update(kw))
    video = _tiny_mp4_bytes()   # helper: build 1s synthetic clip via fixture builder
    res = client.post("/api/shots/upload-reel",
                      files={"file": ("My Reel (1).mp4", video, "video/mp4")})
    assert res.status_code == 200
    body = res.json()
    assert body["job_id"]
    saved = tmp_output / "source"
    assert any(p.suffix == ".mp4" for p in saved.iterdir())
    _wait_for_job(client, body["job_id"])      # poll /api/jobs/{id}/status
    assert Path(calls["video_path"]).parent == saved


def test_upload_reel_rejects_non_mp4(client):
    res = client.post("/api/shots/upload-reel",
                      files={"file": ("x.avi", b"xx", "video/avi")})
    assert res.status_code == 400
```

- [x] **Step 2: Implement** — `RunRequest` gains `input_path: str | None = None`. In `_run_job`, pass `video_path=Path(params.input_path)` to `run_pipeline` when set (confirm actual plumbing while editing — `run_pipeline(**stage_kwargs)` already forwards `video_path` to stages). Endpoint:

```python
@app.post("/api/shots/upload-reel")
async def upload_reel(file: UploadFile = File(...)):
    """Save a full highlights reel to output/source/ and spawn a
    prepare_shots job that splits it into shots (split/auto mode)."""
    # validate .mp4 (as upload_shots), sanitise stem, stream to
    # output/source/<stem>.mp4 (mkdir), reject if a source with the same
    # name exists AND the manifest already ingested it; then spawn the job
    # exactly like upload_shots but with
    # RunRequest(stages="prepare_shots", from_stage="prepare_shots",
    #            input_path=str(dest))
```

Validate in the run endpoint(s): when `input_path` is set it must resolve under `output_dir / "source"` (400 otherwise) — the dashboard never sends arbitrary paths.

- [x] **Step 3: Run** — `pytest tests/test_web_api_reel.py -q` → PASS.
- [x] **Step 4: Commit** — `git commit -m "feat(web): highlights reel upload endpoint spawning split-mode prepare_shots"`

---

### Task 12: Server — bulk shot edits, group reconciliation, sync v2, auto-align, features/thumbs, active-aware listings

**Files:**
- Modify: `src/web/server.py`
- Test: `tests/test_web_api_groups.py` (new)

- [x] **Step 1: Failing tests** (TestClient; fixture builds an output dir with the Task 8 synthetic layout — import `build_reel` + run the stage once per module via `pytest.fixture(scope="module")`):

```python
def test_bulk_patch_discard_and_restore(client):
    r = client.patch("/api/shots/bulk", json={"updates": [
        {"shot_id": "s001", "excluded": True, "exclude_reason": "manual"}]})
    assert r.status_code == 200
    m = client.get("/api/shots/manifest").json()
    s = next(x for x in m["shots"] if x["id"] == "s001")
    assert s["excluded"] and s["exclude_reason"] == "manual"

def test_bulk_patch_move_shot_reconciles_groups(client):
    r = client.patch("/api/shots/bulk", json={"updates": [
        {"shot_id": "s003", "group_id": "g02"}]})
    m = client.get("/api/shots/manifest").json()
    by_id = {g["id"]: g["shot_ids"] for g in m["groups"]}
    assert "s003" in by_id["g02"] and "s003" not in by_id.get("g01", [])

def test_bulk_patch_new_group_id_creates_group(client):
    client.patch("/api/shots/bulk", json={"updates": [
        {"shot_id": "s005", "group_id": "g99"}]})
    m = client.get("/api/shots/manifest").json()
    assert any(g["id"] == "g99" for g in m["groups"])

def test_bulk_patch_unknown_shot_400(client):
    assert client.patch("/api/shots/bulk", json={"updates": [
        {"shot_id": "nope", "excluded": True}]}).status_code == 400

def test_sync_get_is_group_scoped(client):
    sync = client.get("/api/sync").json()
    assert "groups" in sync and sync["version"] == 2

def test_sync_post_v2_round_trip(client):
    sync = client.get("/api/sync").json()
    gid = sync["groups"][0]["group_id"]
    payload = {"group_id": gid,
               "reference_shot": sync["groups"][0]["reference_shot"],
               "alignments": [
                   {"shot_id": sid, "frame_offset": i * 3,
                    "method": "manual", "confidence": 1.0}
                   for i, sid in enumerate(
                       a["shot_id"] for a in sync["groups"][0]["alignments"])]}
    assert client.post("/api/sync", json=payload).status_code == 200
    again = client.get("/api/sync").json()
    g = next(g for g in again["groups"] if g["group_id"] == gid)
    assert g["alignments"][1]["frame_offset"] == 3

def test_sync_auto_recomputes_group(client):
    sync = client.get("/api/sync").json()
    gid = next(g["group_id"] for g in sync["groups"]
               if len(g["alignments"]) > 1)
    r = client.post("/api/sync/auto", json={"group_id": gid, "force": True})
    assert r.status_code == 200 and r.json()["aligned"] >= 1

def test_features_endpoint(client):
    feats = client.get("/api/shots/features").json()
    assert "s001" in feats and "scale" in feats["s001"]

def test_thumb_endpoint(client):
    assert client.get("/api/shots/s001/thumb").status_code == 200

def test_output_shots_lists_active_only(client):
    ids = client.get("/api/output/shots").json()["shots"]
    assert "s002" not in ids  # the excluded reaction shot
```

- [x] **Step 2: Implement** in `server.py`:
- `class ShotUpdate(BaseModel): shot_id: str; excluded: bool | None = None; exclude_reason: str | None = None; group_id: str | None = None` + `class BulkShotsPayload(BaseModel): updates: list[ShotUpdate]`.
- `PATCH /api/shots/bulk` under `_match_manifest_lock`: load manifest; validate every shot_id exists and `group_id` matches `^[A-Za-z0-9_-]{1,32}$|^$` (400 before any write); apply updates (replace `Shot` fields); reconcile groups: rebuild each `HighlightGroup.shot_ids` from shots' `group_id` preserving reel order (manifest shot order), drop emptied groups, append new `HighlightGroup(id=<new gid>, label=f"Highlight {n}", boundary_rule="manual", boundary_confidence=1.0)` for unseen ids; prune sync-map alignments whose shot left the group; save manifest (+ sync map if pruned); return `asdict(manifest)`.
- `GET /api/sync` → load v2 (migration in schema), then top-up: every manifest group gets a `GroupSync` (reference = manifest group's first shot if unset) and every member an `Alignment(offset 0)` if missing; ungrouped shots → `group_id ""` GroupSync. Replaces the v1 top-up block.
- `POST /api/sync` → payload `{group_id, reference_shot, alignments[]}` (one group per call); validate ids against the manifest group membership, reference pinned to offset 0, methods via `validate_method`; upsert that `GroupSync` only, save.
- `POST /api/sync/auto` `{group_id, force=False}` → resolve group's active member clips, run `align_group` (config thresholds from `app.state` config load — same pattern as other endpoints reading config), upsert non-manual (or all when force) alignments, return `{"aligned": n, "alignments": [...]}`.
- `GET /api/shots/features` → serve `shots/shot_features.json` or `{}`.
- `GET /api/shots/{shot_id}/thumb` → `FileResponse shots/thumbs/{id}.jpg`; if missing and the clip exists, generate via `extract_thumbnail` (midpoint) then serve; 404 otherwise. Validate shot id `^[A-Za-z0-9_-]+$`.
- `GET /api/output/shots` → when manifest exists, return `[s.id for s in manifest.active_shots()]` (sorted by manifest order); fallback to the existing glob.
- Update the `default_sync_map` import/call site to the new helper (Task 2).

- [x] **Step 3: Run** — `pytest tests/test_web_api_groups.py tests/test_web_api.py tests/test_web_api_output_dirs.py -q` → PASS.
- [x] **Step 4: Commit** — `git commit -m "feat(web): group-aware shot editing, sync v2, auto-align, features/thumb endpoints"`

---

### Task 13: Frontend — prepare_shots panel module (ingest card, groups board, dropped tray)

**Files:**
- Create: `src/web/static/js/prepare_shots_panel.js`
- Modify: `src/web/static/index.html` (add `<script src="/static/js/prepare_shots_panel.js"></script>` before the inline module; delete `renderPrepareShots`/`renderShotSync`/`_buildSyncVideoColumn`/`_buildSyncTimeline`/`_label` bodies from the inline script — they move; keep the `renderPrepareShots(panel)` call signature identical so `selectStage` keeps working)
- Test: `tests/test_web_static_panel.py` (new — asserts `/static/js/prepare_shots_panel.js` serves 200 and `index.html` references it; UI behaviour is manually verified in Task 15)

- [x] **Step 1: Failing test**

```python
def test_panel_script_served_and_referenced(client):
    res = client.get("/static/js/prepare_shots_panel.js")
    assert res.status_code == 200 and "renderPrepareShots" in res.text
    index = client.get("/").text
    assert "/static/js/prepare_shots_panel.js" in index
```

- [x] **Step 2: Implement the module.** Globals consumed (already in `index.html`): `makePanel`, `makeTable`, `addRow`, `makeToolbarBtn`, `emptyNote`, `fetchJsonOrNull`, `attachToJob`, `loadStages`, `selectStage`, `currentStage`, `renderMatchInfoForm`, `renderMultiShotStatus`. Structure (~600 lines, plain script, functions on `window`):

```js
// prepare_shots_panel.js — Prepare Shots dashboard panel.
// Loaded as a plain script before index.html's inline code; exposes
// window.renderPrepareShots(panel). Talks to:
//   GET  /api/shots/manifest   PATCH /api/shots/bulk
//   GET  /api/shots/features   GET  /api/sync   POST /api/sync
//   POST /api/sync/auto        POST /api/shots/upload-reel
//   POST /api/shots/upload     GET  /api/shots/{id}/thumb

async function renderPrepareShots(panel) {
  await renderMatchInfoForm(panel);
  const [manifest, features, sync] = await Promise.all([
    fetchJsonOrNull("/api/shots/manifest"),
    fetchJsonOrNull("/api/shots/features"),
    fetchJsonOrNull("/api/sync"),
  ]);
  _renderIngestCard(panel, manifest);            // reel upload + Add Shots
  if (!manifest || !manifest.shots.length) { emptyNote(panel, "…"); return; }
  const model = _buildModel(manifest, features, sync);  // groups, ungrouped, dropped
  _renderGroupsBoard(panel, model);              // cards + tiles + DnD
  _renderDroppedTray(panel, model);
  await renderMultiShotStatus(panel, model.activeIds);
  _renderGroupSyncEditor(panel, model);          // Task 14
}
```

Key pieces (write fully in this task):
- `_buildModel(manifest, features, sync)` — joins shots ↔ features ↔ group sync; derives per-shot badges (`scale`, `REPLAY ×sf` when `speed_factor ≥ 1.25`, alignment method+confidence), group time ranges from `source_start_s/end_s`, `dropped` list (excluded shots with reason), `ungrouped` pseudo-group (`group_id === ""`).
- `_renderIngestCard` — drop-zone + file picker → `POST /api/shots/upload-reel` (FormData, single file) → `attachToJob(job_id, "prepare_shots", refresh)`; keep the existing multi-clip "Add Shots" button beside it (same handlers as today, moved here).
- `_renderGroupsBoard` — one card per group (`Highlight 1 · N shots · m:ss–m:ss`, boundary-confidence chip when < 0.9, ✕ Discard group, ⟳ Re-align (POST `/api/sync/auto {group_id, force:true}`)); tiles: `<img src="/api/shots/{id}/thumb">` swapping to a muted autoplay `<video>` on mouseenter, id + duration + badges, ✕ discard, ◀/▶ move to adjacent group, ⋮ menu (*make reference*, *move to new group*, *split group here* — split = this tile and all later tiles in the card move to a fresh `gNN`).
- All mutations funnel through `_patchShots(updates)` → `PATCH /api/shots/bulk` → full panel re-render (`selectStage("prepare_shots")`-style refresh via a local `refresh()` that re-runs `renderPrepareShots` on a cleared panel) + toast (`_toast(msg, ok)` — fixed-position bottom-right, matches dark theme).
- HTML5 DnD: tiles `draggable=true` (`dataTransfer.setData("text/shot-id", id)`); group cards + "New group" drop target handle `dragover`/`drop` → `_patchShots([{shot_id, group_id}])`.
- `_renderDroppedTray` — `<details>` ("Dropped shots (N)"), rows: thumb, id, reason badge, Restore button (`excluded:false, exclude_reason:""`).
- Group discard = one bulk PATCH excluding every member (`exclude_reason:"manual"`); the group record disappears when emptied — note this in the tray restore flow (restored shots keep their old `group_id`, recreating the group server-side).

- [x] **Step 3: Wire index.html** — script tag + delete the moved functions (`renderPrepareShots`, `renderShotSync`, `_buildSyncVideoColumn`, `_buildSyncTimeline`, `_label`) from the inline script. Grep to confirm no remaining references: `grep -n "renderShotSync\|_buildSyncTimeline" src/web/static/index.html` → only the script tag's file.
- [x] **Step 4: Run** — `pytest tests/test_web_static_panel.py -q` → PASS; `node --check src/web/static/js/prepare_shots_panel.js` → syntax OK.
- [x] **Step 5: Commit** — `git commit -m "feat(dashboard): highlight groups board with discard/restore + reel ingest"`

---

### Task 14: Frontend — group-scoped sync editor with confidence badges

**Files:**
- Modify: `src/web/static/js/prepare_shots_panel.js`

- [x] **Step 1: Implement `_renderGroupSyncEditor(panel, model)`** — port `renderShotSync` + `_buildSyncVideoColumn` + `_buildSyncTimeline` from the old inline code with these changes:
  - Group tabs across the top (groups with ≥ 2 active shots; ungrouped shown when it has ≥ 2); switching tabs rebuilds the editor for that group's shots only.
  - State seeds from `model.sync.groups[gid]`; reference select restricted to group members; default = group's `reference_shot`.
  - Each timeline row gains a method/confidence chip (`auto ·0.82` amber when < 0.5, `manual` indigo); any offset edit (drag, nudge, lock, input) flips that shot's chip to `manual` locally.
  - Save posts the **v2 per-group payload** `{group_id, reference_shot, alignments}` (method `manual` for operator-touched rows, original method+confidence preserved for untouched rows).
  - "Re-align group" button → `POST /api/sync/auto {group_id, force:true}` → refresh editor.
  - Keep: side-by-side scrub columns, play-both with drift re-sync, draggable blocks, ruler, cursor. The timeline code is moved as-is apart from row chips + group scoping.
- [x] **Step 2: Verify** — `node --check` passes; `pytest tests/test_web_static_panel.py tests/test_web_api_groups.py -q` green; manual check deferred to Task 15.
- [x] **Step 3: Commit** — `git commit -m "feat(dashboard): group-scoped sync timeline with alignment confidence chips"`

---

### Task 15: E2E on the Liverpool reel + threshold tuning

**Files:**
- Possibly tune: `config/default.yaml` (split/classify thresholds)
- Notes: `docs/superpowers/notes/highlights-ingestion-e2e.md` (new)

- [x] **Step 1: Run the stage on the real reel**

```bash
cd /Users/joebower/workplace/football-perspectives/.claude/worktrees/highlights-ingestion
/Users/joebower/workplace/football-perspectives/.venv/bin/python recon.py run \
  --input "test-media/Liverpool vs Barcelona (4-0) _ Epic Comeback Completed At Anfield _ UEFA Champions League Highlights.mp4" \
  --output ./output-highlights/ --stages prepare_shots
```

Expected: tens of shots in `output-highlights/shots/`, several groups, reaction shots excluded. Inspect `shots_manifest.json`, `shot_features.json`, thumbnails.

- [x] **Step 2: Compare grouping against ground truth** — the Origi corner goal (`origi01–04` in `test-media/cleaned_up/`) must land in one group: live wide angle + replays. Record mismatches; tune `classify`/`group` thresholds in `config/default.yaml`; re-run with `--clean`. Iterate ≤ 3 times; log each iteration's shot/group counts in the notes file.
- [x] **Step 3: Dashboard review on port 8001**

```bash
/Users/joebower/workplace/football-perspectives/.venv/bin/python recon.py serve \
  --output ./output-highlights/ --port 8001
```

(Confirm `serve` exposes `--port`; if not, add a click option defaulting to 8000.) Walk through: reel upload of the same file into a fresh output dir, job progress, groups board rendering, discard/restore, move shot between groups, group sync editor (drag, nudge, play-both, save, re-align). Screenshot-by-API not required — verify via curl + browser.
- [x] **Step 4: Commit tuning + notes** — `git commit -m "feat(prepare): threshold tuning from Liverpool-reel e2e + notes"`

---

### Task 16: Docs + finish

**Files:**
- Modify: `CLAUDE.md` (pipeline table row for stage 1; new "highlights ingestion" paragraph; config keys)
- Modify: `docs/football-reconstruction-pipeline-design.md` only if its stage table contradicts the new behaviour (check `grep -n "prepare_shots" docs/football-reconstruction-pipeline-design.md`)

- [x] **Step 1: Update docs** — CLAUDE.md: stage 1 row becomes "trimmed clip *or* full highlights reel → per-shot clips + manifest + groups + sync map"; add commands (`--stages prepare_shots` on a reel; serve `--port`); document `prepare_shots.mode` and the review UX in the dashboard section.
- [x] **Step 2: Full suite** — `pytest -q` → green minus the pre-existing Blender failure; `git commit -m "docs: highlights ingestion stage + dashboard review flow"`.
- [x] **Step 3: Finish** — superpowers:finishing-a-development-branch (present merge/PR options; worktree cleanup decision stays with the user).

---

## Self-review

- **Spec coverage:** splitting (T4), classification/dropping (T5, T8), grouping (T6, T8), alignment (T7, T8), schema (T1–T2), ffmpeg (T3), downstream filtering (T9), quality report (T10), reel upload + jobs (T11), bulk edits/sync v2/auto-align/features/thumbs/active listings (T12), groups-board UX (T13), group sync timeline UX (T14), e2e + tuning on the real reel (T15), docs (T16). Config block lands in T8 alongside the stage that reads it.
- **Placeholders:** Task 9's test acknowledges adapting to the found iteration structure, and T13/T14 describe JS by responsibility with the data contracts pinned — acceptable: the executor reads the moved code in-place; everything else has concrete code/commands.
- **Type consistency:** `ShotSpan(start_frame, end_frame, start_s, end_s)` used in T4/T5/T8; `AlignmentResult.frame_offset` sign matches `sync_map.py`'s convention (T7 pulse test pins it); `HighlightGroup` fields match T1 across T8/T12; bulk-PATCH payload identical in T12 tests and T13 client calls.
