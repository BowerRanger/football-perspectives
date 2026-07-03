# Ball Phase 4 — WASB Detector Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute spec §4.3 — fine-tune the WASB ball detector on this project's own footage (145 gold labels across 4 clips + solved-track weak labels), evaluate against the recorded baselines, and on success swap the checkpoint and re-measure the two detector-gated features (`ball.foot_guided`, `ball.touch_attribution`).

**Architecture:** The vendored WASB trainer is unusable here (CUDA-locked asserts, `Trainer` commented out of the runner factory, hard-coded `nn.DataParallel`), so training lives repo-side — the established pattern (`wasb_ball_detector.py` already rebuilds the model config as a dict and loads HRNet device-agnostically). We import only *pure* vendored pieces and never edit vendored source. Three new units: a **weak-label densifier** (solved-track world positions projected back to pixels inside ±window of gold anchors — physics-corrected labels exactly where the detector failed), a **corpus builder** CLI (frames + merged gold∪weak CVAT XML per clip, kroupi01 held out), and a **fine-tune harness** (torch Dataset over 3-consecutive-labelled-frame runs, preprocessing byte-identical to inference, TrackNetV2 weighted-BCE loss, MPS/CPU device fallback). Then two operational tasks: run the training, and evaluate/ship behind the recorded acceptance bars.

**Tech Stack:** Python 3.11, pytest, torch (MPS), OpenCV, vendored WASB HRNet + heatmap utilities (import-only).

## Scope notes — conscious v1 simplifications (do not "fix" these)

- **The spec's "author `train.yaml` for the vendored WASB" step is deliberately replaced** by the repo-side harness: the vendored `Trainer` is commented out of the runner factory, asserts `device=='cuda'` + GPU availability (`third_party/wasb_sbdt/src/runners/train_and_test.py:51-55`), and hard-codes `nn.DataParallel` — unusable on this Mac without editing vendored source, which the project's integration rules forbid. The harness imports only pure vendored pieces so training targets/preprocessing match the original.

- **Train/eval overlap is accepted and disclosed**: this is a per-project specialist detector; in-domain gains are the product. kroupi01 (12 gold labels) is fully held out as the honest generalization check, and gberch's recall eval is against touch *moments* the detector never sees as such (it trains on ball pixels, not touch labels).
- **No SoccerNet mixing in v1** (spec lists it as optional): the corpus builder is designed so a public-data importer can be added later; regularization against forgetting comes from a low LR + few epochs + init from `wasb_soccer_best`.
- **Weak labels come from the solved track only within ±window of gold anchors** — the spans where the solve is operator-anchored and trustworthy. No detector-echo self-distillation.
- **Training is time-boxed** (default 30 epochs, ~small model, ~500–1500 samples on MPS ≈ 30–90 min). If MPS is unstable, the harness falls back to CPU with the same semantics; if that is impractically slow the run task reports the timing honestly instead of shipping a half-trained checkpoint.

## Global Constraints

- **Never edit vendored source** under `third_party/wasb_sbdt/` — import pure pieces only (`HRNet` model, `gen_binary_map`/`gen_heatmap`, optionally `BCELoss`). Mirror `src/utils/wasb_ball_detector.py`'s import mechanics exactly.
- **Train/infer preprocessing parity is load-bearing.** Inference (`wasb_ball_detector.py:_preprocess_buffer`, :~240-260) does: `cv2.cvtColor(BGR→RGB)` → letterbox warp via `_get_affine_transform(center=(w/2,h/2), scale=float(max(h,w)), (inp_w,inp_h))` + `cv2.warpAffine(..., INTER_LINEAR)` → `float32/255` → `(x − _IMAGENET_MEAN)/_IMAGENET_STD` → `transpose(2,0,1)` → concat 3 frames → `(9, inp_h, inp_w)`; model outputs `(1, 3, inp_h, inp_w)` logits, sigmoid at inference. Training must replicate this pipeline exactly (reuse the module's `_get_affine_transform`, `_IMAGENET_MEAN`, `_IMAGENET_STD` by import, input size `[512, 288]` from `ball.wasb.input_size`).
- Labels map into model space through the same forward affine; targets are `gen_binary_map`-style heatmaps at output resolution (= input resolution), sigma 2.5 (`configs/dataloader/default.yaml`).
- Checkpoint format: `{"model_state_dict": ...}`; the pipeline loader strips `module.` prefixes and accepts bare dicts (`wasb_ball_detector.py` load code) — save in the dict form.
- Corpus and checkpoints are **untracked data** (`output/…`, `third_party/wasb_sbdt/pretrained_weights/…` weights are gitignored); code + tests + docs are committed.
- Baselines to beat (all recorded 2026-07-02/03 with shipped Phase-3 defaults): gberch union recall 0.250 (2/8), break_only fp 11; coverage totals origi01 0.3636, origi02 0.5120; kroupi/japan coverage to be captured pre-swap in Task 5.
- Acceptance for the checkpoint swap (spec §4.3): gberch union recall ≥ 4/8; origi02 `detection_coverage.total` ≥ 0.562 (+0.05); origi01/kroupi01/s013 coverage each within −0.02 of pre-swap; anchor-accuracy harness green. Precision ≥ 0.5 is reported but adjudicated with the known GT-non-exhaustiveness caveat (spec §5.2).
- Commit format `<type>: <description>`, no attribution trailers. Tests via `.venv/bin/python -m pytest` from the repo root. Paths relative to `/Users/joebower/workplace/football-perspectives`.

---

### Task 1: Weak-label densifier (`src/utils/ball_weak_labels.py`)

**Files:**
- Create: `src/utils/ball_weak_labels.py`
- Test: `tests/test_ball_weak_labels.py`

**Interfaces:**
- Consumes: `BallTrack`/`BallFrame` (`src/schemas/ball_track.py`: frame, world_xyz|None, state ∈ grounded/flight/occluded/missing, confidence); `project_world_to_image(K, R, t, distortion, world_points) -> (N,2)` from `src/utils/camera_projection.py`; `anchors_to_cvat_xml` XML shape (`src/utils/ball_finetune_export.py` — `<track><points frame=… outside="0" occluded="0" points="x,y"><attribute name="used_in_game">1`).
- Produces (Tasks 2–3 rely on exactly these):
  - `weak_labels_from_track(track: "BallTrack", *, per_frame_K: dict[int, np.ndarray], per_frame_R: dict[int, np.ndarray], per_frame_t: dict[int, np.ndarray], distortion: tuple[float, float], image_size: tuple[int, int], gold_frames: set[int], window: int = 20, min_conf: float = 0.5, edge_margin_px: float = 4.0) -> dict[int, tuple[float, float]]` — solved-track frames that are within ±`window` of some gold frame, have `world_xyz` and `state in ("grounded", "flight")` and `confidence >= min_conf`, are NOT gold frames themselves, and project inside the image with `edge_margin_px` to spare.
  - `merge_labels(gold: Mapping[int, tuple[float, float]], weak: Mapping[int, tuple[float, float]]) -> dict[int, tuple[float, float]]` — union, gold wins on collision.
  - `labels_to_cvat_xml(clip_id: str, labels: Mapping[int, tuple[float, float]]) -> str` — same XML dialect as `anchors_to_cvat_xml` (every label visible, `used_in_game=1`), frames ascending.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ball_weak_labels.py`:

```python
"""Weak-label densification from the solved ball track."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.schemas.ball_track import BallFrame, BallTrack
from src.utils.ball_weak_labels import (
    labels_to_cvat_xml,
    merge_labels,
    weak_labels_from_track,
)

IMG = (1280, 720)


def _camera():
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _track(frames: list[BallFrame]) -> BallTrack:
    return BallTrack(clip_id="play", fps=30.0, frames=tuple(frames),
                     flight_segments=())


def _bf(frame: int, world, state="grounded", conf=0.9) -> BallFrame:
    return BallFrame(frame=frame, world_xyz=world, state=state,
                     confidence=conf)


def _mats(n: int):
    K, R, t = _camera()
    return ({i: K for i in range(n)}, {i: R for i in range(n)},
            {i: t for i in range(n)})


def test_window_and_gold_exclusion():
    Ks, Rs, ts = _mats(100)
    frames = [_bf(i, (30.0 + 0.2 * i, 34.0, 0.11)) for i in range(100)]
    out = weak_labels_from_track(
        _track(frames), per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0), image_size=IMG,
        gold_frames={50}, window=5,
    )
    assert set(out) == {45, 46, 47, 48, 49, 51, 52, 53, 54, 55}


def test_state_conf_and_missing_world_gates():
    Ks, Rs, ts = _mats(10)
    frames = [
        _bf(0, (40.0, 34.0, 0.11)),                       # ok
        _bf(1, (40.2, 34.0, 0.11), conf=0.2),             # low conf
        _bf(2, None, state="missing"),                    # no world
        _bf(3, (40.6, 34.0, 0.11), state="occluded"),     # bad state
        _bf(4, (40.8, 34.0, 2.0), state="flight"),        # ok (flight)
    ]
    out = weak_labels_from_track(
        _track(frames), per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0), image_size=IMG,
        gold_frames={2}, window=10,
    )
    assert set(out) == {0, 4}


def test_off_image_projection_rejected():
    Ks, Rs, ts = _mats(4)
    frames = [
        _bf(0, (52.5, 34.0, 0.11)),      # centre-ish, in image
        _bf(1, (52.5, 300.0, 0.11)),     # projects far outside
    ]
    out = weak_labels_from_track(
        _track(frames), per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0), image_size=IMG,
        gold_frames={0, 1}, window=3,
    )
    # both frames are gold -> excluded anyway; use neighbours instead
    frames2 = [
        _bf(0, (52.5, 34.0, 0.11)),
        _bf(1, (52.5, 300.0, 0.11)),
        _bf(2, (52.5, 34.5, 0.11)),
    ]
    out = weak_labels_from_track(
        _track(frames2), per_frame_K=Ks, per_frame_R=Rs, per_frame_t=ts,
        distortion=(0.0, 0.0), image_size=IMG,
        gold_frames={2}, window=3,
    )
    assert 0 in out and 1 not in out


def test_merge_gold_wins():
    merged = merge_labels({5: (1.0, 2.0)}, {5: (9.0, 9.0), 6: (3.0, 4.0)})
    assert merged == {5: (1.0, 2.0), 6: (3.0, 4.0)}


def test_xml_parses_and_matches_exporter_dialect():
    xml = labels_to_cvat_xml("play", {7: (100.5, 200.25), 3: (1.0, 2.0)})
    root = ET.fromstring(xml)
    track = root.find("track")
    assert track is not None and track.attrib["label"] == "ball"
    pts = track.findall("points")
    assert [int(p.attrib["frame"]) for p in pts] == [3, 7]  # ascending
    for p in pts:
        assert p.attrib["outside"] == "0"
        assert p.attrib["occluded"] == "0"
        attr = p.find("attribute")
        assert attr is not None and attr.attrib["name"] == "used_in_game"
        assert attr.text == "1"
    assert pts[1].attrib["points"] == "100.50,200.25"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_weak_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.ball_weak_labels'`

- [ ] **Step 3: Implement the module**

Create `src/utils/ball_weak_labels.py`:

```python
"""Weak training labels from the solved ball track (spec §4.3 step 1).

The 145 gold labels (operator-clicked pixels) are too thin to fine-tune on
alone; WASB also trains best on labelled RUNS (3-frame stacks want all three
frames labelled). Near a manual anchor the piecewise/events solve is
operator-anchored and physically constrained, so its per-frame world
positions are trustworthy exactly where the raw detector failed — the
hard examples. This module projects those positions back to pixels inside
±window of each gold frame. Pure and torch-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping
from xml.sax.saxutils import escape

import numpy as np

from src.utils.camera_projection import project_world_to_image

if TYPE_CHECKING:  # pragma: no cover — typing only
    from src.schemas.ball_track import BallTrack

_WEAK_STATES = frozenset({"grounded", "flight"})


def weak_labels_from_track(
    track: "BallTrack",
    *,
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    image_size: tuple[int, int],
    gold_frames: set[int],
    window: int = 20,
    min_conf: float = 0.5,
    edge_margin_px: float = 4.0,
) -> dict[int, tuple[float, float]]:
    """Solved-track pixels usable as weak labels around gold anchors."""
    w, h = float(image_size[0]), float(image_size[1])
    out: dict[int, tuple[float, float]] = {}
    for bf in track.frames:
        f = bf.frame
        if f in gold_frames:
            continue
        if not any(abs(f - g) <= window for g in gold_frames):
            continue
        if bf.world_xyz is None or bf.state not in _WEAK_STATES:
            continue
        if bf.confidence < min_conf:
            continue
        K, R, t = per_frame_K.get(f), per_frame_R.get(f), per_frame_t.get(f)
        if K is None or R is None or t is None:
            continue
        uv = project_world_to_image(
            K, R, t, distortion,
            np.asarray([bf.world_xyz], dtype=float),
        )[0]
        u, v = float(uv[0]), float(uv[1])
        if not (edge_margin_px <= u <= w - edge_margin_px
                and edge_margin_px <= v <= h - edge_margin_px):
            continue
        out[f] = (u, v)
    return out


def merge_labels(
    gold: Mapping[int, tuple[float, float]],
    weak: Mapping[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    """Union of label maps; gold wins on frame collision."""
    merged: dict[int, tuple[float, float]] = dict(weak)
    merged.update(gold)
    return merged


def labels_to_cvat_xml(
    clip_id: str, labels: Mapping[int, tuple[float, float]],
) -> str:
    """Render a label map in the same CVAT dialect as anchors_to_cvat_xml
    (validated against the vendored WASB soccer loader)."""
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<annotations>",
        f'  <track id="0" label="ball" source="{escape(str(clip_id))}">',
    ]
    for frame in sorted(labels):
        u, v = labels[frame]
        lines.append(
            f'    <points frame="{int(frame)}" outside="0" occluded="0" '
            f'points="{u:.2f},{v:.2f}">'
        )
        lines.append('      <attribute name="used_in_game">1</attribute>')
        lines.append("    </points>")
    lines.append("  </track>")
    lines.append("</annotations>")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_weak_labels.py tests/test_ball_finetune_export.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_weak_labels.py tests/test_ball_weak_labels.py
git commit -m "feat: solved-track weak labels for detector fine-tune corpus"
```

---

### Task 2: Corpus builder (`scripts/build_finetune_corpus.py`)

**Files:**
- Create: `scripts/build_finetune_corpus.py`
- Test: `tests/test_build_finetune_corpus.py`

**Interfaces:**
- Consumes: Task 1 (`weak_labels_from_track`, `merge_labels`, `labels_to_cvat_xml`); `extract_frames(clip_path, frames_dir) -> int` from `src/utils/ball_finetune_export.py`; `BallAnchorSet.load`; `BallTrack.load`; `CameraTrack.load` (per-frame K/R/t construction — copy the `/joints-near` pattern in `src/web/server.py`); clip discovery `output/shots/{clip}.mp4`, camera `output/camera/{clip}_camera_track.json`, track `output/ball/{clip}_ball_track.json`, anchors `output/ball/{clip}_ball_anchors.json`.
- Produces:
  - `build_clip_entry(output_dir: Path, clip_id: str, corpus_root: Path, *, window: int, min_conf: float, skip_frames: bool = False) -> dict` — extracts frames to `corpus_root/frames/{clip_id}/` (unless `skip_frames`), writes merged gold∪weak XML to `corpus_root/annos/{clip_id}.xml`, returns `{"clip_id", "n_gold", "n_weak", "n_frames"}`.
  - CLI: `python scripts/build_finetune_corpus.py --pairs output:gberch output-origi:origi01 output-kroupi:kroupi01 output-japan:s013 --corpus-root output/ball_finetune_corpus --holdout kroupi01 [--weak-window 20] [--weak-min-conf 0.5]` — builds every pair, then writes `corpus_root/manifest.json`: `{"clips": {clip_id: entry}, "holdout": ["kroupi01"], "train": [others]}`.
  - Gold labels = anchors with `image_xy` (all states; the clicked pixel is the ball).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_finetune_corpus.py`:

```python
"""Corpus builder: gold+weak merge per clip, manifest with holdout split."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from src.schemas.ball_anchor import BallAnchor, BallAnchorSet
from src.schemas.ball_track import BallFrame, BallTrack
from src.schemas.camera_track import CameraFrame, CameraTrack
from scripts.build_finetune_corpus import build_clip_entry

N = 30


def _camera():
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _fake_output(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    K, R, t = _camera()
    clip = out / "shots" / "play.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720))
    for _ in range(N):
        writer.write(np.full((720, 1280, 3), 90, dtype=np.uint8))
    writer.release()
    CameraTrack(
        clip_id="play", fps=30.0, image_size=(1280, 720),
        t_world=t.tolist(),
        frames=tuple(CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                                 confidence=1.0, is_anchor=(i == 0))
                     for i in range(N)),
    ).save(out / "camera" / "play_camera_track.json")
    BallAnchorSet(
        clip_id="play", image_size=(1280, 720),
        anchors=(BallAnchor(frame=10, image_xy=(640.0, 400.0),
                            state="grounded"),),
    ).save(out / "ball" / "play_ball_anchors.json")
    BallTrack(
        clip_id="play", fps=30.0,
        frames=tuple(
            BallFrame(frame=i, world_xyz=(30.0 + 0.2 * i, 34.0, 0.11),
                      state="grounded", confidence=0.9)
            for i in range(N)
        ),
        flight_segments=(),
    ).save(out / "ball" / "play_ball_track.json")
    return out


def test_build_clip_entry_merges_gold_and_weak(tmp_path: Path):
    out = _fake_output(tmp_path)
    corpus = tmp_path / "corpus"
    entry = build_clip_entry(out, "play", corpus, window=5, min_conf=0.5)
    assert entry["clip_id"] == "play"
    assert entry["n_gold"] == 1
    assert entry["n_weak"] == 10  # frames 5..15 minus the gold frame
    assert entry["n_frames"] == N
    # Frames extracted with the 5-digit naming the WASB layout expects.
    assert (corpus / "frames" / "play" / "00000.png").exists()
    # XML contains gold + weak, gold pixel authoritative at frame 10.
    root = ET.parse(corpus / "annos" / "play.xml").getroot()
    pts = {int(p.attrib["frame"]): p.attrib["points"]
           for p in root.find("track").findall("points")}
    assert len(pts) == 11
    assert pts[10] == "640.00,400.00"


def test_skip_frames_reuses_existing(tmp_path: Path):
    out = _fake_output(tmp_path)
    corpus = tmp_path / "corpus"
    build_clip_entry(out, "play", corpus, window=5, min_conf=0.5)
    marker = corpus / "frames" / "play" / "00000.png"
    before = marker.stat().st_mtime
    entry = build_clip_entry(out, "play", corpus, window=5, min_conf=0.5,
                             skip_frames=True)
    assert marker.stat().st_mtime == before
    assert entry["n_gold"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_build_finetune_corpus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.build_finetune_corpus'`

- [ ] **Step 3: Implement the script**

Create `scripts/build_finetune_corpus.py`:

```python
"""Build the WASB fine-tune corpus from annotated clips (spec §4.3 step 1).

For each --pairs OUTPUT_DIR:CLIP_ID this extracts every clip frame to
<corpus>/frames/<clip>/{fid:05d}.png and writes <corpus>/annos/<clip>.xml
containing the operator's gold anchor pixels UNION solved-track weak labels
within ±window of each gold frame (gold wins on collision). A manifest
records the train/holdout split.

Usage:
    python scripts/build_finetune_corpus.py \
        --pairs output:gberch output-origi:origi01 \
                output-kroupi:kroupi01 output-japan:s013 \
        --corpus-root output/ball_finetune_corpus \
        --holdout kroupi01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.schemas.ball_anchor import BallAnchorSet  # noqa: E402
from src.schemas.ball_track import BallTrack  # noqa: E402
from src.schemas.camera_track import CameraTrack  # noqa: E402
from src.utils.ball_finetune_export import extract_frames  # noqa: E402
from src.utils.ball_weak_labels import (  # noqa: E402
    labels_to_cvat_xml,
    merge_labels,
    weak_labels_from_track,
)


def build_clip_entry(
    output_dir: Path,
    clip_id: str,
    corpus_root: Path,
    *,
    window: int,
    min_conf: float,
    skip_frames: bool = False,
) -> dict:
    """Frames + merged gold∪weak XML for one clip; returns the manifest entry."""
    anchors = BallAnchorSet.load(
        output_dir / "ball" / f"{clip_id}_ball_anchors.json")
    gold = {
        a.frame: (float(a.image_xy[0]), float(a.image_xy[1]))
        for a in anchors.anchors if a.image_xy is not None
    }

    camera = CameraTrack.load(
        output_dir / "camera" / f"{clip_id}_camera_track.json")
    per_frame_K = {f.frame: np.array(f.K) for f in camera.frames}
    per_frame_R = {f.frame: np.array(f.R) for f in camera.frames}
    t_world = np.array(camera.t_world)
    per_frame_t = {
        f.frame: (np.array(f.t) if f.t is not None else t_world)
        for f in camera.frames
    }
    track = BallTrack.load(output_dir / "ball" / f"{clip_id}_ball_track.json")
    weak = weak_labels_from_track(
        track,
        per_frame_K=per_frame_K, per_frame_R=per_frame_R,
        per_frame_t=per_frame_t, distortion=camera.distortion,
        image_size=camera.image_size, gold_frames=set(gold),
        window=window, min_conf=min_conf,
    )
    merged = merge_labels(gold, weak)

    anno_path = corpus_root / "annos" / f"{clip_id}.xml"
    anno_path.parent.mkdir(parents=True, exist_ok=True)
    anno_path.write_text(labels_to_cvat_xml(clip_id, merged))

    frames_dir = corpus_root / "frames" / clip_id
    if skip_frames and frames_dir.exists():
        n_frames = len(list(frames_dir.glob("*.png")))
    else:
        n_frames = extract_frames(
            output_dir / "shots" / f"{clip_id}.mp4", frames_dir)

    return {
        "clip_id": clip_id,
        "n_gold": len(gold),
        "n_weak": len(weak),
        "n_frames": n_frames,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", required=True,
                    metavar="OUTPUT_DIR:CLIP_ID")
    ap.add_argument("--corpus-root", type=Path, required=True)
    ap.add_argument("--holdout", nargs="*", default=[])
    ap.add_argument("--weak-window", type=int, default=20)
    ap.add_argument("--weak-min-conf", type=float, default=0.5)
    ap.add_argument("--skip-frames", action="store_true",
                    help="reuse already-extracted frames")
    args = ap.parse_args()

    clips: dict[str, dict] = {}
    for pair in args.pairs:
        out_dir, _, clip_id = pair.partition(":")
        if not clip_id:
            ap.error(f"--pairs entries must be OUTPUT_DIR:CLIP_ID; got {pair!r}")
        entry = build_clip_entry(
            Path(out_dir), clip_id, args.corpus_root,
            window=args.weak_window, min_conf=args.weak_min_conf,
            skip_frames=args.skip_frames,
        )
        clips[clip_id] = entry
        print(f"{clip_id}: gold={entry['n_gold']} weak={entry['n_weak']} "
              f"frames={entry['n_frames']}")

    unknown = [h for h in args.holdout if h not in clips]
    if unknown:
        ap.error(f"--holdout clip(s) not in --pairs: {unknown}")
    manifest = {
        "clips": clips,
        "holdout": list(args.holdout),
        "train": [c for c in clips if c not in set(args.holdout)],
    }
    (args.corpus_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"manifest: train={manifest['train']} holdout={manifest['holdout']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_build_finetune_corpus.py -v`
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_finetune_corpus.py tests/test_build_finetune_corpus.py
git commit -m "feat: fine-tune corpus builder (gold+weak labels, holdout split)"
```

---

### Task 3: Fine-tune harness (`src/utils/ball_finetune_train.py` + `scripts/finetune_wasb.py`)

**Files:**
- Create: `src/utils/ball_finetune_train.py`
- Create: `scripts/finetune_wasb.py`
- Modify: `src/utils/wasb_ball_detector.py` — extract a public `load_wasb_model(checkpoint_path, device="auto") -> tuple[torch.nn.Module, str]` from the existing constructor logic (build HRNet from `_WASB_MODEL_CFG`, load checkpoint handling both dict forms + `module.` stripping, move to resolved device; the constructor then calls it — behaviour unchanged).
- Test: `tests/test_ball_finetune_train.py`

**Interfaces:**
- Consumes: corpus layout from Task 2 (`frames/{clip}/{fid:05d}.png`, `annos/{clip}.xml`, `manifest.json`); `_get_affine_transform`, `_IMAGENET_MEAN`, `_IMAGENET_STD` imported from `src.utils.wasb_ball_detector`; the vendored heatmap generator (import `gen_binary_map` from the vendored `third_party/wasb_sbdt/src/utils/heatmap.py` the same way `wasb_ball_detector.py` imports vendored modules — READ that module's import mechanics first and mirror them; if `gen_binary_map` differs in signature from `gen_heatmap(wh, cxy, r, ...)`, read the file and bind whichever generates the binary fixed-size map with sigma 2.5).
- Produces:
  - `parse_labels_xml(path: Path) -> dict[int, tuple[float, float]]` — reads the CVAT dialect (only `outside="0"` + `used_in_game=1` points), mirroring the vendored loader's semantics.
  - `build_runs(labels: Mapping[int, tuple[float, float]], frames_in: int = 3) -> list[list[int]]` — all windows of `frames_in` CONSECUTIVE labelled frames (stride 1); e.g. labels {4,5,6,7} → [[4,5,6],[5,6,7]]; sparse labels yield no runs.
  - `class FinetuneDataset(torch.utils.data.Dataset)` — `__init__(corpus_root: Path, clips: list[str], input_size: tuple[int, int] = (512, 288), sigma: float = 2.5)`; `__getitem__` returns `(x, y)` with `x: float32 (9, inp_h, inp_w)` built by the EXACT inference preprocessing (cv2.imread BGR → RGB → letterbox `_get_affine_transform(center=(w/2,h/2), scale=max(h,w))` warp INTER_LINEAR → /255 → ImageNet norm → transpose → concat) and `y: float32 (3, inp_h, inp_w)` binary heatmaps: each frame's label pixel mapped through the SAME forward affine, then sigma-2.5 map at input resolution.
  - `wbce_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor` — TrackNetV2 weighted BCE on sigmoid-clamped predictions `ŷ = clamp(sigmoid(x), 1e-4, 1-1e-4)`: `mean( -( (1-ŷ)**2 * y * log(ŷ) + ŷ**2 * (1-y) * log(1-ŷ) ) )` (equivalent to the vendored `BCELoss`'s documented focal-γ=2 form; implement directly — the vendored class needs its cfg plumbing).
  - `evaluate_hit_rate(model, dataset, device, *, tol_px: float = 5.0, max_samples: int = 200) -> float` — fraction of samples whose LAST output frame's argmax peak lies within `tol_px` (model space) of the label.
  - `scripts/finetune_wasb.py` CLI: `--corpus-root PATH --run-dir PATH [--epochs 30] [--batch 4] [--lr 1e-4] [--device auto] [--init third_party/wasb_sbdt/pretrained_weights/wasb_soccer_best.pth.tar] [--val-frac 0.1] [--limit-samples N]` — trains on manifest `train` clips (random val split), evaluates holdout clips each epoch, saves `{run_dir}/best.pth.tar` as `{"model_state_dict": ...}` on best holdout hit-rate (falls back to val hit-rate when no holdout), writes `{run_dir}/history.json` (per-epoch loss/val/holdout metrics + wall time). Device via `_pick_device`-style resolution: auto → mps if available else cpu (NOT the detector's conservative cpu default — training wants mps; implement locally in the script).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ball_finetune_train.py`:

```python
"""Fine-tune harness: XML parsing, run building, dataset parity, loss."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.utils.ball_finetune_train import (
    FinetuneDataset,
    build_runs,
    parse_labels_xml,
    wbce_loss,
)
from src.utils.ball_weak_labels import labels_to_cvat_xml
from src.utils.wasb_ball_detector import _get_affine_transform


def _mini_corpus(tmp_path: Path, labels: dict[int, tuple[float, float]],
                 n_frames: int = 8, size=(64, 48)) -> Path:
    corpus = tmp_path / "corpus"
    fdir = corpus / "frames" / "clipA"
    fdir.mkdir(parents=True)
    for i in range(n_frames):
        img = np.full((size[1], size[0], 3), 30, dtype=np.uint8)
        if i in labels:
            u, v = labels[i]
            cv2.circle(img, (int(u), int(v)), 2, (255, 255, 255), -1)
        cv2.imwrite(str(fdir / f"{i:05d}.png"), img)
    (corpus / "annos").mkdir(parents=True)
    (corpus / "annos" / "clipA.xml").write_text(
        labels_to_cvat_xml("clipA", labels))
    return corpus


def test_parse_labels_roundtrip(tmp_path: Path):
    labels = {3: (10.0, 20.0), 4: (11.0, 21.0)}
    corpus = _mini_corpus(tmp_path, labels)
    parsed = parse_labels_xml(corpus / "annos" / "clipA.xml")
    assert parsed == {3: (10.0, 20.0), 4: (11.0, 21.0)}


def test_build_runs_consecutive_only():
    labels = {4: (0, 0), 5: (0, 0), 6: (0, 0), 7: (0, 0), 20: (0, 0)}
    assert build_runs(labels) == [[4, 5, 6], [5, 6, 7]]
    assert build_runs({1: (0, 0), 3: (0, 0)}) == []


def test_dataset_shapes_and_label_mapping(tmp_path: Path):
    labels = {2: (40.0, 24.0), 3: (41.0, 24.0), 4: (42.0, 24.0)}
    corpus = _mini_corpus(tmp_path, labels)
    ds = FinetuneDataset(corpus, ["clipA"], input_size=(128, 72), sigma=2.5)
    assert len(ds) == 1
    x, y = ds[0]
    assert x.shape == (9, 72, 128) and x.dtype == torch.float32
    assert y.shape == (3, 72, 128) and y.dtype == torch.float32
    # The target peak must sit where the SAME forward affine maps the label.
    trans = _get_affine_transform((64 / 2, 48 / 2), 64.0, (128, 72), inv=False)
    lbl = np.array([40.0, 24.0, 1.0])
    exp = trans @ lbl
    peak = np.unravel_index(int(torch.argmax(y[0])), y[0].shape)
    assert abs(peak[1] - exp[0]) <= 3 and abs(peak[0] - exp[1]) <= 3
    assert float(y.max()) == 1.0 and float(y.min()) == 0.0


def test_wbce_loss_decreases_toward_target():
    torch.manual_seed(0)
    target = torch.zeros(1, 3, 8, 8)
    target[0, :, 4, 4] = 1.0
    good = torch.full((1, 3, 8, 8), -6.0)
    good[0, :, 4, 4] = 6.0
    bad = torch.full((1, 3, 8, 8), 6.0)
    bad[0, :, 4, 4] = -6.0
    assert wbce_loss(good, target) < wbce_loss(bad, target)
    assert wbce_loss(good, target).item() >= 0.0


def test_one_training_step_runs_on_cpu(tmp_path: Path):
    labels = {2: (30.0, 20.0), 3: (31.0, 20.0), 4: (32.0, 20.0)}
    corpus = _mini_corpus(tmp_path, labels)
    ds = FinetuneDataset(corpus, ["clipA"], input_size=(128, 72))
    x, y = ds[0]
    # Tiny stand-in model with the WASB io contract (9ch in, 3ch out).
    model = torch.nn.Conv2d(9, 3, kernel_size=3, padding=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = wbce_loss(model(x.unsqueeze(0)), y.unsqueeze(0))
    for _ in range(20):
        opt.zero_grad()
        loss = wbce_loss(model(x.unsqueeze(0)), y.unsqueeze(0))
        loss.backward()
        opt.step()
    after = wbce_loss(model(x.unsqueeze(0)), y.unsqueeze(0))
    assert after < before
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_finetune_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.ball_finetune_train'`

- [ ] **Step 3: Implement**

1. In `src/utils/wasb_ball_detector.py`, extract the model-construction + checkpoint-loading block of the detector's `__init__` into:

```python
def load_wasb_model(
    checkpoint_path: str | Path, device: str = "auto",
) -> tuple["torch.nn.Module", str]:
    """Build the WASB HRNet and load a checkpoint onto the resolved device.

    Accepts both ``{"model_state_dict": ...}`` and bare state dicts, strips
    DataParallel ``module.`` prefixes, returns ``(model, device_str)``.
    Shared by the inference detector and the fine-tune harness so the two
    can never drift.
    """
```

(read the constructor; move its existing lines, then call `load_wasb_model` from the constructor — behaviour byte-identical, existing tests must stay green).

2. Create `src/utils/ball_finetune_train.py` implementing the Interfaces block exactly. Key implementation notes (the implementer writes the code; these pin the semantics):
   - `parse_labels_xml`: ElementTree; only `<points>` with `outside=="0"` and child attribute `used_in_game` text `"1"`; return `{int(frame): (float(x), float(y))}`.
   - `FinetuneDataset`: enumerate manifest-independent — takes explicit `clips`; per clip, `parse_labels_xml` + `build_runs`; sample = (clip, [f0,f1,f2]). `__getitem__` loads the three pngs with `cv2.imread` (assert not None), applies the parity preprocessing (import `_get_affine_transform`, `_IMAGENET_MEAN`, `_IMAGENET_STD` from `src.utils.wasb_ball_detector`; the frame's own (w,h) drives center/scale exactly as `_preprocess_buffer` does), builds targets by mapping each frame's label through the forward `trans` and stamping a sigma-2.5 gaussian-binary map at input resolution — import the vendored generator (`gen_binary_map` or `gen_heatmap` from `third_party/wasb_sbdt/src/utils/heatmap.py`, mirroring `wasb_ball_detector.py`'s vendored-import mechanics) and clamp/binarize consistently with the vendored `binary_fixed_size` behaviour (read `heatmaps.py`); labels mapped outside the model canvas produce an all-zero target for that frame.
   - `wbce_loss`: exactly the formula in Interfaces (mean over all elements; `clamp(sigmoid(x), 1e-4, 1-1e-4)`).
   - `evaluate_hit_rate`: model in eval/no_grad; last-frame heatmap argmax vs label in model space.
3. Create `scripts/finetune_wasb.py` (same `sys.path.insert` header as sibling scripts): loads manifest, builds train dataset (+ random `val_frac` split, seed 0), holdout dataset; `load_wasb_model(args.init, args.device)`; Adam(lr); epochs loop with per-epoch `wbce_loss` train, val + holdout `evaluate_hit_rate`; saves `best.pth.tar` (`{"model_state_dict": model.state_dict()}`) on best holdout (or val) hit rate; writes `history.json`; prints a per-epoch line. `--limit-samples` truncates the dataset (smoke runs). MPS note: keep `pin_memory=False`, `num_workers=0` (macOS default-safe).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_finetune_train.py tests/test_wasb_ball_detector.py -v`
Expected: all PASS (detector suite guards the `load_wasb_model` refactor).

- [ ] **Step 5: Smoke the CLI end-to-end on the tiny corpus (CPU, seconds)**

Run:
```bash
.venv/bin/python - <<'EOF'
# Build a 3-label mini corpus in a temp dir and run 1 CPU epoch end-to-end.
import sys, subprocess, json, tempfile
from pathlib import Path
sys.path.insert(0, ".")
import cv2, numpy as np
from src.utils.ball_weak_labels import labels_to_cvat_xml

root = Path(tempfile.mkdtemp())
corpus = root / "corpus"
labels = {2: (30.0, 20.0), 3: (31.0, 20.0), 4: (32.0, 20.0)}
fdir = corpus / "frames" / "clipA"
fdir.mkdir(parents=True)
for i in range(8):
    img = np.full((48, 64, 3), 30, dtype=np.uint8)
    if i in labels:
        u, v = labels[i]
        cv2.circle(img, (int(u), int(v)), 2, (255, 255, 255), -1)
    cv2.imwrite(str(fdir / f"{i:05d}.png"), img)
(corpus / "annos").mkdir()
(corpus / "annos" / "clipA.xml").write_text(labels_to_cvat_xml("clipA", labels))
(corpus / "manifest.json").write_text(json.dumps(
    {"clips": {"clipA": {}}, "train": ["clipA"], "holdout": []}))
r = subprocess.run([sys.executable, "scripts/finetune_wasb.py",
    "--corpus-root", str(corpus), "--run-dir", str(root / "run"),
    "--epochs", "1", "--batch", "1", "--device", "cpu", "--val-frac", "0"],
    capture_output=True, text=True)
print(r.stdout[-2000:]); print(r.stderr[-2000:])
assert r.returncode == 0, "CLI failed"
assert (root / "run" / "best.pth.tar").exists()
EOF
```
Expected: exit 0, `best.pth.tar` written. (This loads the REAL HRNet + stock checkpoint — one epoch over ~1 sample is seconds on CPU.)

- [ ] **Step 6: Commit**

```bash
git add src/utils/ball_finetune_train.py src/utils/wasb_ball_detector.py scripts/finetune_wasb.py tests/test_ball_finetune_train.py
git commit -m "feat: repo-side WASB fine-tune harness (parity preprocessing, wBCE, holdout eval)"
```

---

### Task 4: Build the real corpus and run the fine-tune (operational, MPS)

No new code. Budget: corpus build ~10 min (4 clips × frame extraction), training 30–90 min on MPS.

- [ ] **Step 1: Build the corpus**

```bash
.venv/bin/python scripts/build_finetune_corpus.py \
  --pairs output:gberch output-origi:origi01 output-kroupi:kroupi01 output-japan:s013 \
  --corpus-root output/ball_finetune_corpus --holdout kroupi01
```

Record the per-clip gold/weak/frames counts. Expect gold ≈ 59/60/12/14 and weak in the hundreds total. If a clip is missing its `_ball_track.json` or camera track, report it and proceed with the remaining clips (holdout must still exist).

- [ ] **Step 2: Sanity-check sample counts**

```bash
.venv/bin/python -c "
from pathlib import Path
from src.utils.ball_finetune_train import FinetuneDataset
import json
m = json.loads(Path('output/ball_finetune_corpus/manifest.json').read_text())
tr = FinetuneDataset(Path('output/ball_finetune_corpus'), m['train'])
ho = FinetuneDataset(Path('output/ball_finetune_corpus'), m['holdout'])
print('train samples:', len(tr), 'holdout samples:', len(ho))
"
```
Expected: train samples in the hundreds (weak densification creates consecutive runs); holdout ≥ 5. If train < 100, lower `--weak-min-conf` to 0.4 and rebuild once; if still < 100, report the numbers and continue (small-corpus fine-tune is still informative).

- [ ] **Step 3: Train**

```bash
.venv/bin/python scripts/finetune_wasb.py \
  --corpus-root output/ball_finetune_corpus \
  --run-dir output/ball_finetune_runs/run1 \
  --epochs 30 --batch 4 --lr 1e-4 --device auto
```

(Use a 600000 ms bash timeout and run in the background / re-poll; the script prints per-epoch lines and writes `history.json` incrementally.) Watch epoch 1: if loss is NaN or MPS errors out, retry once with `--device cpu --epochs 10`; if CPU epoch time × 10 exceeds ~2 h, stop and report timings (BLOCKED-style) instead of shipping a partial run. Record the final holdout hit-rate curve.

- [ ] **Step 4: Commit the run record (not the weights)**

Append the corpus counts + final history summary to the task report. Nothing to git-commit in this task (corpus + runs are untracked data).

---

### Task 5: Evaluate the fine-tuned checkpoint against the recorded baselines

No new code; one throwaway override config. Budget ~45 min of stage runs on MPS.

- [ ] **Step 1: Capture pre-swap coverage for kroupi01 and s013** (origi/gberch baselines are already recorded): re-run the ball stage on `output-kroupi` and `output-japan` with the STOCK checkpoint and read each `quality_report.json` ball shot's `detection_coverage.total` (the `ball.shots` key is a LIST of per-shot dicts). Record.

- [ ] **Step 2: Evaluate with the fine-tuned checkpoint** via a scratchpad override yaml:

```yaml
ball:
  wasb:
    checkpoint: output/ball_finetune_runs/run1/best.pth.tar
```

Then, passing `--config <that yaml>`:
- `recon.py run --output output-origi --stages ball` → origi01/origi02 coverage
- `recon.py run --output output-kroupi --stages ball` and `--output output-japan` → holdout + s013 coverage
- `scripts/run_touch_recall_validation.py --output output --shot gberch --config <yaml>` → recall table
- `.venv/bin/python -m pytest tests/test_ball_anchor_accuracy.py -q` (anchored frames are detector-independent; must stay green)

- [ ] **Step 3: Adjudicate against the bars** (Global Constraints): gberch union recall ≥ 4/8; origi02 total ≥ 0.562; origi01/kroupi01/s013 within −0.02 of pre-swap; report precision with the GT caveat. **kroupi01 is the honest holdout number — headline it.** Produce a before/after table in the task report. If bars FAIL: do not proceed to Task 6; commit nothing config-side; write the table + a one-paragraph diagnosis (e.g. overfit signature: train clips improve, holdout regresses) and stop — the harness itself remains merged and re-runnable with more labels.

- [ ] **Step 4 (only on PASS): promote the checkpoint**

```bash
cp output/ball_finetune_runs/run1/best.pth.tar \
   third_party/wasb_sbdt/pretrained_weights/wasb_soccer_finetuned_v1.pth.tar
```

Update `config/default.yaml` `ball.wasb.checkpoint` to the new path with a comment naming the run + date + the stock path as fallback. Commit: `feat: swap ball.wasb.checkpoint to fine-tuned v1 (measured tables in body)`.

---

### Task 6 (only if Task 5 passed): Re-enable the detector-gated features and re-measure

- [ ] **Step 1:** Flip `ball.foot_guided.enabled: true` and `ball.touch_attribution.enabled: true` in `config/default.yaml` (update their comments: gated on detector quality — re-enabled after fine-tune v1, date). Update `TouchAttributionCfg.enabled` default to `True` and the two tests that pin the defaults (`test_config_block_keys` in `tests/test_ball_touch_attribution.py`; check `tests/test_ball_stage_attribution_wiring.py` disabled-test still passes since it sets the flag explicitly).
- [ ] **Step 2:** Re-measure with everything on: gberch two-config recall + origi coverage (same commands as Task 5). Acceptance: union recall ≥ its Task-5 value (foot_guided + attribution must not regress it); origi coverage within −0.02 of Task-5 values. If either feature regresses, flip THAT feature back off, note it, and keep the other.
- [ ] **Step 3:** Run the full suite `.venv/bin/python -m pytest -q` (known Blender-env failure excepted) + ruff over touched files.
- [ ] **Step 4:** Docs commit: update the spec §4.3 with a "Measured outcome" paragraph (tables), CLAUDE.md's ball bullets (new checkpoint, re-enabled flags — whichever survived), and `docs/superpowers/specs/2026-06-15-ball-finetune-README.md`'s operator steps (now: `build_finetune_corpus.py` → `finetune_wasb.py` → eval commands). Commit `docs: phase-4 measured outcomes + fine-tune runbook`.
