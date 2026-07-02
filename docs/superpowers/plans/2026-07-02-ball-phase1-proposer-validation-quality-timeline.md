# Ball Phase 1 — Proposer Validation + Quality Timeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 1 of `docs/superpowers/specs/2026-07-02-ball-stage-improvement-design.md`: (a) a stage-level test guarding that the body-kinematics touch proposer actually fires inside `BallStage.run`, plus a one-command two-config recall-validation runner for the GPU box (spec §4.1); (b) a ball quality timeline in the ball anchor editor, fed by a new read-only `GET /ball-quality/{shot_id}` endpoint that aggregates the existing per-shot sidecars (spec §5.1).

**Architecture:** No pipeline behaviour changes. New pure module `src/utils/ball_quality.py` builds the timeline payload from the three sidecars the ball stage already writes (`*_ball_observations.json`, `*_ball_diag.json`, `*_ball_keyframes.json`); a thin FastAPI endpoint loads the JSON and delegates; the editor renders a confidence strip (same visual language as the camera editor's strip) plus a ranked "annotate next" list. The recall runner drives `BallStage` directly twice (proposer off/on), snapshots the auto-anchor sidecars under the names `scripts/ball_touch_recall_report.py` already documents, and prints the table.

**Tech Stack:** Python 3.11, pytest, FastAPI + TestClient, vanilla JS/canvas in a single HTML file (no framework), numpy only in tests.

## Global Constraints

- Type annotations on all new function signatures; frozen/immutable patterns (never mutate an input dict — return a new one).
- New utility modules must be torch-free and import-light (the web server imports them).
- Sidecars are enrichment: the endpoint must degrade to an empty payload when files are missing/corrupt, never 500.
- Commit format `<type>: <description>` (feat/fix/test/docs/chore). No attribution trailers.
- Run tests with the repo venv: `.venv/bin/python -m pytest` (plain `pytest` if the venv is already active).
- All file paths below are relative to the repo root `/Users/joebower/workplace/football-perspectives`.

---

### Task 1: Stage-level wiring test — the kinematic proposer fires inside `BallStage.run`

The proposer injection in `src/stages/ball.py:1477-1503` is wrapped in `try/except` and gated on `kin_cfg.enabled and player_ctx.player_ids`; a regression (bad import, signature drift, config-key typo) would silently disable it. This task adds the guard tests. Expected outcome: they PASS against current code; a failure means a real wiring bug — fix `src/stages/ball.py`, not the test.

**Files:**
- Test (create): `tests/test_ball_stage_kinematic_wiring.py`

**Interfaces:**
- Consumes: `BallStage` (`src/stages/ball.py`), `FakeBallDetector` (`src/utils/ball_detector.py`), `BallEvent` (`src/utils/ball_auto_events.py`), `SmplWorldTrack` (`src/schemas/smpl_world.py`), monkeypatch target `src.stages.ball.propose_touches`.
- Produces: nothing used by later tasks (pure regression guard).

- [ ] **Step 1: Write the three wiring tests**

Create `tests/test_ball_stage_kinematic_wiring.py`:

```python
"""BallStage wiring guard for the body-kinematics touch proposer.

The proposer injection (src/stages/ball.py, "Body-kinematics touch
proposer" block) is gated on config + player tracks and wrapped in a
swallow-all try/except; these tests pin that (a) it actually fires with
default config, (b) its events reach the diag sidecar, (c) the config
flag disables it, and (d) a proposer crash degrades with a warning
instead of killing the stage."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.shots import Shot, ShotsManifest
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.smpl_world import SmplWorldTrack
from src.stages.ball import BallStage
from src.utils.ball_auto_events import BallEvent
from src.utils.ball_detector import FakeBallDetector

N_FRAMES = 60
FPS = 30.0


def _camera_pose() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    look = np.array([0.0, 64.0, -30.0])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _project(p: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    cam = R @ p + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _build_scene(tmp_path: Path) -> tuple[Path, list[tuple[float, float, float] | None]]:
    """Multi-shot output dir with one shot 'play': camera track, blank clip,
    manifest, one SMPL player track (so player_ctx.player_ids is non-empty —
    the proposer gate requires it), and a grounded rolling-ball detection set."""
    out = tmp_path / "out"
    K, R, t = _camera_pose()

    clip = out / "shots" / "play.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(clip), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    for _ in range(N_FRAMES):
        writer.write(np.full((720, 1280, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()

    CameraTrack(
        clip_id="play", fps=FPS, image_size=(1280, 720),
        t_world=t.tolist(),
        frames=tuple(
            CameraFrame(frame=i, K=K.tolist(), R=R.tolist(),
                        confidence=1.0, is_anchor=(i == 0))
            for i in range(N_FRAMES)
        ),
    ).save(out / "camera" / "play_camera_track.json")

    ShotsManifest(
        source_file="fake.mp4", fps=FPS, total_frames=N_FRAMES,
        shots=[Shot(id="play", clip_file="shots/play.mp4",
                    start_frame=0, end_frame=N_FRAMES - 1,
                    start_time=0.0, end_time=(N_FRAMES - 1) / FPS)],
    ).save(out / "shots" / "shots_manifest.json")

    base_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    thetas0 = np.zeros((24, 3), dtype=np.float32)
    frames = np.arange(N_FRAMES, dtype=np.int64)
    SmplWorldTrack(
        player_id="P001", frames=frames,
        betas=np.zeros(10, dtype=np.float32),
        thetas=np.stack([thetas0] * N_FRAMES),
        root_R=np.stack([base_R.astype(np.float32)] * N_FRAMES),
        root_t=np.stack([np.array([40.0, 34.0, 1.0], dtype=np.float32)] * N_FRAMES),
        confidence=np.full(N_FRAMES, 0.8, dtype=np.float32),
        shot_id="play",
    ).save(out / "hmr_world" / "play__P001_smpl_world.npz")

    detections: list[tuple[float, float, float] | None] = []
    for i in range(N_FRAMES):
        p = np.array([30.0 + 0.2 * i, 34.0, 0.11])
        u, v = _project(p, K, R, t)
        detections.append((u, v, 0.9))
    return out, detections


def _ball_cfg(**overrides) -> dict:
    cfg = {
        "ball": {
            "detector": "fake",
            # the all-green clip gives a uniform NCC surface that confuses
            # the appearance bridge; irrelevant to the wiring under test
            "appearance_bridge": {"enabled": False},
        },
        "pitch": {"length_m": 105.0, "width_m": 68.0},
    }
    cfg["ball"].update(overrides)
    return cfg


@pytest.mark.integration
def test_proposer_fires_and_touch_reaches_diag(tmp_path: Path, monkeypatch):
    out, detections = _build_scene(tmp_path)
    calls: dict = {}

    def fake_propose(**kwargs):
        calls["kwargs"] = kwargs
        return [BallEvent(frame=20, kind="touch", score=0.9,
                          player_id="P001", bone="head")]

    monkeypatch.setattr("src.stages.ball.propose_touches", fake_propose)
    BallStage(config=_ball_cfg(), output_dir=out,
              ball_detector=FakeBallDetector(detections)).run()

    assert "kwargs" in calls, "propose_touches was never invoked by BallStage.run"
    assert calls["kwargs"]["cfg"].enabled is True
    assert calls["kwargs"]["ball_uvs"], "proposer received no ball pixels"

    diag = json.loads((out / "ball" / "play_ball_diag.json").read_text())
    assert any(
        e["kind"] == "touch" and e["frame"] == 20
        and e["player_id"] == "P001" and e["bone"] == "head"
        for e in diag["events"]
    ), f"sentinel proposer touch missing from diag events: {diag['events']}"


@pytest.mark.integration
def test_proposer_disabled_by_config_flag(tmp_path: Path, monkeypatch):
    out, detections = _build_scene(tmp_path)
    calls: dict = {}

    def fake_propose(**kwargs):
        calls["kwargs"] = kwargs
        return []

    monkeypatch.setattr("src.stages.ball.propose_touches", fake_propose)
    BallStage(
        config=_ball_cfg(kinematic_touch={"enabled": False}),
        output_dir=out, ball_detector=FakeBallDetector(detections),
    ).run()
    assert "kwargs" not in calls, "proposer ran despite enabled=false"


@pytest.mark.integration
def test_proposer_crash_degrades_with_warning(tmp_path: Path, monkeypatch, caplog):
    out, detections = _build_scene(tmp_path)

    def broken_propose(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.stages.ball.propose_touches", broken_propose)
    with caplog.at_level("WARNING"):
        BallStage(config=_ball_cfg(), output_dir=out,
                  ball_detector=FakeBallDetector(detections)).run()
    assert (out / "ball" / "play_ball_track.json").exists()
    assert any("kinematic touch proposer failed" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_ball_stage_kinematic_wiring.py -v`
Expected: 3 PASSED. These pin current behaviour; a failure is a genuine wiring bug in `src/stages/ball.py` — investigate the injection block at `src/stages/ball.py:1474-1503` before touching the test. (Known acceptable variance: if the default `events` solver rejects this synthetic scene for scene-specific reasons, add `"solver": "piecewise"` to `_ball_cfg` — the detect_events → proposer-merge → diag path under test is identical in both solvers.)

- [ ] **Step 3: Run the neighbouring ball-stage suites to confirm no interference**

Run: `.venv/bin/python -m pytest tests/test_ball_stage.py tests/test_ball_kinematic_touch.py tests/test_ball_kinematic_recall.py -q`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ball_stage_kinematic_wiring.py
git commit -m "test: guard kinematic touch proposer wiring in BallStage.run"
```

---

### Task 2: Two-config recall-validation runner (`scripts/run_touch_recall_validation.py`)

Spec §4.1 requires running the ball stage twice (proposer off/on) and scoring both auto-anchor sets against the shot's manual anchors. Today that's four manual commands plus file renames; this task makes it one command so the GPU validation is a single operator step.

**Files:**
- Create: `scripts/run_touch_recall_validation.py`
- Test: `tests/test_run_touch_recall_validation.py`

**Interfaces:**
- Consumes: `recall_table`, `proposer_only_touches`, `_print_table` from `scripts/ball_touch_recall_report.py`; `touches_from_anchor_set(path: str | Path) -> list[tuple[int, str, str]]` from `src/utils/ball_touch_recall.py`; `load_config(path: Path | None) -> dict` from `src/pipeline/config.py`; `BallStage` (supports `stage.shot_filter = "<shot_id>"` to restrict to one shot, see `src/stages/ball.py:701-707`).
- Produces: `with_kinematic_toggle(cfg: dict, enabled: bool) -> dict` and `snapshot_auto_anchors(ball_dir: Path, shot_id: str, label: str) -> Path` (unit-tested pure helpers); snapshot files `<shot>_ball_anchors_auto_break_only.json` / `<shot>_ball_anchors_auto_union.json` (the exact names `ball_touch_recall_report.py`'s usage docstring documents); report JSON `<shot>_touch_recall.json`.

- [ ] **Step 1: Write the failing unit tests for the pure helpers**

Create `tests/test_run_touch_recall_validation.py`:

```python
"""Pure helpers of the two-config touch-recall validation runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_touch_recall_validation import (
    snapshot_auto_anchors,
    with_kinematic_toggle,
)


def test_toggle_sets_flag_without_mutating_input():
    cfg = {"ball": {"kinematic_touch": {"enabled": True, "kin_window": 2}}}
    off = with_kinematic_toggle(cfg, False)
    assert off["ball"]["kinematic_touch"]["enabled"] is False
    assert off["ball"]["kinematic_touch"]["kin_window"] == 2
    # input untouched (immutability)
    assert cfg["ball"]["kinematic_touch"]["enabled"] is True


def test_toggle_creates_missing_subtrees():
    on = with_kinematic_toggle({}, True)
    assert on["ball"]["kinematic_touch"]["enabled"] is True


def test_snapshot_copies_auto_sidecar(tmp_path: Path):
    ball_dir = tmp_path / "ball"
    ball_dir.mkdir()
    payload = {"clip_id": "play", "image_size": [1280, 720], "anchors": []}
    (ball_dir / "play_ball_anchors_auto.json").write_text(json.dumps(payload))
    dst = snapshot_auto_anchors(ball_dir, "play", "break_only")
    assert dst == ball_dir / "play_ball_anchors_auto_break_only.json"
    assert json.loads(dst.read_text()) == payload


def test_snapshot_missing_sidecar_raises(tmp_path: Path):
    ball_dir = tmp_path / "ball"
    ball_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        snapshot_auto_anchors(ball_dir, "play", "union")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_run_touch_recall_validation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_touch_recall_validation'`

- [ ] **Step 3: Write the runner**

Create `scripts/run_touch_recall_validation.py`:

```python
"""Two-config touch-recall validation for one shot (Phase 1, spec
docs/superpowers/specs/2026-07-02-ball-stage-improvement-design.md §4.1).

Runs the ball stage twice on an existing output directory — body-kinematics
touch proposer disabled, then enabled — snapshots the auto-anchor sidecar
after each run under the names ball_touch_recall_report.py documents, and
prints the break-only / proposer-only / union recall table against the
shot's manual anchors (pseudo-ground-truth). Requires the shot's upstream
artifacts (camera track, refined_poses/hmr_world) and the real detector,
so the stage runs need the GPU box.

Usage (gberch example):
    python scripts/run_touch_recall_validation.py \
        --output output-gberch --shot gberch

    # re-print the table from existing snapshots, no GPU needed:
    python scripts/run_touch_recall_validation.py \
        --output output-gberch --shot gberch --report-only
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

from scripts.ball_touch_recall_report import (
    _print_table,
    proposer_only_touches,
    recall_table,
)
from src.utils.ball_touch_recall import touches_from_anchor_set


def with_kinematic_toggle(cfg: dict, enabled: bool) -> dict:
    """New deep-copied config with ``ball.kinematic_touch.enabled`` set."""
    out = copy.deepcopy(cfg)
    out.setdefault("ball", {}).setdefault("kinematic_touch", {})["enabled"] = enabled
    return out


def snapshot_auto_anchors(ball_dir: Path, shot_id: str, label: str) -> Path:
    """Copy ``<shot>_ball_anchors_auto.json`` to the labelled snapshot the
    recall report consumes (``..._auto_<label>.json``)."""
    src = ball_dir / f"{shot_id}_ball_anchors_auto.json"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} missing — did the ball stage run produce auto anchors?")
    dst = ball_dir / f"{shot_id}_ball_anchors_auto_{label}.json"
    shutil.copyfile(src, dst)
    return dst


def _run_ball_stage(output_dir: Path, shot_id: str, cfg: dict) -> None:
    # Import inside the run path: the stage pulls detector deps (torch/WASB)
    # that the pure helpers and their tests must not require.
    from src.stages.ball import BallStage

    stage = BallStage(config=cfg, output_dir=output_dir)
    stage.shot_filter = shot_id  # only re-detect/solve the shot under test
    stage.run()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path,
                    help="pipeline output dir (e.g. output-gberch)")
    ap.add_argument("--shot", required=True, help="shot id (e.g. gberch)")
    ap.add_argument("--config", type=Path, default=None,
                    help="optional YAML override merged with defaults")
    ap.add_argument("--report-only", action="store_true",
                    help="skip the stage runs; score existing snapshots")
    args = ap.parse_args()

    ball_dir = args.output / "ball"
    manual_path = ball_dir / f"{args.shot}_ball_anchors.json"
    if not manual_path.exists():
        ap.error(f"no manual anchors at {manual_path} — nothing to score against")

    if not args.report_only:
        from src.pipeline.config import load_config

        cfg = load_config(args.config)
        print("run 1/2: ball stage with kinematic_touch DISABLED (break-only)")
        _run_ball_stage(args.output, args.shot, with_kinematic_toggle(cfg, False))
        snapshot_auto_anchors(ball_dir, args.shot, "break_only")
        print("run 2/2: ball stage with kinematic_touch ENABLED (union)")
        _run_ball_stage(args.output, args.shot, with_kinematic_toggle(cfg, True))
        snapshot_auto_anchors(ball_dir, args.shot, "union")

    manual = touches_from_anchor_set(manual_path)
    break_only = touches_from_anchor_set(
        ball_dir / f"{args.shot}_ball_anchors_auto_break_only.json")
    union = touches_from_anchor_set(
        ball_dir / f"{args.shot}_ball_anchors_auto_union.json")
    proposer_only = proposer_only_touches(break_only, union)
    table = recall_table(manual, break_only, proposer_only, union)
    _print_table(table)
    report_path = ball_dir / f"{args.shot}_touch_recall.json"
    report_path.write_text(json.dumps(table, indent=2))
    print(f"written {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_run_touch_recall_validation.py tests/test_ball_kinematic_recall.py -v`
Expected: all PASS (the second file guards the report functions this script reuses).

- [ ] **Step 5: Smoke the CLI arg handling (no GPU)**

Run: `.venv/bin/python scripts/run_touch_recall_validation.py --output /nonexistent --shot x --report-only; echo "exit=$?"`
Expected: argparse error `no manual anchors at /nonexistent/ball/x_ball_anchors.json — nothing to score against`, `exit=2`.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_touch_recall_validation.py tests/test_run_touch_recall_validation.py
git commit -m "feat: one-command two-config touch recall validation runner"
```

- [ ] **Step 7 (operator, GPU box — documented handoff, not CI):**

On the GPU machine, with the gberch output dir present (manual anchors `gberch_ball_anchors.json` are the ground truth; a backup lives at `/tmp/gberch_preregen_backup/gberch_ball_anchors.json` on that box):

```bash
python scripts/run_touch_recall_validation.py --output <gberch-output-dir> --shot gberch
```

Acceptance (from the 2026-06-27 spec §8): `union` recall materially above `break_only` with `union` precision ≥ 0.5. If precision is low, raise `ball.kinematic_touch.min_emit_score` (default 0.25); if recall is low, widen `contact_gap_m` (default 0.30) or lower `kin_min_foot_speed` (default 8.0) via a `--config` override YAML, then re-run. Record the final table (committed `<shot>_touch_recall.json` is fine) and the chosen thresholds; if defaults change, update `config/default.yaml` in a follow-up commit.

---

### Task 3: Pure quality-payload builder (`src/utils/ball_quality.py`)

Aggregation + ranking logic as a torch-free pure function, so the endpoint stays a two-line delegate and everything is unit-testable without a server.

**Files:**
- Create: `src/utils/ball_quality.py`
- Test: `tests/test_ball_quality.py`

**Interfaces:**
- Consumes: raw dicts parsed from the three sidecars. Shapes (all already on disk today):
  - observations (`src/stages/ball.py::_write_observations_sidecar`): `{"clip_id", "fps", "frames": [{"frame": int, "uv": [u,v]|null, "confidence": float, "p_flight": float, "gap_fill": bool, "source": str}]}`
  - diag (`src/stages/ball.py:1878-1914`): `{"solver", "derived", "underconstrained_spans": [{"start": int, "end": int, "residual_px": float|null}], "segments", "bounces", "splits", "contact_gaps", "events": [{"frame", "kind", "score", "player_id", "bone", "goal_element", "end_frame"}], "anchors": {...}, "detection_coverage": {...}, "cross_replay": ...}`
  - keyframes (`src/schemas/ball_keyframes.py::BallKeyframeSet.save`): `{"keyframes": [...], "segments": [{"start_frame": int, "end_frame": int, "kind": str, "hints": {...}}], ...}`
- Produces: `build_quality_payload(observations: dict | None, diag: dict | None, keyframes: dict | None, *, min_gap_frames: int = 12) -> dict` returning `{"n_frames": int, "fps": float | None, "frames": [{"frame", "confidence", "gap_fill", "source"}], "events": [...], "underconstrained_spans": [...], "segments": [{"start_frame", "end_frame", "kind"}], "detection_coverage": dict | None, "annotate_next": [{"start": int, "end": int, "reason": "underconstrained_flight"|"detection_gap", "severity": float}]}`. Also exports `detection_gaps(frames: list[dict], min_gap_frames: int) -> list[tuple[int, int]]` and `rank_annotate_next(underconstrained_spans: list[dict], gaps: list[tuple[int, int]]) -> list[dict]`. Tasks 4 and 5 rely on exactly these names and keys.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ball_quality.py`:

```python
"""build_quality_payload: sidecar aggregation + annotate-next ranking."""

from __future__ import annotations

from src.utils.ball_quality import (
    build_quality_payload,
    detection_gaps,
    rank_annotate_next,
)


def _obs_frame(frame: int, conf: float, gap_fill: bool = False) -> dict:
    return {"frame": frame, "uv": [100.0, 200.0], "confidence": conf,
            "p_flight": 0.1, "gap_fill": gap_fill, "source": "detector"}


def test_missing_sidecars_degrade_to_empty_payload():
    payload = build_quality_payload(None, None, None)
    assert payload == {
        "n_frames": 0, "fps": None, "frames": [], "events": [],
        "underconstrained_spans": [], "segments": [],
        "detection_coverage": None, "annotate_next": [],
    }


def test_detection_gap_run_detected():
    frames = [_obs_frame(i, 0.9) for i in range(5)]
    frames += [_obs_frame(5 + i, 0.0) for i in range(15)]      # 15-frame hole
    frames += [_obs_frame(20 + i, 0.9) for i in range(5)]
    assert detection_gaps(frames, min_gap_frames=12) == [(5, 19)]


def test_gap_fill_frames_count_as_missing():
    frames = [_obs_frame(i, 0.8, gap_fill=True) for i in range(12)]
    assert detection_gaps(frames, min_gap_frames=12) == [(0, 11)]


def test_short_gap_ignored():
    frames = [_obs_frame(0, 0.9)] + [_obs_frame(1 + i, 0.0) for i in range(5)] \
        + [_obs_frame(6, 0.9)]
    assert detection_gaps(frames, min_gap_frames=12) == []


def test_underconstrained_span_outranks_gap():
    spans = [{"start": 10, "end": 20, "residual_px": 8.0}]
    gaps = [(40, 60)]
    ranked = rank_annotate_next(spans, gaps)
    assert [it["reason"] for it in ranked] == [
        "underconstrained_flight", "detection_gap"]
    assert ranked[0]["start"] == 10 and ranked[0]["end"] == 20
    assert ranked[0]["severity"] > ranked[1]["severity"]


def test_payload_aggregates_all_three_sidecars():
    observations = {"clip_id": "play", "fps": 30.0,
                    "frames": [_obs_frame(0, 0.9), _obs_frame(1, 0.0)]}
    diag = {
        "underconstrained_spans": [{"start": 0, "end": 1, "residual_px": None}],
        "events": [{"frame": 1, "kind": "touch", "score": 0.8,
                    "player_id": "P1", "bone": "r_foot",
                    "goal_element": None, "end_frame": None}],
        "detection_coverage": {"pass1": 0.5, "second_pass": 0.0,
                               "total": 0.5, "zoom_recoveries": 0},
    }
    keyframes = {"segments": [
        {"start_frame": 0, "end_frame": 1, "kind": "roll", "hints": {}}]}
    payload = build_quality_payload(observations, diag, keyframes)
    assert payload["n_frames"] == 2
    assert payload["fps"] == 30.0
    assert payload["frames"][1] == {
        "frame": 1, "confidence": 0.0, "gap_fill": False, "source": "detector"}
    assert payload["events"][0]["kind"] == "touch"
    assert payload["segments"] == [
        {"start_frame": 0, "end_frame": 1, "kind": "roll"}]
    assert payload["detection_coverage"]["total"] == 0.5
    assert payload["annotate_next"][0]["reason"] == "underconstrained_flight"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_quality.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.ball_quality'`

- [ ] **Step 3: Implement the module**

Create `src/utils/ball_quality.py`:

```python
"""Ball-stage quality payload for the dashboard timeline (spec §5.1).

Aggregates the three per-shot sidecars the ball stage already writes
(observations / diag / keyframes) into one compact payload the ball
anchor editor renders as a per-frame quality strip plus a ranked
"annotate here next" list. Pure and torch-free: the web endpoint only
parses JSON and delegates here.
"""

from __future__ import annotations

DEFAULT_MIN_GAP_FRAMES = 12
_MAX_ANNOTATE_ITEMS = 10
# Underconstrained flight spans are the operator's highest-value fix (one
# bracketing anchor resolves the whole arc); plain detection gaps rank below.
_GAP_SEVERITY_WEIGHT = 0.5


def detection_gaps(
    frames: list[dict], min_gap_frames: int,
) -> list[tuple[int, int]]:
    """Maximal runs of >= ``min_gap_frames`` consecutive frames with no
    accepted detection (zero confidence or IMM gap-fill)."""
    gaps: list[tuple[int, int]] = []
    run_start: int | None = None
    prev: int | None = None
    for rec in sorted(frames, key=lambda r: int(r["frame"])):
        f = int(rec["frame"])
        missing = (
            float(rec.get("confidence", 0.0)) <= 0.0
            or bool(rec.get("gap_fill", False))
        )
        contiguous = prev is not None and f == prev + 1
        if missing and run_start is not None and contiguous:
            pass  # run continues
        elif missing:
            if run_start is not None and prev is not None \
                    and prev - run_start + 1 >= min_gap_frames:
                gaps.append((run_start, prev))
            run_start = f
        else:
            if run_start is not None and prev is not None \
                    and prev - run_start + 1 >= min_gap_frames:
                gaps.append((run_start, prev))
            run_start = None
        prev = f
    if run_start is not None and prev is not None \
            and prev - run_start + 1 >= min_gap_frames:
        gaps.append((run_start, prev))
    return gaps


def rank_annotate_next(
    underconstrained_spans: list[dict], gaps: list[tuple[int, int]],
) -> list[dict]:
    """Ranked "annotate here next" items, most valuable first."""
    items: list[dict] = []
    for span in underconstrained_spans:
        start, end = int(span["start"]), int(span["end"])
        residual = float(span.get("residual_px") or 0.0)
        items.append({
            "start": start, "end": end,
            "reason": "underconstrained_flight",
            "severity": (end - start + 1) * (1.0 + residual / 10.0),
        })
    for start, end in gaps:
        items.append({
            "start": start, "end": end,
            "reason": "detection_gap",
            "severity": _GAP_SEVERITY_WEIGHT * (end - start + 1),
        })
    items.sort(key=lambda it: (-it["severity"], it["start"]))
    return items[:_MAX_ANNOTATE_ITEMS]


def build_quality_payload(
    observations: dict | None,
    diag: dict | None,
    keyframes: dict | None,
    *,
    min_gap_frames: int = DEFAULT_MIN_GAP_FRAMES,
) -> dict:
    """One payload for GET /ball-quality/{shot_id}; every input optional."""
    obs = observations or {}
    dg = diag or {}
    kf = keyframes or {}
    obs_frames = list(obs.get("frames", []))
    spans = list(dg.get("underconstrained_spans", []))
    gaps = detection_gaps(obs_frames, min_gap_frames)
    return {
        "n_frames": len(obs_frames),
        "fps": obs.get("fps"),
        "frames": [
            {
                "frame": int(r["frame"]),
                "confidence": float(r.get("confidence", 0.0)),
                "gap_fill": bool(r.get("gap_fill", False)),
                "source": r.get("source", "none"),
            }
            for r in obs_frames
        ],
        "events": list(dg.get("events", [])),
        "underconstrained_spans": spans,
        "segments": [
            {
                "start_frame": s.get("start_frame"),
                "end_frame": s.get("end_frame"),
                "kind": s.get("kind"),
            }
            for s in kf.get("segments", [])
        ],
        "detection_coverage": dg.get("detection_coverage"),
        "annotate_next": rank_annotate_next(spans, gaps),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_quality.py -v`
Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_quality.py tests/test_ball_quality.py
git commit -m "feat: ball quality payload builder (sidecar aggregation + annotate-next ranking)"
```

---

### Task 4: `GET /ball-quality/{shot_id}` endpoint

**Files:**
- Modify: `src/web/server.py` — insert the new route directly after `get_auto_ball_anchors_for_shot` (which ends at `src/web/server.py:1964`)
- Test: `tests/test_web_ball_quality_api.py`

**Interfaces:**
- Consumes: `build_quality_payload` from Task 3; sidecar files under `<output_dir>/ball/` named `{shot_id}_ball_observations.json`, `{shot_id}_ball_diag.json`, `{shot_id}_ball_keyframes.json` (legacy single-shot fallback: same names without the `{shot_id}_` prefix, matching how legacy runs name `ball_track.json`).
- Produces: `GET /ball-quality/{shot_id}` returning the Task 3 payload; always 200 (missing/corrupt sidecars degrade to the empty payload). The Task 5 UI fetches exactly this route.

- [ ] **Step 1: Write the failing endpoint tests**

Create `tests/test_web_ball_quality_api.py`:

```python
"""GET /ball-quality/{shot_id}: sidecar aggregation, degradation, legacy names."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.web.server import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(output_dir=tmp_path, config_path=None))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_no_sidecars_degrades_to_empty_payload(tmp_path: Path):
    r = _client(tmp_path).get("/ball-quality/play")
    assert r.status_code == 200
    body = r.json()
    assert body["n_frames"] == 0
    assert body["annotate_next"] == []


def test_aggregates_prefixed_sidecars(tmp_path: Path):
    ball = tmp_path / "ball"
    _write(ball / "play_ball_observations.json", {
        "clip_id": "play", "fps": 30.0,
        "frames": [
            {"frame": 0, "uv": [1.0, 2.0], "confidence": 0.9,
             "p_flight": 0.0, "gap_fill": False, "source": "detector"},
            {"frame": 1, "uv": None, "confidence": 0.0,
             "p_flight": 0.0, "gap_fill": True, "source": "none"},
        ],
    })
    _write(ball / "play_ball_diag.json", {
        "underconstrained_spans": [{"start": 0, "end": 1, "residual_px": 5.0}],
        "events": [{"frame": 0, "kind": "touch", "score": 0.7,
                    "player_id": "P1", "bone": "l_foot",
                    "goal_element": None, "end_frame": None}],
        "detection_coverage": {"pass1": 0.5, "second_pass": 0.0,
                               "total": 0.5, "zoom_recoveries": 0},
    })
    _write(ball / "play_ball_keyframes.json", {
        "segments": [{"start_frame": 0, "end_frame": 1,
                      "kind": "roll", "hints": {}}],
    })
    body = _client(tmp_path).get("/ball-quality/play").json()
    assert body["n_frames"] == 2
    assert body["events"][0]["bone"] == "l_foot"
    assert body["segments"] == [
        {"start_frame": 0, "end_frame": 1, "kind": "roll"}]
    assert body["annotate_next"][0]["reason"] == "underconstrained_flight"


def test_legacy_unprefixed_sidecar_fallback(tmp_path: Path):
    _write(tmp_path / "ball" / "ball_observations.json", {
        "clip_id": "play", "fps": 25.0,
        "frames": [{"frame": 0, "uv": [1.0, 2.0], "confidence": 0.9,
                    "p_flight": 0.0, "gap_fill": False, "source": "detector"}],
    })
    body = _client(tmp_path).get("/ball-quality/play").json()
    assert body["n_frames"] == 1
    assert body["fps"] == 25.0


def test_corrupt_sidecar_degrades_not_500(tmp_path: Path):
    p = tmp_path / "ball" / "play_ball_diag.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json")
    r = _client(tmp_path).get("/ball-quality/play")
    assert r.status_code == 200
    assert r.json()["events"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_web_ball_quality_api.py -v`
Expected: 3 of 4 FAIL with 404 (route not registered); the empty-payload test also fails on 404.

- [ ] **Step 3: Add the route**

In `src/web/server.py`, immediately after the `get_auto_ball_anchors_for_shot` function body (after `src/web/server.py:1964`, before `@app.get("/joints-near")`), insert:

```python
    @app.get("/ball-quality/{shot_id}")
    def get_ball_quality_for_shot(shot_id: str):
        """Aggregated ball-stage quality for the editor's timeline strip
        (per-frame detection confidence, events, underconstrained spans,
        ranked annotate-next cues). Read-only over sidecars the ball stage
        already writes; missing or corrupt sidecars degrade to an empty
        payload rather than erroring."""
        from src.utils.ball_quality import build_quality_payload

        def _load_sidecar(name: str):
            path = output_dir / "ball" / f"{shot_id}_{name}.json"
            if not path.exists():
                # Legacy single-shot runs write unprefixed names
                # (ball_track.json et al.).
                path = output_dir / "ball" / f"{name}.json"
            if not path.exists():
                return None
            try:
                return json.loads(path.read_text())
            except Exception as exc:  # noqa: BLE001 — quality is enrichment
                logger.debug("ball-quality: unreadable %s: %s", path, exc)
                return None

        return build_quality_payload(
            _load_sidecar("ball_observations"),
            _load_sidecar("ball_diag"),
            _load_sidecar("ball_keyframes"),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_web_ball_quality_api.py tests/test_web_ball_anchors_api.py -v`
Expected: all PASS (second file guards the neighbouring routes didn't break).

- [ ] **Step 5: Commit**

```bash
git add src/web/server.py tests/test_web_ball_quality_api.py
git commit -m "feat: read-only /ball-quality endpoint aggregating ball sidecars"
```

---

### Task 5: Quality timeline strip + annotate-next list in the ball anchor editor

Same visual language as the camera editor's confidence strip (`src/web/static/anchor_editor.html:1189-1225`): red→green per-frame bars, coloured bands for annotate-next spans, tick stripes for events/anchors, white cursor, click-to-seek.

**Files:**
- Modify: `src/web/static/ball_anchor_editor.html`
- Test: `tests/test_web_ball_quality_timeline.py`

**Interfaces:**
- Consumes: `GET /ball-quality/{shot_id}` (Task 4 payload — `n_frames`, `frames[].{frame,confidence,gap_fill}`, `events[].frame`, `annotate_next[].{start,end,reason}`); existing editor globals `shotId`, `anchors`, `fps`, and functions `currentFrame()`, `seekToFrame(fi)` (`ball_anchor_editor.html:194-295`).
- Produces: DOM ids `qualityStrip`, `qualityCanvas`, `annotateNext`; JS functions `loadQuality()`, `renderQualityStrip()`, `renderAnnotateNext()` (referenced only within this file).

- [ ] **Step 1: Write the failing markup test**

Create `tests/test_web_ball_quality_timeline.py`:

```python
"""The ball anchor editor ships the quality timeline strip wired to
/ball-quality (strip canvas, annotate-next list, click-to-seek)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.web.server import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(output_dir=tmp_path, config_path=None))


def test_editor_served_with_quality_strip(tmp_path: Path):
    html = _client(tmp_path).get("/ball-anchor-editor").text
    assert 'id="qualityStrip"' in html
    assert 'id="qualityCanvas"' in html
    assert 'id="annotateNext"' in html
    # Strip is fed by the new endpoint and seeks on click.
    assert "/ball-quality/" in html
    assert "renderQualityStrip" in html
    assert "annotate_next" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_web_ball_quality_timeline.py -v`
Expected: FAIL — `assert 'id="qualityStrip"' in html`

- [ ] **Step 3: Add the markup**

In `src/web/static/ball_anchor_editor.html`, find the seek-bar row (the `<div>` containing `<input id="seek" type="range" style="flex:1;" min="0" value="0">`, around line 126). Immediately **after that row's closing `</div>`**, insert:

```html
      <div id="qualityStrip" title="Ball quality — click to seek"
           style="height:26px;background:#0a0c12;border:1px solid #252840;border-radius:4px;margin-top:6px;cursor:pointer;">
        <canvas id="qualityCanvas" style="width:100%;height:100%;display:block;"></canvas>
      </div>
      <div id="annotateNext" style="margin-top:4px;font-size:11px;color:#94a3b8;"></div>
```

- [ ] **Step 4: Add the JS**

In the same file's `<script>` block, after the `seekToFrame` definition (`function seekToFrame(fi) { ... }`, around line 295), insert:

```javascript
// ── Ball quality timeline (fed by GET /ball-quality/{shot}) ────────────
let quality = null;  // payload from /ball-quality, null until loaded

async function loadQuality() {
  if (!shotId) { quality = null; renderQualityStrip(); renderAnnotateNext(); return; }
  try {
    const r = await fetch(`/ball-quality/${encodeURIComponent(shotId)}`);
    quality = r.ok ? await r.json() : null;
  } catch (e) {
    quality = null;
  }
  renderQualityStrip();
  renderAnnotateNext();
}

function renderQualityStrip() {
  const strip = document.getElementById("qualityStrip");
  const c = document.getElementById("qualityCanvas");
  const ctx = c.getContext("2d");
  const w = strip.clientWidth, h = strip.clientHeight;
  c.width = w; c.height = h;
  ctx.fillStyle = "#0a0c12";
  ctx.fillRect(0, 0, w, h);
  if (!quality || !quality.n_frames) {
    ctx.fillStyle = "#4b5563";
    ctx.font = "11px -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("(no ball quality yet — run the ball stage)", w / 2, h / 2 + 4);
    return;
  }
  const N = quality.n_frames;
  const barW = Math.max(1, w / N);
  // Per-frame detection confidence: red (0) -> green (1); gap-fill reads as 0.
  for (const f of quality.frames) {
    const c01 = f.gap_fill ? 0 : Math.max(0, Math.min(1, f.confidence || 0));
    ctx.fillStyle = `rgb(${Math.round(255 * (1 - c01))},${Math.round(200 * c01 + 30)},60)`;
    ctx.fillRect(f.frame * barW, 0, Math.ceil(barW), h);
  }
  // Annotate-next bands along the bottom: red = underconstrained flight,
  // orange = detection gap.
  for (const it of quality.annotate_next || []) {
    ctx.fillStyle = it.reason === "underconstrained_flight"
      ? "rgba(239,68,68,0.6)" : "rgba(251,146,60,0.5)";
    ctx.fillRect(it.start * barW, h - 7, (it.end - it.start + 1) * barW, 7);
  }
  // Top stripe ticks: auto events (blue), manual anchors (purple, drawn last).
  for (const e of quality.events || []) {
    ctx.fillStyle = "#38bdf8";
    ctx.fillRect(e.frame * barW, 0, Math.max(1, barW), 4);
  }
  for (const a of anchors) {
    ctx.fillStyle = "#a855f7";
    ctx.fillRect(a.frame * barW, 0, Math.max(1, barW), 4);
  }
  // Cursor.
  const cx = (currentFrame() / Math.max(1, N - 1)) * w;
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.fillRect(Math.max(0, cx - 1), 0, 2, h);
}

function renderAnnotateNext() {
  const el = document.getElementById("annotateNext");
  el.innerHTML = "";
  const items = (quality && quality.annotate_next) || [];
  if (!items.length) return;
  const label = document.createElement("span");
  label.textContent = "Annotate next: ";
  el.appendChild(label);
  for (const it of items.slice(0, 3)) {
    const b = document.createElement("button");
    b.textContent = `${it.reason === "underconstrained_flight" ? "flight" : "gap"} ${it.start}–${it.end}`;
    b.title = it.reason === "underconstrained_flight"
      ? "Flight span with < 2 hard knots — add a bracketing kick/bounce/grounded anchor inside it"
      : "Long detection gap — confirm the ball state through it";
    b.style.cssText = "margin-right:6px;background:#1e293b;color:#e2e8f0;border:1px solid #475569;border-radius:3px;font-size:11px;padding:1px 6px;cursor:pointer;";
    b.onclick = () => seekToFrame(it.start);
    el.appendChild(b);
  }
}

document.getElementById("qualityStrip").addEventListener("click", (ev) => {
  if (!quality || !quality.n_frames) return;
  const rect = ev.currentTarget.getBoundingClientRect();
  const frac = (ev.clientX - rect.left) / Math.max(1, rect.width);
  seekToFrame(Math.round(frac * (quality.n_frames - 1)));
});
```

- [ ] **Step 5: Wire the load + redraw hooks**

Still in `ball_anchor_editor.html`:

1. Find where the editor loads anchors for the current shot (the `fetch` of `` `/ball-anchors/${...}` `` and its `/auto` sibling — the editor's init/shot-change path). Add a `loadQuality();` call immediately after those loads complete (same function, after `renderAnchors()` is triggered). If the editor reloads anchors when `shotSelect` changes, add `loadQuality();` in that change handler too.
2. Find the per-frame UI update where the seek slider is synced (`seek.value = String(fi);`, around line 311). Add `renderQualityStrip();` at the end of that handler so the cursor tracks playback.
3. In the save-success path of the anchors POST (where `setDirty(false)` is called after a successful save), add `loadQuality();` so manual-anchor ticks refresh.

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_web_ball_quality_timeline.py tests/test_web_ball_editor_touch_panel.py -v`
Expected: all PASS.

- [ ] **Step 7: Manual smoke check (if a populated output dir is available locally)**

Run: `.venv/bin/python recon.py serve --output output-origi --port 8001` and open `http://127.0.0.1:8001/ball-anchor-editor?shot=<shot_id>`.
Expected: the strip renders red→green bars with orange/red bottom bands where the diag sidecar flags gaps/underconstrained spans; clicking the strip seeks; "Annotate next" buttons seek to span starts. If no output dir is available, skip — the endpoint test in Task 4 covers the data path.

- [ ] **Step 8: Commit**

```bash
git add src/web/static/ball_anchor_editor.html tests/test_web_ball_quality_timeline.py
git commit -m "feat: ball quality timeline strip + annotate-next cues in anchor editor"
```

---

### Task 6: Full-suite verification

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: everything passes except the known env-dependent Blender test (`test_blender_export_smpl_skeleton.py`) when Blender is absent. No new failures relative to `git stash && pytest -q` baseline if in doubt.

- [ ] **Step 2: Lint the touched Python**

Run: `.venv/bin/python -m ruff check src/utils/ball_quality.py scripts/run_touch_recall_validation.py tests/test_ball_quality.py tests/test_web_ball_quality_api.py tests/test_web_ball_quality_timeline.py tests/test_ball_stage_kinematic_wiring.py tests/test_run_touch_recall_validation.py`
Expected: clean (fix anything it flags before committing).

- [ ] **Step 3: Commit any fixups**

```bash
git add -A && git commit -m "chore: phase-1 lint/test fixups"   # only if needed
```
