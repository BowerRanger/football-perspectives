"""Regression guard: the ball stage's pure-Python event/anchor tail
(detect_events -> kinematic touch proposer -> touch attribution ->
generate_auto_anchors -> merge_anchors) must be BIT-IDENTICAL run to run
given identical inputs.

Context (2026-09-03 determinism investigation, see
docs/superpowers/notes/ball-accuracy/2026-09-03-ball-stage-determinism.md):
run_touch_recall_validation.py produced different touch-recall numbers
(0.500 vs 0.375) across two runs with byte-identical config on the same
clip. The investigation's cheapest, most direct probe is this one: replay
the pipeline from an ALREADY-CACHED ``*_ball_observations.json`` sidecar
(no detector, no video decode -- second_pass/foot_guided/detection_cache
disabled and a dummy detector injected) twice in separate subprocesses
with different ``PYTHONHASHSEED`` values, and assert the resulting
``*_ball_anchors_auto.json`` is byte-identical.

A different PYTHONHASHSEED per subprocess specifically targets set/dict
iteration-order bugs (unseeded str/tuple hashing changes set iteration
order per process) -- the classic "same code, same inputs, different
answer" Python footgun. See the linked note for the companion WASB/MPS
forward-pass determinism findings (bit-identical on this machine/torch
version, both same- and cross-process) -- this test only covers the
Python-level half of the investigation.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# (clip_id, out_dir, shot_id) -- same fixtures as test_ball_anchor_accuracy.py.
CLIPS = [
    ("gberch", "output", "gberch"),
    ("kroupi01", "output-kroupi", "kroupi01"),
    ("origi01", "output-origi", "origi01"),
]

# Driver executed in a fresh subprocess: monkeypatches BallStage._detect_loop
# to return the cached observations sidecar verbatim (no detector, no video
# decode), disables second_pass/foot_guided/detection_cache (which would
# otherwise want a real detector), and runs the stage once into the given
# scratch output dir. Written to a temp file per test invocation so it can
# be re-run with a different PYTHONHASHSEED via a fresh `python` process
# (in-process re-import wouldn't change hash seed -- that's fixed at
# interpreter start).
_REPLAY_DRIVER = '''
import json
import sys
from pathlib import Path

sys.path.insert(0, {root!r})

from src.stages.ball import BallStage, TrackerStep
from src.pipeline.config import load_config

shot_id = {shot_id!r}
obs_path = Path({obs_path!r})
out_dir = Path(sys.argv[1])

data = json.loads(obs_path.read_text())
steps, raw_confidences, sources = [], {{}}, {{}}
for rec in data["frames"]:
    f = rec["frame"]
    uv = tuple(rec["uv"]) if rec["uv"] is not None else None
    steps.append(TrackerStep(
        frame=f, uv=uv, p_flight=rec["p_flight"],
        is_outlier=False, is_gap_fill=bool(rec["gap_fill"]), pos_cov=None,
    ))
    raw_confidences[f] = rec["confidence"]
    if rec["source"] != "none":
        sources[f] = rec["source"]


class _DummyDetector:
    SUPPORTS_REDETECT = False

    def detect(self, frame):
        raise RuntimeError("dummy detector invoked -- replay driver is wrong")

    def detect_candidates(self, frame, min_score, top_k=5):
        raise RuntimeError("dummy detector invoked -- replay driver is wrong")


def _fake_detect_loop(self, clip_path, cfg, detector, anchor_by_frame,
                       prior=None, prior_drop_below=0.0):
    return (list(steps), dict(raw_confidences), dict(sources))


BallStage._detect_loop = _fake_detect_loop

cfg = load_config(None)
ball_cfg = cfg.setdefault("ball", {{}})
ball_cfg.setdefault("kinematic_touch", {{}})["enabled"] = True
ball_cfg.setdefault("second_pass", {{}})["enabled"] = False
ball_cfg.setdefault("foot_guided", {{}})["enabled"] = False
ball_cfg.setdefault("detection_cache", {{}})["enabled"] = False

stage = BallStage(config=cfg, output_dir=out_dir, ball_detector=_DummyDetector())
stage.shot_filter = shot_id
stage.run()
'''


def _run_replay(tmp_path: Path, out_dir: Path, shot_id: str,
                 obs_path: Path, label: str, hashseed: str) -> Path:
    """Symlink the read-only fixture subdirs into a scratch output dir,
    copy the manual anchors, write + run the replay driver with the given
    PYTHONHASHSEED, and return the resulting auto-anchors path."""
    scratch = tmp_path / label
    scratch.mkdir()
    (scratch / "ball").mkdir()
    for sub in ("camera", "tracks", "refined_poses", "hmr_world", "shots"):
        src = out_dir / sub
        if src.exists():
            (scratch / sub).symlink_to(src)
    manual = out_dir / "ball" / f"{shot_id}_ball_anchors.json"
    (scratch / "ball" / f"{shot_id}_ball_anchors.json").write_bytes(
        manual.read_bytes())

    script = tmp_path / f"replay_{label}.py"
    script.write_text(_REPLAY_DRIVER.format(
        root=str(ROOT), shot_id=shot_id, obs_path=str(obs_path)))

    import os
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    result = subprocess.run(
        [sys.executable, str(script), str(scratch)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 0, (
        f"replay driver ({label}, PYTHONHASHSEED={hashseed}) failed:\\n"
        f"stdout: {result.stdout}\\nstderr: {result.stderr}"
    )
    auto_path = scratch / "ball" / f"{shot_id}_ball_anchors_auto.json"
    assert auto_path.exists(), f"{label}: no auto-anchors written"
    return auto_path


@pytest.mark.integration
@pytest.mark.parametrize("clip_id,out_dir_name,shot_id", CLIPS)
def test_anchor_pipeline_is_hashseed_independent(
    tmp_path, clip_id, out_dir_name, shot_id,
):
    out_dir = ROOT / out_dir_name
    obs_path = out_dir / "ball" / f"{shot_id}_ball_observations.json"
    manual_path = out_dir / "ball" / f"{shot_id}_ball_anchors.json"
    cam_path = out_dir / "camera" / f"{shot_id}_camera_track.json"
    if not (obs_path.exists() and manual_path.exists() and cam_path.exists()):
        pytest.skip(f"{clip_id}: cached observations/camera fixtures not present")

    path_a = _run_replay(tmp_path, out_dir, shot_id, obs_path, "run_a", "0")
    path_b = _run_replay(tmp_path, out_dir, shot_id, obs_path, "run_b", "4276993775")

    bytes_a = path_a.read_bytes()
    bytes_b = path_b.read_bytes()
    assert bytes_a == bytes_b, (
        f"{clip_id}: ball_anchors_auto.json differs between PYTHONHASHSEED=0 "
        f"and PYTHONHASHSEED=4276993775 runs from the SAME cached "
        f"observations -- Python-level (set/dict iteration order) "
        f"nondeterminism in the event/anchor pipeline"
    )
