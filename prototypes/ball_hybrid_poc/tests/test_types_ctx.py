"""T0 spike tests: round-trip every shared type, and sanity-check
``ClipContext`` against each of the four real clips (skipped when its
main-repo output dir isn't present)."""

from __future__ import annotations

import math

import pytest

from src.utils.ball_anchor_heights import GROUND_LEVEL_STATES
from src.utils.ball_eval import ray_plane_z

from ..ctx import CLIPS, load_clip
from ..types import (
    Observation,
    SynthRun,
    Track,
    TrackFrame,
    TruthEvent,
    TruthFrame,
    TruthTrack,
    load_json,
    save_json,
    validate_results,
)

BALL_RADIUS_M = 0.11


# ---------------------------------------------------------------------------
# Round-trip every contract type
# ---------------------------------------------------------------------------

def test_observation_round_trip():
    o = Observation(frame=12, uv=(100.5, 200.25), conf=0.8, source="detector")
    assert Observation.from_json(o.to_json()) == o


def test_truth_frame_and_event_round_trip():
    f = TruthFrame(frame=5, xyz=(1.0, 2.0, 0.11), state="ground")
    assert TruthFrame.from_json(f.to_json()) == f

    e = TruthEvent(frame=5, kind="touch", xyz=(1.0, 2.0, 0.11),
                    player_id="P001", bone="right_ankle")
    assert TruthEvent.from_json(e.to_json()) == e

    e_no_player = TruthEvent(frame=9, kind="bounce", xyz=(3.0, 4.0, 0.11))
    assert TruthEvent.from_json(e_no_player.to_json()) == e_no_player


def test_truth_track_round_trip():
    tt = TruthTrack(
        clip_id="gberch",
        scenario="baseline",
        fps=30.0,
        frames=(
            TruthFrame(frame=0, xyz=(0.0, 0.0, 0.11), state="ground"),
            TruthFrame(frame=1, xyz=(0.1, 0.0, 0.5), state="air"),
        ),
        events=(TruthEvent(frame=1, kind="touch", xyz=(0.1, 0.0, 0.5),
                            player_id="P001", bone="right_foot"),),
        seed_anchor_frames=(0, 1),
        physics={"drag_cd": 0.25, "restitution": 0.6},
    )
    assert TruthTrack.from_json(tt.to_json()) == tt


def test_synth_run_round_trip():
    sr = SynthRun(
        clip_id="gberch",
        scenario="baseline",
        observations=(Observation(frame=1, uv=(1.0, 2.0), conf=0.9,
                                   source="synthetic"),),
        anchors=({"frame": 0, "image_xy": [1.0, 2.0], "state": "grounded"},),
        noise_model={"px_sigma": 1.5},
    )
    assert SynthRun.from_json(sr.to_json()) == sr


def test_track_round_trip():
    tr = Track(
        clip_id="gberch",
        method="hybrid",
        frames=(
            TrackFrame(frame=0, xyz=(0.0, 0.0, 0.11), mode="faithful",
                       conf=0.9),
            TrackFrame(frame=1, xyz=None, mode="simulated", conf=None),
        ),
    )
    assert Track.from_json(tr.to_json()) == tr


def test_save_load_json_round_trip(tmp_path):
    tt = TruthTrack(
        clip_id="gberch", scenario="baseline", fps=30.0,
        frames=(TruthFrame(frame=0, xyz=(0.0, 0.0, 0.11), state="ground"),),
    )
    path = tmp_path / "nested" / "truth_baseline.json"
    save_json(path, tt)
    assert TruthTrack.from_json(load_json(path)) == tt


def test_validate_results_flags_missing_keys():
    problems = validate_results({})
    assert any("clip_id" in p for p in problems)
    assert any("scenarios" in p for p in problems)


def test_validate_results_accepts_minimal_shape():
    results = {
        "clip_id": "gberch",
        "fps": 30.0,
        "image_size": [1920, 1080],
        "scenarios": {
            "baseline": {
                "truth": {"clip_id": "gberch"},
                "tracks": {"current": {"frames": []}},
                "metrics": {"current": {}},
            },
        },
        "real": {"tracks": {}, "metrics": {}},
    }
    assert validate_results(results) == []


def test_validate_results_not_a_dict():
    assert validate_results([1, 2, 3]) == ["results is not a dict"]


# ---------------------------------------------------------------------------
# Per-clip ClipContext sanity checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("clip_id", sorted(CLIPS))
def test_load_clip(clip_id):
    output_dir, _shot_id = CLIPS[clip_id]
    from pathlib import Path
    if not Path(output_dir).exists():
        pytest.skip(f"{output_dir} not present on this machine")

    ctx = load_clip(clip_id)

    assert ctx.n_frames > 0
    assert len(ctx.anchors.anchors) > 0
    assert ctx.fps > 0
    assert ctx.image_size[0] > 0 and ctx.image_size[1] > 0

    # --- ray/project round trip on a grounded anchor -----------------
    grounded = [a for a in ctx.anchors.anchors
                if a.state in GROUND_LEVEL_STATES and a.image_xy is not None]
    assert grounded, "expected at least one grounded anchor with a click"
    anchor = grounded[0]

    C, d_hat = ctx.ray(anchor.frame, anchor.image_xy)
    xyz = ray_plane_z(C, d_hat, BALL_RADIUS_M)
    assert xyz is not None, "ground-plane intersection failed"

    uv2 = ctx.project(anchor.frame, xyz)
    err_px = math.hypot(uv2[0] - anchor.image_xy[0],
                         uv2[1] - anchor.image_xy[1])
    assert err_px < 0.5, (
        f"{clip_id} f{anchor.frame}: reprojection error {err_px:.3f}px >= 0.5px")

    # --- camera_centres() sanity ---------------------------------------
    centres = ctx.camera_centres()
    assert len(centres) == ctx.n_frames
    assert all(c is not None for c in centres)

    # --- player_context() + at least one touch anchor's joint ----------
    touches = [a for a in ctx.anchors.anchors if a.state == "player_touch"]
    if touches:
        pc = ctx.player_context()
        found = False
        for a in touches:
            for df in range(-3, 4):
                world = pc.joint_world(a.frame + df, a.player_id, a.bone)
                if world is not None and all(math.isfinite(x) for x in world):
                    found = True
                    break
            if found:
                break
        assert found, (
            f"{clip_id}: no player_touch anchor resolved a finite joint "
            "world position within +/-3 frames")
