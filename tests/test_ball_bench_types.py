"""Round-trip every shared bench dataclass + ``validate_results``."""

from __future__ import annotations

from src.utils.ball_bench_types import (
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
        scenario="mismatch",
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
        scenario="mismatch",
        observations=(Observation(frame=1, uv=(1.0, 2.0), conf=0.9,
                                   source="detector"),),
        anchors=({"frame": 0, "image_xy": [1.0, 2.0], "state": "grounded"},),
        noise_model={"sigma_px": 1.5},
    )
    assert SynthRun.from_json(sr.to_json()) == sr


def test_track_round_trip():
    tr = Track(
        clip_id="gberch",
        method="reference",
        frames=(
            TrackFrame(frame=0, xyz=(0.0, 0.0, 0.11), mode="faithful",
                       conf=0.9),
            TrackFrame(frame=1, xyz=None, mode="simulated", conf=None),
        ),
    )
    assert Track.from_json(tr.to_json()) == tr


def test_save_load_json_round_trip(tmp_path):
    tt = TruthTrack(
        clip_id="gberch", scenario="mismatch", fps=30.0,
        frames=(TruthFrame(frame=0, xyz=(0.0, 0.0, 0.11), state="ground"),),
    )
    path = tmp_path / "nested" / "truth_mismatch.json"
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
            "mismatch": {
                "truth": {"clip_id": "gberch"},
                "tracks": {"reference": {"frames": []}},
                "metrics": {"reference": {}},
            },
        },
        "real": {"tracks": {}, "metrics": {}},
    }
    assert validate_results(results) == []


def test_validate_results_not_a_dict():
    assert validate_results([1, 2, 3]) == ["results is not a dict"]
