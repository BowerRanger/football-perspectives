"""ShortsStage orchestration with Blender / ffmpeg / audio mocked."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.schemas.shorts import load_sidecar, save_sidecar, sidecar_path
from src.stages import shorts as stage_mod
from src.stages.shorts import ShortsStage
from src.utils.shorts_framing import FramingFailure, FramingResult
from src.utils.shorts_pipeline import audio_plan, pass_fingerprint, resolve_captions

MOMENTS = {"strike": 371, "line_cross": 394, "impact": 402, "keeper_dive": 380,
           "buildup_start": 300, "scorer_pid": "P006", "keeper_pid": "P005",
           "goal_end": "left", "sources": {}}


class Calls:
    def __init__(self):
        self.renders: list[str] = []
        self.composed: list[Path] = []
        self.audio: list[tuple] = []


@pytest.fixture
def env(tmp_path, monkeypatch):
    calls = Calls()
    (tmp_path / "shots").mkdir()
    (tmp_path / "shots" / "g1.mp4").write_bytes(b"x")
    (tmp_path / "ball").mkdir()
    (tmp_path / "ball" / "g1_ball_track.json").write_text("{}")

    monkeypatch.setattr(stage_mod, "derive_moments", lambda out, shot: dict(MOMENTS))
    monkeypatch.setattr(stage_mod, "make_framing_check",
                        lambda *a, **k: (lambda spec, a_, b_, subj, excl, ov:
                                         FramingResult(True, (), {"m": 1.0})))
    monkeypatch.setattr(stage_mod, "inputs_digest", lambda out, shot: "digest0")
    monkeypatch.setattr(stage_mod.rpr, "resolve_style_payload",
                        lambda *a, **k: {"teams": {}})

    def fake_render(out, shot, spec, cfg, quality, out_dir=None, vertical_only=True):
        assert vertical_only is True
        calls.renders.append(spec["id"])
        p = Path(out_dir) / f"{spec['camera'].replace(':', '_')}_9x16.mp4"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"mp4")
        return p
    monkeypatch.setattr(stage_mod.rpr, "render_pass", fake_render)

    class _Cam:
        fps = 25.0
    monkeypatch.setattr(stage_mod.rpr, "_load_broadcast_camera", lambda out, shot: _Cam())

    def fake_compose(edl, out_path, audio=None):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"short")
        calls.composed.append(out_path)
        return {"duration_s": 12.5}
    monkeypatch.setattr(stage_mod, "compose", fake_compose)

    def fake_audio(src, windows, events, dur, cfg, out_path=None):
        calls.audio.append((windows, events, dur))
        Path(out_path).write_bytes(b"wav")
        return Path(out_path)
    monkeypatch.setattr(stage_mod.shorts_audio, "build_audio", fake_audio)
    return tmp_path, calls


def make(tmp, **shorts_cfg):
    return ShortsStage({"shorts": shorts_cfg}, tmp)


def test_runs_all_default_templates_and_writes_sidecar(env):
    tmp, calls = env
    st = make(tmp)
    assert not st.is_complete()
    st.run()
    for t in ("matchday", "keeper", "comic"):
        assert (tmp / "shorts" / f"g1_{t}.mp4").exists()
    assert len(calls.composed) == 3
    assert st.is_complete()
    side = load_sidecar(sidecar_path(tmp, "g1"))
    assert set(side["templates"]) == {"matchday", "keeper", "comic"}
    assert side["moments"]["strike"] == 371
    out = side["templates"]["matchday"]["outputs"]
    assert out["mp4"] == "shorts/g1_matchday.mp4" and out["audio"] is True
    # audio events landed on the output timeline for strike + impact
    kinds = {k for _, evs, _ in calls.audio for k, _ in evs}
    assert kinds == {"strike", "impact"}


def test_rerun_skips_cached_passes(env):
    tmp, calls = env
    make(tmp, templates=["matchday"]).run()
    n = len(calls.renders)
    assert n > 0
    make(tmp, templates=["matchday"]).run()
    assert len(calls.renders) == n          # every pass cache-hit
    side = load_sidecar(sidecar_path(tmp, "g1"))
    assert all(v["cached"] for v in side["templates"]["matchday"]["outputs"]["pass_cache"].values())


def test_changed_inputs_invalidate_cache(env, monkeypatch):
    tmp, calls = env
    make(tmp, templates=["matchday"]).run()
    n = len(calls.renders)
    monkeypatch.setattr(stage_mod, "inputs_digest", lambda out, shot: "digest1")
    make(tmp, templates=["matchday"]).run()
    assert len(calls.renders) == 2 * n


def test_operator_block_preserved_and_pins_win(env):
    tmp, calls = env
    side = {"version": 1, "shot": "g1", "moments": {}, "templates": {},
            "operator": {"moments": {"strike": 365}, "captions": {"matchday": [
                {"text": "MY CAPTION", "style": "title", "start": 0}]},
                "templates": ["matchday"], "candidates": {}}}
    save_sidecar(sidecar_path(tmp, "g1"), side)
    make(tmp).run()
    after = load_sidecar(sidecar_path(tmp, "g1"))
    assert after["operator"]["moments"] == {"strike": 365}
    assert after["moments"]["strike"] == 365
    assert set(after["templates"]) == {"matchday"}          # operator template pin
    assert [c["text"] for c in after["templates"]["matchday"]["captions"]] == ["MY CAPTION"]
    assert not (tmp / "shorts" / "g1_keeper.mp4").exists()


def test_framing_rejection_falls_back_to_next_candidate(env, monkeypatch):
    tmp, calls = env
    bad = FramingResult(False, (FramingFailure("subject_too_small", (1, 2), "tiny"),), {})

    def check_factory(*a, **k):
        def check(spec, a_, b_, subj, excl, ov):
            return bad if spec["camera"] == "drone" else FramingResult(True, (), {})
        return check
    monkeypatch.setattr(stage_mod, "make_framing_check", check_factory)
    make(tmp, templates=["matchday"]).run()
    side = load_sidecar(sidecar_path(tmp, "g1"))
    slot = next(s for s in side["templates"]["matchday"]["slots"] if s["id"] == "buildup")
    assert slot["chosen"] == 1 and slot["rejected"][0]["framing"]["failures"][0]["check"] == "subject_too_small"
    assert not any(p["camera"] == "drone" and "buildup" in p["id"] for p in side["templates"]["matchday"]["passes"])


def test_unresolved_template_is_not_rendered(env, monkeypatch):
    tmp, calls = env
    bad = FramingResult(False, (FramingFailure("ball_occluded", (1, 2), "x"),), {})
    monkeypatch.setattr(stage_mod, "make_framing_check",
                        lambda *a, **k: (lambda *x: bad))
    make(tmp, templates=["matchday"]).run()
    assert calls.renders == [] and calls.composed == []
    side = load_sidecar(sidecar_path(tmp, "g1"))
    assert side["templates"]["matchday"]["ok"] is False


def test_audio_disabled_passes_none(env, monkeypatch):
    tmp, calls = env
    seen = {}
    monkeypatch.setattr(stage_mod, "compose",
                        lambda edl, out, audio=None: (seen.setdefault("audio", audio),
                                                      out.parent.mkdir(exist_ok=True),
                                                      out.write_bytes(b"s"), {"duration_s": 1})[3])
    make(tmp, templates=["comic"], audio={"enabled": False}).run()
    assert seen["audio"] is None and calls.audio == []


def test_audio_failure_degrades_to_silent(env, monkeypatch):
    tmp, calls = env
    monkeypatch.setattr(stage_mod.shorts_audio, "build_audio",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("no audio decoded")))
    make(tmp, templates=["comic"]).run()
    assert (tmp / "shorts" / "g1_comic.mp4").exists()


def test_shot_filter_and_non_goal_shot_skipped(env, monkeypatch):
    tmp, calls = env
    st = make(tmp)
    st.shot_filter = "other"
    assert st._target_shots() == []
    assert st.is_complete()
    monkeypatch.setattr(stage_mod, "derive_moments",
                        lambda out, shot: {**MOMENTS, "impact": None})
    assert make(tmp)._target_shots() == []


def test_disabled_is_noop(env):
    tmp, calls = env
    st = make(tmp, enabled=False)
    st.run()
    assert calls.renders == [] and st.is_complete()


# --- helpers ---------------------------------------------------------------

def test_pass_fingerprint_sensitive_to_each_input():
    base = dict(shot="g", quality={"w": 1}, digest="d", style_payload={"a": 1})
    spec = {"id": "x", "frames": [1, 2]}
    fp = pass_fingerprint(spec, **base)
    assert fp == pass_fingerprint(dict(spec), **base)
    assert fp != pass_fingerprint({**spec, "frames": [1, 3]}, **base)
    assert fp != pass_fingerprint(spec, **{**base, "digest": "e"})
    assert fp != pass_fingerprint(spec, **{**base, "style_payload": {"a": 2}})
    assert fp != pass_fingerprint(spec, **{**base, "quality": {"w": 2}})


def test_audio_plan_maps_events_through_edl():
    edl = {"fps": 30, "segments": [
        {"from": 360, "to": 390, "first_frame": 350, "stretch": 1, "src": "a"},
        {"from": 366, "to": 378, "first_frame": 366, "stretch": 4, "src": "b"}]}
    windows, events, total = audio_plan(edl, 30.0, {"strike": 371, "impact": 402})
    assert total == pytest.approx(1.0 + 1.6)
    assert windows[0] == [12.0, 13.0] and windows[1][2] == pytest.approx(4.0)
    strikes = [t for k, t in events if k == "strike"]
    assert strikes[0] == pytest.approx(11 / 30)
    assert strikes[1] == pytest.approx(1.0 + (371 - 366) * 4 / 30)
    assert not [e for e in events if e[0] == "impact"]       # 402 is outside both cuts


def test_resolve_captions_precedence():
    tpl = {"captions": [{"text": "A", "style": "title", "start": 0}, {"text": "B", "style": "sub"}]}
    assert resolve_captions(tpl, None, None) is None
    assert resolve_captions(tpl, "What goal is this?", None)[0]["text"] == "What goal is this?"
    assert resolve_captions(tpl, "x", None)[1]["text"] == "B"
    op = [{"text": "OP", "style": "title"}]
    assert resolve_captions(tpl, "x", op) == op


def test_default_yaml_has_shorts_block():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "default.yaml").read_text())
    s = cfg["shorts"]
    assert s["templates"] == ["matchday", "keeper", "comic"] and s["quality"] == "clean"
    assert s["audio"]["enabled"] is True and s["shot"] == "auto"
