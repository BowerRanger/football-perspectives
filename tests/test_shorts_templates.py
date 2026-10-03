"""shorts_templates: moment-relative DSL, candidate fallback, the three
shipped templates vs the hand-made gberch EDLs."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from src.utils import shorts_templates as st
from src.utils.short_compositor import edl_time_map
from src.utils.shorts_framing import FramingFailure, FramingResult

REPO = Path(__file__).resolve().parents[1]
GBERCH = {"strike": 371, "line_cross": 394, "impact": 402, "keeper_dive": 397,
          "buildup_start": 296, "scorer_pid": "P006", "keeper_pid": "P005", "goal_end": "left"}


def _mini(**cand):
    base = {"camera": "chase", "from": "strike-40", "to": "impact+5"}
    base.update(cand)
    return {"name": "t", "slots": [{"id": "a", "candidates": [base]}]}


# --- DSL -------------------------------------------------------------------

def test_expressions_resolve_against_moments():
    assert st.eval_moment_expr("strike-40", GBERCH) == 331
    assert st.eval_moment_expr("impact+5", GBERCH) == 407
    assert st.eval_moment_expr("line_cross", GBERCH) == 394


@pytest.mark.parametrize("bad", [330, "330", "strik-4", "strike*2", True])
def test_literals_and_unknown_moments_are_errors(bad):
    with pytest.raises(st.ShortsTemplateError):
        st.check_moment_expr(bad, "x")


def test_unknown_moment_in_template_fails_at_load():
    with pytest.raises(st.ShortsTemplateError, match="unknown moment 'goal'"):
        st.validate_template(_mini(to="goal+5"))


def test_bare_frame_number_in_template_fails_at_load():
    with pytest.raises(st.ShortsTemplateError, match="bare frame number"):
        st.validate_template(_mini(**{"from": 330}))


def test_underivable_moment_is_an_error_at_resolve():
    with pytest.raises(st.ShortsTemplateError, match="could not be derived"):
        st.resolve_template(_mini(), {**GBERCH, "impact": None})


def test_roles_substitute_in_camera_and_rig():
    t = _mini(camera="eyes:@keeper", rig={"focus": "@scorer", "orbit_start_frame": "strike-11"})
    r = st.resolve_template(t, GBERCH)
    p = r["passes"][0]
    assert p["camera"] == "eyes:P005"
    assert p["rig"] == {"focus": "P006", "orbit_start_frame": 360}
    t = _mini(camera="goal:@goal_end")
    assert st.resolve_template(t, GBERCH)["passes"][0]["camera"] == "goal:left"


def test_unresolvable_role_is_an_error():
    with pytest.raises(st.ShortsTemplateError, match="@keeper"):
        st.resolve_template(_mini(camera="eyes:@keeper"), {**GBERCH, "keeper_pid": None})


def test_pass_window_is_cut_plus_pad_and_spec_shape():
    r = st.resolve_template(_mini(pad=[10, 3], time_stretch=3), GBERCH)
    p, seg = r["passes"][0], r["edl"]["segments"][0]
    assert p["frames"] == [321, 410] and p["time_stretch"] == 3 and p["vertical"] is True
    assert (seg["from"], seg["to"], seg["first_frame"], seg["stretch"]) == (331, 407, 321, 3)


def test_freeze_at_splits_the_cut_and_holds_the_first_half():
    r = st.resolve_template(_mini(**{"from": "strike-9", "to": "strike+22", "freeze_at": "strike+9",
                                     "freeze_s": 1.4, "time_stretch": 4}), GBERCH)
    a, b = r["edl"]["segments"]
    assert (a["from"], a["to"], a["hold_s"]) == (362, 380, 1.4)
    assert (b["from"], b["to"]) == (380, 393) and "hold_s" not in b
    with pytest.raises(st.ShortsTemplateError, match="freeze_at"):
        st.resolve_template(_mini(freeze_at="impact+50"), GBERCH)


# --- candidates + framing ---------------------------------------------------

def _two_candidates():
    return {"name": "t", "slots": [{"id": "a", "candidates": [
        {"camera": "drone", "from": "strike-40", "to": "strike"},
        {"camera": "chase", "from": "strike-40", "to": "strike"}]}]}


def _fc(reject_cameras):
    def check(spec, a, b, subj, excl, over):
        if spec["camera"] in reject_cameras:
            return FramingResult(False, (FramingFailure("subject_too_small", (a, b), "tiny"),))
        return FramingResult(True)
    return check


def test_first_passing_candidate_wins_and_rejections_are_recorded():
    r = st.resolve_template(_two_candidates(), GBERCH, framing_check=_fc({"drone"}))
    assert r["ok"] and r["passes"][0]["camera"] == "chase" and r["passes"][0]["id"] == "t_a_c1"
    slot = r["slots"][0]
    assert slot["chosen"] == 1 and slot["rejected"][0]["framing"]["failures"][0]["check"] == "subject_too_small"


def test_required_slot_with_no_passing_candidate_is_unresolved_optional_is_dropped():
    r = st.resolve_template(_two_candidates(), GBERCH, framing_check=_fc({"drone", "chase"}))
    assert not r["ok"] and r["slots"][0]["status"] == "unresolved" and r["edl"]["segments"] == []
    t = _two_candidates(); t["slots"][0]["optional"] = True
    r2 = st.resolve_template(t, GBERCH, framing_check=_fc({"drone", "chase"}))
    assert r2["ok"] and r2["slots"][0]["status"] == "dropped"


def test_operator_candidate_pin_skips_framing_check():
    r = st.resolve_template(_two_candidates(), GBERCH, framing_check=_fc({"drone"}),
                            candidate_pins={"a": 0})
    assert r["passes"][0]["camera"] == "drone"


def test_framing_limit_overrides_reach_the_callback():
    seen = {}
    def check(spec, a, b, subj, excl, over):
        seen.update(over=over, excl=excl, subj=subj); return None
    t = _mini(camera="eyes:@keeper", framing={"min_subject_px": 90}, subject="@keeper")
    st.resolve_template(t, GBERCH, framing_check=check)
    assert seen == {"over": {"min_subject_px": 90}, "excl": ["P005"], "subj": "P005"}


# --- shipped templates -------------------------------------------------------

def test_shipped_templates_are_free_of_frame_literals():
    for p in (REPO / "config/shorts/templates").glob("*.yaml"):
        for line in p.read_text().splitlines():
            assert not re.match(r"^\s*(from|to|freeze_at):\s*[0-9]+\s*$", line), (p.name, line)


@pytest.mark.parametrize("name", ["matchday", "keeper", "comic"])
def test_shipped_templates_resolve_on_gberch_moments(name):
    r = st.resolve_template(st.load_template(name), GBERCH)
    assert r["ok"] and r["passes"] and all(p["frames"][0] >= 0 for p in r["passes"])
    edl = st.fill_sources(r["edl"], lambda pid: f"/x/{pid}.mp4")
    assert all(s["src"].startswith("/x/") and "pass" not in s for s in edl["segments"])
    edl_time_map(edl)   # compositor-compatible


def _plan(name, drop_first=False):
    r = st.resolve_template(st.load_template(name), GBERCH)
    segs = [(s["first_frame"], s["stretch"], s["from"], s["to"]) for s in r["edl"]["segments"]]
    return segs[1:] if drop_first else segs


def _hand(name, drop_first=False):
    edl = yaml.safe_load((REPO / f"config/shorts/gberch_{name}.yaml").read_text())
    segs = [(s["first_frame"], s.get("stretch", 1), s["from"], s["to"]) for s in edl["segments"]]
    return segs[1:] if drop_first else segs


def test_matchday_reproduces_hand_segments_after_the_build_up():
    # the hand drone opener started at frame 232 (a hand-picked frame); the
    # derived one starts at the possession chain/default build-up instead.
    assert _plan("matchday", True) == _hand("matchday", True)


def test_keeper_reproduces_hand_segments_exactly():
    assert _plan("keeper") == _hand("keeper")


def test_comic_reproduces_hand_segments_after_the_build_up():
    # hand drone: 236..335; derived: strike-135..strike-36 = 236..335 exactly
    assert _plan("comic") == _hand("comic")


def test_keeper_freeze_caption_lands_on_the_hold():
    r = st.resolve_template(st.load_template("keeper"), GBERCH)
    cap = [c for c in r["edl"]["captions"] if c["text"].startswith("Which way")][0]
    tm = edl_time_map({**r["edl"], "segments": [dict(s, src="x") for s in r["edl"]["segments"]]})
    freeze_seg = next(i for i, s in enumerate(r["edl"]["segments"]) if s.get("freeze"))
    assert cap["end"] == pytest.approx(tm[freeze_seg]["end_s"])
    assert cap["end"] - cap["start"] == pytest.approx(1.4)


def test_hand_labels_and_flash_survive():
    r = st.resolve_template(st.load_template("keeper"), GBERCH)
    segs = r["edl"]["segments"]
    assert segs[1].get("flash") and segs[1]["label"] == "KEEPER CAM" and segs[1]["label_style"] == "chip_dark"
