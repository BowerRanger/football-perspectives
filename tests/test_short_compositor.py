"""short_compositor: ffmpeg command-list parity with the original
scripts/compose_short.py (golden captured before the move), the output
time map, and the optional audio input."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from src.utils import short_compositor as sc

REPO = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((REPO / "tests/fixtures/shorts/compose_cmds_golden.json").read_text())
_WORK_RE = re.compile(r"/[^ ]*?/(seg_\d+|concat|body|cap_\d+|label_\d+)\.(mp4|txt|png)")


def _record(monkeypatch, edl, tmp_path, **kw):
    rec: list[list[str]] = []
    monkeypatch.setattr(sc, "_run", lambda cmd: rec.append(cmd))
    monkeypatch.setattr(sc, "_probe_frames", lambda p: 99999)
    sc.compose(edl, tmp_path / "out.mp4", **kw)
    return [[_WORK_RE.sub(r"<WORK>/\1.\2", a) for a in c] for c in rec]


@pytest.mark.parametrize("name", ["matchday", "keeper", "comic"])
def test_hand_edl_command_list_matches_pre_move_golden(name, monkeypatch, tmp_path):
    edl = yaml.safe_load((REPO / f"config/shorts/gberch_{name}.yaml").read_text())
    # compose() ends in a different out path than the golden run; ignore it.
    got = _record(monkeypatch, edl, tmp_path)
    want = GOLDEN[name]
    strip = lambda cmds: [[a for a in c if not a.endswith("out.mp4") and not a.startswith("/tmp/t4out")] for c in cmds]  # noqa: E731
    assert strip(got) == strip(want)


def test_edl_time_map_matches_segment_durations():
    edl = yaml.safe_load((REPO / "config/shorts/gberch_matchday.yaml").read_text())
    tm = sc.edl_time_map(edl)
    assert len(tm) == 4
    assert tm[0]["start_s"] == 0.0
    # seg0: 111 frames @30 fps, no stretch
    assert tm[0]["end_s"] == pytest.approx(111 / 30)
    # seg3: 34 scene frames x3 stretch /30 + 1.6 s hold
    assert tm[3]["end_s"] - tm[3]["start_s"] == pytest.approx(34 * 3 / 30 + 1.6)
    assert tm[1]["start_s"] == pytest.approx(tm[0]["end_s"])


def test_scene_frame_to_out_s_single_appearance():
    edl = yaml.safe_load((REPO / "config/shorts/gberch_matchday.yaml").read_text())
    hits = sc.scene_frame_to_out_s(edl, 400)  # only goal_slow [374,408)
    tm = sc.edl_time_map(edl)
    assert len(hits) == 1
    assert hits[0] == pytest.approx(tm[3]["start_s"] + (400 - 374) * 3 / 30)


def test_scene_frame_in_two_replays():
    edl = yaml.safe_load((REPO / "config/shorts/gberch_matchday.yaml").read_text())
    # 376: chase [343,396), orbit_slow [362,380), goal_slow [374,408)
    assert len(sc.scene_frame_to_out_s(edl, 376)) == 3


def test_audio_input_replaces_anullsrc(monkeypatch, tmp_path):
    edl = yaml.safe_load((REPO / "config/shorts/gberch_comic.yaml").read_text())
    cmds = _record(monkeypatch, edl, tmp_path, audio=tmp_path / "mix.wav")
    final = cmds[-1]
    assert str(tmp_path / "mix.wav") in final
    assert not any("anullsrc" in a for a in final)
    cmds0 = _record(monkeypatch, edl, tmp_path)
    assert any("anullsrc" in a for a in cmds0[-1])
