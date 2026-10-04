"""Stage fingerprints / freshness (D4)."""

import json
from pathlib import Path

import pytest

from src.pipeline import fingerprint as fp


def _write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


@pytest.fixture
def out(tmp_path: Path) -> Path:
    _write(tmp_path / "shots" / "shots_manifest.json", "{}")
    _write(tmp_path / "camera" / "s1_camera_track.json", "{}")
    _write(tmp_path / "refined_poses" / "P001_refined.npz", "x")
    _write(tmp_path / "ball" / "s1_ball_anchors.json", '{"anchors": []}')
    return tmp_path


CFG = {"ball": {"trajectory": "hybrid"}}


@pytest.mark.unit
def test_hash_file_small_and_large_are_stable(tmp_path: Path) -> None:
    small = _write(tmp_path / "a.json", '{"b": 1, "a": 2}')
    assert fp.hash_file(small) == fp.hash_file(small)
    big = tmp_path / "big.npz"
    big.write_bytes(b"a" * (fp.BIG_FILE_BYTES + 10))
    h1 = fp.hash_file(big)
    big.write_bytes(b"a" * (fp.BIG_FILE_BYTES + 11))
    assert fp.hash_file(big) != h1


@pytest.mark.unit
def test_json_hash_ignores_formatting(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.json", '{"a": 1, "b": [1, 2]}')
    b = _write(tmp_path / "b.json", '{\n "b": [1,2],\n "a": 1}')
    assert fp.hash_file(a) == fp.hash_file(b)


@pytest.mark.unit
def test_record_then_fresh(out: Path) -> None:
    fp.record(out, "ball", CFG)
    res = fp.assess(out, "ball", CFG)
    assert res.state == "fresh" and not res.code_drift


@pytest.mark.unit
def test_legacy_without_state_is_unknown(out: Path) -> None:
    res = fp.assess(out, "ball", CFG)
    assert res.state == "unknown"
    assert not (out / "pipeline_state.json").exists()


@pytest.mark.unit
def test_operator_anchor_edit_makes_ball_stale(out: Path) -> None:
    fp.record(out, "ball", CFG)
    _write(out / "ball" / "s1_ball_anchors.json", '{"anchors": [1]}')
    res = fp.assess(out, "ball", CFG)
    assert res.state == "stale"
    assert any("s1_ball_anchors.json" in r for r in res.reasons)


@pytest.mark.unit
def test_config_change_makes_stage_stale(out: Path) -> None:
    fp.record(out, "ball", CFG)
    res = fp.assess(out, "ball", {"ball": {"trajectory": "reference"}})
    assert res.state == "stale" and any("config" in r for r in res.reasons)


@pytest.mark.unit
def test_unrelated_config_does_not_stale(out: Path) -> None:
    fp.record(out, "ball", CFG)
    res = fp.assess(out, "ball", {**CFG, "render": {"x": 1}})
    assert res.state == "fresh"


@pytest.mark.unit
def test_upstream_output_change_makes_stage_stale(out: Path) -> None:
    fp.record(out, "ball", CFG)
    _write(out / "refined_poses" / "P001_refined.npz", "changed!")
    res = fp.assess(out, "ball", CFG)
    assert res.state == "stale"


@pytest.mark.unit
def test_code_only_change_is_drift_not_stale(out: Path, monkeypatch) -> None:
    fp.record(out, "ball", CFG)
    monkeypatch.setattr(fp, "code_hash", lambda stage: "different")
    res = fp.assess(out, "ball", CFG)
    assert res.state == "fresh" and res.code_drift


@pytest.mark.unit
def test_inputs_added_counts_as_change(out: Path) -> None:
    fp.record(out, "ball", CFG)
    _write(out / "ball" / "s2_ball_anchors.json", "{}")
    assert fp.assess(out, "ball", CFG).state == "stale"


@pytest.mark.unit
def test_corrupt_state_is_unknown(out: Path) -> None:
    (out / "pipeline_state.json").write_text("{nope")
    assert fp.assess(out, "ball", CFG).state == "unknown"


@pytest.mark.unit
def test_downstream_of_follows_table() -> None:
    down = fp.downstream_of("ball")
    assert "export" in down and "render" in down and "tracking" not in down


@pytest.mark.unit
def test_all_registered_stages_have_table_entries() -> None:
    from src.pipeline.runner import _STAGE_NAMES

    assert set(_STAGE_NAMES) <= set(fp.STAGE_TABLE)


@pytest.mark.unit
def test_state_file_shape(out: Path) -> None:
    fp.record(out, "ball", CFG)
    data = json.loads((out / "pipeline_state.json").read_text())
    rec = data["stages"]["ball"]
    assert {"config", "inputs", "code", "recorded_at"} <= set(rec)
