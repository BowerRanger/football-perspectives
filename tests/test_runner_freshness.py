"""Runner freshness behaviour: [STALE] re-runs, code drift warn, --stale."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.pipeline import fingerprint as fp
from src.pipeline import runner
from src.pipeline.base import BaseStage

CFG = {"ball": {"trajectory": "hybrid"}}


class _Ball(BaseStage):
    name = "ball"
    runs = 0

    def is_complete(self) -> bool:
        return True

    def run(self) -> None:
        _Ball.runs += 1


class _Export(_Ball):
    name = "export"
    export_runs = 0

    def run(self) -> None:
        _Export.export_runs += 1


@pytest.fixture(autouse=True)
def _patch(monkeypatch, tmp_path):
    _Ball.runs = 0
    _Export.export_runs = 0
    monkeypatch.setattr(
        runner, "_stage_class", lambda n: _Ball if n == "ball" else _Export)
    monkeypatch.setattr(runner, "write_quality_report", lambda out: None)
    (tmp_path / "ball").mkdir()
    (tmp_path / "ball" / "s1_ball_anchors.json").write_text("{}")


def _go(tmp_path, stages="ball,export", **kw):
    runner.run_pipeline(tmp_path, stages, None, CFG, **kw)


@pytest.mark.unit
def test_legacy_dir_is_cached_not_stale(tmp_path, capsys) -> None:
    _go(tmp_path)
    assert _Ball.runs == 0
    assert "[SKIP] ball (cached)" in capsys.readouterr().out
    assert not (tmp_path / "pipeline_state.json").exists()


@pytest.mark.unit
def test_anchor_edit_reruns_ball_with_stale_line(tmp_path, capsys) -> None:
    fp.record(tmp_path, "ball", CFG)
    fp.record(tmp_path, "export", CFG)
    (tmp_path / "ball" / "s1_ball_anchors.json").write_text('{"a": 1}')
    _go(tmp_path)
    out = capsys.readouterr().out
    assert "[STALE] ball (input changed: ball/s1_ball_anchors.json)" in out
    assert _Ball.runs == 1
    assert fp.assess(tmp_path, "ball", CFG).state == "fresh"


@pytest.mark.unit
def test_code_only_change_warns_and_skips(tmp_path, capsys, monkeypatch) -> None:
    fp.record(tmp_path, "ball", CFG)
    monkeypatch.setattr(fp, "code_hash", lambda s: "new")
    _go(tmp_path, stages="ball")
    out = capsys.readouterr().out
    assert _Ball.runs == 0
    assert "[WARN] ball: stage code changed" in out


@pytest.mark.unit
def test_stale_flag_reruns_code_drift_and_cascades(tmp_path, monkeypatch) -> None:
    fp.record(tmp_path, "ball", CFG)
    fp.record(tmp_path, "export", CFG)
    real = fp.code_hash
    monkeypatch.setattr(fp, "code_hash", lambda s: "new" if s == "ball" else real(s))
    _go(tmp_path, stale=True)
    assert _Ball.runs == 1
    assert _Export.export_runs == 1  # cascaded from ball


@pytest.mark.unit
def test_filtered_run_does_not_record_fingerprint(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "_known_shot_ids", lambda out: ["s1"])
    _go(tmp_path, stages="ball", shots=["s1"])
    assert _Ball.runs == 1
    assert not (tmp_path / "pipeline_state.json").exists()


@pytest.mark.unit
def test_shots_flag_skips_other_shots(tmp_path, monkeypatch) -> None:
    seen: list = []

    class _Rec(_Ball):
        def run(self) -> None:
            seen.append(self.shot_filter)

    monkeypatch.setattr(runner, "_stage_class", lambda n: _Rec)
    monkeypatch.setattr(runner, "_known_shot_ids", lambda out: ["a", "b"])
    runner.run_pipeline(tmp_path, "ball", None, CFG, shots=["b"])
    assert seen == ["b"]


@pytest.mark.unit
def test_stage_status_rows(tmp_path) -> None:
    rows = {r["stage"]: r for r in runner.stage_status(tmp_path, CFG)}
    assert rows["ball"]["completeness"] == "complete"
    assert rows["ball"]["freshness"] == "unknown"
    fp.record(tmp_path, "ball", CFG)
    (tmp_path / "ball" / "s1_ball_anchors.json").write_text('{"a": 1}')
    rows = {r["stage"]: r for r in runner.stage_status(tmp_path, CFG)}
    assert rows["ball"]["freshness"] == "stale"
    assert rows["ball"]["reasons"]


@pytest.mark.unit
def test_cli_status_runs(tmp_path) -> None:
    import recon

    res = CliRunner().invoke(recon.cli, ["status", "--output", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "prepare_shots" in res.output and "shorts" in res.output
    json.dumps(res.output)  # plain text, no exceptions


@pytest.mark.unit
def test_cli_run_rejects_unknown_shot(tmp_path, monkeypatch) -> None:
    import recon

    monkeypatch.setattr(runner, "_known_shot_ids", lambda out: ["a"])
    res = CliRunner().invoke(
        recon.cli,
        ["run", "--output", str(tmp_path), "--stages", "ball", "--shots", "zzz"])
    assert res.exit_code != 0 and "zzz" in res.output
