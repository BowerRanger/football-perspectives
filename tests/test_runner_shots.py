"""Stage registration (appearance/shorts) and ``--shots`` scoping."""

import pytest

from src.pipeline import runner
from src.pipeline.base import BaseStage


@pytest.mark.unit
def test_new_stages_registered_in_order() -> None:
    names = runner.resolve_stages("all", None)
    assert names.index("ball") < names.index("appearance") < names.index("export")
    assert names.index("render") < names.index("shorts")
    assert names[-1] == "shorts"


@pytest.mark.unit
def test_missing_stage_module_is_not_implemented_not_crash() -> None:
    # Modules are written by other tasks; absence must yield None.
    for name in ("appearance", "shorts"):
        cls = runner._stage_class(name)
        assert cls is None or issubclass(cls, BaseStage)


class _Recorder(BaseStage):
    name = "ball"
    calls: list = []

    def run(self) -> None:
        _Recorder.calls.append(self.shot_filter)


@pytest.mark.unit
def test_shots_loops_one_shot_at_a_time(tmp_path, monkeypatch) -> None:
    _Recorder.calls = []
    monkeypatch.setattr(runner, "_stage_class", lambda n: _Recorder)
    monkeypatch.setattr(runner, "_known_shot_ids", lambda out: ["a", "b", "c"])
    monkeypatch.setattr(runner, "write_quality_report", lambda out: None)
    runner.run_pipeline(tmp_path, "ball", None, {}, shots=["a", "c"])
    assert _Recorder.calls == ["a", "c"]


@pytest.mark.unit
def test_shots_unknown_id_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "_known_shot_ids", lambda out: ["a"])
    with pytest.raises(ValueError, match="zzz"):
        runner.run_pipeline(tmp_path, "ball", None, {}, shots=["zzz"])
