"""Stages must skip excluded shots (manifest.active_shots())."""
from pathlib import Path

from src.schemas.shots import Shot, ShotsManifest
from src.stages.tracking import PlayerTrackingStage


def _write_manifest(out: Path) -> None:
    (out / "shots").mkdir(parents=True, exist_ok=True)
    shots = [
        Shot("a", 0, 9, 0.0, 0.4, "shots/a.mp4"),
        Shot("b", 0, 9, 0.0, 0.4, "shots/b.mp4", excluded=True,
             exclude_reason="reaction", kind="reaction"),
    ]
    ShotsManifest("reel.mp4", 25.0, 20, shots).save(
        out / "shots" / "shots_manifest.json")


class _StubResult:
    tracks: list = []

    def save(self, path: Path) -> None:
        Path(path).write_text("{}")


def test_tracking_runs_only_active_shots(tmp_path, monkeypatch):
    out = tmp_path
    _write_manifest(out)
    stage = PlayerTrackingStage(
        config={"tracking": {"team_classifier": "none"}},
        output_dir=out,
        player_detector=object(),  # never used: _track_shot is stubbed
    )
    processed: list[str] = []

    def fake_track(shot_id, clip_file, *a, **k):
        processed.append(shot_id)
        return _StubResult()

    monkeypatch.setattr(stage, "_track_shot", fake_track)
    stage.run()
    assert processed == ["a"]


def test_tracking_is_complete_ignores_excluded(tmp_path):
    out = tmp_path
    _write_manifest(out)
    (out / "tracks").mkdir()
    (out / "tracks" / "a_tracks.json").write_text("{}")
    # no b_tracks.json — but b is excluded, so the stage is complete
    stage = PlayerTrackingStage(config={"tracking": {}}, output_dir=out)
    assert stage.is_complete() is True
