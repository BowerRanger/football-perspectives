"""Tests for the manual shot-sync schema and dashboard endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.schemas.sync_map import (
    Alignment,
    SyncMap,
    default_sync_map,
    validate_method,
)
from src.web.server import create_app


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(output_dir=tmp_path, config_path=None)
    return TestClient(app), tmp_path


def _write_manifest(tmp_path: Path, shot_ids: list[str]) -> None:
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    shots_json = ",".join(
        '{"id":"' + sid + '","start_frame":0,"end_frame":0,'
        '"start_time":0.0,"end_time":0.0,"clip_file":"shots/' + sid + '.mp4"}'
        for sid in shot_ids
    )
    (shots_dir / "shots_manifest.json").write_text(
        '{"source_file":"x","fps":30,"total_frames":0,"shots":['
        + shots_json + "]}"
    )


@pytest.mark.unit
def test_sync_map_round_trips(tmp_path: Path) -> None:
    sm = SyncMap(
        reference_shot="origi01",
        alignments=[
            Alignment(shot_id="origi01", frame_offset=0),
            Alignment(shot_id="origi02", frame_offset=1240),
        ],
    )
    path = tmp_path / "sync_map.json"
    sm.save(path)
    loaded = SyncMap.load(path)
    assert loaded.reference_shot == "origi01"
    assert loaded.offset_for("origi02") == 1240
    assert loaded.offset_for("origi01") == 0
    assert loaded.offset_for("missing") == 0


@pytest.mark.unit
def test_default_sync_map_zeroes_every_shot() -> None:
    sm = default_sync_map(
        reference_shot="alpha", shot_ids=["beta", "alpha"],
    )
    assert sm.reference_shot == "alpha"
    assert {a.shot_id for a in sm.alignments} == {"alpha", "beta"}
    assert all(a.frame_offset == 0 for a in sm.alignments)
    assert all(a.method == "manual" for a in sm.alignments)


@pytest.mark.unit
def test_with_alignment_upserts_by_shot_id() -> None:
    sm = SyncMap(
        reference_shot="a",
        alignments=[
            Alignment(shot_id="a", frame_offset=0),
            Alignment(shot_id="b", frame_offset=10),
        ],
    )
    out = sm.with_alignment(Alignment(shot_id="b", frame_offset=42))
    assert out.offset_for("b") == 42
    assert len(out.alignments) == 2  # b replaced, not duplicated


@pytest.mark.unit
def test_validate_method_rejects_unknown() -> None:
    assert validate_method("manual") == "manual"
    with pytest.raises(ValueError, match="unknown sync method"):
        validate_method("guesswork")


@pytest.mark.integration
def test_get_sync_returns_default_when_file_missing(client) -> None:
    """Empty/no manifest → returns ``{reference_shot: '', alignments: []}``."""
    c, _ = client
    body = c.get("/api/sync").json()
    assert body == {"reference_shot": "", "alignments": []}


@pytest.mark.integration
def test_get_sync_seeds_default_from_manifest(client) -> None:
    c, tmp_path = client
    _write_manifest(tmp_path, ["origi01", "origi02"])
    body = c.get("/api/sync").json()
    assert body["reference_shot"] == "origi01"
    assert sorted(a["shot_id"] for a in body["alignments"]) == ["origi01", "origi02"]
    assert all(a["frame_offset"] == 0 for a in body["alignments"])


@pytest.mark.integration
def test_post_sync_round_trips(client) -> None:
    c, tmp_path = client
    _write_manifest(tmp_path, ["origi01", "origi02"])
    payload = {
        "reference_shot": "origi01",
        "alignments": [
            {"shot_id": "origi01", "frame_offset": 0, "method": "manual",
             "confidence": 1.0},
            {"shot_id": "origi02", "frame_offset": 1240, "method": "manual",
             "confidence": 1.0},
        ],
    }
    r = c.post("/api/sync", json=payload)
    assert r.status_code == 200
    assert r.json()["count"] == 2

    saved = c.get("/api/sync").json()
    assert saved["reference_shot"] == "origi01"
    by_id = {a["shot_id"]: a["frame_offset"] for a in saved["alignments"]}
    assert by_id == {"origi01": 0, "origi02": 1240}


@pytest.mark.integration
def test_post_sync_rejects_non_zero_reference_offset(client) -> None:
    c, tmp_path = client
    _write_manifest(tmp_path, ["origi01"])
    r = c.post("/api/sync", json={
        "reference_shot": "origi01",
        "alignments": [{"shot_id": "origi01", "frame_offset": 5}],
    })
    assert r.status_code == 400
    assert "frame_offset=0" in r.json()["detail"]


@pytest.mark.integration
def test_post_sync_rejects_unknown_shot_id(client) -> None:
    c, tmp_path = client
    _write_manifest(tmp_path, ["origi01"])
    r = c.post("/api/sync", json={
        "reference_shot": "origi01",
        "alignments": [
            {"shot_id": "origi01", "frame_offset": 0},
            {"shot_id": "phantom", "frame_offset": 100},
        ],
    })
    assert r.status_code == 400
    assert "phantom" in r.json()["detail"]


@pytest.mark.integration
def test_shots_manifest_endpoint(client) -> None:
    """The dashboard's sync timeline reads ``/api/shots/manifest`` to
    size each clip block by its actual frame count."""
    c, tmp_path = client
    # Empty case
    body = c.get("/api/shots/manifest").json()
    assert body["shots"] == []
    # With manifest
    _write_manifest(tmp_path, ["alpha", "beta"])
    body = c.get("/api/shots/manifest").json()
    ids = [s["id"] for s in body["shots"]]
    assert ids == ["alpha", "beta"]
    # Each shot row has the fields the timeline needs.
    for s in body["shots"]:
        assert "start_frame" in s and "end_frame" in s


@pytest.mark.integration
def test_get_sync_appends_new_shots_added_after_save(client) -> None:
    """Adding a shot to the manifest after a sync_map was saved should
    surface the new shot at offset=0 in subsequent GETs without forcing
    the operator to manually re-save first."""
    c, tmp_path = client
    _write_manifest(tmp_path, ["alpha"])
    c.post("/api/sync", json={
        "reference_shot": "alpha",
        "alignments": [{"shot_id": "alpha", "frame_offset": 0}],
    })
    _write_manifest(tmp_path, ["alpha", "beta"])
    saved = c.get("/api/sync").json()
    by_id = {a["shot_id"]: a["frame_offset"] for a in saved["alignments"]}
    assert by_id == {"alpha": 0, "beta": 0}
