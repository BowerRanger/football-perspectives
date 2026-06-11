"""Tests for the group-scoped shot-sync schema and dashboard endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.schemas.sync_map import (
    Alignment,
    GroupSync,
    SyncMap,
    default_group_sync,
    validate_method,
)
from src.web.server import create_app


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(output_dir=tmp_path, config_path=None)
    return TestClient(app), tmp_path


def _write_manifest(
    tmp_path: Path,
    shot_ids: list[str],
    group_ids: dict[str, str] | None = None,
    groups: list[dict] | None = None,
) -> None:
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    shots = [
        {"id": sid, "start_frame": 0, "end_frame": 0,
         "start_time": 0.0, "end_time": 0.0,
         "clip_file": f"shots/{sid}.mp4",
         "group_id": (group_ids or {}).get(sid, "")}
        for sid in shot_ids
    ]
    (shots_dir / "shots_manifest.json").write_text(json.dumps({
        "source_file": "x", "fps": 30, "total_frames": 0,
        "shots": shots, "groups": groups or [],
    }))


# ── Schema unit tests ────────────────────────────────────────────────


@pytest.mark.unit
def test_v2_round_trip(tmp_path: Path) -> None:
    sm = SyncMap(groups=[
        GroupSync(group_id="g01", reference_shot="s001",
                  alignments=[Alignment("s001", 0),
                              Alignment("s002", 37, "motion_profile", 0.8)]),
    ])
    p = tmp_path / "sync_map.json"
    sm.save(p)
    loaded = SyncMap.load(p)
    assert loaded.version == 2
    assert loaded.groups[0].alignments[1].frame_offset == 37
    assert loaded.offset_for("g01", "s002") == 37
    assert loaded.offset_for("g01", "missing") == 0
    assert loaded.offset_for("nope", "s002") == 0


@pytest.mark.unit
def test_v1_flat_file_migrates_to_ungrouped(tmp_path: Path) -> None:
    p = tmp_path / "sync_map.json"
    p.write_text(json.dumps({
        "reference_shot": "a",
        "alignments": [
            {"shot_id": "a", "frame_offset": 0,
             "method": "manual", "confidence": 1.0},
            {"shot_id": "b", "frame_offset": -4,
             "method": "manual", "confidence": 1.0},
        ],
    }))
    sm = SyncMap.load(p)
    assert sm.version == 2
    assert sm.groups[0].group_id == ""
    assert sm.groups[0].reference_shot == "a"
    assert sm.offset_for("", "b") == -4


@pytest.mark.unit
def test_with_group_alignment_upserts() -> None:
    sm = SyncMap()
    sm2 = sm.with_group_alignment("g01", "ref", Alignment("x", 5))
    assert sm2.offset_for("g01", "x") == 5
    assert sm.groups == []  # original untouched
    sm3 = sm2.with_group_alignment("g01", "ref", Alignment("x", 9))
    assert sm3.offset_for("g01", "x") == 9
    assert len(sm3.group("g01").alignments) == 1


@pytest.mark.unit
def test_default_group_sync_zeroes_every_shot() -> None:
    gs = default_group_sync("g01", "alpha", ["beta", "alpha"])
    assert gs.group_id == "g01"
    assert gs.reference_shot == "alpha"
    assert {a.shot_id for a in gs.alignments} == {"alpha", "beta"}
    assert all(a.frame_offset == 0 for a in gs.alignments)
    assert all(a.method == "manual" for a in gs.alignments)


@pytest.mark.unit
def test_validate_method_accepts_motion_profile_rejects_unknown() -> None:
    assert validate_method("manual") == "manual"
    assert validate_method("motion_profile") == "motion_profile"
    with pytest.raises(ValueError, match="unknown sync method"):
        validate_method("guesswork")


# ── /api/sync endpoint tests ─────────────────────────────────────────


@pytest.mark.integration
def test_get_sync_returns_default_when_file_missing(client) -> None:
    c, _ = client
    body = c.get("/api/sync").json()
    assert body == {"version": 2, "groups": []}


@pytest.mark.integration
def test_get_sync_seeds_ungrouped_from_manifest(client) -> None:
    c, tmp_path = client
    _write_manifest(tmp_path, ["origi01", "origi02"])
    body = c.get("/api/sync").json()
    assert len(body["groups"]) == 1
    g = body["groups"][0]
    assert g["group_id"] == ""
    assert g["reference_shot"] == "origi01"
    assert sorted(a["shot_id"] for a in g["alignments"]) == [
        "origi01", "origi02"]
    assert all(a["frame_offset"] == 0 for a in g["alignments"])


@pytest.mark.integration
def test_get_sync_seeds_manifest_groups(client) -> None:
    c, tmp_path = client
    _write_manifest(
        tmp_path, ["s001", "s002", "s003"],
        group_ids={"s001": "g01", "s002": "g01", "s003": "g02"},
        groups=[
            {"id": "g01", "label": "Highlight 1",
             "shot_ids": ["s001", "s002"]},
            {"id": "g02", "label": "Highlight 2", "shot_ids": ["s003"]},
        ],
    )
    body = c.get("/api/sync").json()
    by_gid = {g["group_id"]: g for g in body["groups"]}
    assert set(by_gid) == {"g01", "g02"}
    assert {a["shot_id"] for a in by_gid["g01"]["alignments"]} == {
        "s001", "s002"}
    assert by_gid["g01"]["reference_shot"] == "s001"


@pytest.mark.integration
def test_post_sync_round_trips_per_group(client) -> None:
    c, tmp_path = client
    _write_manifest(
        tmp_path, ["s001", "s002"],
        group_ids={"s001": "g01", "s002": "g01"},
        groups=[{"id": "g01", "label": "Highlight 1",
                 "shot_ids": ["s001", "s002"]}],
    )
    payload = {
        "group_id": "g01",
        "reference_shot": "s001",
        "alignments": [
            {"shot_id": "s001", "frame_offset": 0, "method": "manual",
             "confidence": 1.0},
            {"shot_id": "s002", "frame_offset": 1240, "method": "manual",
             "confidence": 1.0},
        ],
    }
    r = c.post("/api/sync", json=payload)
    assert r.status_code == 200
    assert r.json()["count"] == 2

    saved = c.get("/api/sync").json()
    g = next(g for g in saved["groups"] if g["group_id"] == "g01")
    by_id = {a["shot_id"]: a["frame_offset"] for a in g["alignments"]}
    assert by_id == {"s001": 0, "s002": 1240}


@pytest.mark.integration
def test_post_sync_rejects_non_zero_reference_offset(client) -> None:
    c, tmp_path = client
    _write_manifest(tmp_path, ["origi01"])
    r = c.post("/api/sync", json={
        "group_id": "",
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
        "group_id": "",
        "reference_shot": "origi01",
        "alignments": [
            {"shot_id": "origi01", "frame_offset": 0},
            {"shot_id": "phantom", "frame_offset": 100},
        ],
    })
    assert r.status_code == 400
    assert "phantom" in r.json()["detail"]


@pytest.mark.integration
def test_post_sync_preserves_other_groups(client) -> None:
    c, tmp_path = client
    _write_manifest(
        tmp_path, ["s001", "s002", "s003"],
        group_ids={"s001": "g01", "s002": "g01", "s003": "g02"},
        groups=[
            {"id": "g01", "label": "Highlight 1",
             "shot_ids": ["s001", "s002"]},
            {"id": "g02", "label": "Highlight 2", "shot_ids": ["s003"]},
        ],
    )
    for gid, sids in (("g01", ["s001", "s002"]), ("g02", ["s003"])):
        c.post("/api/sync", json={
            "group_id": gid,
            "reference_shot": sids[0],
            "alignments": [
                {"shot_id": sid, "frame_offset": 0} for sid in sids
            ],
        })
    saved = c.get("/api/sync").json()
    assert {g["group_id"] for g in saved["groups"]} == {"g01", "g02"}


@pytest.mark.integration
def test_shots_manifest_endpoint(client) -> None:
    """The dashboard's sync timeline reads ``/api/shots/manifest`` to
    size each clip block by its actual frame count."""
    c, tmp_path = client
    body = c.get("/api/shots/manifest").json()
    assert body["shots"] == []
    _write_manifest(tmp_path, ["alpha", "beta"])
    body = c.get("/api/shots/manifest").json()
    ids = [s["id"] for s in body["shots"]]
    assert ids == ["alpha", "beta"]
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
        "group_id": "",
        "reference_shot": "alpha",
        "alignments": [{"shot_id": "alpha", "frame_offset": 0}],
    })
    _write_manifest(tmp_path, ["alpha", "beta"])
    saved = c.get("/api/sync").json()
    g = next(g for g in saved["groups"] if g["group_id"] == "")
    by_id = {a["shot_id"]: a["frame_offset"] for a in g["alignments"]}
    assert by_id == {"alpha": 0, "beta": 0}
