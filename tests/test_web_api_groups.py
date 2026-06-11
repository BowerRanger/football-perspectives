"""Group-aware shot editing API: bulk PATCH, auto-align, features, thumbs."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.stages.prepare_shots import PrepareShotsStage
from src.web.server import create_app
from tests.fixtures.synthetic_reel import build_reel

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="ffmpeg not on PATH")

CFG = {"prepare_shots": {
    "mode": "split",
    "split": {"detector": "content", "threshold": 27.0,
              "min_scene_len_frames": 8, "min_shot_duration_s": 1.0,
              "min_input_duration_s": 5,
              "merge_max_gap_s": 0.08,
              "merge_short_shots_max_duration_s": 0.6},
    "classify": {"sample_points": [0.2, 0.5, 0.8],
                 "replay_min_speed_factor": 1.25},
    "group": {"gap_boundary_s": 5.0},
    "align": {"enabled": True, "curve_width_px": 96,
              "smooth_sigma_frames": 2.0, "min_overlap_s": 1.0,
              "min_confidence": 0.5},
}}

SEGMENTS = [("green", 3.0), ("crowd", 2.0), ("green_slow", 3.0),
            ("black", 1.2), ("green", 3.0)]
# Produces: s001 live | s002 reaction (excluded) | s003 replay |
#           s004 transition (excluded) | s005 live
# Groups: g01=[s001,s003], g02=[s005]


@pytest.fixture()
def client(tmp_path: Path):
    reel = tmp_path / "reel.mp4"
    build_reel(reel, SEGMENTS)
    PrepareShotsStage(config=CFG, output_dir=tmp_path,
                      video_path=reel).run()
    app = create_app(output_dir=tmp_path, config_path=None)
    return TestClient(app), tmp_path


@pytest.mark.integration
def test_bulk_patch_discard_and_restore(client) -> None:
    c, _ = client
    r = c.patch("/api/shots/bulk", json={"updates": [
        {"shot_id": "s001", "excluded": True, "exclude_reason": "manual"}]})
    assert r.status_code == 200, r.text
    m = c.get("/api/shots/manifest").json()
    s = next(x for x in m["shots"] if x["id"] == "s001")
    assert s["excluded"] is True and s["exclude_reason"] == "manual"

    r = c.patch("/api/shots/bulk", json={"updates": [
        {"shot_id": "s001", "excluded": False, "exclude_reason": ""}]})
    assert r.status_code == 200
    m = c.get("/api/shots/manifest").json()
    s = next(x for x in m["shots"] if x["id"] == "s001")
    assert s["excluded"] is False


@pytest.mark.integration
def test_bulk_patch_move_shot_reconciles_groups(client) -> None:
    c, _ = client
    r = c.patch("/api/shots/bulk", json={"updates": [
        {"shot_id": "s003", "group_id": "g02"}]})
    assert r.status_code == 200, r.text
    m = r.json()
    by_id = {g["id"]: g["shot_ids"] for g in m["groups"]}
    assert "s003" in by_id["g02"]
    assert "s003" not in by_id.get("g01", [])
    # sync map prunes the moved shot from its old group
    sync = c.get("/api/sync").json()
    g01 = next((g for g in sync["groups"] if g["group_id"] == "g01"), None)
    if g01 is not None:
        assert all(a["shot_id"] != "s003" for a in g01["alignments"]
                   if a["shot_id"] == "s003")


@pytest.mark.integration
def test_bulk_patch_new_group_id_creates_group(client) -> None:
    c, _ = client
    r = c.patch("/api/shots/bulk", json={"updates": [
        {"shot_id": "s005", "group_id": "g99"}]})
    assert r.status_code == 200
    m = r.json()
    g99 = next(g for g in m["groups"] if g["id"] == "g99")
    assert g99["shot_ids"] == ["s005"]
    assert g99["boundary_rule"] == "manual"
    # old singleton group g02 emptied -> dropped
    assert all(g["id"] != "g02" for g in m["groups"])


@pytest.mark.integration
def test_bulk_patch_unknown_shot_400(client) -> None:
    c, _ = client
    r = c.patch("/api/shots/bulk", json={"updates": [
        {"shot_id": "nope", "excluded": True}]})
    assert r.status_code == 400


@pytest.mark.integration
def test_bulk_patch_bad_group_id_400(client) -> None:
    c, _ = client
    r = c.patch("/api/shots/bulk", json={"updates": [
        {"shot_id": "s001", "group_id": "bad id!"}]})
    assert r.status_code == 400


@pytest.mark.integration
def test_sync_auto_recomputes_group(client) -> None:
    c, _ = client
    r = c.post("/api/sync/auto", json={"group_id": "g01", "force": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["group_id"] == "g01"
    assert body["aligned"] >= 2
    sync = c.get("/api/sync").json()
    g01 = next(g for g in sync["groups"] if g["group_id"] == "g01")
    assert {a["shot_id"] for a in g01["alignments"]} == {"s001", "s003"}


@pytest.mark.integration
def test_sync_auto_preserves_manual_without_force(client) -> None:
    c, _ = client
    c.post("/api/sync", json={
        "group_id": "g01", "reference_shot": "s001",
        "alignments": [
            {"shot_id": "s001", "frame_offset": 0, "method": "manual"},
            {"shot_id": "s003", "frame_offset": 99, "method": "manual"},
        ],
    })
    r = c.post("/api/sync/auto", json={"group_id": "g01", "force": False})
    assert r.status_code == 200
    sync = c.get("/api/sync").json()
    g01 = next(g for g in sync["groups"] if g["group_id"] == "g01")
    s003 = next(a for a in g01["alignments"] if a["shot_id"] == "s003")
    assert s003["frame_offset"] == 99 and s003["method"] == "manual"


@pytest.mark.integration
def test_sync_auto_unknown_group_404(client) -> None:
    c, _ = client
    r = c.post("/api/sync/auto", json={"group_id": "zzz"})
    assert r.status_code == 404


@pytest.mark.integration
def test_features_endpoint(client) -> None:
    c, _ = client
    feats = c.get("/api/shots/features").json()
    assert "s001" in feats
    assert "scale" in feats["s001"] and "speed_factor" in feats["s001"]


@pytest.mark.integration
def test_thumb_endpoint(client) -> None:
    c, _ = client
    r = c.get("/api/shots/s001/thumb")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")


@pytest.mark.integration
def test_thumb_generated_on_demand_for_legacy_clip(client) -> None:
    c, tmp = client
    (tmp / "shots" / "thumbs" / "s005.jpg").unlink()
    r = c.get("/api/shots/s005/thumb")
    assert r.status_code == 200


@pytest.mark.integration
def test_thumb_unknown_shot_404(client) -> None:
    c, _ = client
    assert c.get("/api/shots/zzz/thumb").status_code == 404


@pytest.mark.integration
def test_output_shots_lists_active_only(client) -> None:
    c, _ = client
    ids = c.get("/api/output/shots").json()["shots"]
    assert "s002" not in ids and "s004" not in ids
    assert {"s001", "s003", "s005"} <= set(ids)
