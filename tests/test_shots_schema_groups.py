"""Schema additions for highlights ingestion: Shot flags + groups."""
import json
from pathlib import Path

from src.schemas.shots import HighlightGroup, Shot, ShotsManifest


def _shot(sid: str, **kw) -> Shot:
    base = dict(id=sid, start_frame=0, end_frame=99, start_time=0.0,
                end_time=4.0, clip_file=f"shots/{sid}.mp4")
    base.update(kw)
    return Shot(**base)


def test_shot_new_fields_default():
    s = _shot("s001")
    assert (s.kind, s.excluded, s.exclude_reason, s.group_id) == (
        "gameplay", False, "", "")
    assert s.source_start_s == -1.0 and s.source_end_s == -1.0


def test_manifest_groups_round_trip(tmp_path: Path):
    m = ShotsManifest(
        source_file="reel.mp4", fps=25.0, total_frames=200,
        shots=[_shot("s001", group_id="g01"),
               _shot("s002", group_id="g01", kind="reaction", excluded=True,
                     exclude_reason="reaction")],
        groups=[HighlightGroup(id="g01", label="Highlight 1",
                               shot_ids=["s001", "s002"],
                               boundary_rule="start",
                               boundary_confidence=1.0)],
    )
    p = tmp_path / "m.json"
    m.save(p)
    loaded = ShotsManifest.load(p)
    assert loaded.groups[0].id == "g01"
    assert loaded.groups[0].shot_ids == ["s001", "s002"]
    assert loaded.shots[1].excluded is True


def test_active_shots_filters_excluded():
    m = ShotsManifest(source_file="x", fps=25.0, total_frames=0,
                      shots=[_shot("a"), _shot("b", excluded=True)])
    assert [s.id for s in m.active_shots()] == ["a"]


def test_legacy_manifest_without_new_fields_loads(tmp_path: Path):
    legacy = {"source_file": "x", "fps": 25.0, "total_frames": 100,
              "shots": [{"id": "a", "start_frame": 0, "end_frame": 99,
                         "start_time": 0.0, "end_time": 4.0,
                         "clip_file": "shots/a.mp4"}]}
    p = tmp_path / "m.json"
    p.write_text(json.dumps(legacy))
    m = ShotsManifest.load(p)
    assert m.groups == [] and m.shots[0].excluded is False


def test_groups_with_unknown_fields_load(tmp_path: Path):
    """A newer writer's extra group keys must not break the loader."""
    data = {"source_file": "x", "fps": 25.0, "total_frames": 0, "shots": [],
            "groups": [{"id": "g01", "label": "Highlight 1",
                        "shot_ids": [], "boundary_rule": "start",
                        "boundary_confidence": 1.0, "future_key": 1}]}
    p = tmp_path / "m.json"
    p.write_text(json.dumps(data))
    assert ShotsManifest.load(p).groups[0].id == "g01"
