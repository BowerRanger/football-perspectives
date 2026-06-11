"""prepare_shots split mode: full-reel ingestion end to end (synthetic)."""
import json
import shutil
from pathlib import Path

import pytest

from src.schemas.shots import ShotsManifest
from src.schemas.sync_map import SyncMap
from src.stages.prepare_shots import PrepareShotsStage
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
# Expected shots: s001 green live | s002 crowd -> reaction (excluded)
# s003 slow replay | s004 black -> transition (excluded) | s005 green live
# Groups: g01=[s001,s003] (replay extends the live shot's group),
#         g02=[s005] (transition boundary).


@pytest.fixture(scope="module")
def split_run(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("split")
    reel = tmp / "reel.mp4"
    build_reel(reel, SEGMENTS)
    out = tmp / "out"
    PrepareShotsStage(config=CFG, output_dir=out, video_path=reel).run()
    return out


def _manifest(out: Path) -> ShotsManifest:
    return ShotsManifest.load(out / "shots" / "shots_manifest.json")


def test_split_writes_clips_and_manifest(split_run):
    m = _manifest(split_run)
    assert len(m.shots) == 5
    assert [s.id for s in m.shots] == [f"s{i:03d}" for i in range(1, 6)]
    assert all((split_run / s.clip_file).exists() for s in m.shots)
    kinds = [s.kind for s in m.shots]
    assert kinds.count("reaction") == 1
    assert kinds.count("transition") == 1


def test_reaction_and_transition_excluded(split_run):
    m = _manifest(split_run)
    excluded = {s.id: s.exclude_reason for s in m.shots if s.excluded}
    assert sorted(excluded.values()) == ["reaction", "transition"]
    assert len(m.active_shots()) == 3


def test_source_span_recorded(split_run):
    m = _manifest(split_run)
    s1 = m.shots[0]
    assert s1.source_start_s == pytest.approx(0.0, abs=0.2)
    assert s1.source_end_s == pytest.approx(3.0, abs=0.4)


def test_grouping_and_sync_map(split_run):
    m = _manifest(split_run)
    assert len(m.groups) == 2
    assert m.groups[0].shot_ids == ["s001", "s003"]
    assert m.groups[1].shot_ids == ["s005"]
    assert m.shots[0].group_id == m.groups[0].id
    # excluded shots stay ungrouped
    assert all(s.group_id == "" for s in m.shots if s.excluded)

    sm = SyncMap.load(split_run / "shots" / "sync_map.json")
    g1 = sm.group(m.groups[0].id)
    assert g1 is not None
    assert len(g1.alignments) == 2
    assert g1.reference_shot == "s001"


def test_slowmo_keeps_native_timing_with_factor_as_metadata(split_run):
    """Retiming on noisy speed estimates proved destructive (real-reel
    estimates hit the 0.3/4.0 clamps on most shots), so clips keep their
    native timing; the estimated factor is metadata only."""
    m = _manifest(split_run)
    slow = [s for s in m.shots if s.speed_factor > 1.25]
    assert [s.id for s in slow] == ["s003"]
    # the 3.0 s slow-mo segment keeps its ~75 source frames
    n = slow[0].end_frame + 1
    assert abs(n - 3.0 * 25.0) <= 4


def test_split_rerun_is_idempotent(split_run):
    manifest_path = split_run / "shots" / "shots_manifest.json"
    before = manifest_path.read_text()
    reel = split_run.parent / "reel.mp4"
    PrepareShotsStage(config=CFG, output_dir=split_run,
                      video_path=reel).run()
    assert manifest_path.read_text() == before


def test_thumbnails_written(split_run):
    thumbs = list((split_run / "shots" / "thumbs").glob("*.jpg"))
    assert len(thumbs) == 5


def test_features_sidecar_written(split_run):
    data = json.loads(
        (split_run / "shots" / "shot_features.json").read_text())
    assert set(data) == {f"s{i:03d}" for i in range(1, 6)}
    row = data["s001"]
    for key in ("pitch_ratio_median", "motion_rate", "speed_factor",
                "scale", "kind"):
        assert key in row


def test_copy_mode_still_works_for_short_clip(tmp_path: Path):
    clip = tmp_path / "myclip.mp4"
    build_reel(clip, [("green", 2.0)])
    out = tmp_path / "out"
    cfg = {"prepare_shots": {"mode": "auto",
                             "split": {"min_input_duration_s": 90}}}
    PrepareShotsStage(config=cfg, output_dir=out, video_path=clip).run()
    m = ShotsManifest.load(out / "shots" / "shots_manifest.json")
    assert [s.id for s in m.shots] == ["myclip"]
    assert m.groups == []


def test_auto_mode_splits_long_input(tmp_path: Path):
    reel = tmp_path / "longreel.mp4"
    build_reel(reel, [("green", 3.0), ("crowd", 2.0), ("green", 3.0)])
    out = tmp_path / "out"
    cfg = json.loads(json.dumps(CFG))  # deep copy
    cfg["prepare_shots"]["mode"] = "auto"
    cfg["prepare_shots"]["split"]["min_input_duration_s"] = 5
    PrepareShotsStage(config=cfg, output_dir=out, video_path=reel).run()
    m = ShotsManifest.load(out / "shots" / "shots_manifest.json")
    assert len(m.shots) == 3
