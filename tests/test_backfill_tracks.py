"""Tests for scripts/backfill_tracks.py — applies the backward
track-extension pass (src.utils.track_backfill) to an EXISTING
tracks/<shot>_tracks.json IN PLACE, without re-running the tracking
stage (ball-stage campaign Workstream 4).

The matching algorithm itself (IoU gate, ambiguity handling, appearance
gate) is covered by tests/test_track_backfill.py. These tests exercise
the CLI-specific plumbing: manifest/clip resolution, --player
selection, the .bak safety net, and --dry-run — using an
``_AlwaysMatchDetector`` fake (injected via ``run(args, detector=...)``)
so no real YOLO/model is needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from scripts.backfill_tracks import build_arg_parser, run
from src.schemas.shots import Shot, ShotsManifest
from src.schemas.tracks import Track, TrackFrame, TracksResult
from src.utils.player_detector import Detection, PlayerDetector

FIXED_BBOX = (100.0, 100.0, 200.0, 300.0)


class _AlwaysMatchDetector(PlayerDetector):
    """Ignores frame content entirely — always reports one player
    detection at FIXED_BBOX. Every backfill-eligible track in these
    fixtures is seeded at FIXED_BBOX too, so every backward step is an
    unambiguous IoU=1.0 match; these tests exercise CLI plumbing (arg
    handling, selection, .bak safety, in-place write), not the
    matching algorithm."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [Detection(bbox=FIXED_BBOX, confidence=0.9, class_name="player")]


def _build_output_dir(tmp_path: Path, shot_id: str = "shot_001", n_frames: int = 12) -> Path:
    output_dir = tmp_path / "output"
    shots_dir = output_dir / "shots"
    shots_dir.mkdir(parents=True)

    clip_path = shots_dir / f"{shot_id}.mp4"
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(n_frames):
        writer.write(np.full((240, 320, 3), 50, dtype=np.uint8))
    writer.release()

    shot = Shot(
        id=shot_id, start_frame=0, end_frame=n_frames - 1,
        start_time=0.0, end_time=n_frames / 10.0,
        clip_file=f"shots/{shot_id}.mp4",
    )
    ShotsManifest(
        source_file="test.mp4", fps=10.0, total_frames=n_frames, shots=[shot]
    ).save(shots_dir / "shots_manifest.json")
    return output_dir


def _write_tracks(output_dir: Path, shot_id: str, tracks: list[Track]) -> Path:
    tracks_dir = output_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    path = tracks_dir / f"{shot_id}_tracks.json"
    TracksResult(shot_id=shot_id, tracks=tracks).save(path)
    return path


def _late_track(track_id: str, player_id: str, player_name: str, start_frame: int) -> Track:
    return Track(
        track_id=track_id, class_name="player", team="A",
        player_id=player_id, player_name=player_name,
        frames=[TrackFrame(frame=start_frame, bbox=list(FIXED_BBOX), confidence=0.9, pitch_position=None)],
    )


def _args(output_dir: Path, shot_id: str, **overrides) -> argparse.Namespace:
    argv = ["--output", str(output_dir), "--shot", shot_id]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return build_arg_parser().parse_args(argv)


def test_backfill_extends_track_and_creates_bak(tmp_path: Path) -> None:
    output_dir = _build_output_dir(tmp_path)
    t1 = _late_track("T001", "P001", "Hato", start_frame=8)
    tracks_path = _write_tracks(output_dir, "shot_001", [t1])
    original_bytes = tracks_path.read_bytes()

    rc = run(_args(output_dir, "shot_001"), detector=_AlwaysMatchDetector())
    assert rc == 0

    bak_path = tracks_path.with_name(tracks_path.name + ".bak")
    assert bak_path.exists()
    assert bak_path.read_bytes() == original_bytes

    result = TracksResult.load(tracks_path)
    assert len(result.tracks) == 1
    assert result.tracks[0].track_id == "T001"
    assert result.tracks[0].player_id == "P001"
    assert result.tracks[0].player_name == "Hato"
    assert result.tracks[0].frames[0].frame == 0  # walked all the way back to shot start
    assert result.tracks[0].frames[0].source == "backfill"


def test_bak_is_never_overwritten_by_a_later_run(tmp_path: Path) -> None:
    output_dir = _build_output_dir(tmp_path)
    t1 = _late_track("T001", "P001", "Hato", start_frame=8)
    t2 = _late_track("T002", "P002", "Salah", start_frame=6)
    tracks_path = _write_tracks(output_dir, "shot_001", [t1, t2])
    original_bytes = tracks_path.read_bytes()
    bak_path = tracks_path.with_name(tracks_path.name + ".bak")

    run(_args(output_dir, "shot_001", player="P001"), detector=_AlwaysMatchDetector())
    assert bak_path.read_bytes() == original_bytes
    after_first_bytes = tracks_path.read_bytes()
    assert after_first_bytes != original_bytes

    # A second edit (different track) must NOT touch the .bak, which
    # still has to hold the PRE-FIRST-RUN original.
    run(_args(output_dir, "shot_001", player="P002"), detector=_AlwaysMatchDetector())
    assert bak_path.read_bytes() == original_bytes

    result = TracksResult.load(tracks_path)
    by_id = {t.track_id: t for t in result.tracks}
    assert by_id["T001"].frames[0].frame == 0  # from the first run, preserved
    assert by_id["T002"].frames[0].frame == 0  # from the second run


def test_player_filter_only_touches_the_selected_track(tmp_path: Path) -> None:
    output_dir = _build_output_dir(tmp_path)
    t1 = _late_track("T001", "P001", "Hato", start_frame=8)
    t2 = _late_track("T002", "P002", "Salah", start_frame=8)
    tracks_path = _write_tracks(output_dir, "shot_001", [t1, t2])

    run(_args(output_dir, "shot_001", player="P001"), detector=_AlwaysMatchDetector())

    result = TracksResult.load(tracks_path)
    by_id = {t.track_id: t for t in result.tracks}
    assert by_id["T001"].frames[0].frame == 0
    assert len(by_id["T002"].frames) == 1
    assert by_id["T002"].frames[0].frame == 8  # not selected -> untouched


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    output_dir = _build_output_dir(tmp_path)
    t1 = _late_track("T001", "P001", "Hato", start_frame=8)
    tracks_path = _write_tracks(output_dir, "shot_001", [t1])
    original_bytes = tracks_path.read_bytes()

    rc = run(_args(output_dir, "shot_001", dry_run=True), detector=_AlwaysMatchDetector())

    assert rc == 0
    assert tracks_path.read_bytes() == original_bytes
    assert not tracks_path.with_name(tracks_path.name + ".bak").exists()


def test_missing_tracks_file_returns_error_code(tmp_path: Path) -> None:
    output_dir = _build_output_dir(tmp_path)
    rc = run(_args(output_dir, "shot_001"), detector=_AlwaysMatchDetector())
    assert rc == 1
