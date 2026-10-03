"""Tests for src.utils.kit_evidence: kit-colour evidence extraction and
scoring for the automatic match-suggestion feature (W4).

Fully offline: a tiny synthetic clip (solid pitch-green background plus
solid red/blue "shirt" rectangles at known bboxes) is generated in-process
with cv2.VideoWriter, alongside a matching shots_manifest.json and
<shot_id>_tracks.json built from the real dataclasses (Shot/ShotsManifest,
Track/TrackFrame/TracksResult) so the on-disk shape always matches the
production schema. Everything lives under tmp_path; no network, no real
pipeline artifacts.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.shots import Shot, ShotsManifest
from src.schemas.tracks import Track, TrackFrame, TracksResult
from src.utils.kit_evidence import KitEvidence, extract_kit_evidence, kit_match_score

FRAME_SIZE = (320, 240)  # (width, height)
N_FRAMES = 20
FPS = 25.0

GREEN_BGR = (0, 170, 0)  # pitch background
RED_BGR = (0, 0, 220)  # "red" shirt, BGR
BLUE_BGR = (220, 0, 0)  # "blue" shirt, BGR
RED_BBOX = [40.0, 40.0, 110.0, 200.0]
BLUE_BBOX = [180.0, 40.0, 250.0, 200.0]


def _write_synthetic_clip(path: Path) -> None:
    """Static pitch-green frame with two solid-colour rectangles, written
    N_FRAMES times (unmoving "players" keep the tracks fixture trivial)."""
    w, h = FRAME_SIZE
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (w, h))
    assert writer.isOpened(), "failed to open synthetic video writer"
    try:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = GREEN_BGR
        x1, y1, x2, y2 = (int(v) for v in RED_BBOX)
        frame[y1:y2, x1:x2] = RED_BGR
        x1, y1, x2, y2 = (int(v) for v in BLUE_BBOX)
        frame[y1:y2, x1:x2] = BLUE_BGR
        for _ in range(N_FRAMES):
            writer.write(frame)
    finally:
        writer.release()


def _build_output_dir(
    tmp_path: Path,
    *,
    shot_id: str = "shot0",
    with_clip: bool = True,
    with_tracks: bool = True,
) -> Path:
    output_dir = tmp_path / "output"
    shots_dir = output_dir / "shots"
    tracks_dir = output_dir / "tracks"
    shots_dir.mkdir(parents=True)
    tracks_dir.mkdir(parents=True)

    clip_rel = f"shots/{shot_id}.mp4"
    if with_clip:
        _write_synthetic_clip(shots_dir / f"{shot_id}.mp4")

    manifest = ShotsManifest(
        source_file="",
        fps=FPS,
        total_frames=N_FRAMES,
        shots=[
            Shot(
                id=shot_id,
                start_frame=0,
                end_frame=N_FRAMES - 1,
                start_time=0.0,
                end_time=N_FRAMES / FPS,
                clip_file=clip_rel,
            )
        ],
    )
    manifest.save(shots_dir / "shots_manifest.json")

    if with_tracks:
        red_track = Track(
            track_id="T001",
            class_name="player",
            team="unknown",
            frames=[
                TrackFrame(frame=i, bbox=list(RED_BBOX), confidence=0.9, pitch_position=None)
                for i in range(N_FRAMES)
            ],
        )
        blue_track = Track(
            track_id="T002",
            class_name="player",
            team="unknown",
            frames=[
                TrackFrame(frame=i, bbox=list(BLUE_BBOX), confidence=0.9, pitch_position=None)
                for i in range(N_FRAMES)
            ],
        )
        tracks_result = TracksResult(shot_id=shot_id, tracks=[red_track, blue_track])
        tracks_result.save(tracks_dir / f"{shot_id}_tracks.json")

    return output_dir


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _rgb_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


class TestExtractKitEvidence:
    def test_recovers_painted_colours_permutation_agnostic(self, tmp_path):
        output_dir = _build_output_dir(tmp_path)
        evidence = extract_kit_evidence(output_dir)

        assert evidence is not None
        assert isinstance(evidence, KitEvidence)
        assert evidence.n_frames_sampled > 0
        assert len(evidence.team_hexes) == 2
        assert len(evidence.cluster_sizes) == 2
        assert all(c > 0 for c in evidence.cluster_sizes)

        red_rgb = (RED_BGR[2], RED_BGR[1], RED_BGR[0])
        blue_rgb = (BLUE_BGR[2], BLUE_BGR[1], BLUE_BGR[0])
        recovered = [_hex_to_rgb(h) for h in evidence.team_hexes]

        # Permutation-agnostic + generous tolerance: mp4v is lossy, so a
        # solid painted colour may come back a handful of RGB units off.
        tolerance = 40.0
        for target in (red_rgb, blue_rgb):
            distances = [_rgb_distance(target, r) for r in recovered]
            assert min(distances) < tolerance, (target, recovered, distances)

        # And the two recovered hexes should map to *different* painted
        # colours (not both closest to the same one).
        best_match = [
            min(range(2), key=lambda i: _rgb_distance(recovered[i], t))
            for t in (red_rgb, blue_rgb)
        ]
        assert best_match[0] != best_match[1]

    def test_missing_manifest_returns_none(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        assert extract_kit_evidence(output_dir) is None

    def test_missing_tracks_returns_none(self, tmp_path):
        output_dir = _build_output_dir(tmp_path, with_tracks=False)
        assert extract_kit_evidence(output_dir) is None

    def test_missing_clip_returns_none(self, tmp_path):
        output_dir = _build_output_dir(tmp_path, with_clip=False)
        assert extract_kit_evidence(output_dir) is None

    def test_malformed_manifest_returns_none_not_raise(self, tmp_path):
        output_dir = tmp_path / "output"
        (output_dir / "shots").mkdir(parents=True)
        (output_dir / "shots" / "shots_manifest.json").write_text("{not valid json")
        assert extract_kit_evidence(output_dir) is None

    def test_malformed_tracks_returns_none_not_raise(self, tmp_path):
        output_dir = _build_output_dir(tmp_path, with_tracks=False)
        (output_dir / "tracks" / "shot0_tracks.json").write_text("{not valid json")
        assert extract_kit_evidence(output_dir) is None

    def test_nonexistent_output_dir_returns_none(self, tmp_path):
        assert extract_kit_evidence(tmp_path / "does_not_exist") is None


class TestKitMatchScore:
    LIVERPOOL_RED = "#f73b57"
    CHELSEA_BLUE = "#4248f0"

    def test_permutation_invariance_swap_home_away(self):
        evidence = KitEvidence(
            team_hexes=(self.LIVERPOOL_RED, self.CHELSEA_BLUE),
            cluster_sizes=(10, 8),
            n_frames_sampled=5,
        )
        score_ab = kit_match_score(evidence, self.LIVERPOOL_RED, self.CHELSEA_BLUE)
        score_ba = kit_match_score(evidence, self.CHELSEA_BLUE, self.LIVERPOOL_RED)
        assert score_ab == pytest.approx(score_ba)

    def test_permutation_invariance_swap_evidence_order(self):
        evidence = KitEvidence(
            team_hexes=(self.LIVERPOOL_RED, self.CHELSEA_BLUE),
            cluster_sizes=(10, 8),
            n_frames_sampled=5,
        )
        evidence_swapped = KitEvidence(
            team_hexes=(self.CHELSEA_BLUE, self.LIVERPOOL_RED),
            cluster_sizes=(8, 10),
            n_frames_sampled=5,
        )
        score = kit_match_score(evidence, self.LIVERPOOL_RED, self.CHELSEA_BLUE)
        score_swapped = kit_match_score(evidence_swapped, self.LIVERPOOL_RED, self.CHELSEA_BLUE)
        assert score == pytest.approx(score_swapped)

    def test_range_bounds_random_hexes(self):
        evidence = KitEvidence(
            team_hexes=("#ff0000", "#0000ff"), cluster_sizes=(10, 10), n_frames_sampled=5
        )
        rng = random.Random(42)
        for _ in range(50):
            home = "#%06x" % rng.randrange(16**6)
            away = "#%06x" % rng.randrange(16**6)
            score = kit_match_score(evidence, home, away)
            assert 0.0 <= score <= 1.0

    def test_identical_colours_score_high(self):
        evidence = KitEvidence(
            team_hexes=("#ff0000", "#0000ff"), cluster_sizes=(10, 10), n_frames_sampled=5
        )
        score = kit_match_score(evidence, "#ff0000", "#0000ff")
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_opposite_colours_score_low(self):
        # Green vs. blue is the max-distance pair used to derive the
        # module's Lab-distance normalisation constant.
        evidence = KitEvidence(
            team_hexes=("#00ff00", "#00ff00"), cluster_sizes=(10, 10), n_frames_sampled=5
        )
        score = kit_match_score(evidence, "#0000ff", "#0000ff")
        assert score < 0.05

    def test_empty_home_hex_ignored_not_raised(self):
        evidence = KitEvidence(
            team_hexes=("#ff0000", "#0000ff"), cluster_sizes=(10, 10), n_frames_sampled=5
        )
        score = kit_match_score(evidence, "", "#0000ff")
        assert 0.0 <= score <= 1.0

    def test_malformed_hex_ignored_not_raised(self):
        evidence = KitEvidence(
            team_hexes=("#ff0000", "#0000ff"), cluster_sizes=(10, 10), n_frames_sampled=5
        )
        score = kit_match_score(evidence, "not-a-colour", "#0000ff")
        assert 0.0 <= score <= 1.0

    def test_both_hex_empty_or_malformed_returns_zero(self):
        evidence = KitEvidence(
            team_hexes=("#ff0000", "#0000ff"), cluster_sizes=(10, 10), n_frames_sampled=5
        )
        assert kit_match_score(evidence, "", "") == 0.0
        assert kit_match_score(evidence, "bogus", "also bogus") == 0.0
