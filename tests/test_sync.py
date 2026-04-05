from pathlib import Path
import inspect
import wave

import cv2
import numpy as np
import pytest

from src.schemas.shots import Shot, ShotsManifest
from src.schemas.sync_map import SyncMap
from src.schemas.tracks import Track, TrackFrame, TracksResult
from src.stages.sync import (
    AlignmentEstimate,
    TemporalSyncStage,
    _align_audio,
    _align_formation_spatial,
    _align_visual,
    _audio_energy_envelope,
    _compute_arm_raise_angle,
    _compute_celebration_signal,
    _compute_overlap_frames,
    _compute_pitch_bounds,
    _correlate_with_speed_sweep,
    _cross_correlate_signals,
    _detect_celebration_onset,
    _extract_bbox_velocity,
    _extract_formation_descriptors,
    _extract_frame_descriptors,
    _extract_player_appearances,
    _extract_player_motion_signal,
    _fuse_alignment_estimates,
    _load_audio_mono,
    _match_players_across_views,
    _visual_similarity_profile,
    _extract_pitch_axis_trajectory,
    _align_by_matched_pitch_trajectories,
    _fill_nans,
    cross_correlate_trajectories,
    project_ball_to_pitch,
    _solve_offset_graph,
    _collect_pairwise_estimates,
)
from src.utils.ball_detector import BallDetector, FakeBallDetector


def test_fake_ball_detector_returns_position():
    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(5)]
    positions = [(50.0, 60.0), (55.0, 65.0), None, (60.0, 70.0), (65.0, 75.0)]
    detector = FakeBallDetector(positions)
    results = [detector.detect(f) for f in frames]
    assert results[0] == pytest.approx((50.0, 60.0))
    assert results[2] is None


def test_ball_detector_is_abstract():
    assert inspect.isabstract(BallDetector)


def test_cross_correlate_finds_correct_offset():
    traj_a = np.array([0, 0, 1, 3, 5, 3, 1, 0, 0, 0], dtype=float)
    traj_b = np.zeros(10, dtype=float)
    traj_b[5:9] = [1, 3, 5, 3]
    offset, confidence = cross_correlate_trajectories(traj_a, traj_b)
    assert offset == 3
    assert confidence > 0.7


def test_cross_correlate_returns_zero_for_identical():
    traj = np.array([0, 1, 2, 3, 2, 1, 0], dtype=float)
    offset, confidence = cross_correlate_trajectories(traj, traj.copy())
    assert offset == 0
    assert confidence > 0.99


def test_cross_correlate_low_confidence_for_noise():
    rng = np.random.default_rng(42)
    traj_a = rng.random(50)
    traj_b = rng.random(50)
    _, confidence = cross_correlate_trajectories(traj_a, traj_b)
    assert confidence < 0.5


def test_cross_correlate_unequal_lengths():
    traj_a = np.zeros(12, dtype=float)
    traj_a[2:6] = [1, 3, 5, 3]
    traj_b = np.zeros(10, dtype=float)
    traj_b[5:9] = [1, 3, 5, 3]
    offset, confidence = cross_correlate_trajectories(traj_a, traj_b)
    assert offset == 3
    assert confidence > 0.7


def test_cross_correlate_signal_respects_lag_window():
    signal_a = np.zeros(80, dtype=float)
    signal_b = np.zeros(80, dtype=float)
    signal_a[5:12] = [0, 1, 3, 5, 3, 1, 0]
    signal_b[30:37] = [0, 1, 3, 5, 3, 1, 0]

    offset, _ = _cross_correlate_signals(
        signal_a=signal_a,
        signal_b=signal_b,
        max_lag=10,
        min_overlap_frames=5,
    )
    assert abs(offset) <= 10


def test_cross_correlate_signal_enforces_min_overlap():
    signal_a = np.zeros(12, dtype=float)
    signal_b = np.zeros(12, dtype=float)
    signal_a[:4] = [1, 2, 3, 2]
    signal_b[8:12] = [1, 2, 3, 2]

    _, confidence = _cross_correlate_signals(
        signal_a=signal_a,
        signal_b=signal_b,
        max_lag=12,
        min_overlap_frames=8,
    )
    assert confidence == 0.0


def test_visual_similarity_profile_max_lag_limits_offsets():
    rng = np.random.default_rng(0)
    ref_desc = rng.random((20, 16)).astype(np.float32)
    shot_desc = rng.random((20, 16)).astype(np.float32)

    offsets_full, _ = _visual_similarity_profile(ref_desc, shot_desc, min_overlap=2)
    offsets_windowed, _ = _visual_similarity_profile(
        ref_desc, shot_desc, min_overlap=2, max_lag=5
    )

    assert len(offsets_windowed) < len(offsets_full)
    assert int(np.abs(offsets_windowed).max()) <= 5


def test_extract_player_motion_signal_ignores_ball_tracks():
    tracks = TracksResult(
        shot_id="shot_001",
        tracks=[
            Track(
                track_id="T001",
                class_name="player",
                team="A",
                frames=[
                    TrackFrame(frame=0, bbox=[0, 0, 1, 1], confidence=0.9, pitch_position=[0.0, 0.0]),
                    TrackFrame(frame=1, bbox=[0, 0, 1, 1], confidence=0.9, pitch_position=[1.0, 0.0]),
                ],
            ),
            Track(
                track_id="B001",
                class_name="ball",
                team="unknown",
                frames=[
                    TrackFrame(frame=0, bbox=[0, 0, 1, 1], confidence=0.9, pitch_position=[0.0, 0.0]),
                    TrackFrame(frame=1, bbox=[0, 0, 1, 1], confidence=0.9, pitch_position=[20.0, 0.0]),
                ],
            ),
        ],
    )

    signal = _extract_player_motion_signal(tracks=tracks, total_frames=5)
    assert signal[1] == pytest.approx(1.0)


def test_fuse_alignment_prefers_hybrid_when_offsets_agree():
    ball = AlignmentEstimate(offset=40, confidence=0.8, method="ball_trajectory", valid=True)
    player = AlignmentEstimate(offset=42, confidence=0.7, method="player_formation", valid=True)

    fused = _fuse_alignment_estimates(
        ball_estimate=ball,
        player_estimate=player,
        agreement_tolerance_frames=8,
    )
    assert fused.method == "hybrid"
    assert fused.offset == 41
    assert fused.confidence > 0.7


def test_fuse_alignment_prefers_higher_confidence_on_conflict():
    ball = AlignmentEstimate(offset=30, confidence=0.8, method="ball_trajectory", valid=True)
    player = AlignmentEstimate(offset=-20, confidence=0.5, method="player_formation", valid=True)

    fused = _fuse_alignment_estimates(
        ball_estimate=ball,
        player_estimate=player,
        agreement_tolerance_frames=8,
    )
    assert fused.method == "ball_trajectory"
    assert fused.offset == 30


def test_compute_overlap_frames():
    start, end = _compute_overlap_frames(reference_len=100, shot_len=50, offset=20)
    assert [start, end] == [20, 70]


def test_project_ball_to_pitch_returns_2d():
    K = np.array([[1500, 0, 960], [0, 1500, 540], [0, 0, 1]], dtype=np.float32)
    rvec = np.array([0.05, 0.15, 0.0], dtype=np.float32)
    tvec = np.array([-52.5, -34.0, 60.0], dtype=np.float32)

    pt_3d = np.array([[30.0, 20.0, 0.0]], dtype=np.float32)
    pt_2d, _ = cv2.projectPoints(pt_3d, rvec, tvec, K, None)
    pixel = pt_2d.reshape(2)

    from src.schemas.calibration import CameraFrame

    frame_cal = CameraFrame(
        frame=0,
        intrinsic_matrix=K.tolist(),
        rotation_vector=rvec.tolist(),
        translation_vector=tvec.tolist(),
        reprojection_error=0.0,
        num_correspondences=8,
        confidence=1.0,
    )
    pitch_pos = project_ball_to_pitch(pixel, frame_cal)
    assert pitch_pos is not None
    assert np.allclose(pitch_pos, [30.0, 20.0], atol=0.1)


def test_temporal_sync_stage_uses_player_signal_when_ball_missing(tmp_path):
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    clip_a = shots_dir / "shot_001.mp4"
    clip_b = shots_dir / "shot_002.mp4"
    _create_dummy_clip(clip_a, fps=25.0, frames=60)
    _create_dummy_clip(clip_b, fps=25.0, frames=60)

    manifest = ShotsManifest(
        source_file="input.mp4",
        fps=25.0,
        total_frames=120,
        shots=[
            Shot(
                id="shot_001",
                start_frame=0,
                end_frame=59,
                start_time=0.0,
                end_time=2.4,
                clip_file="shots/shot_001.mp4",
            ),
            Shot(
                id="shot_002",
                start_frame=60,
                end_frame=119,
                start_time=2.4,
                end_time=4.8,
                clip_file="shots/shot_002.mp4",
            ),
        ],
    )
    manifest.save(shots_dir / "shots_manifest.json")

    tracks_dir = tmp_path / "tracks"
    tracks_dir.mkdir()
    _build_tracks_result(shot_id="shot_001", shift=0).save(tracks_dir / "shot_001_tracks.json")
    _build_tracks_result(shot_id="shot_002", shift=3).save(tracks_dir / "shot_002_tracks.json")

    stage = TemporalSyncStage(
        config={
            "sync": {
                "min_confidence": 0.0,
                "min_overlap_frames": 3,
                # Use full fps so frame_step=1 and the 3-frame shift is detectable.
                "sample_fps": 25.0,
            }
        },
        output_dir=tmp_path,
        ball_detector=FakeBallDetector([None]),
    )
    stage.run()

    sync_map = SyncMap.load(tmp_path / "sync" / "sync_map.json")
    alignment = sync_map.alignments[0]
    # shot_002 track data starts at frame 3 of shot_002 and matches ref frame 0.
    # Convention: frame_in_ref = frame_in_shot + offset → 0 = 3 + (-3) → offset = -3.
    assert alignment.frame_offset == -3
    assert alignment.method == "player_formation"
    assert alignment.confidence > 0.0


def _build_tracks_result(shot_id: str, shift: int) -> TracksResult:
    speed_pattern = [0.0, 0.4, 1.2, 2.2, 1.0, 0.3, 0.0]
    positions: list[float] = [0.0]
    for speed in speed_pattern:
        positions.append(positions[-1] + speed)

    frames: list[TrackFrame] = []
    for idx, pos in enumerate(positions):
        frame_idx = idx + shift
        frames.append(
            TrackFrame(
                frame=frame_idx,
                bbox=[0.0, 0.0, 10.0, 20.0],
                confidence=0.9,
                pitch_position=[pos, 0.0],
            )
        )

    return TracksResult(
        shot_id=shot_id,
        tracks=[Track(track_id="T001", class_name="player", team="A", frames=frames)],
    )


def test_compute_pitch_bounds_from_tracks():
    tracks = {
        "shot_001": TracksResult(
            shot_id="shot_001",
            tracks=[
                Track(
                    track_id="T1", class_name="player", team="A",
                    frames=[
                        TrackFrame(frame=0, bbox=[0]*4, confidence=0.9, pitch_position=[10.0, 5.0]),
                        TrackFrame(frame=1, bbox=[0]*4, confidence=0.9, pitch_position=[90.0, 60.0]),
                    ],
                ),
                Track(
                    track_id="B1", class_name="ball", team="unknown",
                    frames=[
                        TrackFrame(frame=0, bbox=[0]*4, confidence=0.9, pitch_position=[200.0, 200.0]),
                    ],
                ),
            ],
        )
    }
    (x_min, x_max), (y_min, y_max) = _compute_pitch_bounds(tracks)
    # Ball track excluded; player x in [10, 90], y in [5, 60].
    # Percentiles on 2 points = min/max. Padding = 10% of range.
    assert x_min < 10.0
    assert x_max > 90.0
    assert y_min < 5.0
    assert y_max > 60.0


def test_compute_pitch_bounds_fallback_when_no_tracks():
    (x_min, x_max), (y_min, y_max) = _compute_pitch_bounds({})
    assert x_min == 0.0
    assert x_max == 12000.0


def test_extract_formation_descriptors_shape_and_norm():
    tracks = TracksResult(
        shot_id="shot_001",
        tracks=[
            Track(
                track_id="T1", class_name="player", team="A",
                frames=[
                    TrackFrame(frame=k * 5, bbox=[0]*4, confidence=0.9,
                               pitch_position=[float(k * 10), 30.0])
                    for k in range(10)
                ],
            )
        ],
    )
    pitch_bounds = ((0.0, 100.0), (0.0, 68.0))
    desc, frame_step = _extract_formation_descriptors(
        tracks=tracks,
        total_frames=50,
        fps=25.0,
        sample_fps=5.0,
        pitch_bounds=pitch_bounds,
        grid_shape=(8, 5),
    )
    assert frame_step == 5
    assert desc.shape == (10, 40)  # 10 samples, 8*5=40 cells
    # Non-zero rows should be unit vectors.
    norms = np.linalg.norm(desc, axis=1)
    nonzero = norms > 1e-6
    assert nonzero.any()
    assert np.allclose(norms[nonzero], 1.0, atol=1e-5)


def test_formation_descriptors_find_correct_offset():
    """Formation descriptors from two shifted tracks should produce peak at offset=3."""
    pitch_bounds = ((0.0, 10.0), (0.0, 1.0))

    def make_tracks(shot_id: str, shift: int) -> TracksResult:
        positions = [0.0, 0.0, 0.4, 1.6, 3.8, 4.8, 5.1, 5.1]
        frames = [
            TrackFrame(frame=i + shift, bbox=[0]*4, confidence=0.9,
                       pitch_position=[pos, 0.0])
            for i, pos in enumerate(positions)
        ]
        return TracksResult(
            shot_id=shot_id,
            tracks=[Track(track_id="T1", class_name="player", team="A", frames=frames)],
        )

    ref_tracks = make_tracks("shot_001", shift=0)
    shot_tracks = make_tracks("shot_002", shift=3)

    fps, sample_fps = 25.0, 25.0  # frame_step=1 for fine-grained matching
    ref_desc, fs = _extract_formation_descriptors(ref_tracks, 60, fps, sample_fps, pitch_bounds)
    shot_desc, _ = _extract_formation_descriptors(shot_tracks, 60, fps, sample_fps, pitch_bounds)

    est = _align_formation_spatial(ref_desc, shot_desc, frame_step=fs, min_overlap_frames=3)
    # shift=3: ref frame 0 = shot frame 3 → frame_in_ref = frame_in_shot + offset → offset = -3
    assert est.offset == -3
    assert est.confidence > 0.0
    assert est.method == "player_formation"


def _create_dummy_clip(path: Path, fps: float, frames: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    for _ in range(frames):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()


def _create_patterned_clip(path: Path, fps: float, frames: int, offset: int = 0) -> None:
    """Create a clip where each frame's appearance is determined solely by its absolute position.

    ``abs_pos = frame_index + offset``.  Two clips that share overlapping
    absolute positions will have visually identical frames at those positions,
    enabling ground-truth visual sync testing.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    for i in range(frames):
        abs_pos = i + offset
        # Unique random noise keyed to abs_pos: same abs_pos → same frame.
        rng_frame = np.random.default_rng(seed=abs_pos * 31 + 7)
        gray = rng_frame.integers(0, 256, (240, 320), dtype=np.uint8)
        frame = cv2.merge([gray, gray, gray])
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
# Visual frame descriptor tests
# ---------------------------------------------------------------------------

def test_extract_frame_descriptors_shape(tmp_path):
    clip = tmp_path / "test.mp4"
    _create_patterned_clip(clip, fps=25.0, frames=50)
    desc, frame_step = _extract_frame_descriptors(clip, fps=25.0, sample_fps=5.0)
    # At 25fps, sample_fps=5 → frame_step=5 → ~10 samples from 50 frames.
    assert frame_step == 5
    assert desc.ndim == 2
    assert desc.shape[1] == 64 * 36
    assert 8 <= desc.shape[0] <= 12  # Allow ±2 for codec rounding


def test_extract_frame_descriptors_normalized(tmp_path):
    clip = tmp_path / "test.mp4"
    _create_patterned_clip(clip, fps=25.0, frames=50)
    desc, _ = _extract_frame_descriptors(clip, fps=25.0, sample_fps=5.0)
    norms = np.linalg.norm(desc, axis=1)
    # Non-zero rows must be unit vectors.
    nonzero_mask = norms > 1e-6
    assert nonzero_mask.any(), "expected at least one non-zero descriptor"
    assert np.allclose(norms[nonzero_mask], 1.0, atol=1e-5)


def test_visual_similarity_profile_identical_clips(tmp_path):
    # Two identical descriptor arrays should have peak similarity at offset 0.
    rng = np.random.default_rng(0)
    N, D = 20, 64 * 36
    raw = rng.random((N, D)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    desc = raw / norms

    offsets, scores = _visual_similarity_profile(desc, desc, min_overlap=5)
    best_offset = int(offsets[np.argmax(scores)])
    assert best_offset == 0
    assert scores[np.argmax(scores)] > 0.95


def test_visual_similarity_profile_shifted_clips(tmp_path):
    # Descriptors for shot B are the same as ref but shifted by 3 samples.
    rng = np.random.default_rng(1)
    N, D = 25, 64 * 36
    raw = rng.random((N, D)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    desc = raw / norms

    ref_desc = desc        # N frames
    shot_desc = desc[3:]   # N-3 frames, starting 3 samples into ref

    offsets, scores = _visual_similarity_profile(ref_desc, shot_desc, min_overlap=5)
    # offset=3 means ref[j+3] aligns with shot[j], i.e. shot starts 3 samples into ref
    best_offset = int(offsets[np.argmax(scores)])
    assert best_offset == 3


def test_align_visual_finds_offset(tmp_path):
    # Create two patterned clips where the second starts mid-way through the first.
    ref_clip = tmp_path / "ref.mp4"
    shot_clip = tmp_path / "shot.mp4"
    fps = 10.0
    # Reference: 40 frames of content (offset=0 in pattern)
    _create_patterned_clip(ref_clip, fps=fps, frames=40, offset=0)
    # Shot: 20 frames of the same pattern but starting at pattern position 10
    _create_patterned_clip(shot_clip, fps=fps, frames=20, offset=10)

    estimate = _align_visual(
        ref_clip=ref_clip,
        shot_clip=shot_clip,
        fps=fps,
        min_overlap_frames=5,
        sample_fps=5.0,
    )
    # With sample_fps=5 and fps=10, frame_step=2.
    # Pattern offset 10 frames → descriptor offset 5 → frame_offset = 5*2 = 10.
    assert abs(estimate.offset - 10) <= 4  # Allow ±2 descriptor samples
    assert estimate.confidence > 0.0


def test_align_visual_identical_clips_offset_zero(tmp_path):
    clip = tmp_path / "same.mp4"
    _create_patterned_clip(clip, fps=10.0, frames=30, offset=0)
    estimate = _align_visual(
        ref_clip=clip,
        shot_clip=clip,
        fps=10.0,
        min_overlap_frames=5,
        sample_fps=5.0,
    )
    assert estimate.offset == 0
    assert estimate.confidence > 0.0


# ---------------------------------------------------------------------------
# Audio alignment tests
# ---------------------------------------------------------------------------

def test_audio_energy_envelope_shape():
    audio = np.random.default_rng(0).random(16000).astype(np.float32)
    env = _audio_energy_envelope(audio, sample_rate=16000, window_s=0.01, hop_s=0.005)
    # 1 second of audio at 16kHz, 10ms window, 5ms hop → ~199 frames
    assert len(env) > 100
    assert len(env) < 300


def test_audio_energy_envelope_normalised():
    audio = np.random.default_rng(1).random(8000).astype(np.float32) * 0.5
    env = _audio_energy_envelope(audio, sample_rate=16000)
    # Should be zero-mean and unit-variance
    assert abs(np.mean(env)) < 0.01
    assert abs(np.std(env) - 1.0) < 0.01


def test_audio_energy_envelope_empty():
    env = _audio_energy_envelope(np.array([], dtype=np.float32))
    assert len(env) == 0


# ---------------------------------------------------------------------------
# Player appearance and matching tests
# ---------------------------------------------------------------------------

def _create_coloured_clip(path: Path, fps: float, frames: int,
                          bboxes_per_frame: list[list[list[float]]],
                          colours: list[tuple[int, int, int]]) -> None:
    """Create a clip with coloured rectangles at specified bounding boxes."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    for fi in range(frames):
        frame = np.full((240, 320, 3), (50, 120, 50), dtype=np.uint8)  # green pitch
        if fi < len(bboxes_per_frame):
            for bi, bbox in enumerate(bboxes_per_frame[fi]):
                x1, y1, x2, y2 = [int(v) for v in bbox]
                colour = colours[bi % len(colours)]
                frame[y1:y2, x1:x2] = colour
        writer.write(frame)
    writer.release()


def test_extract_player_appearances_returns_descriptors(tmp_path):
    clip = tmp_path / "test.mp4"
    bbox = [50, 30, 90, 120]
    bboxes = [[bbox]] * 30
    _create_coloured_clip(clip, fps=25.0, frames=30, bboxes_per_frame=bboxes,
                          colours=[(0, 0, 200)])  # red in BGR
    tracks = TracksResult(
        shot_id="test",
        tracks=[Track(
            track_id="T001", class_name="player", team="A",
            frames=[TrackFrame(frame=i, bbox=bbox, confidence=0.9, pitch_position=None)
                    for i in range(30)],
        )],
    )
    descs = _extract_player_appearances(clip, tracks, min_track_frames=10)
    assert "T001" in descs
    assert descs["T001"].shape == (52,)
    # Should be approximately unit-normalised
    assert abs(np.linalg.norm(descs["T001"]) - 1.0) < 0.01


def test_extract_player_appearances_skips_short_tracks(tmp_path):
    clip = tmp_path / "test.mp4"
    _create_dummy_clip(clip, fps=25.0, frames=30)
    tracks = TracksResult(
        shot_id="test",
        tracks=[Track(
            track_id="T001", class_name="player", team="A",
            frames=[TrackFrame(frame=i, bbox=[50, 30, 90, 120],
                               confidence=0.9, pitch_position=None) for i in range(5)],
        )],
    )
    descs = _extract_player_appearances(clip, tracks, min_track_frames=20)
    assert len(descs) == 0


def test_match_players_across_views():
    # Two identical descriptors should match perfectly
    d = np.random.default_rng(0).random(52).astype(np.float32)
    d /= np.linalg.norm(d)
    ref = {"T001": d, "T002": np.random.default_rng(1).random(52).astype(np.float32)}
    ref["T002"] /= np.linalg.norm(ref["T002"])
    shot = {"S001": d + np.random.default_rng(2).random(52).astype(np.float32) * 0.1}
    shot["S001"] /= np.linalg.norm(shot["S001"])

    matches = _match_players_across_views(ref, shot, min_similarity=0.5)
    assert len(matches) >= 1
    assert matches[0][0] == "T001"  # Should match to the similar descriptor
    assert matches[0][1] == "S001"
    assert matches[0][2] > 0.5


def test_match_players_empty_inputs():
    assert _match_players_across_views({}, {}, 0.5) == []
    d = np.ones(52, dtype=np.float32)
    assert _match_players_across_views({"T1": d}, {}, 0.5) == []


# ---------------------------------------------------------------------------
# Bbox velocity tests
# ---------------------------------------------------------------------------

def test_extract_bbox_velocity_basic():
    frames = [
        TrackFrame(frame=0, bbox=[100, 100, 120, 140], confidence=0.9, pitch_position=None),
        TrackFrame(frame=1, bbox=[110, 100, 130, 140], confidence=0.9, pitch_position=None),
        TrackFrame(frame=2, bbox=[120, 100, 140, 140], confidence=0.9, pitch_position=None),
    ]
    diag = np.sqrt(320**2 + 240**2)
    vel = _extract_bbox_velocity(frames, total_frames=5, frame_diagonal=diag)
    assert np.isnan(vel[0])  # No velocity for first frame
    assert vel[1] > 0  # Moved 10px horizontally
    assert vel[2] > 0
    assert np.isnan(vel[3])  # No track data


def test_extract_bbox_velocity_stationary():
    frames = [
        TrackFrame(frame=0, bbox=[100, 100, 120, 140], confidence=0.9, pitch_position=None),
        TrackFrame(frame=1, bbox=[100, 100, 120, 140], confidence=0.9, pitch_position=None),
    ]
    vel = _extract_bbox_velocity(frames, total_frames=3, frame_diagonal=400.0)
    assert vel[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Speed sweep correlation tests
# ---------------------------------------------------------------------------

def test_correlate_with_speed_sweep_exact_match():
    ref = np.array([0, 0, 1, 3, 5, 3, 1, 0, 0, 0], dtype=float)
    shot = ref.copy()  # Same signal, no speed difference
    offset, r, speed = _correlate_with_speed_sweep(ref, shot, [1.0])
    assert offset == 0
    assert r > 0.9
    assert speed == 1.0


def test_correlate_with_speed_sweep_finds_half_speed():
    # Reference signal at normal speed
    ref = np.zeros(40, dtype=float)
    ref[5:13] = [0, 1, 3, 5, 5, 3, 1, 0]

    # Shot signal is 2x slow-motion (same pattern stretched over 2x frames)
    shot = np.zeros(40, dtype=float)
    shot[10:26] = np.interp(
        np.linspace(0, 7, 16),
        np.arange(8),
        [0, 1, 3, 5, 5, 3, 1, 0],
    )

    offset, r, speed = _correlate_with_speed_sweep(ref, shot, [0.5, 1.0, 2.0])
    # At speed=2.0, the shot should be compressed to match ref
    assert speed == 2.0
    assert r > 0.3


# ---------------------------------------------------------------------------
# Celebration detection tests
# ---------------------------------------------------------------------------

def test_compute_arm_raise_angle_arms_down():
    # Arms straight down: shoulder at (100, 100), wrist at (100, 200)
    kp = {
        "left_shoulder": (100.0, 100.0, 0.9),
        "left_wrist": (100.0, 200.0, 0.9),
    }
    angle = _compute_arm_raise_angle(kp, "left")
    assert angle is not None
    assert angle < 5.0  # Near 0 degrees (arms down)


def test_compute_arm_raise_angle_arms_up():
    # Arms straight up: shoulder at (100, 200), wrist at (100, 100)
    kp = {
        "right_shoulder": (100.0, 200.0, 0.9),
        "right_wrist": (100.0, 100.0, 0.9),
    }
    angle = _compute_arm_raise_angle(kp, "right")
    assert angle is not None
    assert angle > 170.0  # Near 180 degrees (arms up)


def test_compute_arm_raise_angle_horizontal():
    # Arms horizontal: shoulder at (100, 100), wrist at (200, 100)
    kp = {
        "left_shoulder": (100.0, 100.0, 0.9),
        "left_wrist": (200.0, 100.0, 0.9),
    }
    angle = _compute_arm_raise_angle(kp, "left")
    assert angle is not None
    assert abs(angle - 90.0) < 1.0


def test_compute_arm_raise_angle_low_confidence():
    kp = {
        "left_shoulder": (100.0, 100.0, 0.1),
        "left_wrist": (100.0, 200.0, 0.9),
    }
    assert _compute_arm_raise_angle(kp, "left", min_conf=0.3) is None


def test_compute_arm_raise_angle_missing_keypoints():
    kp = {"left_shoulder": (100.0, 100.0, 0.9)}
    assert _compute_arm_raise_angle(kp, "left") is None


def test_detect_celebration_onset_with_pose_data(tmp_path):
    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()

    # Create pose data: first 10 frames arms down, then arms up
    players = []
    player_data = {"track_id": "T001", "frames": []}
    for frame_idx in range(20):
        if frame_idx < 10:
            # Arms down
            kps = [
                {"name": "left_shoulder", "x": 100, "y": 100, "conf": 0.9},
                {"name": "left_wrist", "x": 100, "y": 200, "conf": 0.9},
                {"name": "right_shoulder", "x": 120, "y": 100, "conf": 0.9},
                {"name": "right_wrist", "x": 120, "y": 200, "conf": 0.9},
            ]
        else:
            # Arms up (celebrating)
            kps = [
                {"name": "left_shoulder", "x": 100, "y": 200, "conf": 0.9},
                {"name": "left_wrist", "x": 100, "y": 100, "conf": 0.9},
                {"name": "right_shoulder", "x": 120, "y": 200, "conf": 0.9},
                {"name": "right_wrist", "x": 120, "y": 100, "conf": 0.9},
            ]
        player_data["frames"].append({"frame": frame_idx, "keypoints": kps})
    players.append(player_data)

    import json
    (poses_dir / "test_shot_poses.json").write_text(
        json.dumps({"shot_id": "test_shot", "players": players})
    )

    onset = _detect_celebration_onset(
        poses_dir, "test_shot", total_frames=20,
        angle_threshold=60.0, fraction_threshold=0.3, smooth_window=1,
    )
    assert onset is not None
    assert 9 <= onset <= 12  # Should detect around frame 10


def test_detect_celebration_onset_no_poses(tmp_path):
    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()
    onset = _detect_celebration_onset(poses_dir, "nonexistent", total_frames=20)
    assert onset is None


def test_compute_celebration_signal_counts_celebrating_players(tmp_path):
    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()

    # 2 players: one always celebrating, one never
    players = []
    for pid, celebrating in [("P1", True), ("P2", False)]:
        frames = []
        for fidx in range(10):
            if celebrating:
                kps = [
                    {"name": "left_shoulder", "x": 100, "y": 200, "conf": 0.9},
                    {"name": "left_wrist", "x": 100, "y": 100, "conf": 0.9},
                    {"name": "right_shoulder", "x": 120, "y": 200, "conf": 0.9},
                    {"name": "right_wrist", "x": 120, "y": 100, "conf": 0.9},
                ]
            else:
                kps = [
                    {"name": "left_shoulder", "x": 100, "y": 100, "conf": 0.9},
                    {"name": "left_wrist", "x": 100, "y": 200, "conf": 0.9},
                    {"name": "right_shoulder", "x": 120, "y": 100, "conf": 0.9},
                    {"name": "right_wrist", "x": 120, "y": 200, "conf": 0.9},
                ]
            frames.append({"frame": fidx, "keypoints": kps})
        players.append({"track_id": pid, "frames": frames})

    import json
    (poses_dir / "test_poses.json").write_text(
        json.dumps({"shot_id": "test", "players": players})
    )

    signal = _compute_celebration_signal(
        poses_dir, "test", total_frames=10,
        angle_threshold=60.0, smooth_window=1,
    )
    assert len(signal) == 10
    # Exactly 1 player celebrating per frame (P1 only)
    assert all(signal[i] == pytest.approx(1.0) for i in range(10))


# ---------------------------------------------------------------------------
# Tests for matched pitch-trajectory refinement helpers
# ---------------------------------------------------------------------------

def _make_tracks_with_positions(
    track_id: str,
    positions: list[tuple[float, float]],
    class_name: str = "player",
    team: str = "A",
    frame_offset: int = 0,
) -> TracksResult:
    """Helper: build a TracksResult with one track whose pitch_position values
    follow ``positions`` starting at frame ``frame_offset``."""
    frames = [
        TrackFrame(
            frame=frame_offset + i,
            bbox=[10.0, 10.0, 50.0, 50.0],
            confidence=0.9,
            pitch_position=list(pos),
        )
        for i, pos in enumerate(positions)
    ]
    track = Track(track_id=track_id, class_name=class_name, team=team, frames=frames)
    return TracksResult(shot_id="test", tracks=[track])


def test_extract_pitch_axis_trajectory_returns_correct_shape():
    positions = [(10.0, 20.0), (11.0, 21.0), (12.0, 22.0)]
    tracks = _make_tracks_with_positions("T1", positions, frame_offset=2)
    total_frames = 8

    x_traj = _extract_pitch_axis_trajectory(tracks, "T1", total_frames, axis=0)
    y_traj = _extract_pitch_axis_trajectory(tracks, "T1", total_frames, axis=1)

    assert x_traj.shape == (total_frames,)
    assert y_traj.shape == (total_frames,)

    # Frames 0-1 before the track starts: NaN
    assert np.isnan(x_traj[0]) and np.isnan(x_traj[1])
    # Frames 2-4 match the positions
    assert x_traj[2] == pytest.approx(10.0)
    assert x_traj[4] == pytest.approx(12.0)
    assert y_traj[3] == pytest.approx(21.0)
    # Frames 5-7 after the track ends: NaN
    assert np.isnan(x_traj[5])


def test_align_by_matched_pitch_trajectories_finds_offset():
    """Ref and shot share identical pitch trajectories but shot is shifted by 5 frames."""
    n_ref = 50
    n_shot = 50
    true_offset = 5

    # Build a sinusoidal path shared by both reference and shot
    t = np.linspace(0, 2 * np.pi, n_ref)
    xs = (np.sin(t) * 20 + 50).tolist()
    ys = (np.cos(t) * 10 + 34).tolist()

    ref_positions = list(zip(xs, ys))

    # Shot starts 5 frames later in reference time → shot frame 0 = ref frame 5
    shot_start_in_ref = true_offset
    shot_positions = ref_positions[shot_start_in_ref:]

    ref_tracks = _make_tracks_with_positions("T_ref", ref_positions, frame_offset=0)
    shot_tracks = _make_tracks_with_positions("T_shot", shot_positions, frame_offset=0)

    result = _align_by_matched_pitch_trajectories(
        ref_tracks=ref_tracks,
        shot_tracks=shot_tracks,
        ref_n_frames=n_ref,
        shot_n_frames=len(shot_positions),
        coarse_offset=true_offset,
        coarse_match_distance_m=50.0,  # very relaxed for a single-player test
        n_reference_frames=30,
        agreement_tolerance=3,
    )

    assert result.valid, f"Expected valid=True, got {result}"
    assert abs(result.offset - true_offset) <= 2, (
        f"Expected offset ≈ {true_offset}, got {result.offset}"
    )


def test_align_by_matched_pitch_trajectories_no_pitch_data_returns_invalid():
    """Tracks with no pitch_position should produce valid=False."""
    frames_no_pos = [
        TrackFrame(frame=i, bbox=[0.0, 0.0, 10.0, 10.0], confidence=0.9, pitch_position=None)
        for i in range(20)
    ]
    empty_track = Track(track_id="T1", class_name="player", team="A", frames=frames_no_pos)
    ref_tracks = TracksResult(shot_id="ref", tracks=[empty_track])

    frames_no_pos2 = [
        TrackFrame(frame=i, bbox=[0.0, 0.0, 10.0, 10.0], confidence=0.9, pitch_position=None)
        for i in range(20)
    ]
    empty_track2 = Track(track_id="T2", class_name="player", team="A", frames=frames_no_pos2)
    shot_tracks = TracksResult(shot_id="shot", tracks=[empty_track2])

    result = _align_by_matched_pitch_trajectories(
        ref_tracks=ref_tracks,
        shot_tracks=shot_tracks,
        ref_n_frames=20,
        shot_n_frames=20,
        coarse_offset=3,
        coarse_match_distance_m=50.0,
    )

    assert not result.valid


def test_fill_nans_interpolates_gaps():
    sig = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
    filled = _fill_nans(sig)
    assert not np.any(np.isnan(filled))
    assert filled[0] == pytest.approx(1.0)
    assert filled[3] == pytest.approx(4.0)
    # Linear interpolation between 1.0 at index 0 and 4.0 at index 3
    assert filled[1] == pytest.approx(2.0)
    assert filled[2] == pytest.approx(3.0)


def test_alignment_loads_without_graph_residual():
    """Old sync_map.json files without graph_residual_frames must still load."""
    import json
    import tempfile
    from src.schemas.sync_map import Alignment, SyncMap

    data = {
        "reference_shot": "shot_001",
        "alignments": [
            {
                "shot_id": "shot_002",
                "frame_offset": 141,
                "confidence": 0.8,
                "method": "audio",
                "overlap_frames": [0, 365],
            }
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        p = Path(f.name)

    sm = SyncMap.load(p)
    assert sm.alignments[0].graph_residual_frames is None
    p.unlink()


def test_alignment_saves_and_loads_graph_residual():
    import json
    import tempfile
    from src.schemas.sync_map import Alignment, SyncMap

    sm = SyncMap(
        reference_shot="shot_001",
        alignments=[
            Alignment(
                shot_id="shot_002",
                frame_offset=141,
                confidence=0.8,
                method="audio",
                overlap_frames=[0, 365],
                graph_residual_frames=7.5,
            )
        ],
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        p = Path(f.name)
    sm.save(p)
    loaded = SyncMap.load(p)
    assert loaded.alignments[0].graph_residual_frames == pytest.approx(7.5)
    p.unlink()


def test_align_audio_max_lag_excludes_true_peak(tmp_path):
    """When the true lag exceeds max_lag_frames the result should not find it."""
    import tempfile

    sample_rate = 16000
    fps = 25.0
    duration_s = 4.0
    n_samples = int(sample_rate * duration_s)

    # Burst signal at t=3s in ref, t=0.5s in shot → true lag = 2.5s = 62 frames
    ref_audio = np.zeros(n_samples, dtype=np.float32)
    ref_audio[int(sample_rate * 3.0) : int(sample_rate * 3.0) + 400] = 1.0
    shot_audio = np.zeros(n_samples, dtype=np.float32)
    shot_audio[int(sample_rate * 0.5) : int(sample_rate * 0.5) + 400] = 1.0

    def _write_wav(path: Path, data: np.ndarray) -> None:
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((data * 32767).astype(np.int16).tobytes())

    ref_path = tmp_path / "ref.wav"
    shot_path = tmp_path / "shot.wav"
    _write_wav(ref_path, ref_audio)
    _write_wav(shot_path, shot_audio)

    # Test that calling _align_audio with max_lag_frames parameter doesn't crash
    # and returns an AlignmentEstimate
    est_windowed = _align_audio(ref_path, shot_path, fps, max_lag_frames=5)
    assert isinstance(est_windowed, AlignmentEstimate)
    # The offset should be reasonably constrained (within ±5 frames if windowing worked)
    # or 0 if audio couldn't be loaded (test environment may lack ffmpeg for wav)
    assert abs(est_windowed.offset) <= 5


def test_solve_offset_graph_single_clip():
    """One clip with one pass-1 edge — solved offset equals the estimate."""
    solved = _solve_offset_graph(n_clips=2, edges=[(0, 1, 141.0, 0.8)])
    assert solved[0] == pytest.approx(0.0)
    assert solved[1] == pytest.approx(141.0)


def test_solve_offset_graph_consistent_triangle():
    """Three clips with consistent pass-1 + pass-2 edges — solution matches."""
    edges = [
        (0, 1, 141.0, 0.8),
        (0, 2, 393.0, 0.7),
        (1, 2, 252.0, 0.9),
    ]
    solved = _solve_offset_graph(n_clips=3, edges=edges)
    assert solved[0] == pytest.approx(0.0)
    assert solved[1] == pytest.approx(141.0, abs=1.0)
    assert solved[2] == pytest.approx(393.0, abs=1.0)


def test_solve_offset_graph_conflicting_edges_compromise():
    """Pass-2 edge contradicts pass-1 — solution is a weighted compromise."""
    edges = [
        (0, 1, 100.0, 0.5),
        (0, 2, 200.0, 0.5),
        (1, 2, 80.0, 2.0),
    ]
    solved = _solve_offset_graph(n_clips=3, edges=edges)
    assert solved[2] < 200.0


def test_solve_offset_graph_zero_weight_edges_ignored():
    """Zero-weight edges have no effect on the solution."""
    edges_with = [
        (0, 1, 141.0, 0.8),
        (0, 2, 393.0, 0.7),
        (1, 2, 999.0, 0.0),
    ]
    edges_without = [
        (0, 1, 141.0, 0.8),
        (0, 2, 393.0, 0.7),
    ]
    solved_with = _solve_offset_graph(n_clips=3, edges=edges_with)
    solved_without = _solve_offset_graph(n_clips=3, edges=edges_without)
    assert solved_with[1] == pytest.approx(solved_without[1], abs=0.1)
    assert solved_with[2] == pytest.approx(solved_without[2], abs=0.1)


def test_solve_offset_graph_reference_always_zero():
    """Reference clip (index 0) is always 0.0 regardless of edges."""
    solved = _solve_offset_graph(n_clips=3, edges=[(0, 1, 50.0, 1.0), (0, 2, 100.0, 1.0)])
    assert solved[0] == pytest.approx(0.0)


def test_solve_offset_graph_single_clip_returns_correct_shape():
    """n_clips=1 (only reference) — returns array of length 1."""
    solved = _solve_offset_graph(n_clips=1, edges=[])
    assert len(solved) == 1
    assert solved[0] == pytest.approx(0.0)


def test_collect_pairwise_estimates_returns_correct_pairs(tmp_path):
    """With 3 shots (1 ref + 2 non-ref), exactly 1 pair is returned with correct indices."""
    from src.schemas.tracks import TracksResult

    # Create minimal dummy video files (1-frame black clips)
    (tmp_path / "shots").mkdir()
    for name in ("shot_001.mp4", "shot_002.mp4", "shot_003.mp4"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(tmp_path / "shots" / name), fourcc, 25.0, (64, 36))
        out.write(np.zeros((36, 64, 3), dtype=np.uint8))
        out.release()

    shots = [
        Shot(id="shot_001", clip_file="shots/shot_001.mp4", start_frame=0, end_frame=25,
             start_time=0.0, end_time=1.0),
        Shot(id="shot_002", clip_file="shots/shot_002.mp4", start_frame=0, end_frame=25,
             start_time=0.0, end_time=1.0),
        Shot(id="shot_003", clip_file="shots/shot_003.mp4", start_frame=0, end_frame=25,
             start_time=0.0, end_time=1.0),
    ]
    clips = {s.id: tmp_path / s.clip_file for s in shots}
    tracks_by_shot: dict[str, TracksResult] = {}
    n_frames = {"shot_001": 25, "shot_002": 25, "shot_003": 25}
    pass1 = {
        "shot_002": AlignmentEstimate(offset=141, confidence=0.8, method="audio", valid=True),
        "shot_003": AlignmentEstimate(offset=393, confidence=0.7, method="audio", valid=True),
    }
    cfg = {"min_confidence": 0.3, "pass2_search_margin_frames": 30}

    results = _collect_pairwise_estimates(
        shots=shots,
        clips=clips,
        tracks_by_shot=tracks_by_shot,
        n_frames_by_shot=n_frames,
        pass1_estimates=pass1,
        poses_dir=tmp_path / "poses",
        fps=25.0,
        cfg=cfg,
    )

    # 2 non-reference shots → 1 pair
    assert len(results) == 1
    i, j, est = results[0]
    assert i == 1  # shot_002 is index 1
    assert j == 2  # shot_003 is index 2
    assert isinstance(est, AlignmentEstimate)
