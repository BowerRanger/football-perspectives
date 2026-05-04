import pytest

pytest.skip("prepare_shots stage removed", allow_module_level=True)


def _prepared_manifest_path(root: Path) -> Path:
    return root / "prepare_shots" / "shots_manifest.json"


def test_shot_prep_stage_accepts_extra_kwargs_from_pipeline(tmp_path):
    """Stage should accept extra kwargs like video_path and device without error."""
    stage = ShotPrepStage(
        config={},
        output_dir=tmp_path,
        video_path=Path("dummy.mp4"),
        device="cpu",
    )
    assert stage.output_dir == tmp_path
    assert stage.config == {}


def _create_moving_square_clip(path: Path, fps: float, frames: int, step: int) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (320, 240),
    )
    for idx in range(frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        x = 20 + (idx * step)
        x = max(0, min(280, x))
        cv2.rectangle(frame, (x, 100), (x + 40, 160), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()


def _create_static_clip(path: Path, fps: float, frames: int) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (320, 240),
    )
    frame = np.full((240, 320, 3), 32, dtype=np.uint8)
    for _ in range(frames):
        writer.write(frame)
    writer.release()


def _frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def _create_moving_square_clip_sized(
    path: Path,
    fps: float,
    frames: int,
    step: int,
    square_size: int = 40,
    width: int = 320,
    height: int = 240,
) -> None:
    """Moving square clip with configurable square size — used to simulate zoom differences."""
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    max_x = max(0, width - square_size - 20)
    for idx in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x = min(20 + idx * step, max_x)
        cv2.rectangle(frame, (x, 80), (x + square_size, 80 + square_size), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()


def test_speed_factor_identical_clips_returns_one(tmp_path):
    clip = tmp_path / "same.mp4"
    _create_moving_square_clip(clip, fps=25.0, frames=50, step=6)

    factor = _estimate_speed_factor(clip, clip, n_samples=10, min_flow_magnitude=0.5)
    assert 0.95 <= factor <= 1.05


def test_speed_factor_slow_clip_returns_gt_one(tmp_path):
    ref_clip = tmp_path / "ref.mp4"
    slow_clip = tmp_path / "slow.mp4"
    _create_moving_square_clip(ref_clip, fps=25.0, frames=50, step=8)
    _create_moving_square_clip(slow_clip, fps=25.0, frames=50, step=4)

    factor = _estimate_speed_factor(ref_clip, slow_clip, n_samples=10, min_flow_magnitude=0.5)
    assert factor > 1.2


def test_speed_factor_static_scene_returns_one(tmp_path):
    ref_clip = tmp_path / "ref.mp4"
    shot_clip = tmp_path / "static.mp4"
    _create_moving_square_clip(ref_clip, fps=25.0, frames=50, step=8)
    _create_static_clip(shot_clip, fps=25.0, frames=50)

    factor = _estimate_speed_factor(ref_clip, shot_clip, n_samples=10, min_flow_magnitude=0.5)
    assert factor == 1.0


def test_retime_clip_shortens_slow_clip(tmp_path):
    clip = tmp_path / "clip.mp4"
    _create_moving_square_clip(clip, fps=25.0, frames=60, step=4)
    before = _frame_count(clip)

    retime_clip(path=clip, speed_factor=2.0, fps=25.0)
    after = _frame_count(clip)

    assert before > 0
    assert after < before


def test_stage_skips_reference_clip(tmp_path, monkeypatch):
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir(parents=True)

    clip_1 = shots_dir / "shot_001.mp4"
    clip_2 = shots_dir / "shot_002.mp4"
    _create_moving_square_clip(clip_1, fps=25.0, frames=40, step=8)
    _create_moving_square_clip(clip_2, fps=25.0, frames=40, step=4)

    manifest = ShotsManifest(
        source_file="dummy.mp4",
        fps=25.0,
        total_frames=80,
        shots=[
            Shot("shot_001", 0, 39, 0.0, 1.6, "shots/shot_001.mp4"),
            Shot("shot_002", 40, 79, 1.6, 3.2, "shots/shot_002.mp4"),
        ],
    )
    manifest.save(shots_dir / "shots_manifest.json")

    monkeypatch.setattr("src.stages.prepare_shots._estimate_speed_factor", lambda *args, **kwargs: 2.0)

    stage = ShotPrepStage(
        config={
            "prepare_shots": {
                "remove_duplicate_frames": False,
                "normalise_speed": True,
                "speed_factor_threshold": 0.15,
            }
        },
        output_dir=tmp_path,
    )
    stage.run()

    updated = ShotsManifest.load(_prepared_manifest_path(tmp_path))
    assert updated.shots[0].speed_factor == 1.0
    # Source clips must be untouched
    assert clip_1.exists()
    assert clip_2.exists()
    # Processed copies are in prepare_shots/
    assert (tmp_path / "prepare_shots" / "shot_001.mp4").exists()
    assert (tmp_path / "prepare_shots" / "shot_002.mp4").exists()


def test_stage_normalises_slow_shot(tmp_path, monkeypatch):
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir(parents=True)

    clip_1 = shots_dir / "shot_001.mp4"
    clip_2 = shots_dir / "shot_002.mp4"
    _create_moving_square_clip(clip_1, fps=25.0, frames=40, step=8)
    _create_moving_square_clip(clip_2, fps=25.0, frames=40, step=4)

    manifest = ShotsManifest(
        source_file="dummy.mp4",
        fps=25.0,
        total_frames=80,
        shots=[
            Shot("shot_001", 0, 39, 0.0, 1.6, "shots/shot_001.mp4"),
            Shot("shot_002", 40, 79, 1.6, 3.2, "shots/shot_002.mp4"),
        ],
    )
    manifest.save(shots_dir / "shots_manifest.json")

    monkeypatch.setattr("src.stages.prepare_shots._estimate_speed_factor", lambda *args, **kwargs: 2.0)
    before = _frame_count(clip_2)

    stage = ShotPrepStage(
        config={
            "prepare_shots": {
                "remove_duplicate_frames": False,
                "normalise_speed": True,
                "speed_factor_threshold": 0.15,
            }
        },
        output_dir=tmp_path,
    )
    stage.run()

    updated = ShotsManifest.load(_prepared_manifest_path(tmp_path))
    # Source clip is unchanged; the retimed copy is in prepare_shots/.
    assert _frame_count(clip_2) == before
    prepared_clip_2 = tmp_path / "prepare_shots" / "shot_002.mp4"
    assert prepared_clip_2.exists()
    assert _frame_count(prepared_clip_2) < before
    # speed_factor now records total temporal change (orig_frames / final_frames).
    # With a 2.0 retiming the clip roughly halves → speed_factor ≈ 2.0.
    assert abs(updated.shots[1].speed_factor - 2.0) < 0.3
    assert updated.shots[1].clip_file == "prepare_shots/shot_002.mp4"


def test_stage_infers_manifest_when_missing(tmp_path):
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir(parents=True)

    _create_moving_square_clip(shots_dir / "origi01.mp4", fps=25.0, frames=30, step=6)
    _create_moving_square_clip(shots_dir / "origi02.mp4", fps=25.0, frames=30, step=6)

    stage = ShotPrepStage(
        config={"prepare_shots": {"remove_duplicate_frames": False, "normalise_speed": False}},
        output_dir=tmp_path,
    )
    stage.run()

    manifest_path = _prepared_manifest_path(tmp_path)
    assert manifest_path.exists()
    manifest = ShotsManifest.load(manifest_path)
    assert len(manifest.shots) == 2


def test_stage_updates_shot_end_time_after_retime(tmp_path, monkeypatch):
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir(parents=True)

    clip_1 = shots_dir / "shot_001.mp4"
    clip_2 = shots_dir / "shot_002.mp4"
    _create_moving_square_clip(clip_1, fps=25.0, frames=40, step=8)
    _create_moving_square_clip(clip_2, fps=25.0, frames=60, step=4)

    manifest = ShotsManifest(
        source_file="dummy.mp4",
        fps=25.0,
        total_frames=100,
        shots=[
            Shot("shot_001", 0, 39, 0.0, 1.6, "shots/shot_001.mp4"),
            Shot("shot_002", 40, 99, 1.6, 4.0, "shots/shot_002.mp4"),
        ],
    )
    manifest.save(shots_dir / "shots_manifest.json")

    monkeypatch.setattr("src.stages.prepare_shots._estimate_speed_factor", lambda *args, **kwargs: 2.0)

    stage = ShotPrepStage(
        config={"prepare_shots": {"remove_duplicate_frames": False, "normalise_speed": True}},
        output_dir=tmp_path,
    )
    stage.run()

    # end_time should reflect the retimed copy in prepare_shots/, not the original.
    prepared_clip_2 = tmp_path / "prepare_shots" / "shot_002.mp4"
    updated = ShotsManifest.load(_prepared_manifest_path(tmp_path))
    expected_end_time = updated.shots[1].start_time + (_frame_count(prepared_clip_2) / 25.0)
    assert abs(updated.shots[1].end_time - expected_end_time) < 0.02


def test_stage_prefers_01_suffix_shot_as_reference(tmp_path, monkeypatch):
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir(parents=True)

    clip_1 = shots_dir / "shot_100.mp4"
    clip_2 = shots_dir / "shot_001.mp4"
    _create_moving_square_clip(clip_1, fps=25.0, frames=40, step=8)
    _create_moving_square_clip(clip_2, fps=25.0, frames=40, step=4)

    manifest = ShotsManifest(
        source_file="dummy.mp4",
        fps=25.0,
        total_frames=80,
        shots=[
            Shot("shot_100", 0, 39, 0.0, 1.6, "shots/shot_100.mp4"),
            Shot("shot_001", 40, 79, 1.6, 3.2, "shots/shot_001.mp4"),
        ],
    )
    manifest.save(shots_dir / "shots_manifest.json")

    monkeypatch.setattr("src.stages.prepare_shots._estimate_speed_factor", lambda *args, **kwargs: 2.0)

    stage = ShotPrepStage(
        config={
            "prepare_shots": {
                "remove_duplicate_frames": False,
                "normalise_speed": True,
                "speed_factor_threshold": 0.15,
            }
        },
        output_dir=tmp_path,
    )
    stage.run()

    updated = ShotsManifest.load(_prepared_manifest_path(tmp_path))
    # shot_001 is the reference (01-suffix) → never retimed; no dedup → speed_factor 1.0
    assert updated.shots[1].speed_factor == 1.0
    # shot_100 is non-reference → retimed; speed_factor ≈ 2.0 (total change ≈ orig/final)
    assert abs(updated.shots[0].speed_factor - 2.0) < 0.3
    # Both clips should now point to prepare_shots/
    assert updated.shots[0].clip_file == "prepare_shots/shot_100.mp4"
    assert updated.shots[1].clip_file == "prepare_shots/shot_001.mp4"


def test_stage_is_complete_after_run(tmp_path):
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir(parents=True)

    _create_moving_square_clip(shots_dir / "shot_001.mp4", fps=25.0, frames=30, step=6)

    stage = ShotPrepStage(
        config={"prepare_shots": {"remove_duplicate_frames": False, "normalise_speed": False}},
        output_dir=tmp_path,
    )
    assert not stage.is_complete()
    stage.run()
    assert stage.is_complete()
    assert (tmp_path / "prepare_shots" / "shots_manifest.json").exists()


def test_speed_factor_zoom_invariant(tmp_path):
    """Gradient normalisation prevents zoom from inverting the slow-motion detection.

    Without normalisation a zoomed shot has larger raw optical-flow magnitudes than
    a wide-angle reference even when the zoomed shot is slow-motion — producing a
    speed_factor < 1.0 (wrong direction).  With gradient normalisation the spatial
    detail density cancels the zoom factor so the result is > 1.0 (correct).

    Scenario:
      ref      : small 40px square, step=4px/frame
      slow_zoom: large 80px square (simulates 2× zoom), step=6px/frame
        raw flow: 6 > 4  → without normalisation factor = 4/6 = 0.67 (WRONG)
        gradients: 80px square has ~2× more edge pixels than 40px square
        normalised: (4/grad_ref) / (6/(2×grad_ref)) = 4/(6/2) = 1.33 (CORRECT)
    """
    ref_clip = tmp_path / "ref.mp4"
    slow_zoomed_clip = tmp_path / "slow_zoomed.mp4"

    _create_moving_square_clip_sized(ref_clip, fps=25.0, frames=30, step=4, square_size=40)
    _create_moving_square_clip_sized(slow_zoomed_clip, fps=25.0, frames=30, step=6, square_size=80)

    factor = _estimate_speed_factor(
        ref_clip, slow_zoomed_clip, n_samples=8, min_flow_magnitude=0.5
    )
    assert factor > 1.0, (
        f"Zoomed slow-motion clip should yield speed_factor > 1.0, got {factor:.3f}. "
        "Gradient normalisation may not be functioning correctly."
    )


def test_stage_dedup_applies_to_reference(tmp_path):
    """Dedup (freeze-frame removal) runs on the reference clip; speed_factor stays 1.0."""
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir(parents=True)

    clip_1 = shots_dir / "shot_001.mp4"
    clip_2 = shots_dir / "shot_002.mp4"
    _create_moving_square_clip(clip_1, fps=25.0, frames=40, step=8)
    _create_moving_square_clip(clip_2, fps=25.0, frames=40, step=4)

    manifest = ShotsManifest(
        source_file="dummy.mp4",
        fps=25.0,
        total_frames=80,
        shots=[
            Shot("shot_001", 0, 39, 0.0, 1.6, "shots/shot_001.mp4"),
            Shot("shot_002", 40, 79, 1.6, 3.2, "shots/shot_002.mp4"),
        ],
    )
    manifest.save(shots_dir / "shots_manifest.json")

    stage = ShotPrepStage(
        config={
            "prepare_shots": {
                "remove_duplicate_frames": True,
                "normalise_speed": False,
            }
        },
        output_dir=tmp_path,
    )
    stage.run()

    updated = ShotsManifest.load(_prepared_manifest_path(tmp_path))
    # Reference always 1.0, even with dedup enabled.
    assert updated.shots[0].speed_factor == 1.0
    # Dedup was attempted on the reference — the prepared copy must exist.
    assert (tmp_path / "prepare_shots" / "shot_001.mp4").exists()
    # Source clips are untouched.
    assert clip_1.exists()


def test_stage_speed_factor_excludes_dedup(tmp_path, monkeypatch):
    """Dedup frame-count changes do not bleed into speed_factor for non-reference shots."""
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir(parents=True)

    clip_1 = shots_dir / "shot_001.mp4"
    clip_2 = shots_dir / "shot_002.mp4"
    _create_moving_square_clip(clip_1, fps=25.0, frames=40, step=8)
    _create_moving_square_clip(clip_2, fps=25.0, frames=40, step=4)

    manifest = ShotsManifest(
        source_file="dummy.mp4",
        fps=25.0,
        total_frames=80,
        shots=[
            Shot("shot_001", 0, 39, 0.0, 1.6, "shots/shot_001.mp4"),
            Shot("shot_002", 40, 79, 1.6, 3.2, "shots/shot_002.mp4"),
        ],
    )
    manifest.save(shots_dir / "shots_manifest.json")

    # Optical flow sees no speed difference — speed_factor must be 1.0 regardless
    # of any frame-count change that dedup introduces.
    monkeypatch.setattr(
        "src.stages.prepare_shots._estimate_speed_factor",
        lambda *args, **kwargs: 1.0,
    )

    stage = ShotPrepStage(
        config={
            "prepare_shots": {
                "remove_duplicate_frames": True,
                "normalise_speed": True,
                "speed_factor_threshold": 0.15,
            }
        },
        output_dir=tmp_path,
    )
    stage.run()

    updated = ShotsManifest.load(_prepared_manifest_path(tmp_path))
    # Optical flow returned 1.0 → no tempo difference → speed_factor == 1.0.
    # Dedup may have changed frame count but that must NOT affect speed_factor.
    assert updated.shots[1].speed_factor == 1.0
