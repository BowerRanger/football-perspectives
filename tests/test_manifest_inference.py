"""TDD tests for manifest inference from pre-prepared clips."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.schemas.shots import ShotsManifest


def test_infer_from_clips_basic(tmp_path):
    """Infer manifest from clip filenames with sequential IDs."""
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    
    # Create dummy clips
    _create_dummy_clip(shots_dir / "shot_001.mp4", fps=25.0, frames=250)
    _create_dummy_clip(shots_dir / "shot_002.mp4", fps=25.0, frames=300)
    _create_dummy_clip(shots_dir / "shot_003.mp4", fps=25.0, frames=200)
    
    manifest = ShotsManifest.infer_from_clips(shots_dir)
    
    assert manifest.source_file == "unknown"  # No original source known
    assert manifest.fps == 25.0  # Extracted from first clip
    assert len(manifest.shots) == 3
    assert manifest.shots[0].id == "shot_001"
    assert manifest.shots[1].id == "shot_002"
    assert manifest.shots[2].id == "shot_003"
    assert manifest.shots[0].clip_file == "shots/shot_001.mp4"


def test_infer_from_clips_supports_prepared_clip_names(tmp_path):
    """Infer manifest entries from arbitrary prepared clip names."""
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()

    _create_dummy_clip(shots_dir / "origi02.mp4", fps=25.0, frames=300)
    _create_dummy_clip(shots_dir / "origi01.mp4", fps=25.0, frames=250)

    manifest = ShotsManifest.infer_from_clips(shots_dir)

    assert [shot.id for shot in manifest.shots] == ["origi01", "origi02"]
    assert [shot.clip_file for shot in manifest.shots] == [
        "shots/origi01.mp4",
        "shots/origi02.mp4",
    ]


def test_infer_from_clips_extracts_fps_from_video(tmp_path):
    """Extract fps and frame count from actual clip using cv2."""
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    
    _create_dummy_clip(shots_dir / "shot_001.mp4", fps=30.0, frames=600)
    
    manifest = ShotsManifest.infer_from_clips(shots_dir)
    
    assert manifest.fps == 30.0
    assert manifest.total_frames == 600


def test_infer_from_clips_empty_directory(tmp_path):
    """Empty shots directory returns empty manifest."""
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    
    manifest = ShotsManifest.infer_from_clips(shots_dir)
    
    assert len(manifest.shots) == 0
    assert manifest.fps == 25.0  # Default fallback


def test_infer_from_clips_non_sequential_ids(tmp_path):
    """Handle non-sequential shot IDs (shot_001, shot_005, shot_012)."""
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    
    _create_dummy_clip(shots_dir / "shot_001.mp4", fps=25.0, frames=250)
    _create_dummy_clip(shots_dir / "shot_005.mp4", fps=25.0, frames=300)
    _create_dummy_clip(shots_dir / "shot_012.mp4", fps=25.0, frames=200)
    
    manifest = ShotsManifest.infer_from_clips(shots_dir)
    
    assert len(manifest.shots) == 3
    assert manifest.shots[0].id == "shot_001"
    assert manifest.shots[1].id == "shot_005"
    assert manifest.shots[2].id == "shot_012"


def test_infer_from_clips_ignores_non_shot_files(tmp_path):
    """Ignore files that don't match shot_XXX.mp4 pattern."""
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    
    _create_dummy_clip(shots_dir / "shot_001.mp4", fps=25.0, frames=250)
    (shots_dir / "random.mp4").write_text("")  # Not a shot file
    (shots_dir / "shots_manifest.json").write_text("{}")  # Manifest file
    
    manifest = ShotsManifest.infer_from_clips(shots_dir)
    
    assert len(manifest.shots) == 1
    assert manifest.shots[0].id == "shot_001"


def test_infer_from_clips_prefers_numbered_shots_before_other_names(tmp_path):
    """Numbered shot clips should retain numeric ordering ahead of arbitrary names."""
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()

    _create_dummy_clip(shots_dir / "origi01.mp4", fps=25.0, frames=250)
    _create_dummy_clip(shots_dir / "shot_010.mp4", fps=25.0, frames=250)
    _create_dummy_clip(shots_dir / "shot_002.mp4", fps=25.0, frames=250)

    manifest = ShotsManifest.infer_from_clips(shots_dir)

    assert [shot.id for shot in manifest.shots] == ["shot_002", "shot_010", "origi01"]


def test_infer_from_clips_skips_unreadable_video_files(tmp_path):
    """Unreadable prepared clips should be ignored rather than poisoning inference."""
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()

    _create_dummy_clip(shots_dir / "shot_001.mp4", fps=25.0, frames=250)
    (shots_dir / "broken.mp4").write_text("not a video")

    manifest = ShotsManifest.infer_from_clips(shots_dir)

    assert [shot.id for shot in manifest.shots] == ["shot_001"]


def test_infer_from_clips_rejects_mixed_fps(tmp_path):
    """Prepared clips with different FPS values should fail loudly."""
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()

    _create_dummy_clip(shots_dir / "shot_001.mp4", fps=25.0, frames=250)
    _create_dummy_clip(shots_dir / "shot_002.mp4", fps=30.0, frames=300)

    with pytest.raises(ValueError, match="common FPS"):
        ShotsManifest.infer_from_clips(shots_dir)


def test_calibration_stage_uses_inferred_manifest_when_missing(tmp_path):
    """Calibration stage falls back to inferred manifest when shots_manifest.json missing."""
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    
    # Create pre-prepared clips (no manifest)
    _create_dummy_clip(shots_dir / "shot_001.mp4", fps=25.0, frames=250)
    _create_dummy_clip(shots_dir / "shot_002.mp4", fps=25.0, frames=300)
    
    # Calibration stage should infer manifest and process
    # (This will be implemented after the schema method exists)
    from src.stages.calibration import CameraCalibrationStage
    
    stage = CameraCalibrationStage(
        config={},
        output_dir=tmp_path,
        detector=None  # Skip detection
    )
    
    # Should not crash - will infer manifest
    stage.run()
    
    # Verify calibration files were created
    cal_dir = tmp_path / "calibration"
    assert (cal_dir / "shot_001_calibration.json").exists()
    assert (cal_dir / "shot_002_calibration.json").exists()
    assert (shots_dir / "shots_manifest.json").exists()


def test_sync_stage_uses_inferred_manifest_when_missing(tmp_path):
    """Sync stage falls back to inferred manifest when shots_manifest.json missing."""
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    
    _create_dummy_clip(shots_dir / "shot_001.mp4", fps=25.0, frames=250)
    _create_dummy_clip(shots_dir / "shot_002.mp4", fps=25.0, frames=300)
    
    from src.stages.sync import TemporalSyncStage
    
    stage = TemporalSyncStage(
        config={},
        output_dir=tmp_path,
        ball_detector=None
    )
    
    # Should not crash - will infer manifest
    stage.run()
    
    # Verify sync map was created
    assert (tmp_path / "sync" / "sync_map.json").exists()
    assert (shots_dir / "shots_manifest.json").exists()


# Helper
def _create_dummy_clip(path: Path, fps: float, frames: int):
    """Create a minimal valid video file using cv2."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (640, 480))
    
    for _ in range(frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        writer.write(frame)
    
    writer.release()
