"""Tests for src.pipeline.runner.resolve_stages — single-mode stage resolution."""

from pathlib import Path

import numpy as np
import pytest

from src.pipeline.runner import resolve_stages


@pytest.mark.unit
def test_resolve_all():
    assert resolve_stages("all", None) == [
        "prepare_shots",
        "tracking",
        "camera",
        "hmr_world",
        "ball",
        "export",
    ]


@pytest.mark.unit
def test_resolve_subset():
    assert resolve_stages("camera,hmr_world", None) == ["camera", "hmr_world"]


@pytest.mark.unit
def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_stages("calibration", None)


@pytest.mark.unit
def test_resolve_with_from_stage_skips_earlier():
    result = resolve_stages("all", "hmr_world")
    assert result == ["hmr_world", "ball", "export"]


@pytest.mark.integration
def test_pipeline_two_shots_end_to_end(tmp_path: Path) -> None:
    """Stage two synthetic clips with anchors and hand-rolled SMPL fixtures,
    then run camera + export through ``run_pipeline``. Asserts that each
    shot produces its own ``{shot_id}_camera_track.json`` and
    ``{shot_id}_scene.glb`` and that the artefacts do not collide.

    Tracking and hmr_world are skipped — both call out to YOLO/GVHMR which
    are too heavy for CI. The plumbing is what's under test, not the ML
    weights, so we stage the SmplWorldTrack outputs directly.
    """
    from src.pipeline.runner import run_pipeline
    from src.schemas.shots import Shot, ShotsManifest
    from src.schemas.smpl_world import SmplWorldTrack
    from tests.fixtures.synthetic_clip import render_synthetic_clip
    from tests.test_camera_stage import (
        _LANDMARK_WORLD, _build_anchor_set, _write_clip_mp4,
    )

    clip_a = render_synthetic_clip(n_frames=20)
    clip_b = render_synthetic_clip(n_frames=20)
    shots = tmp_path / "shots"
    shots.mkdir()
    _write_clip_mp4(clip_a, shots / "alpha.mp4")
    _write_clip_mp4(clip_b, shots / "beta.mp4")
    fps = clip_a.fps if clip_a.fps > 0 else 25.0
    ShotsManifest(
        source_file=str(shots),
        fps=fps,
        total_frames=40,
        shots=[
            Shot(id="alpha", start_frame=0, end_frame=19, start_time=0.0,
                 end_time=20 / fps, clip_file="shots/alpha.mp4"),
            Shot(id="beta", start_frame=0, end_frame=19, start_time=0.0,
                 end_time=20 / fps, clip_file="shots/beta.mp4"),
        ],
    ).save(shots / "shots_manifest.json")

    n = len(clip_a.frames)
    for sid, clip in (("alpha", clip_a), ("beta", clip_b)):
        anchor_set = _build_anchor_set(clip, [0, n // 2, n - 1], _LANDMARK_WORLD)
        anchor_set.save(tmp_path / "camera" / f"{sid}_anchors.json")

    # Stage hand-rolled hmr_world output so export has something to package.
    hmr_dir = tmp_path / "hmr_world"
    hmr_dir.mkdir()
    for sid in ("alpha", "beta"):
        SmplWorldTrack(
            player_id=f"{sid}_T1",
            frames=np.array([0]),
            betas=np.zeros(10),
            thetas=np.zeros((1, 24, 3)),
            root_R=np.tile(np.eye(3), (1, 1, 1)),
            root_t=np.zeros((1, 3)),
            confidence=np.full(1, 0.9),
            shot_id=sid,
        ).save(hmr_dir / f"{sid}_T1_smpl_world.npz")

    run_pipeline(
        output_dir=tmp_path,
        stages="camera,export",
        from_stage=None,
        config={
            "camera": {"static_camera": False},
            "export": {"gltf_enabled": True, "fbx_enabled": False},
        },
    )

    # Per-shot camera tracks must exist and be distinct files.
    assert (tmp_path / "camera" / "alpha_camera_track.json").exists()
    assert (tmp_path / "camera" / "beta_camera_track.json").exists()
    # Per-shot GLBs.
    assert (tmp_path / "export" / "gltf" / "alpha_scene.glb").exists()
    assert (tmp_path / "export" / "gltf" / "beta_scene.glb").exists()
    # Legacy unprefixed file must NOT be written when manifest is present.
    assert not (tmp_path / "export" / "gltf" / "scene.glb").exists()


@pytest.mark.unit
def test_run_pipeline_shot_filter_propagates_to_stage(
    tmp_path: Path, monkeypatch,
) -> None:
    """run_pipeline(stages='camera', shot_filter='alpha') sets
    stage.shot_filter='alpha' on the constructed CameraStage."""
    from src.pipeline.runner import run_pipeline
    from src.pipeline.base import BaseStage

    captured: dict = {}

    class FakeCameraStage(BaseStage):
        name = "camera"

        def is_complete(self) -> bool:
            return False

        def run(self) -> None:
            captured["shot_filter"] = self.shot_filter

    def fake_stage_class(name: str):
        if name == "camera":
            return FakeCameraStage
        return None

    monkeypatch.setattr("src.pipeline.runner._stage_class", fake_stage_class)
    run_pipeline(
        output_dir=tmp_path,
        stages="camera",
        from_stage=None,
        config={},
        shot_filter="alpha",
    )
    assert captured["shot_filter"] == "alpha"
