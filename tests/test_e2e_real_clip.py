"""End-to-end smoke test against a real broadcast clip.

This test skips automatically when no fixture is provided. To enable it, drop
a clip + (optional) anchors file at::

    tests/fixtures/real_clip/play.mp4
    tests/fixtures/real_clip/anchors.json   # optional

The test exercises the full pipeline from tracking through export. It requires
GVHMR weights and is GPU-bound — expect minutes, not seconds.
"""

from pathlib import Path

import pytest

from src.pipeline.runner import run_pipeline


@pytest.mark.e2e
def test_full_pipeline_on_real_clip(tmp_path: Path):
    fixture = Path("tests/fixtures/real_clip")
    if not fixture.exists() or not (fixture / "play.mp4").exists():
        pytest.skip("real-clip fixture not provided")

    # Copy clip + (optional) anchors into tmp output, mirroring the layout the
    # pipeline expects after `prepare_shots`.
    shots_dir = tmp_path / "shots"
    cam_dir = tmp_path / "camera"
    shots_dir.mkdir()
    cam_dir.mkdir()
    (shots_dir / "play.mp4").write_bytes((fixture / "play.mp4").read_bytes())
    if (fixture / "anchors.json").exists():
        (cam_dir / "anchors.json").write_text((fixture / "anchors.json").read_text())

    config: dict = {}  # default config from /config/default.yaml at runtime
    run_pipeline(
        output_dir=tmp_path,
        stages="tracking,camera,pose_2d,hmr_world,ball,export",
        from_stage=None,
        config=config,
        video_path=tmp_path / "shots" / "play.mp4",
        device="auto",
    )

    assert (tmp_path / "camera" / "camera_track.json").exists()
    assert any((tmp_path / "hmr_world").glob("*_smpl_world.npz"))
    assert (tmp_path / "ball" / "ball_track.json").exists()
    assert (tmp_path / "export" / "gltf" / "scene.glb").exists()
