"""Tests for :class:`src.utils.ball_detector.WASBBallDetector`.

Two layers:

* a unit test that builds the HRNet from a randomly-initialised
  checkpoint to confirm the wrapper's construction, preprocessing,
  forward, and postprocessing wiring all work end-to-end without a real
  pretrained model;
* an integration smoke test against the actual ``wasb_soccer_best``
  checkpoint that auto-skips when the file isn't present.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_WASB_SUBMODULE = _REPO_ROOT / "third_party" / "wasb_sbdt" / "src" / "main.py"
_WASB_CHECKPOINT = (
    _REPO_ROOT / "third_party" / "wasb_sbdt" / "pretrained_weights"
    / "wasb_soccer_best.pth.tar"
)


@pytest.mark.unit
def test_wasb_ball_detector_runs_with_random_weights(tmp_path: Path):
    """Build an untrained HRNet checkpoint and run a frame through it.

    Verifies the construction path, frame preprocessing, forward pass,
    and postprocessing return shape — without depending on real model
    weights. An untrained network won't detect anything meaningfully,
    so the test only asserts that ``detect`` returns either ``None`` or
    a well-formed tuple."""
    if not _WASB_SUBMODULE.exists():
        pytest.skip("WASB-SBDT submodule not initialised")
    import sys
    sys.path.insert(0, str(_WASB_SUBMODULE.parent))
    import torch
    from omegaconf import OmegaConf
    from models.hrnet import HRNet  # noqa: F401  — ensures upstream import works

    from src.utils.wasb_ball_detector import _WASB_MODEL_CFG

    model = HRNet(OmegaConf.create(_WASB_MODEL_CFG))
    ckpt_path = tmp_path / "wasb_random.pth.tar"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    from src.utils.ball_detector import WASBBallDetector
    detector = WASBBallDetector(
        checkpoint=ckpt_path, confidence=0.99, input_size=(512, 288),
    )

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(frame, (640, 360), 14, (240, 240, 240), -1)

    out = detector.detect(frame)
    if out is not None:
        u, v, conf = out
        assert 0.0 <= u <= 1280.0
        assert 0.0 <= v <= 720.0
        assert 0.0 <= conf <= 1.0


@pytest.mark.integration
def test_ball_stage_constructs_wasb_detector_from_config(tmp_path: Path):
    """End-to-end: ``ball.detector: wasb`` should drive BallStage to
    instantiate WASBBallDetector and run inference without errors."""
    if not _WASB_SUBMODULE.exists():
        pytest.skip("WASB-SBDT submodule not initialised")
    if not _WASB_CHECKPOINT.exists():
        pytest.skip(f"WASB checkpoint not present at {_WASB_CHECKPOINT}")

    from src.schemas.camera_track import CameraFrame, CameraTrack
    from src.schemas.shots import Shot, ShotsManifest
    from src.stages.ball import BallStage

    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([-52.5, 30.0, -30.0])
    n = 5
    CameraTrack(
        clip_id="clip", fps=30.0, image_size=(1280, 720), t_world=t.tolist(),
        frames=tuple(
            CameraFrame(frame=i, K=K.tolist(), R=R.tolist(), confidence=1.0,
                        is_anchor=(i == 0))
            for i in range(n)
        ),
    ).save(tmp_path / "camera" / "clip_camera_track.json")

    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    clip_path = shots_dir / "clip.mp4"
    writer = cv2.VideoWriter(
        str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240)
    )
    for _ in range(n):
        writer.write(np.full((240, 320, 3), [50, 120, 50], dtype=np.uint8))
    writer.release()
    ShotsManifest(
        source_file="x", fps=30.0, total_frames=n,
        shots=[Shot(id="clip", start_frame=0, end_frame=n - 1, start_time=0.0,
                    end_time=n / 30.0, clip_file="shots/clip.mp4")],
    ).save(shots_dir / "shots_manifest.json")

    cfg = {"ball": {
        "detector": "wasb",
        "wasb": {
            "checkpoint": str(_WASB_CHECKPOINT),
            "confidence": 0.3,
            "input_size": (512, 288),
        },
    }}
    BallStage(config=cfg, output_dir=tmp_path).run()
    assert (tmp_path / "ball" / "clip_ball_track.json").exists()


@pytest.mark.integration
def test_wasb_ball_detector_locates_ball_with_real_checkpoint():
    """Real-checkpoint smoke test on a synthetic pitch + ball scene.

    The pretrained network was trained on actual broadcast footage so
    confidence on a synthetic disc is intrinsically low (≈1 %), but the
    spatial localisation is still accurate to within a few pixels.
    We assert on that — not on a broadcast-grade confidence value."""
    if not _WASB_SUBMODULE.exists():
        pytest.skip("WASB-SBDT submodule not initialised")
    if not _WASB_CHECKPOINT.exists():
        pytest.skip(
            f"WASB-SBDT checkpoint not present at {_WASB_CHECKPOINT}. "
            "Run third_party/wasb_sbdt/src/setup_scripts/setup_weights.sh "
            "to download."
        )

    from src.utils.ball_detector import WASBBallDetector

    # Low threshold so the synthetic scene's weak heatmap response
    # still triggers a detection — real broadcast frames clear the
    # default 0.3 gate comfortably.
    detector = WASBBallDetector(
        checkpoint=str(_WASB_CHECKPOINT), confidence=0.005,
        input_size=(512, 288),
    )

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (50, 120, 50)  # pitch-green backdrop
    cv2.circle(frame, (640, 360), 12, (240, 240, 240), -1)

    out = detector.detect(frame)
    assert out is not None, "WASB returned no detection at threshold=0.005"
    u, v, conf = out
    assert abs(u - 640) < 10, f"u off by {abs(u - 640):.1f} px"
    assert abs(v - 360) < 10, f"v off by {abs(v - 360):.1f} px"
    assert 0.0 <= conf <= 1.0
