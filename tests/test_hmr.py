"""Tests for HMR schema, stage, and runner integration."""

import cv2
import numpy as np
import pytest
from pathlib import Path

from src.pipeline.config import load_config
from src.schemas.hmr_result import (
    SMPL22_BONES,
    SMPL22_JOINT_NAMES,
    SMPL22_PARENTS,
    HmrPlayerTrack,
    HmrResult,
)
from src.schemas.shots import Shot, ShotsManifest
from src.schemas.tracks import Track, TrackFrame, TracksResult
from src.utils.gvhmr_estimator import FakeGVHMREstimator


# ── Schema tests ──────────────────────────────────────────────────────


class TestHmrResult:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        n_frames = 20
        player = HmrPlayerTrack(
            track_id="T001",
            player_id="P001",
            player_name="Test Player",
            team="A",
            frame_indices=np.arange(n_frames, dtype=np.int32),
            global_orient=np.random.randn(n_frames, 3).astype(np.float32),
            body_pose=np.random.randn(n_frames, 63).astype(np.float32),
            betas=np.random.randn(10).astype(np.float32),
            transl=np.random.randn(n_frames, 3).astype(np.float32),
            joints_3d=np.random.randn(n_frames, 22, 3).astype(np.float32),
            pred_cam=np.random.randn(n_frames, 3).astype(np.float32),
            bbx_xys=np.random.randn(n_frames, 3).astype(np.float32),
            confidences=np.random.rand(n_frames).astype(np.float32),
            kp2d=np.random.randn(n_frames, 17, 3).astype(np.float32),
        )
        result = HmrResult(shot_id="shot_001", fps=30.0, players=[player])

        result.save(tmp_path)
        loaded = HmrResult.load(tmp_path, "shot_001")

        assert loaded.shot_id == "shot_001"
        assert loaded.fps == 30.0
        assert len(loaded.players) == 1

        lp = loaded.players[0]
        assert lp.track_id == "T001"
        assert lp.player_id == "P001"
        assert lp.player_name == "Test Player"
        assert lp.team == "A"
        np.testing.assert_array_equal(lp.frame_indices, player.frame_indices)
        np.testing.assert_allclose(lp.global_orient, player.global_orient, atol=1e-6)
        np.testing.assert_allclose(lp.body_pose, player.body_pose, atol=1e-6)
        np.testing.assert_allclose(lp.betas, player.betas, atol=1e-6)
        np.testing.assert_allclose(lp.transl, player.transl, atol=1e-6)
        np.testing.assert_allclose(lp.joints_3d, player.joints_3d, atol=1e-6)
        np.testing.assert_allclose(lp.kp2d, player.kp2d, atol=1e-6)
        assert lp.kp2d.shape == (n_frames, 17, 3)

    def test_save_multiple_players(self, tmp_path: Path) -> None:
        players = []
        for i in range(3):
            players.append(
                HmrPlayerTrack(
                    track_id=f"T{i:03d}",
                    player_id=f"P{i:03d}",
                    player_name=f"Player {i}",
                    team="A" if i % 2 == 0 else "B",
                    frame_indices=np.arange(10, dtype=np.int32),
                    global_orient=np.zeros((10, 3), dtype=np.float32),
                    body_pose=np.zeros((10, 63), dtype=np.float32),
                    betas=np.zeros(10, dtype=np.float32),
                    transl=np.zeros((10, 3), dtype=np.float32),
                    joints_3d=np.zeros((10, 22, 3), dtype=np.float32),
                    pred_cam=np.zeros((10, 3), dtype=np.float32),
                    bbx_xys=np.zeros((10, 3), dtype=np.float32),
                    confidences=np.ones(10, dtype=np.float32),
                    kp2d=np.zeros((10, 17, 3), dtype=np.float32),
                )
            )
        result = HmrResult(shot_id="shot_002", fps=25.0, players=players)
        result.save(tmp_path)

        loaded = HmrResult.load(tmp_path, "shot_002")
        assert len(loaded.players) == 3
        assert {p.track_id for p in loaded.players} == {"T000", "T001", "T002"}

    def test_load_all(self, tmp_path: Path) -> None:
        for sid in ["shot_001", "shot_002"]:
            p = HmrPlayerTrack(
                track_id="T001", player_id="P001", player_name="", team="A",
                frame_indices=np.array([0], dtype=np.int32),
                global_orient=np.zeros((1, 3), dtype=np.float32),
                body_pose=np.zeros((1, 63), dtype=np.float32),
                betas=np.zeros(10, dtype=np.float32),
                transl=np.zeros((1, 3), dtype=np.float32),
                joints_3d=np.zeros((1, 22, 3), dtype=np.float32),
                pred_cam=np.zeros((1, 3), dtype=np.float32),
                bbx_xys=np.zeros((1, 3), dtype=np.float32),
                confidences=np.ones(1, dtype=np.float32),
                kp2d=np.zeros((1, 17, 3), dtype=np.float32),
            )
            HmrResult(shot_id=sid, fps=30.0, players=[p]).save(tmp_path)

        all_results = HmrResult.load_all(tmp_path)
        assert len(all_results) == 2
        assert {r.shot_id for r in all_results} == {"shot_001", "shot_002"}

    def test_empty_load(self, tmp_path: Path) -> None:
        result = HmrResult.load(tmp_path, "nonexistent")
        assert result.players == []


class TestSmpl22Skeleton:
    def test_joint_count(self) -> None:
        assert len(SMPL22_JOINT_NAMES) == 22

    def test_parent_count(self) -> None:
        assert len(SMPL22_PARENTS) == 22

    def test_root_parent(self) -> None:
        assert SMPL22_PARENTS[0] == -1

    def test_bones(self) -> None:
        assert len(SMPL22_BONES) == 21  # 22 joints - 1 root


# ── Fake estimator tests ─────────────────────────────────────────────


class TestFakeGVHMREstimator:
    def test_produces_correct_shapes(self) -> None:
        est = FakeGVHMREstimator()
        frames = [np.zeros((240, 320, 3), dtype=np.uint8)] * 15
        bboxes = [[50.0, 30.0, 150.0, 200.0]] * 15

        result = est.estimate_sequence(frames, bboxes)

        assert result["global_orient"].shape == (15, 3)
        assert result["body_pose"].shape == (15, 63)
        assert result["betas"].shape == (10,)
        assert result["transl"].shape == (15, 3)
        assert result["joints_3d"].shape == (15, 22, 3)
        assert result["pred_cam"].shape == (15, 3)
        assert result["bbx_xys"].shape == (15, 3)
        assert result["kp2d"].shape == (15, 17, 3)

    def test_empty_input(self) -> None:
        est = FakeGVHMREstimator()
        result = est.estimate_sequence([], [])
        assert result["global_orient"].shape == (0, 3)
        assert result["betas"].shape == (10,)


# ── Stage tests ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hmr_test_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("hmr_stage")
    shots_dir = root / "shots"
    shots_dir.mkdir()
    tracks_dir = root / "tracks"
    tracks_dir.mkdir()

    # Create a minimal video
    clip_path = shots_dir / "shot_001.mp4"
    writer = cv2.VideoWriter(
        str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240)
    )
    for _ in range(20):
        writer.write(np.full((240, 320, 3), [50, 200, 50], dtype=np.uint8))
    writer.release()

    shot = Shot(
        id="shot_001", start_frame=0, end_frame=19,
        start_time=0.0, end_time=2.0, clip_file="shots/shot_001.mp4",
    )
    ShotsManifest(
        source_file="test.mp4", fps=10.0, total_frames=20, shots=[shot]
    ).save(shots_dir / "shots_manifest.json")

    frames = [
        TrackFrame(
            frame=i, bbox=[50.0, 30.0, 150.0, 200.0],
            confidence=0.9, pitch_position=None,
        )
        for i in range(20)
    ]
    track = Track(track_id="T001", class_name="player", team="A", frames=frames)
    TracksResult(shot_id="shot_001", tracks=[track]).save(
        tracks_dir / "shot_001_tracks.json"
    )
    return root


def test_hmr_stage_writes_output(hmr_test_dir: Path) -> None:
    from src.stages.hmr import MonocularHMRStage

    cfg = load_config()
    cfg.setdefault("hmr", {})["min_track_frames"] = 5

    stage = MonocularHMRStage(
        config=cfg,
        output_dir=hmr_test_dir,
        hmr_estimator=FakeGVHMREstimator(),
    )
    stage.run()

    hmr_dir = hmr_test_dir / "hmr"
    assert hmr_dir.exists()
    npz_files = list(hmr_dir.glob("*_hmr.npz"))
    assert len(npz_files) == 1
    assert "shot_001_T001_hmr.npz" == npz_files[0].name


def test_hmr_stage_is_complete(hmr_test_dir: Path) -> None:
    from src.stages.hmr import MonocularHMRStage

    cfg = load_config()
    stage = MonocularHMRStage(
        config=cfg,
        output_dir=hmr_test_dir,
        hmr_estimator=FakeGVHMREstimator(),
    )
    assert stage.is_complete()


def test_hmr_output_loads_correctly(hmr_test_dir: Path) -> None:
    loaded = HmrResult.load(hmr_test_dir / "hmr", "shot_001")
    assert len(loaded.players) == 1

    p = loaded.players[0]
    assert p.track_id == "T001"
    assert p.team == "A"
    assert p.joints_3d.shape[1] == 22
    assert p.joints_3d.shape[2] == 3
    assert p.body_pose.shape[1] == 63
    assert p.betas.shape == (10,)


# ── Runner mode tests ────────────────────────────────────────────────


def test_resolve_stages_hmr_mode() -> None:
    from src.pipeline.runner import resolve_stages

    cfg = {"pipeline": {"mode": "hmr"}}
    names = resolve_stages("all", from_stage=None, config=cfg)
    assert names == ["segmentation", "tracking", "hmr", "export"]


def test_resolve_stages_triangulation_mode() -> None:
    from src.pipeline.runner import resolve_stages

    cfg = {"pipeline": {"mode": "triangulation"}}
    names = resolve_stages("all", from_stage=None, config=cfg)
    assert "calibration" in names
    assert "pose" in names
    assert "triangulation" in names


def test_resolve_stages_default_mode() -> None:
    from src.pipeline.runner import resolve_stages

    names = resolve_stages("all", from_stage=None, config={})
    assert "calibration" in names
    assert "hmr" not in names


def test_resolve_stages_hmr_aliases() -> None:
    from src.pipeline.runner import resolve_stages

    cfg = {"pipeline": {"mode": "hmr"}}
    names = resolve_stages("3", from_stage=None, config=cfg)
    assert names == ["hmr"]


def test_resolve_stages_hmr_from_stage() -> None:
    from src.pipeline.runner import resolve_stages

    cfg = {"pipeline": {"mode": "hmr"}}
    names = resolve_stages("all", from_stage="hmr", config=cfg)
    assert names == ["hmr", "export"]


def test_resolve_stages_auto_switches_to_hmr_mode() -> None:
    """When the caller requests 'hmr' without setting mode, auto-flip mode."""
    from src.pipeline.runner import resolve_stages

    cfg: dict = {}
    names = resolve_stages("hmr", from_stage=None, config=cfg)
    assert names == ["hmr"]
    assert cfg.get("pipeline", {}).get("mode") == "hmr"


def test_resolve_stages_auto_switches_with_triangulation_default() -> None:
    """Explicit mode=triangulation default should also auto-switch for hmr."""
    from src.pipeline.runner import resolve_stages

    cfg = {"pipeline": {"mode": "triangulation"}}
    names = resolve_stages("hmr", from_stage=None, config=cfg)
    assert names == ["hmr"]
    assert cfg["pipeline"]["mode"] == "hmr"


def test_resolve_stages_does_not_flip_for_shared_stage() -> None:
    """Stages that exist in both modes (e.g. tracking) should not flip mode."""
    from src.pipeline.runner import resolve_stages

    cfg = {"pipeline": {"mode": "triangulation"}}
    names = resolve_stages("tracking", from_stage=None, config=cfg)
    assert names == ["tracking"]
    assert cfg["pipeline"]["mode"] == "triangulation"


def test_run_pipeline_invokes_hmr_when_called_with_triangulation_default(
    tmp_path: Path,
) -> None:
    """Regression: run_pipeline must compute stage_order AFTER resolve_stages
    (which auto-flips mode to 'hmr' when 'hmr' is in the requested stages).
    Previously stage_order was computed first and locked in the triangulation
    list, so the loop silently iterated past every stage without matching
    active=['hmr'].
    """
    from src.pipeline.runner import run_pipeline

    invoked: list[str] = []

    class FakeStage:
        name = "fake"

        def __init__(self, config, output_dir, **_):
            self.config = config
            self.output_dir = output_dir

        def is_complete(self) -> bool:
            return False

        def run(self) -> None:
            invoked.append("ran")

    # Monkeypatch the lazy hmr stage import in the runner.
    import src.pipeline.runner as runner_mod
    original = runner_mod._get_hmr_stages
    runner_mod._get_hmr_stages = lambda: [("hmr", FakeStage)]
    try:
        run_pipeline(
            output_dir=tmp_path,
            stages="hmr",
            from_stage=None,
            config={"pipeline": {"mode": "triangulation"}},  # default
        )
    finally:
        runner_mod._get_hmr_stages = original

    assert invoked == ["ran"], (
        "HMR stage should have been invoked despite config defaulting to "
        "triangulation mode"
    )


# ── Export stage HMR-mode tests ─────────────────────────────────────


def test_export_is_complete_hmr_mode_with_results(tmp_path: Path) -> None:
    """Export reports complete in HMR mode when hmr/ has at least one .npz."""
    from src.stages.export import ExportStage

    hmr_dir = tmp_path / "hmr"
    hmr_dir.mkdir()
    (hmr_dir / "shot_001_T001_hmr.npz").write_bytes(b"\x00")

    stage = ExportStage(
        config={"pipeline": {"mode": "hmr"}},
        output_dir=tmp_path,
    )
    assert stage.is_complete() is True


def test_export_is_complete_hmr_mode_without_results(tmp_path: Path) -> None:
    """Export reports incomplete in HMR mode when hmr/ is missing or empty."""
    from src.stages.export import ExportStage

    stage = ExportStage(
        config={"pipeline": {"mode": "hmr"}},
        output_dir=tmp_path,
    )
    assert stage.is_complete() is False


def test_export_is_complete_triangulation_mode_falls_back_to_glb(tmp_path: Path) -> None:
    """Triangulation mode behaviour unchanged — looks for export/gltf/scene.glb."""
    from src.stages.export import ExportStage

    stage = ExportStage(
        config={"pipeline": {"mode": "triangulation"}},
        output_dir=tmp_path,
    )
    assert stage.is_complete() is False

    glb_path = tmp_path / "export" / "gltf" / "scene.glb"
    glb_path.parent.mkdir(parents=True)
    glb_path.write_bytes(b"glb")
    assert stage.is_complete() is True


def test_export_run_in_hmr_mode_does_not_call_gltf_builder(tmp_path: Path) -> None:
    """In HMR mode, run() should return without calling glTF builder."""
    from src.stages.export import ExportStage

    hmr_dir = tmp_path / "hmr"
    hmr_dir.mkdir()
    (hmr_dir / "shot_001_T001_hmr.npz").write_bytes(b"\x00")

    stage = ExportStage(
        config={"pipeline": {"mode": "hmr"}},
        output_dir=tmp_path,
    )
    stage.run()  # Must not raise — and must not produce a GLB.
    assert not (tmp_path / "export" / "gltf" / "scene.glb").exists()
