from src.schemas.anchor import Anchor, AnchorSet, LandmarkObservation
from src.stages.camera import CameraStage


def _anchor_set():
    lms = tuple(
        LandmarkObservation(name=f"pnl_kp_{i}", image_xy=(float(i), float(i)),
                            world_xyz=(float(i), float(i), 0.0))
        for i in range(1, 7)
    )
    return AnchorSet(clip_id="s1", image_size=(1920, 1080),
                     anchors=(Anchor(frame=0, landmarks=lms, lines=()),))


def test_ensure_anchors_writes_generated_set(tmp_path, monkeypatch):
    stage = CameraStage.__new__(CameraStage)
    stage.output_dir = tmp_path
    anchors_path = tmp_path / "camera" / "s1_anchors.json"
    cfg = {"auto_anchors": {"enabled": True, "mode": "replace_when_empty"}}
    monkeypatch.setattr(
        "src.stages.camera._generate_auto_anchors",
        lambda shot_id, clip_path, cfg: _anchor_set(),
    )
    stage._ensure_anchors("s1", anchors_path, tmp_path / "s1.mp4", cfg)
    assert anchors_path.exists()
    assert len(AnchorSet.load(anchors_path).anchors) == 1


def test_ensure_anchors_skips_when_manual_exists(tmp_path):
    stage = CameraStage.__new__(CameraStage)
    stage.output_dir = tmp_path
    anchors_path = tmp_path / "camera" / "s1_anchors.json"
    anchors_path.parent.mkdir(parents=True)
    _anchor_set().save(anchors_path)
    before = anchors_path.read_text()
    cfg = {"auto_anchors": {"enabled": True, "mode": "replace_when_empty"}}
    stage._ensure_anchors("s1", anchors_path, tmp_path / "s1.mp4", cfg)
    assert anchors_path.read_text() == before


def test_ensure_anchors_noop_when_disabled(tmp_path):
    stage = CameraStage.__new__(CameraStage)
    stage.output_dir = tmp_path
    anchors_path = tmp_path / "camera" / "s1_anchors.json"
    cfg = {"auto_anchors": {"enabled": False}}
    stage._ensure_anchors("s1", anchors_path, tmp_path / "s1.mp4", cfg)
    assert not anchors_path.exists()
