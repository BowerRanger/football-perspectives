import json
import numpy as np
import pytest

from src.utils import blender_scene_io as bio


@pytest.mark.unit
def test_load_shot_ids_empty_when_no_manifest(tmp_path):
    assert bio.load_shot_ids(tmp_path) == set()


@pytest.mark.unit
def test_load_shot_ids_reads_manifest(tmp_path):
    shots = {"shots": [{"id": "shot01"}, {"id": "shot02"}, {"noid": True}]}
    (tmp_path / "shots").mkdir()
    (tmp_path / "shots" / "shots_manifest.json").write_text(json.dumps(shots))
    assert bio.load_shot_ids(tmp_path) == {"shot01", "shot02"}


@pytest.mark.unit
def test_load_camera_track_error_names_path(tmp_path):
    bad = tmp_path / "camera_track.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="camera_track.json"):
        bio.load_camera_track(bad)


@pytest.mark.unit
def test_load_smpl_body_data_missing_returns_none(tmp_path):
    data, pelvis = bio.load_smpl_body_data(tmp_path, np)
    assert data is None
    assert pelvis.shape == (3,)


@pytest.mark.unit
def test_load_smpl_body_data_present_but_missing_v_template_returns_none(tmp_path):
    # npz exists but is missing v_template — present-but-invalid must
    # collapse to the same safe (None, zeros) state as absent, not leak
    # an incomplete dict a caller might try to index.
    models_dir = tmp_path / "data" / "models"
    models_dir.mkdir(parents=True)
    np.savez(
        models_dir / "smpl_neutral.npz",
        joint_positions=np.zeros((24, 3), dtype=np.float32),
    )
    data, pelvis = bio.load_smpl_body_data(tmp_path, np)
    assert data is None
    assert pelvis.shape == (3,)
    assert np.array_equal(pelvis, np.zeros(3))


@pytest.mark.unit
def test_fbx_script_reexports_readers():
    # Existing tests/tools import these names from the script module —
    # the move must keep them resolvable there.
    from scripts import blender_export_fbx as bef
    assert bef.iter_player_fbx_entries is bio.iter_player_fbx_entries
    assert bef.prepare_ball_keys is bio.prepare_ball_keys
