"""Projection and dispatch regressions for the stadium camera selection."""
import numpy as np
import pytest

from src.utils import virtual_cameras as vc
from src.schemas.render_selection import RenderSelection, RenderSelectionError
from src.stages.render import RenderStage
from tests.test_virtual_cameras_dolly import _static_track

CAMERAS = ['tactical', 'sideline:near', 'sideline:far', 'corner:left', 'corner:right',
           'goal:left', 'goal:right', 'goalline:left', 'goalline:right', 'orbit', 'chase', 'dolly']


@pytest.mark.parametrize('camera_id', CAMERAS)
def test_selected_camera_dispatches_to_valid_track(camera_id, tmp_path):
    selection = RenderSelection.from_dict({'cameras':[camera_id]})
    stage = RenderStage({}, tmp_path)
    track = stage._build_one_virtual_camera(selection.cameras[0],
        {'P001':_static_track('P001',30,34)}, None, stage._virtual_camera_cfg(),
        (960,540),25,'clip')
    assert len(track.frames) == 10
    for f in track.frames:
        R, t = np.asarray(f.R), np.asarray(f.t)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) == pytest.approx(1.)
        assert np.isfinite(t).all()


@pytest.mark.parametrize('size', [(1920,1080),(1080,1920),(1000,1000)])
def test_tactical_contains_whole_pitch_with_margin(size):
    track = vc.build_stadium_track('tactical',[_static_track('P001',10,10)],
                                  None,vc.RigConfig(),size,25,'clip')
    f=track.frames[0]
    xyz = np.array([[x,y,0.] for x in (0,105) for y in (0,68)])
    cam = xyz @ np.asarray(f.R).T + f.t
    assert (cam[:,2] > 0).all()
    uvw = cam @ np.asarray(f.K).T
    uv = uvw[:,:2]/uvw[:,2:]
    assert (uv>0).all()
    assert (uv[:,0]<size[0]).all() and (uv[:,1]<size[1]).all()
    assert all(other.R == f.R and other.t == f.t for other in track.frames)


@pytest.mark.parametrize('name',['goal:near','corner:far','sideline:left','tactical\n','orbit:P001'])
def test_rejects_malformed_rig_id(name):
    with pytest.raises(RenderSelectionError):
        RenderSelection.from_dict({'cameras':[name]})
