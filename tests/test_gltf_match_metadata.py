"""``build_glb`` must mirror ``SceneBundle.match`` into the metadata
sidecar so the viewer can read kit colours and roster without a second
fetch."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import KitColors, MatchInfo, MomentInfo, RosterEntry
from src.schemas.smpl_world import SmplWorldTrack
from src.utils.gltf_builder import SceneBundle, build_glb


def _minimal_camera() -> CameraTrack:
    eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return CameraTrack(
        clip_id="play",
        fps=30.0,
        image_size=(1280, 720),
        t_world=[0.0, -10.0, 5.0],
        frames=tuple(
            CameraFrame(
                frame=i, K=eye, R=eye, confidence=0.9, is_anchor=False,
            )
            for i in range(2)
        ),
    )


def _minimal_player() -> SmplWorldTrack:
    return SmplWorldTrack(
        player_id="P001",
        frames=np.arange(2),
        betas=np.zeros(10),
        thetas=np.zeros((2, 24, 3)),
        root_R=np.tile(np.eye(3), (2, 1, 1)),
        root_t=np.zeros((2, 3)),
        confidence=np.full(2, 0.8),
    )


@pytest.mark.unit
def test_metadata_includes_match_when_set_on_bundle() -> None:
    match = MatchInfo(
        home_team="Liverpool",
        away_team="Real Madrid",
        home_score=0,
        away_score=1,
        venue="Stade de France",
        date="2022-05-28",
        moment=MomentInfo(minute=59, description="Vinicius"),
        kits=KitColors(home_primary="#c8102e", away_primary="#ffffff"),
        roster=[
            RosterEntry(name="Salah", team="A", position="FWD", shirt_number=11),
            RosterEntry(name="Vinicius", team="B", position="FWD", shirt_number=20),
        ],
    )
    bundle = SceneBundle(
        camera_track=_minimal_camera(),
        players=(_minimal_player(),),
        ball_track=None,
        match=asdict(match),
    )
    _glb, metadata = build_glb(bundle)
    assert "match" in metadata
    assert metadata["match"]["home_team"] == "Liverpool"
    assert metadata["match"]["kits"]["home_primary"] == "#c8102e"
    assert metadata["match"]["roster"][0]["name"] == "Salah"
    assert metadata["match"]["moment"]["minute"] == 59


@pytest.mark.unit
def test_metadata_omits_match_when_unset_on_bundle() -> None:
    bundle = SceneBundle(
        camera_track=_minimal_camera(),
        players=(_minimal_player(),),
        ball_track=None,
    )
    _glb, metadata = build_glb(bundle)
    assert "match" not in metadata
