"""Tests for src/utils/ball_hybrid_types.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_hybrid_types import CueEvidence, HybridShotCtx, Knot, SpinFit


def test_knot_requires_3vec_xyz():
    Knot(frame=1, xyz=(1.0, 2.0, 3.0), kind="bounce", depth_hard=True,
         source="manual")
    with pytest.raises(ValueError):
        Knot(frame=1, xyz=(1.0, 2.0), kind="bounce", depth_hard=True,  # type: ignore[arg-type]
             source="manual")


def test_knot_uv_must_be_2vec_when_present():
    Knot(frame=1, xyz=(0.0, 0.0, 0.11), kind="airborne_mid", depth_hard=False,
         source="auto", uv=(100.0, 200.0))
    with pytest.raises(ValueError):
        Knot(frame=1, xyz=(0.0, 0.0, 0.11), kind="airborne_mid",
             depth_hard=False, source="auto", uv=(100.0,))  # type: ignore[arg-type]


def test_knot_is_frozen_and_immutable():
    k = Knot(frame=1, xyz=(0.0, 0.0, 0.11), kind="fix", depth_hard=True,
              source="fix", weight=5.0)
    with pytest.raises(Exception):
        k.frame = 2  # type: ignore[misc]


def test_cue_evidence_defaults():
    c = CueEvidence(frame=10, kind="bounce", cue="audio_impact", conf=0.6)
    assert c.xyz is None
    assert c.uv is None


def test_spin_fit_fields():
    s = SpinFit(omega_world=(0.0, 0.0, 12.5), rad_s=12.5, delta_bic=3.2)
    assert s.rad_s == pytest.approx(12.5)
    assert s.delta_bic > 0


def _pinhole_camera(image_size=(1920, 1080)):
    fx = fy = 1000.0
    cx, cy = image_size[0] / 2.0, image_size[1] / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    R = np.eye(3)
    C = np.array([0.0, -20.0, 5.0])
    t = -R @ C
    return K, R, t


def test_hybrid_shot_ctx_project_and_ray_round_trip():
    K, R, t = _pinhole_camera()
    ctx = HybridShotCtx(
        clip_id="clip", fps=30.0, image_size=(1920, 1080),
        per_frame_K={0: K}, per_frame_R={0: R}, per_frame_t={0: t},
        distortion=(0.0, 0.0),
    )
    world = np.array([1.0, 2.0, 0.11])
    uv = ctx.project(0, world)
    assert uv.shape == (2,)

    C, d_hat = ctx.ray(0, (float(uv[0]), float(uv[1])))
    # The known 3-D point must lie on its own back-projected ray.
    v = world - C
    along = float(np.dot(v, d_hat))
    perp = np.linalg.norm(v - along * d_hat)
    assert perp < 1e-6


def test_hybrid_shot_ctx_has_frame():
    K, R, t = _pinhole_camera()
    ctx = HybridShotCtx(
        clip_id="clip", fps=30.0, image_size=(1920, 1080),
        per_frame_K={5: K}, per_frame_R={5: R}, per_frame_t={5: t},
        distortion=(0.0, 0.0),
    )
    assert ctx.has_frame(5)
    assert not ctx.has_frame(6)
