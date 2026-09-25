"""Sanity-check ``ClipContext``/``load_clip`` against each real clip
(skipped when its main-repo output dir isn't present locally)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.utils.ball_anchor_heights import GROUND_LEVEL_STATES
from src.utils.ball_bench_clip import CLIPS, load_clip
from src.utils.ball_eval import ray_plane_z

BALL_RADIUS_M = 0.11


@pytest.mark.parametrize("clip_id", sorted(CLIPS))
def test_load_clip(clip_id):
    output_dir, _shot_id = CLIPS[clip_id]
    if not Path(output_dir).exists():
        pytest.skip(f"{output_dir} not present on this machine")

    ctx = load_clip(clip_id)

    assert ctx.n_frames > 0
    assert len(ctx.anchors.anchors) > 0
    assert ctx.fps > 0
    assert ctx.image_size[0] > 0 and ctx.image_size[1] > 0

    # --- ray/project round trip on a grounded anchor -----------------
    grounded = [a for a in ctx.anchors.anchors
                if a.state in GROUND_LEVEL_STATES and a.image_xy is not None]
    assert grounded, "expected at least one grounded anchor with a click"
    anchor = grounded[0]

    C, d_hat = ctx.ray(anchor.frame, anchor.image_xy)
    xyz = ray_plane_z(C, d_hat, BALL_RADIUS_M)
    assert xyz is not None, "ground-plane intersection failed"

    uv2 = ctx.project(anchor.frame, xyz)
    err_px = math.hypot(uv2[0] - anchor.image_xy[0],
                         uv2[1] - anchor.image_xy[1])
    assert err_px < 0.5, (
        f"{clip_id} f{anchor.frame}: reprojection error {err_px:.3f}px >= 0.5px")

    # --- camera_centres() sanity ---------------------------------------
    centres = ctx.camera_centres()
    assert len(centres) == ctx.n_frames
    assert all(c is not None for c in centres)

    # --- player_context() + at least one touch anchor's joint ----------
    touches = [a for a in ctx.anchors.anchors if a.state == "player_touch"]
    if touches:
        pc = ctx.player_context()
        found = False
        for a in touches:
            for df in range(-3, 4):
                world = pc.joint_world(a.frame + df, a.player_id, a.bone)
                if world is not None and all(math.isfinite(x) for x in world):
                    found = True
                    break
            if found:
                break
        assert found, (
            f"{clip_id}: no player_touch anchor resolved a finite joint "
            "world position within +/-3 frames")
