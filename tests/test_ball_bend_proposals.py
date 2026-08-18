"""W5k (sub-20cm campaign): natural-motion violations on the resolved
track become second-chance event proposals — a real bend without a
covering event gets a touch (joint nearby) or bounce (at ground)
candidate, which then runs the normal evidence/contact/flight gates."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_FPS = 30.0


class _Ctx:
    def __init__(self, joints_by_frame):
        self._j = joints_by_frame

    def joints_at(self, frame):
        return list(self._j.get(frame, []))


class _Joint:
    def __init__(self, pid, bone, world, conf=0.9):
        self.player_id = pid
        self.bone = bone
        self.world_xyz = tuple(world)
        self.confidence = conf


def _bent_track(bend_frame=15, n=31):
    """Ground track with a sharp unexplained 90° bend at ``bend_frame``."""
    world, state = {}, {}
    for f in range(n):
        if f <= bend_frame:
            world[f] = (0.25 * f, 0.0, 0.11)
        else:
            world[f] = (0.25 * bend_frame, 0.25 * (f - bend_frame), 0.11)
        state[f] = "grounded"
    return world, state


def test_bend_with_nearby_joint_proposes_touch():
    from src.utils.ball_bend_proposals import propose_bend_events

    world, state = _bent_track()
    bend_pos = np.asarray(world[15])
    ctx = _Ctx({15: [_Joint("P007", "l_foot", bend_pos + [0.1, 0.05, 0.0])]})
    out = propose_bend_events(
        world_by_frame=world, state_by_frame=state, event_frames=set(),
        fps=_FPS, player_ctx=ctx, contact_max_gap_m=0.6,
    )
    assert len(out) == 1
    e = out[0]
    assert e.kind == "touch" and abs(e.frame - 15) <= 1
    assert e.player_id == "P007" and e.bone == "l_foot"


def test_bend_without_joint_at_ground_proposes_bounce():
    from src.utils.ball_bend_proposals import propose_bend_events

    world, state = _bent_track()
    out = propose_bend_events(
        world_by_frame=world, state_by_frame=state, event_frames=set(),
        fps=_FPS, player_ctx=_Ctx({}), contact_max_gap_m=0.6,
    )
    assert len(out) == 1
    assert out[0].kind == "bounce" and abs(out[0].frame - 15) <= 1


def test_covered_bend_proposes_nothing():
    from src.utils.ball_bend_proposals import propose_bend_events

    world, state = _bent_track()
    out = propose_bend_events(
        world_by_frame=world, state_by_frame=state, event_frames={15},
        fps=_FPS, player_ctx=_Ctx({}), contact_max_gap_m=0.6,
    )
    assert out == ()


def test_clean_track_proposes_nothing():
    from src.utils.ball_bend_proposals import propose_bend_events

    world = {f: (0.25 * f, 0.0, 0.11) for f in range(31)}
    state = {f: "grounded" for f in range(31)}
    out = propose_bend_events(
        world_by_frame=world, state_by_frame=state, event_frames=set(),
        fps=_FPS, player_ctx=_Ctx({}), contact_max_gap_m=0.6,
    )
    assert out == ()
