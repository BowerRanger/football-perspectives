"""The refined_poses stage's gated PhysPT takeover hook.

Drives ``_apply_physpt_takeover`` directly with a stubbed refiner —
the real PhysPT checkout/weights/torch are optional dependencies the
hook must degrade around, so none of these tests touch them.
"""
import numpy as np
import pytest

from src.schemas.refined_pose import RefinedPose
from src.schemas.sync_map import SyncMap
from src.stages.refined_poses import _apply_physpt_takeover
from src.utils.physpt_hybrid import TakeoverConfig


def _track(n=80, spike=40):
    t = np.c_[np.linspace(0, 4, n), np.zeros(n), np.full(n, 0.9)]
    if spike is not None:
        t[spike, 0] += 0.35
    return RefinedPose(
        player_id='P001', frames=np.arange(n),
        betas=np.zeros(10), thetas=np.zeros((n, 24, 3), dtype=np.float32),
        root_R=np.tile(np.eye(3), (n, 1, 1)).astype(np.float32),
        root_t=t.astype(np.float32),
        confidence=np.ones(n), view_count=np.ones(n, dtype=int),
        contributing_shots=('gberch',),
    )


class _StubRefiner:
    def refine(self, track):
        smooth = _track(len(track.frames), spike=None)
        return smooth, [{'status': 'processed', 'start': 0,
                         'end': int(track.frames[-1])}]


def test_unavailable_physpt_skips_once_and_caches_the_verdict(tmp_path, monkeypatch, caplog):
    from src.utils import physpt_refiner
    monkeypatch.setattr(physpt_refiner, 'physpt_available', lambda *a, **k: False)
    track = _track()
    state = {}
    out, stats = _apply_physpt_takeover(
        track, tmp_path, SyncMap(), TakeoverConfig(), state)
    assert out is track and stats is None and state['failed']
    # Second call short-circuits without re-probing.
    monkeypatch.setattr(physpt_refiner, 'physpt_available',
                        lambda *a, **k: pytest.fail('re-probed'))
    out2, stats2 = _apply_physpt_takeover(
        track, tmp_path, SyncMap(), TakeoverConfig(), state)
    assert out2 is track and stats2 is None


def test_stubbed_takeover_splices_flagged_span_and_reports_spans(tmp_path):
    track = _track()
    state = {'refiner': _StubRefiner()}
    # No kp2d sidecar in tmp_path: the hook must disable the occlusion
    # trigger rather than flag the whole track.
    out, stats = _apply_physpt_takeover(
        track, tmp_path, SyncMap(), TakeoverConfig(), state)
    assert stats is not None and len(stats['spans']) == 1
    span = stats['spans'][0]
    assert span['translation'] == 'accepted'
    assert 'occlusion' not in span['triggers']
    a = span['start']; b = span['end'] + 1
    np.testing.assert_array_equal(out.root_t[:a], track.root_t[:a])
    np.testing.assert_array_equal(out.root_t[b:], track.root_t[b:])
    assert not np.array_equal(out.root_t[a:b], track.root_t[a:b])
    assert isinstance(stats['penetration_frames_raised'], int)
