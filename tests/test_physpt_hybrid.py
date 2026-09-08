import numpy as np
from scipy.spatial.transform import Rotation
from src.schemas.refined_pose import RefinedPose
from src.utils.physpt_hybrid import (
    TakeoverConfig, blend_weight, build_hybrid, flag_spans,
    reanchor_translation, slerp_blend,
)


def _make_track(root_t, root_R=None, n=None):
    n = len(root_t) if n is None else n
    return RefinedPose(
        player_id='P001', frames=np.arange(n),
        betas=np.zeros(10), thetas=np.zeros((n, 24, 3), dtype=np.float32),
        root_R=(np.tile(np.eye(3), (n, 1, 1)) if root_R is None else root_R).astype(np.float32),
        root_t=np.asarray(root_t, dtype=np.float32),
        confidence=np.ones(n), view_count=np.ones(n, dtype=int),
        contributing_shots=('gberch',),
    )


def test_flag_spans_dilates_merges_and_drops_short():
    bad = np.zeros(60, bool)
    bad[10] = bad[14] = True   # two hits 4 apart -> one merged span
    bad[40] = True
    spans = flag_spans(bad, dilate=2, merge_gap=3, min_len=3)
    assert spans == [(8, 17), (38, 43)]
    assert flag_spans(np.zeros(10, bool)) == []
    # A lone hit with no dilation is shorter than min_len and dropped.
    lone = np.zeros(10, bool); lone[5] = True
    assert flag_spans(lone, dilate=0, merge_gap=0, min_len=3) == []


def test_blend_weight_ramps_and_saturates():
    w = blend_weight(20, ease=5)
    assert w[0] < 0.3 and w[9] == 1.0 and np.allclose(w, w[::-1])
    short = blend_weight(4, ease=6)   # ramps never overlap
    assert short.max() < 1.0 and np.allclose(short, short[::-1])


def test_reanchor_matches_current_at_both_edges():
    n = 30
    cur = np.c_[np.linspace(0, 3, n), np.zeros(n), np.full(n, 0.9)]
    phys = cur + [5.0, -2.0, 0.1]           # constant drift
    phys[:, 0] += np.linspace(0, 1, n) ** 2  # plus nonlinear accumulating drift
    out = reanchor_translation(cur, phys, 5, 25)
    np.testing.assert_allclose(out[0], cur[5], atol=1e-12)
    np.testing.assert_allclose(out[-1], cur[24], atol=1e-12)
    # Interior keeps PhysPT's local shape, not the current animation's.
    assert not np.allclose(out[10], cur[15], atol=1e-3)


def test_takeover_config_from_mapping_ignores_stage_keys():
    cfg = TakeoverConfig.from_mapping({'enabled': True, 'device': 'mps', 'acc_hi': 20.0})
    assert cfg.acc_hi == 20.0 and cfg.step_hi == TakeoverConfig().step_hi


def _spiky_current(n=80, spike=40):
    t = np.c_[np.linspace(0, 4, n), np.zeros(n), np.full(n, 0.9)]
    t[spike, 0] += 0.35   # single-frame teleport blip -> acceleration trigger
    return _make_track(t)


def test_build_hybrid_accepts_an_improving_takeover_and_keeps_the_rest():
    cur = _spiky_current()
    phys = _make_track(np.c_[np.linspace(0, 4, 80), np.zeros(80), np.full(80, 0.9)])
    hybrid, spans = build_hybrid(cur, phys, np.ones(80), 30., TakeoverConfig())
    assert len(spans) == 1 and spans[0]['translation'] == 'accepted'
    assert spans[0]['triggers'].get('acc')
    a = spans[0]['start']; b = spans[0]['end'] + 1
    # Outside the span: bit-identical. Inside: the blip is gone.
    np.testing.assert_array_equal(hybrid.root_t[:a], cur.root_t[:a])
    np.testing.assert_array_equal(hybrid.root_t[b:], cur.root_t[b:])
    def peak_acc(t): return np.linalg.norm(np.diff(t, 2, axis=0), axis=1).max() * 900
    assert peak_acc(hybrid.root_t) < 0.5 * peak_acc(cur.root_t)


def test_build_hybrid_rejects_a_worsening_takeover_per_channel():
    n = 80
    line = np.c_[np.linspace(0, 4, n), np.zeros(n), np.full(n, 0.9)]
    flip = np.tile(np.eye(3), (n, 1, 1))
    flip[40] = Rotation.from_euler('z', 30, degrees=True).as_matrix()
    cur = _make_track(line, root_R=flip)          # rotation-spike trigger only
    # PhysPT offers clean rotations but a translation with real curvature
    # (savgol order 2 passes a parabola through untouched) on a span
    # whose current translation is perfectly linear -> must be rejected.
    x = np.arange(n) / (n - 1)
    bump = line.copy(); bump[:, 1] += 3 * x * (1 - x)
    hybrid, spans = build_hybrid(cur, _make_track(bump), np.ones(n), 30., TakeoverConfig())
    assert len(spans) == 1
    assert spans[0]['triggers'].get('step')
    assert spans[0]['translation'] == 'rejected'
    assert spans[0]['rotation'] == 'accepted'
    np.testing.assert_array_equal(hybrid.root_t, cur.root_t)
    assert not np.array_equal(hybrid.root_R, cur.root_R)


def test_slerp_blend_endpoints_and_midpoint():
    a = Rotation.from_euler('z', [10, 20], degrees=True).as_matrix()
    b = Rotation.from_euler('z', [50, 80], degrees=True).as_matrix()
    np.testing.assert_allclose(slerp_blend(a, b, [0, 0]), a, atol=1e-12)
    np.testing.assert_allclose(slerp_blend(a, b, [1, 1]), b, atol=1e-12)
    mid = Rotation.from_matrix(slerp_blend(a, b, [0.5, 0.5])).as_euler('zyx', degrees=True)[:, 0]
    np.testing.assert_allclose(mid, [30, 50], atol=1e-9)
