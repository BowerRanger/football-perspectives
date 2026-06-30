# Body-Kinematics Touch Proposer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a body-kinematics touch *proposer* that generates candidate ball touches from limb motion (independent of ball-pixel breaks), so we recover contacts the ball detector drops at the moment of contact.

**Architecture:** A new pure module `src/utils/ball_kinematic_touch.py` finds, per (player, bone), the frames where the bone's 3-D world position is closest to the ball's pixel sight-line (depth-robust ray-gap minima), gates each on a kinematic contact signature (foot-speed peak / head-into-line / keeper hand), uses the ball only as a confidence *modifier* (boost on agreement, no penalty when occluded, downweight when visible-but-unchanged), and emits `BallEvent(kind="touch", …)`. The ball stage unions these with the existing ball-break-attributed touches (operator/manual always wins) behind a config flag; everything downstream (body-pin resolution → `BallKeyframeSet`) is reused unchanged.

**Tech Stack:** Python 3, NumPy, pytest. No torch in the proposer (pure geometry over `PlayerContext` + ball pixel dict). Reuses `src/utils/camera_projection.py`, `src/utils/ball_pose_touch.py`, `src/utils/ball_player_context.py`, `src/utils/ball_auto_events.py`.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-27-body-kinematics-touch-proposer-design.md`. Decisions K1–K7 are binding.
- **Additive only:** the proposer never deletes or overrides an existing or manual touch. Merge is union + NMS; manual wins (handled downstream by the existing `merge_anchors`).
- **Body is the trigger:** ball-pixel signal is a *modifier*, never a gate. A candidate where the ball is occluded through the contact must NOT be penalised.
- **High-recall:** tune for recall; the web editor prunes false positives. Default `min_emit_score` is a low floor.
- **Immutability / style:** frozen dataclasses for config; type annotations on every signature; PEP 8; functions < 50 lines; pure functions where possible (no I/O in the proposer module).
- **Coordinate system:** all 3-D positions in pitch-metres; `point_to_pixel_ray_distance` returns metres.
- **Config namespace:** new block `ball.kinematic_touch.*` in `config/default.yaml`, read in `ball.py` via `cfg.get("kinematic_touch", {})`.
- **Bone vocabulary:** `BONE_TO_SMPL_INDEX` from `src/utils/ball_anchor_heights.py` (`l_foot, r_foot, l_knee, r_knee, chest, head, l_shoulder, r_shoulder, l_hand, r_hand`).
- **Commit style:** `feat(ball): …` / `test(ball): …` / `docs(ball): …`, one commit per task step group.

---

## File Structure

| File | Responsibility | Tasks |
|------|----------------|-------|
| `src/utils/ball_kinematic_touch.py` (new) | The pure proposer: cfg, ball-pixel interpolation, ray-gap series + minima, kinematic gate, ball-confirmation, scoring, `propose_touches`, `nms_touches`, `merge_touch_events`. | 1–5, 7 |
| `tests/test_ball_kinematic_touch.py` (new) | Unit tests for every function above + synthetic touch scenarios. | 1–5 |
| `config/default.yaml` (modify) | `ball.kinematic_touch.*` defaults. | 6 |
| `src/stages/ball.py` (modify ~1405, ~1424–1443) | Build `KinematicTouchCfg` from config; call `propose_touches`; union via `merge_touch_events` behind the flag. | 6–7 |
| `tests/test_ball_kinematic_touch_cfg.py` (new) | `_kinematic_touch_cfg` builder. | 6 |
| `scripts/ball_touch_recall_report.py` (new) | Three-config recall/precision table (break-only / proposer-only / union) via `match_touches`. | 8 |
| `tests/test_ball_kinematic_recall.py` (new) | Synthetic assertion: union recall ≥ break-only recall. | 8 |

---

## Phase 1 — The pure proposer module (TDD, no stage wiring)

### Task 1: Config + ball-pixel gap interpolation

**Files:**
- Create: `src/utils/ball_kinematic_touch.py`
- Test: `tests/test_ball_kinematic_touch.py`

**Interfaces:**
- Produces:
  - `KinematicTouchCfg` (frozen dataclass) — fields below.
  - `interpolate_ball_uvs(ball_uvs: dict[int, np.ndarray], max_gap_frames: int) -> tuple[dict[int, np.ndarray], frozenset[int]]` — returns `(filled, interpolated_frames)`. Fills gaps of length ≤ `max_gap_frames` between bracketing detections by linear pixel interpolation; longer gaps are left empty. `interpolated_frames` is the set of newly-filled frames.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ball_kinematic_touch.py
import numpy as np
import pytest

from src.utils.ball_kinematic_touch import KinematicTouchCfg, interpolate_ball_uvs


def test_cfg_defaults_are_high_recall():
    cfg = KinematicTouchCfg()
    assert cfg.enabled is True
    assert cfg.contact_gap_m == pytest.approx(0.30)
    assert cfg.min_emit_score == pytest.approx(0.25)
    assert cfg.max_ball_gap_frames == 6


def test_interpolate_fills_short_gap_and_flags_it():
    uvs = {0: np.array([0.0, 0.0]), 3: np.array([3.0, 6.0])}
    filled, interp = interpolate_ball_uvs(uvs, max_gap_frames=6)
    assert set(filled) == {0, 1, 2, 3}
    assert filled[1] == pytest.approx(np.array([1.0, 2.0]))
    assert filled[2] == pytest.approx(np.array([2.0, 4.0]))
    assert interp == frozenset({1, 2})


def test_interpolate_leaves_long_gap_empty():
    uvs = {0: np.array([0.0, 0.0]), 10: np.array([10.0, 0.0])}
    filled, interp = interpolate_ball_uvs(uvs, max_gap_frames=6)
    assert set(filled) == {0, 10}
    assert interp == frozenset()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ball_kinematic_touch.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.ball_kinematic_touch'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/utils/ball_kinematic_touch.py
"""Body-kinematics touch proposer.

Generates candidate ball touches from limb motion, independent of whether a
ball-pixel velocity break exists at the contact. The body is the trigger; the
ball is only a confidence modifier. See
docs/superpowers/specs/2026-06-27-body-kinematics-touch-proposer-design.md.

Pure and torch-free: PlayerContext samples + a ball pixel dict in, BallEvents
out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KinematicTouchCfg:
    """Thresholds + score weights for the kinematic touch proposer.

    Pixel units are px/frame; gaps are 3-D metres; speeds for feet are
    px/frame, for the head metres/frame.
    """

    enabled: bool = True
    contact_gap_m: float = 0.30
    touch_relaxed_px: float = 60.0
    max_ball_gap_frames: int = 6
    min_fk_conf: float = 0.3
    kin_window: int = 2
    kin_min_foot_speed: float = 8.0
    kin_min_head_speed_m: float = 0.05
    confirm_window: int = 3
    nms_window: int = 2
    w_gap: float = 0.35
    w_kin: float = 0.30
    w_confirm: float = 0.25
    w_fk: float = 0.10
    w_interp: float = 0.15
    min_emit_score: float = 0.25


def interpolate_ball_uvs(
    ball_uvs: dict[int, np.ndarray], max_gap_frames: int
) -> tuple[dict[int, np.ndarray], frozenset[int]]:
    """Linear-fill ball-pixel gaps of length <= ``max_gap_frames``.

    Returns ``(filled, interpolated_frames)``. Frames present in ``ball_uvs``
    are copied through; gaps longer than the cap are left empty.
    """
    if not ball_uvs:
        return {}, frozenset()
    frames = sorted(ball_uvs)
    filled: dict[int, np.ndarray] = {f: np.asarray(ball_uvs[f], dtype=float) for f in frames}
    interp: set[int] = set()
    for a, b in zip(frames[:-1], frames[1:]):
        span = b - a
        if span <= 1 or span - 1 > max_gap_frames:
            continue
        pa, pb = filled[a], filled[b]
        for f in range(a + 1, b):
            w = (f - a) / span
            filled[f] = pa * (1.0 - w) + pb * w
            interp.add(f)
    return filled, frozenset(interp)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ball_kinematic_touch.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_kinematic_touch.py tests/test_ball_kinematic_touch.py
git commit -m "feat(ball): kinematic-touch cfg + ball-pixel gap interpolation"
```

---

### Task 2: Ray-gap series + closest-approach minima

**Files:**
- Modify: `src/utils/ball_kinematic_touch.py`
- Test: `tests/test_ball_kinematic_touch.py`

**Interfaces:**
- Consumes: `PlayerContext` (`joints_at(frame) -> tuple[JointSample, ...]`, each with `.player_id, .bone, .world_xyz, .uv, .confidence`); `camera_projection.point_to_pixel_ray_distance`.
- Produces:
  - `ray_gap_series(player_ctx, ball_uvs, per_frame_K, per_frame_R, per_frame_t, distortion, min_fk_conf) -> dict[tuple[str, str], dict[int, tuple[float, float, float]]]` — keyed by `(player_id, bone)`, value maps frame → `(gap3d_m, pixgap_px, fk_conf)`. Only frames where the bone exists with `confidence >= min_fk_conf`, has a projected `uv`, a ball pixel exists, and the camera is known.
  - `local_minima_below(series: dict[int, float], threshold: float) -> list[int]` — frames that are strict local minima (over present neighbours) with value ≤ `threshold`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ball_kinematic_touch.py
from src.utils.ball_kinematic_touch import local_minima_below, ray_gap_series
from src.utils.ball_player_context import JointSample, PlayerContext
from src.utils.camera_projection import project_world_to_image


def _cam(frames):
    K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.zeros(3)
    return ({f: K for f in frames}, {f: R for f in frames}, {f: t for f in frames})


def _ball_uv(world):
    K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]])
    return np.asarray(project_world_to_image(K, np.eye(3), np.zeros(3), (0.0, 0.0),
                                             np.asarray([world]))[0], dtype=float)


def test_local_minima_below_picks_the_dip():
    series = {0: 0.5, 1: 0.3, 2: 0.1, 3: 0.25, 4: 0.4}
    assert local_minima_below(series, 0.3) == [2]
    assert local_minima_below(series, 0.05) == []


def test_ray_gap_zero_when_bone_on_ball_ray():
    frames = range(5)
    K, R, t = _cam(frames)
    # ball fixed at world (0,0,10) -> uv (960,540); ray is the +z axis.
    ball_world = (0.0, 0.0, 10.0)
    ball_uvs = {f: _ball_uv(ball_world) for f in frames}
    # foot x sweeps toward the z-axis (gap == |x|), nearest at frame 3.
    xs = {0: 0.5, 1: 0.3, 2: 0.12, 3: 0.04, 4: 0.2}
    samples = {
        f: (JointSample("P1", "r_foot", (xs[f], 0.0, 9.0),
                        (960.0, 540.0), 0.9),)
        for f in frames
    }
    ctx = PlayerContext(samples, ("P1",))
    series = ray_gap_series(ctx, ball_uvs, K, R, t, (0.0, 0.0), min_fk_conf=0.3)
    gaps = {f: g for f, (g, _px, _c) in series[("P1", "r_foot")].items()}
    assert gaps[3] == pytest.approx(0.04, abs=1e-6)
    assert local_minima_below(gaps, 0.30) == [3]


def test_ray_gap_skips_low_fk_conf():
    frames = range(3)
    K, R, t = _cam(frames)
    ball_uvs = {f: _ball_uv((0.0, 0.0, 10.0)) for f in frames}
    samples = {
        f: (JointSample("P1", "r_foot", (0.0, 0.0, 9.0), (960.0, 540.0), 0.1),)
        for f in frames
    }
    ctx = PlayerContext(samples, ("P1",))
    series = ray_gap_series(ctx, ball_uvs, K, R, t, (0.0, 0.0), min_fk_conf=0.3)
    assert ("P1", "r_foot") not in series
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ball_kinematic_touch.py -q`
Expected: FAIL — `ImportError: cannot import name 'ray_gap_series'`

- [ ] **Step 3: Write minimal implementation**

```python
# add imports near the top of src/utils/ball_kinematic_touch.py
from math import hypot

from src.utils.camera_projection import point_to_pixel_ray_distance

# append to src/utils/ball_kinematic_touch.py

def ray_gap_series(
    player_ctx,
    ball_uvs: dict[int, np.ndarray],
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float],
    min_fk_conf: float,
) -> dict[tuple[str, str], dict[int, tuple[float, float, float]]]:
    """Per-(player, bone) frame -> (gap3d_m, pixgap_px, fk_conf)."""
    out: dict[tuple[str, str], dict[int, tuple[float, float, float]]] = {}
    for frame, ball_uv in ball_uvs.items():
        K = per_frame_K.get(frame)
        R = per_frame_R.get(frame)
        t = per_frame_t.get(frame)
        if K is None or R is None or t is None:
            continue
        for s in player_ctx.joints_at(frame):
            if s.confidence < min_fk_conf or s.uv is None:
                continue
            world = np.asarray(s.world_xyz, dtype=float)
            gap3d = point_to_pixel_ray_distance(world, ball_uv, K, R, t, distortion)
            pixgap = hypot(s.uv[0] - float(ball_uv[0]), s.uv[1] - float(ball_uv[1]))
            out.setdefault((s.player_id, s.bone), {})[frame] = (
                float(gap3d), float(pixgap), float(s.confidence),
            )
    return out


def local_minima_below(series: dict[int, float], threshold: float) -> list[int]:
    """Frames that are strict local minima over present neighbours and <=
    ``threshold``. Endpoints count if strictly below their one neighbour."""
    frames = sorted(series)
    minima: list[int] = []
    for i, f in enumerate(frames):
        v = series[f]
        if v > threshold:
            continue
        left = series[frames[i - 1]] if i > 0 else float("inf")
        right = series[frames[i + 1]] if i < len(frames) - 1 else float("inf")
        if v < left and v <= right:
            minima.append(f)
    return minima
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ball_kinematic_touch.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_kinematic_touch.py tests/test_ball_kinematic_touch.py
git commit -m "feat(ball): per-bone ray-gap series + closest-approach minima"
```

---

### Task 3: Kinematic gate + strength

**Files:**
- Modify: `src/utils/ball_kinematic_touch.py`
- Test: `tests/test_ball_kinematic_touch.py`

**Interfaces:**
- Consumes: `ball_pose_touch.joint_pixel_velocity(player_ctx, frame, player_id, bone) -> tuple[float, float] | None`; `PlayerContext.joint_world(frame, player_id, bone) -> np.ndarray | None`.
- Produces:
  - `kinematic_gate(player_ctx, frame, player_id, bone, cfg) -> tuple[bool, float]` — `(passed, strength)`. `strength` ∈ [0, 1]. Feet/knees require a foot-pixel-speed peak ≥ `cfg.kin_min_foot_speed` within ±`cfg.kin_window`; head requires a 3-D head speed ≥ `cfg.kin_min_head_speed_m`; hands/shoulders/chest always pass with a fixed 0.5 strength.

Bone groups (module constants): `FOOT_BONES = ("l_foot", "r_foot")`, `KNEE_BONES = ("l_knee", "r_knee")`, `SPEED_GATED_BONES = FOOT_BONES + KNEE_BONES`, `HEAD_BONES = ("head",)`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ball_kinematic_touch.py
from src.utils.ball_kinematic_touch import kinematic_gate


def _foot_ctx(positions_uv, world=(0.0, 0.0, 9.0)):
    # positions_uv: dict frame -> (u, v) for the foot bone of P1.
    samples = {
        f: (JointSample("P1", "r_foot", world, uv, 0.9),)
        for f, uv in positions_uv.items()
    }
    return PlayerContext(samples, ("P1",))


def test_kicking_foot_passes_gate():
    # foot u moves 20 px/frame -> central-diff speed 20 at frame 1.
    ctx = _foot_ctx({0: (900.0, 540.0), 1: (920.0, 540.0), 2: (940.0, 540.0)})
    passed, strength = kinematic_gate(ctx, 1, "P1", "r_foot", KinematicTouchCfg())
    assert passed is True
    assert strength > 0.0


def test_planted_foot_fails_gate():
    ctx = _foot_ctx({0: (900.0, 540.0), 1: (900.2, 540.0), 2: (900.4, 540.0)})
    passed, _ = kinematic_gate(ctx, 1, "P1", "r_foot", KinematicTouchCfg())
    assert passed is False


def test_keeper_hand_always_passes():
    samples = {1: (JointSample("P1", "l_hand", (0.0, 0.0, 1.5), (500.0, 300.0), 0.9),)}
    ctx = PlayerContext(samples, ("P1",))
    passed, strength = kinematic_gate(ctx, 1, "P1", "l_hand", KinematicTouchCfg())
    assert passed is True
    assert strength == pytest.approx(0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ball_kinematic_touch.py -q`
Expected: FAIL — `ImportError: cannot import name 'kinematic_gate'`

- [ ] **Step 3: Write minimal implementation**

```python
# add import near the top of src/utils/ball_kinematic_touch.py
from src.utils.ball_pose_touch import joint_pixel_velocity

# add module constants after the imports
FOOT_BONES = ("l_foot", "r_foot")
KNEE_BONES = ("l_knee", "r_knee")
SPEED_GATED_BONES = FOOT_BONES + KNEE_BONES
HEAD_BONES = ("head",)

# pixel speed (px/frame) at which a foot is "clearly kicking" — strength saturates
_KICK_SPEED_PX = 12.0
# head 3-D speed (m/frame) at which a header strength saturates
_HEAD_SPEED_SAT_M = 0.15

# append to src/utils/ball_kinematic_touch.py

def _peak_foot_speed(player_ctx, frame: int, player_id: str, bone: str,
                     window: int) -> float:
    """Max central-difference foot pixel speed within +/- window of frame."""
    peak = 0.0
    for f in range(frame - window, frame + window + 1):
        vel = joint_pixel_velocity(player_ctx, f, player_id, bone)
        if vel is not None:
            peak = max(peak, hypot(vel[0], vel[1]))
    return peak


def _head_speed_m(player_ctx, frame: int, player_id: str, bone: str) -> float:
    """Central-difference 3-D speed (m/frame) of a bone, 0 if unavailable."""
    prev = player_ctx.joint_world(frame - 1, player_id, bone)
    nxt = player_ctx.joint_world(frame + 1, player_id, bone)
    if prev is None or nxt is None:
        return 0.0
    return float(np.linalg.norm((nxt - prev) / 2.0))


def kinematic_gate(player_ctx, frame: int, player_id: str, bone: str,
                   cfg: KinematicTouchCfg) -> tuple[bool, float]:
    """(passed, strength in [0,1]) for the bone's contact signature."""
    if bone in SPEED_GATED_BONES:
        peak = _peak_foot_speed(player_ctx, frame, player_id, bone, cfg.kin_window)
        return (peak >= cfg.kin_min_foot_speed, min(1.0, peak / _KICK_SPEED_PX))
    if bone in HEAD_BONES:
        speed = _head_speed_m(player_ctx, frame, player_id, bone)
        return (speed >= cfg.kin_min_head_speed_m,
                min(1.0, speed / _HEAD_SPEED_SAT_M))
    # hands / shoulders / chest: a body part on the ball line is a contact;
    # no speed requirement (keeper saves, blocks).
    return (True, 0.5)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ball_kinematic_touch.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_kinematic_touch.py tests/test_ball_kinematic_touch.py
git commit -m "feat(ball): kinematic contact gate (foot/head/hand)"
```

---

### Task 4: Ball-confirmation modifier + score

**Files:**
- Modify: `src/utils/ball_kinematic_touch.py`
- Test: `tests/test_ball_kinematic_touch.py`

**Interfaces:**
- Produces:
  - `ball_confirm(frame, cfg, confirm_frames, interp_frames, detected_frames) -> float` — returns `+1.0` (a ball-pixel break is within ±`cfg.confirm_window`), `-1.0` (the ball is clearly *detected* across the window but no break — visible-but-unchanged), or `0.0` (occluded through the window — the rescue case, no penalty).
  - `touch_score(gap3d_min, kin_strength, confirm, fk_conf, is_interp, cfg) -> float` — clipped to [0, 1].

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ball_kinematic_touch.py
from src.utils.ball_kinematic_touch import ball_confirm, touch_score


def test_confirm_boost_when_break_nearby():
    cfg = KinematicTouchCfg()
    assert ball_confirm(10, cfg, confirm_frames=frozenset({11}),
                        interp_frames=frozenset(),
                        detected_frames=frozenset(range(0, 30))) == 1.0


def test_confirm_no_penalty_when_occluded():
    cfg = KinematicTouchCfg()
    # frame 10 + neighbours not in detected_frames -> occluded -> 0.0
    assert ball_confirm(10, cfg, confirm_frames=frozenset(),
                        interp_frames=frozenset({9, 10, 11}),
                        detected_frames=frozenset()) == 0.0


def test_confirm_downweight_when_visible_unchanged():
    cfg = KinematicTouchCfg()
    assert ball_confirm(10, cfg, confirm_frames=frozenset(),
                        interp_frames=frozenset(),
                        detected_frames=frozenset(range(0, 30))) == -1.0


def test_score_monotonic_and_clipped():
    cfg = KinematicTouchCfg()
    good = touch_score(0.02, 1.0, 1.0, 0.9, False, cfg)
    poor = touch_score(0.29, 0.0, -1.0, 0.3, True, cfg)
    assert 0.0 <= poor < good <= 1.0
    assert good == pytest.approx(
        min(1.0, cfg.w_gap * (1 - 0.02 / cfg.contact_gap_m)
            + cfg.w_kin * 1.0 + cfg.w_confirm * 1.0 + cfg.w_fk * 0.9), abs=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ball_kinematic_touch.py -q`
Expected: FAIL — `ImportError: cannot import name 'ball_confirm'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/utils/ball_kinematic_touch.py

def ball_confirm(
    frame: int,
    cfg: KinematicTouchCfg,
    confirm_frames: frozenset[int],
    interp_frames: frozenset[int],
    detected_frames: frozenset[int],
) -> float:
    """+1 boost / 0 no-penalty (occluded) / -1 downweight (visible-unchanged)."""
    if any(abs(frame - cf) <= cfg.confirm_window for cf in confirm_frames):
        return 1.0
    window = range(frame - cfg.confirm_window, frame + cfg.confirm_window + 1)
    visible = [w for w in window if w in detected_frames and w not in interp_frames]
    if len(visible) >= 2:
        return -1.0
    return 0.0


def touch_score(
    gap3d_min: float,
    kin_strength: float,
    confirm: float,
    fk_conf: float,
    is_interp: bool,
    cfg: KinematicTouchCfg,
) -> float:
    """Blended confidence in [0, 1]."""
    gap_term = max(0.0, 1.0 - gap3d_min / cfg.contact_gap_m)
    score = (
        cfg.w_gap * gap_term
        + cfg.w_kin * kin_strength
        + cfg.w_confirm * confirm
        + cfg.w_fk * fk_conf
        - (cfg.w_interp if is_interp else 0.0)
    )
    return float(min(1.0, max(0.0, score)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ball_kinematic_touch.py -q`
Expected: PASS (13 passed)

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_kinematic_touch.py tests/test_ball_kinematic_touch.py
git commit -m "feat(ball): ball-confirmation modifier + touch score"
```

---

### Task 5: `propose_touches` + `nms_touches` (assembly + headline scenarios)

**Files:**
- Modify: `src/utils/ball_kinematic_touch.py`
- Test: `tests/test_ball_kinematic_touch.py`

**Interfaces:**
- Consumes: all of Tasks 1–4; `ball_auto_events.BallEvent` (fields `frame, kind, score, player_id, bone, goal_element, end_frame`).
- Produces:
  - `propose_touches(*, player_ctx, ball_uvs, per_frame_K, per_frame_R, per_frame_t, distortion=(0.0, 0.0), confirm_frames=frozenset(), detected_frames=None, cfg) -> list[BallEvent]` — body-kinematics touch candidates (`kind="touch"`), score ≥ `cfg.min_emit_score`. `detected_frames` defaults to `frozenset(ball_uvs)` (every supplied pixel treated as a genuine detection).
  - `nms_touches(events, window) -> list[BallEvent]` — temporal NMS keyed by `(player_id, bone)`, higher score wins, kept within `window` frames; returns frame-sorted.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ball_kinematic_touch.py
from src.utils.ball_auto_events import BallEvent
from src.utils.ball_kinematic_touch import nms_touches, propose_touches


def _kick_scene(drop_contact_frame=None):
    """Foot sweeps fast through a fixed ball at frame 3; optionally drop the
    ball detection at the contact frame (the headline occlusion case)."""
    frames = list(range(7))
    K, R, t = _cam(frames)
    ball_world = (0.0, 0.0, 10.0)
    ball_uvs = {f: _ball_uv(ball_world) for f in frames}
    if drop_contact_frame is not None:
        del ball_uvs[drop_contact_frame]
    # foot x sweeps 0.6 -> -0.6 (fast), nearest the z-axis ray at frame 3,
    # and its projected uv moves fast (kick signature).
    xs = {0: 0.6, 1: 0.4, 2: 0.2, 3: 0.03, 4: -0.2, 5: -0.4, 6: -0.6}
    samples = {}
    for f in frames:
        uv = _ball_uv((xs[f], 0.0, 9.0))
        samples[f] = (JointSample("P1", "r_foot", (xs[f], 0.0, 9.0),
                                  (float(uv[0]), float(uv[1])), 0.9),)
    ctx = PlayerContext(samples, ("P1",))
    return ctx, ball_uvs, K, R, t


def test_propose_detects_kick_when_ball_present():
    ctx, ball_uvs, K, R, t = _kick_scene()
    cfg = KinematicTouchCfg()
    # ball visible + no break -> downweight; raise recall floor for the test.
    touches = propose_touches(
        player_ctx=ctx, ball_uvs=ball_uvs, per_frame_K=K, per_frame_R=R,
        per_frame_t=t, confirm_frames=frozenset({3}), cfg=cfg)
    assert any(e.player_id == "P1" and e.bone == "r_foot" and abs(e.frame - 3) <= 1
               for e in touches)


def test_propose_rescues_touch_when_ball_occluded_at_contact():
    ctx, ball_uvs, K, R, t = _kick_scene(drop_contact_frame=3)
    filled = set(ball_uvs)  # detections that survived
    cfg = KinematicTouchCfg()
    touches = propose_touches(
        player_ctx=ctx, ball_uvs=ball_uvs, per_frame_K=K, per_frame_R=R,
        per_frame_t=t, confirm_frames=frozenset(),
        detected_frames=frozenset(filled), cfg=cfg)
    assert any(e.bone == "r_foot" and abs(e.frame - 3) <= 1 for e in touches)


def test_propose_rejects_planted_foot_ball_grazing():
    frames = list(range(7))
    K, R, t = _cam(frames)
    # ball moves across; foot planted at small fixed offset (gap dips < 0.3)
    # but foot pixel speed ~0 -> kinematic gate fails.
    foot_world = (0.08, 0.0, 9.0)
    foot_uv = _ball_uv(foot_world)
    ball_xs = {0: 0.6, 1: 0.4, 2: 0.2, 3: 0.08, 4: -0.1, 5: -0.3, 6: -0.5}
    ball_uvs = {f: _ball_uv((ball_xs[f], 0.0, 10.0)) for f in frames}
    samples = {f: (JointSample("P1", "r_foot", foot_world,
                               (float(foot_uv[0]), float(foot_uv[1])), 0.9),)
               for f in frames}
    ctx = PlayerContext(samples, ("P1",))
    touches = propose_touches(
        player_ctx=ctx, ball_uvs=ball_uvs, per_frame_K=K, per_frame_R=R,
        per_frame_t=t, cfg=KinematicTouchCfg())
    assert touches == []


def test_nms_keeps_highest_score_per_bone_in_window():
    evs = [
        BallEvent(frame=10, kind="touch", score=0.4, player_id="P1", bone="r_foot"),
        BallEvent(frame=11, kind="touch", score=0.8, player_id="P1", bone="r_foot"),
        BallEvent(frame=40, kind="touch", score=0.5, player_id="P1", bone="r_foot"),
        BallEvent(frame=11, kind="touch", score=0.9, player_id="P2", bone="l_foot"),
    ]
    kept = nms_touches(evs, window=2)
    assert (11, "P1", "r_foot") in {(e.frame, e.player_id, e.bone) for e in kept}
    assert (10, "P1", "r_foot") not in {(e.frame, e.player_id, e.bone) for e in kept}
    assert len(kept) == 3  # P1@11, P1@40, P2@11
    assert [e.frame for e in kept] == sorted(e.frame for e in kept)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ball_kinematic_touch.py -q`
Expected: FAIL — `ImportError: cannot import name 'propose_touches'`

- [ ] **Step 3: Write minimal implementation**

```python
# add import near the top of src/utils/ball_kinematic_touch.py
from collections import defaultdict

from src.utils.ball_auto_events import BallEvent

# append to src/utils/ball_kinematic_touch.py

def propose_touches(
    *,
    player_ctx,
    ball_uvs: dict[int, np.ndarray],
    per_frame_K: dict[int, np.ndarray],
    per_frame_R: dict[int, np.ndarray],
    per_frame_t: dict[int, np.ndarray],
    distortion: tuple[float, float] = (0.0, 0.0),
    confirm_frames: frozenset[int] = frozenset(),
    detected_frames: frozenset[int] | None = None,
    cfg: KinematicTouchCfg,
) -> list[BallEvent]:
    """Body-kinematics touch candidates for one shot.

    The body is the trigger: a closest-approach minimum of the bone-to-ball-ray
    gap, gated on a kinematic contact signature. The ball only modifies
    confidence (boost on agreement, no penalty when occluded, downweight when
    visible-but-unchanged).
    """
    if not cfg.enabled or not ball_uvs:
        return []
    if detected_frames is None:
        detected_frames = frozenset(ball_uvs)
    filled, interp_frames = interpolate_ball_uvs(ball_uvs, cfg.max_ball_gap_frames)
    series = ray_gap_series(
        player_ctx, filled, per_frame_K, per_frame_R, per_frame_t,
        distortion, cfg.min_fk_conf,
    )
    out: list[BallEvent] = []
    for (pid, bone), per_frame in series.items():
        gaps = {f: g for f, (g, _px, _c) in per_frame.items()}
        for f in local_minima_below(gaps, cfg.contact_gap_m):
            gap3d, pixgap, fk_conf = per_frame[f]
            if pixgap > cfg.touch_relaxed_px:
                continue
            passed, strength = kinematic_gate(player_ctx, f, pid, bone, cfg)
            if not passed:
                continue
            confirm = ball_confirm(
                f, cfg, confirm_frames, interp_frames, detected_frames)
            score = touch_score(
                gap3d, strength, confirm, fk_conf, f in interp_frames, cfg)
            if score >= cfg.min_emit_score:
                out.append(BallEvent(
                    frame=f, kind="touch", score=score,
                    player_id=pid, bone=bone))
    return sorted(out, key=lambda e: (e.frame, e.player_id, e.bone))


def nms_touches(events, window: int) -> list[BallEvent]:
    """Temporal NMS keyed by (player_id, bone); higher score wins."""
    by_key: dict[tuple[str | None, str | None], list[BallEvent]] = defaultdict(list)
    for e in events:
        by_key[(e.player_id, e.bone)].append(e)
    kept: list[BallEvent] = []
    for evs in by_key.values():
        claimed: list[BallEvent] = []
        for e in sorted(evs, key=lambda e: -e.score):
            if all(abs(e.frame - c.frame) > window for c in claimed):
                claimed.append(e)
        kept.extend(claimed)
    return sorted(kept, key=lambda e: e.frame)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ball_kinematic_touch.py -q`
Expected: PASS (17 passed)

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_kinematic_touch.py tests/test_ball_kinematic_touch.py
git commit -m "feat(ball): propose_touches assembly + temporal NMS"
```

---

## Phase 2 — Wire into the ball stage (behind a flag)

### Task 6: Config builder + defaults

**Files:**
- Modify: `config/default.yaml` (add `ball.kinematic_touch` block)
- Modify: `src/stages/ball.py` (add `_kinematic_touch_cfg` helper near `_auto_event_cfg`, ~line 508)
- Test: `tests/test_ball_kinematic_touch_cfg.py`

**Interfaces:**
- Produces: `_kinematic_touch_cfg(cfg_dict: dict) -> KinematicTouchCfg` in `ball.py` — maps a plain dict (the `ball.kinematic_touch` sub-tree) onto `KinematicTouchCfg`, falling back to dataclass defaults for missing keys.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ball_kinematic_touch_cfg.py
from src.stages.ball import _kinematic_touch_cfg
from src.utils.ball_kinematic_touch import KinematicTouchCfg


def test_empty_dict_gives_defaults():
    assert _kinematic_touch_cfg({}) == KinematicTouchCfg()


def test_overrides_are_applied():
    cfg = _kinematic_touch_cfg({"enabled": False, "contact_gap_m": 0.5,
                                "kin_min_foot_speed": 10.0})
    assert cfg.enabled is False
    assert cfg.contact_gap_m == 0.5
    assert cfg.kin_min_foot_speed == 10.0
    # untouched keys keep defaults
    assert cfg.nms_window == KinematicTouchCfg().nms_window
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ball_kinematic_touch_cfg.py -q`
Expected: FAIL — `ImportError: cannot import name '_kinematic_touch_cfg'`

- [ ] **Step 3: Write minimal implementation**

In `src/stages/ball.py`, add the import alongside the other `ball_*` imports (near line 84):

```python
from src.utils.ball_kinematic_touch import (
    KinematicTouchCfg,
    merge_touch_events,
    propose_touches,
)
```

Add the builder next to `_auto_event_cfg` (after line ~545):

```python
def _kinematic_touch_cfg(cfg_dict: dict) -> KinematicTouchCfg:
    """Build a KinematicTouchCfg from the ``ball.kinematic_touch`` sub-tree,
    falling back to dataclass defaults for any missing key."""
    base = KinematicTouchCfg()
    d = cfg_dict or {}
    return KinematicTouchCfg(
        enabled=bool(d.get("enabled", base.enabled)),
        contact_gap_m=float(d.get("contact_gap_m", base.contact_gap_m)),
        touch_relaxed_px=float(d.get("touch_relaxed_px", base.touch_relaxed_px)),
        max_ball_gap_frames=int(d.get("max_ball_gap_frames", base.max_ball_gap_frames)),
        min_fk_conf=float(d.get("min_fk_conf", base.min_fk_conf)),
        kin_window=int(d.get("kin_window", base.kin_window)),
        kin_min_foot_speed=float(d.get("kin_min_foot_speed", base.kin_min_foot_speed)),
        kin_min_head_speed_m=float(d.get("kin_min_head_speed_m", base.kin_min_head_speed_m)),
        confirm_window=int(d.get("confirm_window", base.confirm_window)),
        nms_window=int(d.get("nms_window", base.nms_window)),
        w_gap=float(d.get("w_gap", base.w_gap)),
        w_kin=float(d.get("w_kin", base.w_kin)),
        w_confirm=float(d.get("w_confirm", base.w_confirm)),
        w_fk=float(d.get("w_fk", base.w_fk)),
        w_interp=float(d.get("w_interp", base.w_interp)),
        min_emit_score=float(d.get("min_emit_score", base.min_emit_score)),
    )
```

In `config/default.yaml`, under the `ball:` section (alongside `auto_anchors:` / `pose_touch:`), add:

```yaml
  kinematic_touch:
    enabled: true            # body-kinematics touch proposer (additive recall)
    contact_gap_m: 0.30      # max bone<->ball-ray 3-D gap for a contact (m)
    touch_relaxed_px: 60.0   # max bone<->ball pixel distance at the minimum
    max_ball_gap_frames: 6   # longest ball-pixel gap we interpolate across
    min_fk_conf: 0.3         # min bone FK confidence to consider
    kin_window: 2            # +/- frames to find the kinematic peak
    kin_min_foot_speed: 8.0  # min foot pixel-speed (px/frame) for a kick
    kin_min_head_speed_m: 0.05  # min head 3-D speed (m/frame) for a header
    confirm_window: 3        # +/- frames to look for a ball-pixel break
    nms_window: 2            # temporal NMS half-width (frames)
    w_gap: 0.35
    w_kin: 0.30
    w_confirm: 0.25
    w_fk: 0.10
    w_interp: 0.15
    min_emit_score: 0.25     # high-recall emission floor
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ball_kinematic_touch_cfg.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add config/default.yaml src/stages/ball.py tests/test_ball_kinematic_touch_cfg.py
git commit -m "feat(ball): kinematic_touch config block + builder"
```

---

### Task 7: `merge_touch_events` + inject into the stage

**Files:**
- Modify: `src/utils/ball_kinematic_touch.py` (add `merge_touch_events`)
- Modify: `src/stages/ball.py` (after the `foot_touches` block, ~line 1443)
- Test: `tests/test_ball_kinematic_touch.py`

**Interfaces:**
- Produces: `merge_touch_events(existing: tuple[BallEvent, ...] | list[BallEvent], kin_touches: list[BallEvent], nms_window: int) -> tuple[BallEvent, ...]` — unions existing **touch** events with `kin_touches`, applies `nms_touches`, and keeps all non-touch events untouched. Frame-sorted.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ball_kinematic_touch.py
from src.utils.ball_kinematic_touch import merge_touch_events


def test_merge_unions_touches_and_preserves_non_touch():
    existing = (
        BallEvent(frame=5, kind="bounce", score=0.6),
        BallEvent(frame=12, kind="touch", score=0.3, player_id="P1", bone="r_foot"),
        BallEvent(frame=20, kind="goal_impact", score=0.7, goal_element="post"),
    )
    kin = [
        BallEvent(frame=12, kind="touch", score=0.9, player_id="P1", bone="r_foot"),
        BallEvent(frame=30, kind="touch", score=0.5, player_id="P2", bone="head"),
    ]
    merged = merge_touch_events(existing, kin, nms_window=2)
    kinds = [e.kind for e in merged]
    assert kinds.count("bounce") == 1 and kinds.count("goal_impact") == 1
    # the higher-score touch at frame 12 wins over the existing 0.3
    t12 = [e for e in merged if e.kind == "touch" and e.frame == 12]
    assert len(t12) == 1 and t12[0].score == pytest.approx(0.9)
    assert any(e.kind == "touch" and e.frame == 30 for e in merged)
    assert [e.frame for e in merged] == sorted(e.frame for e in merged)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ball_kinematic_touch.py::test_merge_unions_touches_and_preserves_non_touch -q`
Expected: FAIL — `ImportError: cannot import name 'merge_touch_events'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/utils/ball_kinematic_touch.py`:

```python
def merge_touch_events(existing, kin_touches, nms_window: int):
    """Union existing touch events with proposer touches (NMS by player+bone);
    keep all non-touch events untouched. Returns a frame-sorted tuple."""
    touches = [e for e in existing if e.kind == "touch"] + list(kin_touches)
    others = [e for e in existing if e.kind != "touch"]
    merged = others + nms_touches(touches, nms_window)
    return tuple(sorted(merged, key=lambda e: (e.frame, e.kind)))
```

In `src/stages/ball.py`, immediately **after** the `if artifacts.foot_touches:` block (ends ~line 1443, just before `auto_by_frame: dict[int, BallAnchor] = {}`), insert:

```python
        # Body-kinematics touch proposer: recover contacts the ball-pixel
        # break path missed (occlusion/blur at contact). Additive recall;
        # operator/manual anchors still win downstream via merge_anchors.
        kin_cfg = _kinematic_touch_cfg(cfg.get("kinematic_touch", {}))
        if kin_cfg.enabled and player_ctx.player_ids:
            ball_uvs = {
                s.frame: np.asarray(s.uv, dtype=float)
                for s in steps if s.uv is not None
            }
            detected_frames = frozenset(
                f for f, c in raw_confidences.items() if c > 0.0
            )
            confirm_frames = frozenset(
                e.frame for e in events
                if e.kind in ("touch", "bounce", "goal_impact", "velocity_break")
            )
            try:
                kin_touches = propose_touches(
                    player_ctx=player_ctx, ball_uvs=ball_uvs,
                    per_frame_K=per_frame_K, per_frame_R=per_frame_R,
                    per_frame_t=per_frame_t, distortion=distortion,
                    confirm_frames=confirm_frames,
                    detected_frames=detected_frames, cfg=kin_cfg,
                )
                events = merge_touch_events(events, kin_touches, kin_cfg.nms_window)
            except Exception as exc:  # noqa: BLE001 — never kill the stage
                logger.warning(
                    "ball stage: kinematic touch proposer failed (%s) — "
                    "continuing with ball-break touches only", exc,
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ball_kinematic_touch.py -q && pytest tests/test_ball_stage.py -q`
Expected: PASS — the new merge test passes; `test_ball_stage.py` still green (proposer is additive and exception-guarded).

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_kinematic_touch.py src/stages/ball.py tests/test_ball_kinematic_touch.py
git commit -m "feat(ball): union kinematic touches into the ball stage (flagged)"
```

---

## Phase 3 — Validation against the recall harness

### Task 8: Three-config recall report + regression test

**Files:**
- Create: `scripts/ball_touch_recall_report.py`
- Create: `tests/test_ball_kinematic_recall.py`

**Interfaces:**
- Consumes: `ball_touch_recall.match_touches(manual, auto, frame_tol, require_bone)`; `ball_touch_recall.touches_from_anchor_set(path)`.
- Produces: `recall_table(manual, break_only, proposer_only, union, frame_tol=2) -> dict[str, dict]` in the script — recall/precision per config; a `__main__` CLI that loads a manual anchor set + an auto set and prints the table.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ball_kinematic_recall.py
from scripts.ball_touch_recall_report import recall_table


def test_union_recall_at_least_break_only():
    # pseudo-ground-truth: three touches
    manual = [(10, "P1", "r_foot"), (40, "P1", "l_foot"), (70, "P2", "head")]
    # ball-break path found only the first
    break_only = [(10, "P1", "r_foot")]
    # proposer recovered the two the ball missed
    proposer_only = [(41, "P1", "l_foot"), (70, "P2", "head")]
    union = break_only + proposer_only
    table = recall_table(manual, break_only, proposer_only, union, frame_tol=2)
    assert table["break_only"]["recall"] <= table["union"]["recall"]
    assert table["union"]["recall"] > table["break_only"]["recall"]
    assert table["union"]["recall"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ball_kinematic_recall.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.ball_touch_recall_report'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/ball_touch_recall_report.py
"""Report touch-detection recall/precision for three configurations
(break-only / proposer-only / union) against a manual anchor set used as
pseudo-ground-truth. See the body-kinematics-touch-proposer spec, section 8.

Usage:
    python scripts/ball_touch_recall_report.py \
        output/ball/<shot>_ball_anchors.json \
        output/ball/<shot>_ball_anchors_auto.json
"""

from __future__ import annotations

import sys

from src.utils.ball_touch_recall import match_touches, touches_from_anchor_set

Touch = tuple[int, str, str]


def recall_table(
    manual: list[Touch],
    break_only: list[Touch],
    proposer_only: list[Touch],
    union: list[Touch],
    *,
    frame_tol: int = 2,
) -> dict[str, dict]:
    """recall/precision for each config against ``manual``."""
    return {
        name: match_touches(manual, auto, frame_tol=frame_tol, require_bone=True)
        for name, auto in (
            ("break_only", break_only),
            ("proposer_only", proposer_only),
            ("union", union),
        )
    }


def _print_table(table: dict[str, dict]) -> None:
    print(f"{'config':<16}{'recall':>8}{'precision':>11}{'tp':>5}{'fp':>5}")
    for name, m in table.items():
        print(f"{name:<16}{m['recall']:>8.3f}{m['precision']:>11.3f}"
              f"{m['true_positive']:>5}{m['false_positive']:>5}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: ball_touch_recall_report.py <manual.json> <auto.json>")
        raise SystemExit(2)
    manual = touches_from_anchor_set(sys.argv[1])
    auto = touches_from_anchor_set(sys.argv[2])
    # With only the merged auto set on disk we report union vs the empty
    # break-only baseline; pass a break-only file as auto to compare paths.
    table = recall_table(manual, [], [], auto)
    _print_table(table)
```

Add an empty `scripts/__init__.py` if `scripts/` is not already importable:

```bash
test -f scripts/__init__.py || touch scripts/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ball_kinematic_recall.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/ball_touch_recall_report.py tests/test_ball_kinematic_recall.py
test -f scripts/__init__.py && git add scripts/__init__.py
git commit -m "feat(ball): three-config touch recall report + regression test"
```

- [ ] **Step 6: Measure on real data (manual, no commit)**

Run the full proposer suite, then the report on a labelled shot (e.g. gberch):

```bash
pytest tests/test_ball_kinematic_touch.py tests/test_ball_kinematic_touch_cfg.py \
       tests/test_ball_kinematic_recall.py -q
# after running the ball stage with kinematic_touch.enabled in config:
python scripts/ball_touch_recall_report.py \
    output/ball/<shot>_ball_anchors.json \
    output/ball/<shot>_ball_anchors_auto.json
```

Record recall/precision in the spec's §8 and tune `contact_gap_m`, `kin_min_foot_speed`, `min_emit_score` if the precision floor (≥0.5 provisional) is breached. This step gates whether the flag ships default-on.

---

## Phase 4 — Verification (reused surfaces)

These are checks, not new code. Run them once Phases 1–3 land:

- [ ] **Editor surfacing:** run `python recon.py serve --output ./output/`, open the ball anchor editor on a labelled shot, confirm proposer touches appear as dashed *suggestions* with confirm/dismiss (they flow through `*_ball_anchors_auto.json` → existing event-list panel; no new code expected). File a follow-up only if the denser auto output breaks the panel.
- [ ] **Carry-span interaction:** confirm the existing carry/possession collapse still behaves with the denser proposer output on a dribble sequence; if micro-touch storms leak through, tune `nms_window` / carry thresholds (no code change expected).
- [ ] **Full ball suite:** `pytest tests/ -k ball -q` stays green.

---

## Self-Review

**Spec coverage:**
- §4 placement / new module → Tasks 1–5, 7. ✓
- §5 step 1 ball sight-line + interpolation → Task 1 (`interpolate_ball_uvs`). ✓
- §5 step 2 ray-gap series → Task 2 (`ray_gap_series`). ✓
- §5 step 3 closest-approach minima → Task 2 (`local_minima_below`) + Task 5 (`contact_gap_m` + `touch_relaxed_px` gates). ✓
- §5 step 4 kinematic gate (feet/head/hands) → Task 3. ✓
- §5 step 5 ball-confirmation (boost/no-penalty/downweight) → Task 4 (`ball_confirm`). ✓
- §5 step 6 score & emit → Task 4 (`touch_score`) + Task 5 (`propose_touches`). ✓
- §6 merge/dedup/NMS, additive, manual wins → Task 5 (`nms_touches`) + Task 7 (`merge_touch_events` + injection before `generate_auto_anchors`/`merge_anchors`). ✓
- §7 config + flag (`use_kinematic_proposer` intent) → Task 6 (`ball.kinematic_touch.enabled` + builder). ✓
- §8 validation (three configs, `match_touches`) → Task 8. ✓
- §9 testing plan (ball-present, gap-at-contact, planted-foot, head/hand, long blackout, NMS/merge precedence) → Tasks 1–5, 7 tests. *(long-blackout is covered by `test_interpolate_leaves_long_gap_empty` + `max_ball_gap_frames` cap.)* ✓
- §10 phasing → Phases 1–4. ✓
- §11 risks (precision/NMS/FK jitter) → `nms_window`, `min_fk_conf` gate, downweight modifier. ✓

**Naming deviations from the spec (intentional, recorded):**
- Spec §7 says "new `AutoEventCfg` fields"; the plan uses a dedicated `KinematicTouchCfg` instead, so the pure module is testable without the large `AutoEventCfg`/config wiring (high cohesion, matches the existing `KickAnchorCfg` pattern). Config namespace is `ball.kinematic_touch.*`.
- Spec used `ball.auto_events.*`; the real config namespace on this branch is `ball.*` sub-trees (`auto_anchors`, `pose_touch`, …), so the block is `ball.kinematic_touch`.

**Placeholder scan:** none — every code/test step carries complete code; no TBD/TODO/"handle edge cases".

**Type consistency:** `KinematicTouchCfg` field names identical across Tasks 1, 6; `BallEvent(frame, kind, score, player_id, bone)` matches `ball_auto_events.BallEvent`; `propose_touches`/`nms_touches`/`merge_touch_events` signatures consistent between definition (Tasks 5, 7) and call site (Task 7). `point_to_pixel_ray_distance`, `joint_pixel_velocity`, `JointSample` fields, `PlayerContext(samples_by_frame, player_ids)` constructor all match the real source read during planning.
