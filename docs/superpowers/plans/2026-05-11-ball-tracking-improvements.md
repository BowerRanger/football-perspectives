# Ball Tracking Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three observed ball-tracking failures on `origi01` (aerial passes projected as grounded, 8 s of detection misses, billion-metre garbage parabola fits) by layering four orthogonal improvements onto the existing IMM + per-segment-fit pipeline.

**Architecture:** Four config-gated layers wrap the existing ball pipeline without restructuring it. Layer 1 (plausibility filter) rejects geometry-absurd fits. Layer 2 (flight promotion) re-classifies grounded runs whose ground projection implies impossible motion. Layer 3 (kick anchor) collapses monocular depth ambiguity by pinning `p0` to a player's foot when a kick is detected. Layer 4 (appearance bridge) plugs short detection holes via NCC template matching. Each layer is independently switchable and independently testable.

**Tech Stack:** Python 3.11, numpy, scipy.optimize.least_squares, OpenCV (cv2.matchTemplate), pytest, existing pipeline conventions in `src/stages/` and `src/utils/`.

**Spec:** `docs/superpowers/specs/2026-05-11-ball-tracking-improvements-design.md`

---

## File Map

| Path | Status | Responsibility |
|---|---|---|
| `src/utils/ball_plausibility.py` | NEW | Pure functions for trajectory plausibility (Layer 1) and implausible-grounded-run detection (Layer 2). |
| `src/utils/ball_kick_anchor.py` | NEW | Pure function: given pixel ankles + ball pixel + camera, return world `p0` if a kick is detected (Layer 3). |
| `src/utils/ball_appearance_bridge.py` | NEW | `AppearanceBridge` class: rolling template + NCC search (Layer 4). |
| `src/utils/bundle_adjust.py` | MODIFY | Add optional `p0_fixed` parameter to `fit_parabola_to_image_observations` and `fit_magnus_trajectory`. |
| `src/stages/ball.py` | MODIFY | Wire the four layers into `_run_shot`; load kp2d sidecars opportunistically. |
| `config/default.yaml` | MODIFY | Add `pitch_margin_m`, `flight_promotion`, `kick_anchor`, `appearance_bridge` blocks under `ball:`. |
| `tests/test_ball_plausibility.py` | NEW | Unit tests for `is_plausible_trajectory`, `find_implausible_grounded_runs`. |
| `tests/test_ball_kick_anchor.py` | NEW | Unit tests for `find_kick_anchor`. |
| `tests/test_ball_appearance_bridge.py` | NEW | Unit tests for `AppearanceBridge`. |
| `tests/test_bundle_adjust_p0_fixed.py` | NEW | Unit tests for the new `p0_fixed` parameter. |
| `tests/test_ball_stage_layered.py` | NEW | Integration test reproducing the `origi01` failure scenarios. |

---

## Task 1: Add `pitch_margin_m` config

**Files:**
- Modify: `config/default.yaml` (the `ball.plausibility` block)

- [ ] **Step 1: Edit the config**

In `config/default.yaml`, locate the `ball.plausibility` block and replace it:

```yaml
ball:
  plausibility:
    z_max_m: 50.0
    horizontal_speed_max_m_s: 40.0
    pitch_margin_m: 5.0
```

- [ ] **Step 2: Sanity check — config loads**

Run: `python -c "import yaml; print(yaml.safe_load(open('config/default.yaml'))['ball']['plausibility']['pitch_margin_m'])"`
Expected output: `5.0`

- [ ] **Step 3: Commit**

```bash
git add config/default.yaml
git commit -m "feat(ball): add pitch_margin_m to plausibility config"
```

---

## Task 2: Layer 1 — `is_plausible_trajectory` pure function

**Files:**
- Create: `src/utils/ball_plausibility.py`
- Create: `tests/test_ball_plausibility.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ball_plausibility.py`:

```python
"""Unit tests for ball trajectory plausibility checks (Layer 1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_plausibility import (
    PitchDims,
    PlausibilityCfg,
    is_plausible_trajectory,
)


def _cfg(**over) -> PlausibilityCfg:
    base = dict(
        z_max_m=50.0,
        horizontal_speed_max_m_s=40.0,
        pitch_margin_m=5.0,
    )
    base.update(over)
    return PlausibilityCfg(**base)


def _pitch() -> PitchDims:
    return PitchDims(length_m=105.0, width_m=68.0)


def test_plausible_parabola_apex_10m_25mps():
    # Apex ~10 m above pitch centre, horizontal speed ~25 m/s along x.
    p0 = np.array([-30.0, 0.0, 0.11])
    v0 = np.array([25.0, 0.0, 14.0])  # 14 m/s upward → apex ≈ 10 m
    duration = 2.5
    assert is_plausible_trajectory(
        p0, v0, omega=None, duration_s=duration, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


def test_rejects_off_pitch_p0():
    p0 = np.array([-200.0, 0.0, 0.11])
    v0 = np.array([0.0, 0.0, 5.0])
    assert not is_plausible_trajectory(
        p0, v0, omega=None, duration_s=1.0, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


def test_rejects_excessive_speed():
    p0 = np.array([0.0, 0.0, 0.11])
    v0 = np.array([250.0, 0.0, 5.0])
    assert not is_plausible_trajectory(
        p0, v0, omega=None, duration_s=0.5, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


def test_rejects_z_above_max():
    p0 = np.array([0.0, 0.0, 0.11])
    v0 = np.array([0.0, 0.0, 60.0])  # ~180 m apex
    assert not is_plausible_trajectory(
        p0, v0, omega=None, duration_s=2.0, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


def test_rejects_z_far_below_ground():
    p0 = np.array([0.0, 0.0, 0.11])
    v0 = np.array([0.0, 0.0, -50.0])  # plunges through ground
    assert not is_plausible_trajectory(
        p0, v0, omega=None, duration_s=1.0, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


def test_accepts_p0_just_inside_margin():
    # Half-length 52.5; +5 m margin → boundary at 57.5. 56 should pass.
    p0 = np.array([56.0, 0.0, 0.11])
    v0 = np.array([0.0, 0.0, 5.0])
    assert is_plausible_trajectory(
        p0, v0, omega=None, duration_s=0.5, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


def test_rejects_p0_just_outside_margin():
    p0 = np.array([59.0, 0.0, 0.11])
    v0 = np.array([0.0, 0.0, 5.0])
    assert not is_plausible_trajectory(
        p0, v0, omega=None, duration_s=0.5, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


def test_zero_duration_returns_false():
    p0 = np.array([0.0, 0.0, 0.11])
    v0 = np.array([10.0, 0.0, 5.0])
    assert not is_plausible_trajectory(
        p0, v0, omega=None, duration_s=0.0, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )


def test_billion_metre_p0_rejected():
    # Reproduces seg-3 garbage from origi01.
    p0 = np.array([-5_690_504.0, 9_399_056.0, -2_218_511.0])
    v0 = np.array([3_745_003.0, 3_366_928.0, -698_927.0])
    assert not is_plausible_trajectory(
        p0, v0, omega=None, duration_s=0.2, fps=30.0,
        cfg=_cfg(), pitch=_pitch(),
    )
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `pytest tests/test_ball_plausibility.py -v`
Expected: `ImportError` / `ModuleNotFoundError: No module named 'src.utils.ball_plausibility'`.

- [ ] **Step 3: Implement the module**

Create `src/utils/ball_plausibility.py`:

```python
"""Plausibility checks for ball trajectories and grounded runs.

Layer 1 of the ball-tracking improvement plan: every fitted parabola or
Magnus segment is sampled at several time points and rejected unless all
samples stay within physical bounds (pitch envelope + speed + height).

Layer 2 lives here too because it shares the pitch geometry helpers:
:func:`find_implausible_grounded_runs` flags contiguous ``grounded``
runs whose ground-projected world positions imply impossible rolling
motion (off-pitch or > 35 m/s).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PlausibilityCfg:
    """Physical bounds for an accepted trajectory."""
    z_max_m: float
    horizontal_speed_max_m_s: float
    pitch_margin_m: float


@dataclass(frozen=True)
class PitchDims:
    length_m: float
    width_m: float


def _sample_positions(
    p0: np.ndarray,
    v0: np.ndarray,
    omega: np.ndarray | None,
    duration_s: float,
    fps: float,
    g: float = -9.81,
    drag_k_over_m: float = 0.005,
) -> np.ndarray:
    """Return (N, 3) sampled positions along the trajectory.

    Without omega: closed-form parabola.
    With omega: forward Euler integration matching
    :func:`src.utils.bundle_adjust._integrate_magnus_positions`.
    """
    n = max(8, int(duration_s * fps))
    times = np.linspace(0.0, duration_s, n)
    g_vec = np.array([0.0, 0.0, g])

    if omega is None:
        positions = p0[None, :] + np.outer(times, v0) + 0.5 * np.outer(times ** 2, g_vec)
        return positions

    # Forward Euler matching the existing Magnus integrator.
    from src.utils.bundle_adjust import _integrate_magnus_positions
    return _integrate_magnus_positions(p0, v0, omega, g_vec, drag_k_over_m, times)


def is_plausible_trajectory(
    p0: np.ndarray,
    v0: np.ndarray,
    *,
    omega: np.ndarray | None,
    duration_s: float,
    fps: float,
    cfg: PlausibilityCfg,
    pitch: PitchDims,
) -> bool:
    """True when the trajectory stays within the physical envelope.

    Samples the trajectory at ≥ 8 points and checks:
      - |x| ≤ pitch_length / 2 + margin
      - |y| ≤ pitch_width / 2 + margin
      - z ∈ [-1.0, z_max_m]
      - per-sample speed ≤ horizontal_speed_max_m_s + 5.0
    """
    if duration_s <= 0.0:
        return False

    half_len = pitch.length_m / 2.0 + cfg.pitch_margin_m
    half_wid = pitch.width_m / 2.0 + cfg.pitch_margin_m
    speed_cap = cfg.horizontal_speed_max_m_s + 5.0

    positions = _sample_positions(p0, v0, omega, duration_s, fps)

    if np.any(np.abs(positions[:, 0]) > half_len):
        return False
    if np.any(np.abs(positions[:, 1]) > half_wid):
        return False
    if np.any(positions[:, 2] < -1.0) or np.any(positions[:, 2] > cfg.z_max_m):
        return False

    # Per-sample speed (forward difference plus initial v0).
    if positions.shape[0] >= 2:
        dt = duration_s / (positions.shape[0] - 1)
        diffs = np.diff(positions, axis=0) / max(dt, 1e-9)
        speeds = np.linalg.norm(diffs, axis=1)
        if np.any(speeds > speed_cap):
            return False
    if float(np.linalg.norm(v0)) > speed_cap:
        return False

    return True
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_ball_plausibility.py -v`
Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_plausibility.py tests/test_ball_plausibility.py
git commit -m "feat(ball): add is_plausible_trajectory pure function (Layer 1)"
```

---

## Task 3: Wire Layer 1 into `BallStage`

**Files:**
- Modify: `src/stages/ball.py` (the flight-segment loop in `_run_shot`)
- Modify: `tests/test_ball_stage.py` — add a regression test

- [ ] **Step 1: Write the failing regression test**

Add to `tests/test_ball_stage.py`:

```python
@pytest.mark.integration
def test_implausible_parabola_fit_is_rejected(tmp_path: Path, monkeypatch):
    """Reproduces origi01 seg-3: a parabola fit lands at billion-metre
    coordinates with tiny pixel residual. Layer 1 must drop it."""
    from src.schemas.ball_track import BallTrack
    from src.stages.ball import BallStage
    from src.utils import bundle_adjust

    # Force fit_parabola_to_image_observations to return garbage with a
    # microscopic residual so the existing flight_max_residual_px gate
    # cannot save us.
    def fake_parab(*args, **kwargs):
        p0 = np.array([-5_690_504.0, 9_399_056.0, -2_218_511.0])
        v0 = np.array([3_745_003.0, 3_366_928.0, -698_927.0])
        return p0, v0, 0.11
    monkeypatch.setattr(
        bundle_adjust, "fit_parabola_to_image_observations", fake_parab,
    )

    K, R, t = _camera_pose()
    out = tmp_path / "out"
    clip = out / "shots" / "play.mp4"
    n_frames = 40
    _write_blank_clip(clip, n=n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    ShotsManifest(
        clip_id="origi-test",
        shots=(Shot(id="play", clip_file="shots/play.mp4", start_frame=0, end_frame=n_frames-1),),
    ).save(out / "shots" / "shots_manifest.json")

    # Drive IMM into flight mode for frames 10..25 with valid-looking detections.
    detections = [None] * 10 + [(640.0 + i, 360.0 + i, 0.9) for i in range(16)] + [None] * (n_frames - 26)
    stage = BallStage(
        config={"ball": {
            "detector": "fake",
            "ball_radius_m": 0.11,
            "max_gap_frames": 6,
            "flight_max_residual_px": 5.0,
            "tracker": {
                "process_noise_grounded_px": 4.0,
                "process_noise_flight_px": 12.0,
                "measurement_noise_px": 2.0,
                "gating_sigma": 4.0,
                "min_flight_frames": 6,
                "max_flight_frames": 90,
            },
            "spin": {"enabled": False, "min_flight_seconds": 0.5, "min_residual_improvement": 0.2, "max_omega_rad_s": 200.0, "drag_k_over_m": 0.005},
            "plausibility": {"z_max_m": 50.0, "horizontal_speed_max_m_s": 40.0, "pitch_margin_m": 5.0},
        }, "pitch": {"length_m": 105.0, "width_m": 68.0}},
        output_dir=out,
        ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # No flight segment should have survived plausibility.
    assert len(track.flight_segments) == 0, (
        f"expected garbage segment to be rejected; got {track.flight_segments}"
    )
```

- [ ] **Step 2: Run the new test — expect FAIL**

Run: `pytest tests/test_ball_stage.py::test_implausible_parabola_fit_is_rejected -v`
Expected: FAIL with the assertion `len(track.flight_segments) == 0`.

- [ ] **Step 3: Wire Layer 1 into `BallStage._run_shot`**

In `src/stages/ball.py`:

(a) Add imports at the top of the file (after the existing imports from `src.utils.bundle_adjust`):

```python
from src.utils.ball_plausibility import (
    PitchDims,
    PlausibilityCfg,
    is_plausible_trajectory,
)
```

(b) Inside `_run_shot`, right after computing `tracker_cfg`, `spin_cfg`, and `max_residual`, build a plausibility cfg + pitch dims:

```python
plaus_cfg = PlausibilityCfg(
    z_max_m=float(cfg.get("plausibility", {}).get("z_max_m", 50.0)),
    horizontal_speed_max_m_s=float(cfg.get("plausibility", {}).get("horizontal_speed_max_m_s", 40.0)),
    pitch_margin_m=float(cfg.get("plausibility", {}).get("pitch_margin_m", 5.0)),
)
pitch_cfg = self.config.get("pitch", {})
pitch_dims = PitchDims(
    length_m=float(pitch_cfg.get("length_m", 105.0)),
    width_m=float(pitch_cfg.get("width_m", 68.0)),
)
```

(c) Modify the flight-segment loop. After the parabola fit and the existing
`if parab_resid > max_residual: continue` guard, add a plausibility check.
Also gate the Magnus-accepted path. Replace the existing block:

```python
            try:
                p0, v0, parab_resid = fit_parabola_to_image_observations(
                    obs, Ks=Ks_seg, Rs=Rs_seg, t_world=ts_seg,
                    fps=camera.fps, distortion=distortion,
                )
            except Exception as exc:
                logger.debug("parabola fit failed on segment %d: %s", sid, exc)
                continue
            if parab_resid > max_residual:
                continue
```

with:

```python
            try:
                p0, v0, parab_resid = fit_parabola_to_image_observations(
                    obs, Ks=Ks_seg, Rs=Rs_seg, t_world=ts_seg,
                    fps=camera.fps, distortion=distortion,
                )
            except Exception as exc:
                logger.debug("parabola fit failed on segment %d: %s", sid, exc)
                continue
            if parab_resid > max_residual:
                continue
            segment_duration_s = (b - a) / camera.fps
            if not is_plausible_trajectory(
                p0, v0, omega=None,
                duration_s=segment_duration_s, fps=camera.fps,
                cfg=plaus_cfg, pitch=pitch_dims,
            ):
                logger.info(
                    "ball seg %d (%d-%d): parabola failed plausibility, dropping",
                    sid, a, b,
                )
                continue
```

(d) Add a Magnus plausibility check. Inside the existing `if spin_enabled
and duration_s >= spin_min_seconds:` branch, change the inner acceptance
condition. Replace:

```python
                    if (
                        omega_mag > 0
                        and omega_mag <= spin_max_omega
                        and improvement >= spin_min_improve
                    ):
                        spin_axis = list((momega / omega_mag).astype(float))
                        ...
                        omega_world = momega
```

with:

```python
                    magnus_plausible = is_plausible_trajectory(
                        mp0, mv0, omega=momega,
                        duration_s=duration_s, fps=camera.fps,
                        cfg=plaus_cfg, pitch=pitch_dims,
                    )
                    if (
                        omega_mag > 0
                        and omega_mag <= spin_max_omega
                        and improvement >= spin_min_improve
                        and magnus_plausible
                    ):
                        spin_axis = list((momega / omega_mag).astype(float))
                        spin_omega = omega_mag
                        duration_factor = min(1.0, duration_s / 1.0)
                        spin_confidence = float(min(1.0, (improvement / 0.5) * duration_factor))
                        effective_p0, effective_v0 = mp0, mv0
                        effective_resid = magnus_resid
                        omega_world = momega
```

- [ ] **Step 4: Run all ball tests — expect pass**

Run: `pytest tests/test_ball_stage.py tests/test_ball_plausibility.py -v`
Expected: all pass, including the new regression test.

- [ ] **Step 5: Commit**

```bash
git add src/stages/ball.py tests/test_ball_stage.py
git commit -m "feat(ball): reject implausible fitted segments (Layer 1)"
```

---

## Task 4: Layer 2 — `find_implausible_grounded_runs` pure function

**Files:**
- Modify: `src/utils/ball_plausibility.py`
- Modify: `tests/test_ball_plausibility.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ball_plausibility.py`:

```python
from src.utils.ball_plausibility import (
    GroundedRun,
    GroundPromotionCfg,
    find_implausible_grounded_runs,
)


def _promote_cfg(**over) -> GroundPromotionCfg:
    base = dict(
        enabled=True,
        min_run_frames=6,
        off_pitch_margin_m=5.0,
        max_ground_speed_m_s=35.0,
    )
    base.update(over)
    return GroundPromotionCfg(**base)


def test_no_runs_when_ground_motion_is_credible():
    # Rolling along the pitch at 5 m/s — well within bounds.
    xyzs = {
        i: (np.array([10.0 + 5.0 * i / 30.0, 0.0, 0.11]), 0.5)
        for i in range(20)
    }
    states = {i: "grounded" for i in range(20)}
    runs = find_implausible_grounded_runs(
        per_frame_xyz=xyzs,
        per_frame_state=states,
        fps=30.0,
        cfg=_promote_cfg(),
        pitch=_pitch(),
    )
    assert runs == []


def test_flags_off_pitch_run():
    # Ground-projection at y=40 (well past 34 + 5 margin = 39).
    xyzs = {
        i: (np.array([0.0, 40.5, 0.11]), 0.5)
        for i in range(10)
    }
    states = {i: "grounded" for i in range(10)}
    runs = find_implausible_grounded_runs(
        per_frame_xyz=xyzs,
        per_frame_state=states,
        fps=30.0,
        cfg=_promote_cfg(),
        pitch=_pitch(),
    )
    assert len(runs) == 1
    assert runs[0].start == 0 and runs[0].end == 9


def test_flags_speed_exceeding_run():
    # 40 m/s ground speed (above 35).
    xyzs = {
        i: (np.array([40.0 * i / 30.0, 0.0, 0.11]), 0.5)
        for i in range(10)
    }
    states = {i: "grounded" for i in range(10)}
    runs = find_implausible_grounded_runs(
        per_frame_xyz=xyzs,
        per_frame_state=states,
        fps=30.0,
        cfg=_promote_cfg(),
        pitch=_pitch(),
    )
    assert len(runs) == 1


def test_ignores_runs_shorter_than_min_run_frames():
    xyzs = {
        i: (np.array([0.0, 40.5, 0.11]), 0.5)
        for i in range(4)
    }
    states = {i: "grounded" for i in range(4)}
    runs = find_implausible_grounded_runs(
        per_frame_xyz=xyzs,
        per_frame_state=states,
        fps=30.0,
        cfg=_promote_cfg(min_run_frames=6),
        pitch=_pitch(),
    )
    assert runs == []


def test_run_terminates_at_non_grounded_state():
    xyzs = {
        i: (np.array([0.0, 40.5, 0.11]), 0.5)
        for i in range(20)
    }
    states = {i: "grounded" for i in range(20)}
    states[8] = "missing"
    states[9] = "missing"
    runs = find_implausible_grounded_runs(
        per_frame_xyz=xyzs,
        per_frame_state=states,
        fps=30.0,
        cfg=_promote_cfg(),
        pitch=_pitch(),
    )
    # Two qualifying runs: 0..7 (length 8) and 10..19 (length 10).
    assert len(runs) == 2
    assert (runs[0].start, runs[0].end) == (0, 7)
    assert (runs[1].start, runs[1].end) == (10, 19)
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `pytest tests/test_ball_plausibility.py -v`
Expected: import errors for `GroundedRun`, `GroundPromotionCfg`, `find_implausible_grounded_runs`.

- [ ] **Step 3: Implement the new functions**

Append to `src/utils/ball_plausibility.py`:

```python
@dataclass(frozen=True)
class GroundPromotionCfg:
    enabled: bool
    min_run_frames: int
    off_pitch_margin_m: float
    max_ground_speed_m_s: float


@dataclass(frozen=True)
class GroundedRun:
    """A contiguous run of grounded frames flagged for refit-as-flight."""
    start: int
    end: int                       # inclusive


def _off_pitch_distance(x: float, y: float, pitch: PitchDims) -> float:
    """Point-to-rectangle distance (0 inside, positive outside) for the
    axis-aligned pitch centred at the origin."""
    dx = max(0.0, abs(x) - pitch.length_m / 2.0)
    dy = max(0.0, abs(y) - pitch.width_m / 2.0)
    return float(np.hypot(dx, dy))


def find_implausible_grounded_runs(
    *,
    per_frame_xyz: dict[int, tuple[np.ndarray, float]],
    per_frame_state: dict[int, str],
    fps: float,
    cfg: GroundPromotionCfg,
    pitch: PitchDims,
) -> list[GroundedRun]:
    """Return runs of ``state="grounded"`` frames whose world positions
    imply impossible rolling motion (off-pitch or > max_ground_speed).
    Empty list when disabled or nothing qualifies."""
    if not cfg.enabled:
        return []
    if not per_frame_state:
        return []

    frames = sorted(per_frame_state)
    out: list[GroundedRun] = []
    run_start: int | None = None
    last_xy: np.ndarray | None = None
    last_frame: int | None = None
    run_max_off_pitch = 0.0
    run_max_speed = 0.0

    def _close_run(end_frame: int) -> None:
        nonlocal run_start, run_max_off_pitch, run_max_speed
        if run_start is None:
            return
        length = end_frame - run_start + 1
        if length >= cfg.min_run_frames and (
            run_max_off_pitch > cfg.off_pitch_margin_m
            or run_max_speed > cfg.max_ground_speed_m_s
        ):
            out.append(GroundedRun(start=run_start, end=end_frame))
        run_start = None
        run_max_off_pitch = 0.0
        run_max_speed = 0.0

    prev_xy: np.ndarray | None = None
    prev_frame: int | None = None
    for fi in frames:
        state = per_frame_state[fi]
        if state != "grounded" or fi not in per_frame_xyz:
            if run_start is not None:
                _close_run(prev_frame if prev_frame is not None else fi - 1)
            prev_xy = None
            prev_frame = None
            continue
        xyz, _conf = per_frame_xyz[fi]
        xy = np.array([xyz[0], xyz[1]])
        if run_start is None:
            run_start = fi
            run_max_off_pitch = _off_pitch_distance(xy[0], xy[1], pitch)
            run_max_speed = 0.0
        else:
            off = _off_pitch_distance(xy[0], xy[1], pitch)
            run_max_off_pitch = max(run_max_off_pitch, off)
            if prev_xy is not None and prev_frame is not None and fi > prev_frame:
                dt = (fi - prev_frame) / fps
                speed = float(np.linalg.norm(xy - prev_xy) / max(dt, 1e-9))
                run_max_speed = max(run_max_speed, speed)
        prev_xy = xy
        prev_frame = fi

    if run_start is not None and prev_frame is not None:
        _close_run(prev_frame)

    return out
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_ball_plausibility.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_plausibility.py tests/test_ball_plausibility.py
git commit -m "feat(ball): add find_implausible_grounded_runs (Layer 2 core)"
```

---

## Task 5: Wire Layer 2 into `BallStage`

**Files:**
- Modify: `src/stages/ball.py`
- Modify: `config/default.yaml` (add `flight_promotion` block)
- Modify: `tests/test_ball_stage.py`

- [ ] **Step 1: Add `flight_promotion` config block**

In `config/default.yaml`, add under `ball:`:

```yaml
ball:
  flight_promotion:
    enabled: true
    min_run_frames: 6
    off_pitch_margin_m: 5.0
    max_ground_speed_m_s: 35.0
```

- [ ] **Step 2: Write the failing integration test**

Add to `tests/test_ball_stage.py`:

```python
@pytest.mark.integration
def test_aerial_arc_promotes_grounded_run_to_flight(tmp_path: Path):
    """Reproduces origi01 frames 101–191: a long aerial pass where IMM
    never trips flight mode. Layer 2 must detect the implausible ground
    motion and refit as a flight segment."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    clip = out / "shots" / "play.mp4"
    n_frames = 60
    _write_blank_clip(clip, n=n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    ShotsManifest(
        clip_id="aerial-test",
        shots=(Shot(id="play", clip_file="shots/play.mp4", start_frame=0, end_frame=n_frames-1),),
    ).save(out / "shots" / "shots_manifest.json")

    # Synthesise pixel detections that follow a clean low-curvature line
    # in pixel space (so the IMM stays in grounded mode), but whose
    # ground-projection — given our camera at (52.5, -30, 30) looking
    # up — implies a >35 m/s ground roll across the pitch.
    detections: list[tuple[float, float, float] | None] = [None] * 5
    for i in range(50):
        u = 200.0 + 18.0 * i  # 18 px/frame ≈ very fast pixel motion
        v = 200.0 + 0.5 * i
        detections.append((u, v, 0.85))
    detections += [None] * (n_frames - len(detections))

    stage = BallStage(
        config={
            "ball": {
                "detector": "fake",
                "ball_radius_m": 0.11,
                "max_gap_frames": 6,
                "flight_max_residual_px": 200.0,
                "tracker": {
                    "process_noise_grounded_px": 4.0,
                    "process_noise_flight_px": 12.0,
                    "measurement_noise_px": 2.0,
                    "gating_sigma": 4.0,
                    "min_flight_frames": 6,
                    "max_flight_frames": 90,
                },
                "spin": {"enabled": False, "min_flight_seconds": 0.5, "min_residual_improvement": 0.2, "max_omega_rad_s": 200.0, "drag_k_over_m": 0.005},
                "plausibility": {"z_max_m": 50.0, "horizontal_speed_max_m_s": 40.0, "pitch_margin_m": 5.0},
                "flight_promotion": {"enabled": True, "min_run_frames": 6, "off_pitch_margin_m": 5.0, "max_ground_speed_m_s": 35.0},
            },
            "pitch": {"length_m": 105.0, "width_m": 68.0},
        },
        output_dir=out,
        ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")

    # Within the detection window we must NOT have a long pure-grounded
    # run anymore: either it was promoted to flight or marked missing.
    grounded_window = [
        f for f in track.frames
        if 5 <= f.frame < 55 and f.state == "grounded"
    ]
    assert len(grounded_window) < 30, (
        f"expected promotion/demotion to break the long grounded run; "
        f"got {len(grounded_window)} grounded frames in 5..55"
    )
```

- [ ] **Step 3: Run new test — expect FAIL**

Run: `pytest tests/test_ball_stage.py::test_aerial_arc_promotes_grounded_run_to_flight -v`
Expected: FAIL — the run is currently emitted as ~50 grounded frames.

- [ ] **Step 4: Implement promotion in `BallStage`**

In `src/stages/ball.py`:

(a) Extend the imports added in Task 3 to include the new symbols:

```python
from src.utils.ball_plausibility import (
    GroundPromotionCfg,
    GroundedRun,
    PitchDims,
    PlausibilityCfg,
    find_implausible_grounded_runs,
    is_plausible_trajectory,
)
```

(b) Refactor `_run_shot` so the per-frame world / flight assembly is
followed by a single "promotion pass" before the final `BallFrame`
construction. Concretely: after the existing `for sid, (a, b) in enumerate(flight_runs):`
loop completes (i.e. just before the `per_frame_out: list[BallFrame] = []` line),
insert this block:

```python
        promote_cfg = GroundPromotionCfg(
            enabled=bool(cfg.get("flight_promotion", {}).get("enabled", True)),
            min_run_frames=int(cfg.get("flight_promotion", {}).get("min_run_frames", 6)),
            off_pitch_margin_m=float(cfg.get("flight_promotion", {}).get("off_pitch_margin_m", 5.0)),
            max_ground_speed_m_s=float(cfg.get("flight_promotion", {}).get("max_ground_speed_m_s", 35.0)),
        )

        # Provisional state map matching what would be emitted below.
        provisional_state: dict[int, str] = {}
        for fi in range(n_frames):
            if fi in per_frame_world:
                provisional_state[fi] = "flight" if fi in flight_membership else "grounded"
            else:
                provisional_state[fi] = "missing"

        runs_to_promote = find_implausible_grounded_runs(
            per_frame_xyz=per_frame_world,
            per_frame_state=provisional_state,
            fps=camera.fps,
            cfg=promote_cfg,
            pitch=pitch_dims,
        )

        next_segment_id = (max(flight_membership.values()) + 1) if flight_membership else 0
        for run in runs_to_promote:
            obs_pairs = [
                (fi, steps[fi].uv) for fi in range(run.start, run.end + 1)
                if 0 <= fi < len(steps) and steps[fi].uv is not None and fi in per_frame_K
            ]
            if len(obs_pairs) < tracker_cfg.get("min_flight_frames", 6) if isinstance(tracker_cfg, dict) else 6:
                continue
            obs = [(o[0], (float(o[1][0]), float(o[1][1]))) for o in obs_pairs]
            Ks_seg = [per_frame_K[o[0]] for o in obs]
            Rs_seg = [per_frame_R[o[0]] for o in obs]
            ts_seg = [per_frame_t[o[0]] for o in obs]
            try:
                p0, v0, parab_resid = fit_parabola_to_image_observations(
                    obs, Ks=Ks_seg, Rs=Rs_seg, t_world=ts_seg,
                    fps=camera.fps, distortion=distortion,
                )
            except Exception as exc:
                logger.debug("promotion refit failed at run %d-%d: %s", run.start, run.end, exc)
                _demote_run_to_missing(per_frame_world, run.start, run.end)
                continue
            seg_duration = (run.end - run.start) / camera.fps
            if not is_plausible_trajectory(
                p0, v0, omega=None,
                duration_s=seg_duration, fps=camera.fps,
                cfg=plaus_cfg, pitch=pitch_dims,
            ):
                logger.info(
                    "ball: promotion refit for run %d-%d failed plausibility; "
                    "marking frames missing",
                    run.start, run.end,
                )
                _demote_run_to_missing(per_frame_world, run.start, run.end)
                continue

            sid = next_segment_id
            next_segment_id += 1
            g_vec = np.array([0.0, 0.0, g])
            for fi in range(run.start, run.end + 1):
                if fi not in per_frame_K:
                    continue
                dt = (fi - run.start) / camera.fps
                pos = p0 + v0 * dt + 0.5 * g_vec * dt ** 2
                prev_conf = per_frame_world.get(fi, (None, 0.5))[1]
                per_frame_world[fi] = (pos, prev_conf)
                flight_membership[fi] = sid
            flight_segments.append(
                FlightSegment(
                    id=sid,
                    frame_range=(run.start, run.end),
                    parabola={
                        "p0": [float(x) for x in p0],
                        "v0": [float(x) for x in v0],
                        "g": g,
                        "spin_axis_world": None,
                        "spin_omega_rad_s": None,
                        "spin_confidence": None,
                    },
                    fit_residual_px=parab_resid,
                )
            )
```

(c) Add the helper at module scope, near the `_build_detector` function:

```python
def _demote_run_to_missing(
    per_frame_world: dict[int, tuple[np.ndarray, float]],
    a: int,
    b: int,
) -> None:
    """Drop world positions for frames [a, b] so they emit state='missing'."""
    for fi in range(a, b + 1):
        per_frame_world.pop(fi, None)
```

- [ ] **Step 5: Run all ball tests — expect pass**

Run: `pytest tests/test_ball_stage.py tests/test_ball_plausibility.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add config/default.yaml src/stages/ball.py tests/test_ball_stage.py
git commit -m "feat(ball): promote implausible grounded runs to flight (Layer 2)"
```

---

## Task 6: `p0_fixed` parameter on `fit_parabola_to_image_observations`

**Files:**
- Modify: `src/utils/bundle_adjust.py`
- Create: `tests/test_bundle_adjust_p0_fixed.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bundle_adjust_p0_fixed.py`:

```python
"""Tests for the p0_fixed kwarg on parabola/Magnus fits."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.bundle_adjust import fit_parabola_to_image_observations


def _camera() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.0])
    return K, R, t


def _synthesise_observations(
    p0: np.ndarray, v0: np.ndarray, K, R, t, n: int, fps: float = 30.0
):
    g_vec = np.array([0.0, 0.0, -9.81])
    obs = []
    for i in range(n):
        dt = i / fps
        pt = p0 + v0 * dt + 0.5 * g_vec * dt ** 2
        cam = R @ pt + t
        u = float(K @ cam)[0] if False else float((K @ cam)[0] / (K @ cam)[2])
        v = float((K @ cam)[1] / (K @ cam)[2])
        obs.append((i, (u, v)))
    return obs


def test_p0_fixed_none_matches_existing_behaviour():
    K, R, t = _camera()
    # Pick a benign aerial scenario well inside the camera frustum.
    p0_true = np.array([0.0, 5.0, 0.11])
    v0_true = np.array([3.0, 0.5, 12.0])
    obs = _synthesise_observations(p0_true, v0_true, K, R, t, n=15)

    p0_a, v0_a, resid_a = fit_parabola_to_image_observations(
        obs,
        Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t,
        fps=30.0,
    )
    p0_b, v0_b, resid_b = fit_parabola_to_image_observations(
        obs,
        Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t,
        fps=30.0,
        p0_fixed=None,
    )
    assert resid_a == pytest.approx(resid_b)
    assert np.allclose(p0_a, p0_b)
    assert np.allclose(v0_a, v0_b)


def test_p0_fixed_pins_p0_exactly():
    K, R, t = _camera()
    p0_true = np.array([0.0, 5.0, 0.11])
    v0_true = np.array([3.0, 0.5, 12.0])
    obs = _synthesise_observations(p0_true, v0_true, K, R, t, n=15)

    p0_anchored, v0_anchored, resid = fit_parabola_to_image_observations(
        obs,
        Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t,
        fps=30.0,
        p0_fixed=p0_true,
    )
    assert np.allclose(p0_anchored, p0_true)
    assert np.allclose(v0_anchored, v0_true, atol=0.1)
    assert resid < 0.5


def test_p0_fixed_with_noisy_observations_recovers_v0():
    rng = np.random.default_rng(7)
    K, R, t = _camera()
    p0_true = np.array([0.0, 5.0, 0.11])
    v0_true = np.array([3.0, 0.5, 12.0])
    obs = _synthesise_observations(p0_true, v0_true, K, R, t, n=15)
    noisy = [(fi, (uv[0] + rng.normal(0, 0.5), uv[1] + rng.normal(0, 0.5))) for fi, uv in obs]

    _, v0_recovered, resid = fit_parabola_to_image_observations(
        noisy,
        Ks=[K] * len(noisy), Rs=[R] * len(noisy), t_world=t,
        fps=30.0,
        p0_fixed=p0_true,
    )
    assert np.linalg.norm(v0_recovered - v0_true) < 1.0
    assert resid < 2.0
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_bundle_adjust_p0_fixed.py -v`
Expected: FAIL — `fit_parabola_to_image_observations` does not accept `p0_fixed`.

- [ ] **Step 3: Add `p0_fixed` to `fit_parabola_to_image_observations`**

In `src/utils/bundle_adjust.py`, modify the function signature:

```python
def fit_parabola_to_image_observations(
    observations: list[tuple[int, tuple[float, float]]],
    *,
    Ks: list[np.ndarray],
    Rs: list[np.ndarray],
    t_world: np.ndarray | list[np.ndarray],
    fps: float,
    g: float = -9.81,
    max_iter: int = 100,
    distortion: tuple[float, float] = (0.0, 0.0),
    p0_fixed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
```

Then change the LM optimisation block. Replace this section:

```python
    result = least_squares(
        _residuals,
        np.concatenate([p0_seed, v0_seed]),
        method="lm",
        max_nfev=max_iter * 50,
    )
    n = len(observations)
```

with:

```python
    if p0_fixed is None:
        result = least_squares(
            _residuals,
            np.concatenate([p0_seed, v0_seed]),
            method="lm",
            max_nfev=max_iter * 50,
        )
    else:
        p0_pin = np.asarray(p0_fixed, dtype=float).copy()

        def _residuals_v0only(params: np.ndarray) -> np.ndarray:
            v0 = params[:3]
            pts = p0_pin + np.outer(dt, v0) + 0.5 * np.outer(dt ** 2, g_vec)
            residuals = []
            for i in range(n_obs):
                cam = Rs[i] @ pts[i] + ts[i]
                pix = Ks[i] @ cam
                uv = pix[:2] / pix[2]
                residuals.append(uv - obs_array[i])
            return np.concatenate(residuals)

        result = least_squares(
            _residuals_v0only,
            v0_seed,
            method="lm",
            max_nfev=max_iter * 50,
        )
    n = len(observations)
```

And modify the return block. Locate:

```python
    p0_opt = result.x[:3]
    v0_opt = result.x[3:6]
```

(or the equivalent — read the existing tail of the function before editing).
Replace with a conditional:

```python
    if p0_fixed is None:
        p0_opt = result.x[:3]
        v0_opt = result.x[3:6]
    else:
        p0_opt = np.asarray(p0_fixed, dtype=float).copy()
        v0_opt = result.x[:3]
```

(If the existing tail of the function uses a different shape, fold the new
branches into it without disturbing the residual computation.)

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_bundle_adjust_p0_fixed.py tests/test_ball_flight.py tests/test_ball_spin_fit.py -v`
Expected: all pass (new tests pass; existing tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/utils/bundle_adjust.py tests/test_bundle_adjust_p0_fixed.py
git commit -m "feat(bundle_adjust): support p0_fixed in parabola fit"
```

---

## Task 7: `p0_fixed` parameter on `fit_magnus_trajectory`

**Files:**
- Modify: `src/utils/bundle_adjust.py`
- Modify: `tests/test_bundle_adjust_p0_fixed.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bundle_adjust_p0_fixed.py`:

```python
from src.utils.bundle_adjust import fit_magnus_trajectory


def test_magnus_p0_fixed_pins_p0_exactly():
    K, R, t = _camera()
    p0_true = np.array([0.0, 5.0, 0.11])
    v0_true = np.array([3.0, 0.5, 12.0])
    obs = _synthesise_observations(p0_true, v0_true, K, R, t, n=20)

    mp0, mv0, momega, resid = fit_magnus_trajectory(
        obs,
        Ks=[K] * len(obs), Rs=[R] * len(obs), t_world=t,
        fps=30.0,
        drag_k_over_m=0.005,
        p0_seed=p0_true,
        v0_seed=v0_true,
        p0_fixed=p0_true,
    )
    assert np.allclose(mp0, p0_true)
    assert np.linalg.norm(mv0 - v0_true) < 1.0
    # No real spin → ω should be tiny.
    assert np.linalg.norm(momega) < 5.0
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_bundle_adjust_p0_fixed.py::test_magnus_p0_fixed_pins_p0_exactly -v`
Expected: FAIL — `fit_magnus_trajectory` does not accept `p0_fixed`.

- [ ] **Step 3: Add `p0_fixed` to `fit_magnus_trajectory`**

Open `src/utils/bundle_adjust.py`, locate `def fit_magnus_trajectory(`, and:

(a) Add the keyword argument:

```python
    p0_fixed: np.ndarray | None = None,
```

(b) Locate the LM optimisation block at the bottom of the function. It
currently optimises 9 parameters `[p0, v0, omega]`. Wrap it conditionally:

```python
    if p0_fixed is None:
        x0 = np.concatenate([p0_seed, v0_seed, np.zeros(3)])
        result = least_squares(_residuals, x0, method="lm",
                               max_nfev=max_iter * 50)
        p0_opt = result.x[:3]
        v0_opt = result.x[3:6]
        omega_opt = result.x[6:9]
    else:
        p0_pin = np.asarray(p0_fixed, dtype=float).copy()

        def _residuals_anchored(params: np.ndarray) -> np.ndarray:
            v0 = params[:3]
            omega = params[3:6]
            positions = _integrate_magnus_positions(
                p0_pin, v0, omega, g_vec, drag_k_over_m, dt,
            )
            residuals = []
            for i in range(n_obs):
                cam = Rs[i] @ positions[i] + ts[i]
                pix = Ks[i] @ cam
                uv = pix[:2] / pix[2]
                residuals.append(uv - obs_array[i])
            return np.concatenate(residuals)

        x0 = np.concatenate([v0_seed, np.zeros(3)])
        result = least_squares(_residuals_anchored, x0, method="lm",
                               max_nfev=max_iter * 50)
        p0_opt = p0_pin
        v0_opt = result.x[:3]
        omega_opt = result.x[3:6]
```

(Read the function body before editing — the variable names `dt`, `g_vec`,
`obs_array`, `n_obs`, `ts` must match the actual locals in the existing
implementation.)

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_bundle_adjust_p0_fixed.py tests/test_ball_spin_fit.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/utils/bundle_adjust.py tests/test_bundle_adjust_p0_fixed.py
git commit -m "feat(bundle_adjust): support p0_fixed in Magnus fit"
```

---

## Task 8: Layer 3 — `find_kick_anchor` pure function

**Files:**
- Create: `src/utils/ball_kick_anchor.py`
- Create: `tests/test_ball_kick_anchor.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ball_kick_anchor.py`:

```python
"""Unit tests for the Layer 3 kick-anchor helper."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_kick_anchor import (
    KickAnchorCfg,
    find_kick_anchor,
)


def _cfg(**over) -> KickAnchorCfg:
    base = dict(
        enabled=True,
        max_pixel_distance_px=30.0,
        lookahead_frames=4,
        min_pixel_acceleration_px_per_frame=6.0,
        foot_anchor_z_m=0.11,
    )
    base.update(over)
    return KickAnchorCfg(**base)


def _camera():
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.0])
    return K, R, t


def test_kick_detected_when_foot_close_and_acceleration_present():
    K, R, t = _camera()
    # Ball pixel positions: stationary, then rapid acceleration.
    ball_uvs = {10: (640.0, 360.0), 11: (645.0, 365.0), 12: (660.0, 380.0), 13: (685.0, 405.0), 14: (720.0, 440.0)}
    foot_uvs = {10: (635.0, 363.0)}
    anchor = find_kick_anchor(
        segment_start_frame=10,
        ball_uvs=ball_uvs,
        foot_uvs_by_frame=foot_uvs,
        K=K, R=R, t=t,
        cfg=_cfg(),
    )
    assert anchor is not None
    # p0.z should match foot_anchor_z_m.
    assert anchor[2] == pytest.approx(0.11, abs=1e-6)


def test_no_kick_when_foot_far():
    K, R, t = _camera()
    ball_uvs = {10: (640.0, 360.0), 11: (645.0, 365.0), 12: (660.0, 380.0), 13: (685.0, 405.0)}
    foot_uvs = {10: (100.0, 100.0)}
    anchor = find_kick_anchor(
        segment_start_frame=10,
        ball_uvs=ball_uvs,
        foot_uvs_by_frame=foot_uvs,
        K=K, R=R, t=t,
        cfg=_cfg(max_pixel_distance_px=30.0),
    )
    assert anchor is None


def test_no_kick_when_no_acceleration():
    K, R, t = _camera()
    # Constant 1 px/frame motion — no acceleration jump.
    ball_uvs = {i: (640.0 + (i - 10), 360.0) for i in range(10, 15)}
    foot_uvs = {10: (640.0, 363.0)}
    anchor = find_kick_anchor(
        segment_start_frame=10,
        ball_uvs=ball_uvs,
        foot_uvs_by_frame=foot_uvs,
        K=K, R=R, t=t,
        cfg=_cfg(min_pixel_acceleration_px_per_frame=6.0),
    )
    assert anchor is None


def test_disabled_returns_none():
    K, R, t = _camera()
    ball_uvs = {10: (640.0, 360.0), 11: (660.0, 380.0)}
    foot_uvs = {10: (640.0, 363.0)}
    anchor = find_kick_anchor(
        segment_start_frame=10,
        ball_uvs=ball_uvs,
        foot_uvs_by_frame=foot_uvs,
        K=K, R=R, t=t,
        cfg=_cfg(enabled=False),
    )
    assert anchor is None


def test_missing_segment_start_frame_returns_none():
    K, R, t = _camera()
    anchor = find_kick_anchor(
        segment_start_frame=10,
        ball_uvs={11: (660.0, 380.0)},
        foot_uvs_by_frame={10: (640.0, 363.0)},
        K=K, R=R, t=t,
        cfg=_cfg(),
    )
    assert anchor is None
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `pytest tests/test_ball_kick_anchor.py -v`
Expected: `ModuleNotFoundError: No module named 'src.utils.ball_kick_anchor'`.

- [ ] **Step 3: Implement `ball_kick_anchor`**

Create `src/utils/ball_kick_anchor.py`:

```python
"""Layer 3 of the ball-tracking improvement plan: anchor flight-segment
p0 to a player's foot when a kick is detected at the segment's seed
frame.

Inputs are in pixel space (ball uv per frame, ankle uv per frame). When
a kick is detected, the closest ankle pixel is ray-cast onto the ground
plane at z = foot_anchor_z_m via :func:`src.utils.foot_anchor.ankle_ray_to_pitch`,
yielding the world anchor point ``p0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.foot_anchor import ankle_ray_to_pitch


@dataclass(frozen=True)
class KickAnchorCfg:
    enabled: bool
    max_pixel_distance_px: float
    lookahead_frames: int
    min_pixel_acceleration_px_per_frame: float
    foot_anchor_z_m: float


def _pixel_acceleration(ball_uvs: dict[int, tuple[float, float]], start: int, lookahead: int) -> float:
    """Max change in pixel-velocity magnitude over the lookahead window."""
    frames = sorted(f for f in ball_uvs if start <= f <= start + lookahead)
    if len(frames) < 3:
        return 0.0
    speeds = []
    for a, b in zip(frames[:-1], frames[1:]):
        du = ball_uvs[b][0] - ball_uvs[a][0]
        dv = ball_uvs[b][1] - ball_uvs[a][1]
        speeds.append(np.hypot(du, dv) / max(b - a, 1))
    return float(max(speeds) - min(speeds))


def find_kick_anchor(
    *,
    segment_start_frame: int,
    ball_uvs: dict[int, tuple[float, float]],
    foot_uvs_by_frame: dict[int, tuple[float, float]],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    cfg: KickAnchorCfg,
    distortion: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray | None:
    """Return a world-space (x, y, foot_anchor_z_m) anchor if a kick is
    detected at ``segment_start_frame``; otherwise ``None``.

    A kick is declared when:
      - a foot pixel is available at the seed frame within
        ``max_pixel_distance_px`` of the ball pixel; AND
      - the ball pixel-speed varies by at least
        ``min_pixel_acceleration_px_per_frame`` across the
        ``lookahead_frames``-frame window starting at the seed frame.
    """
    if not cfg.enabled:
        return None
    if segment_start_frame not in ball_uvs:
        return None
    if segment_start_frame not in foot_uvs_by_frame:
        return None

    ball_uv = ball_uvs[segment_start_frame]
    foot_uv = foot_uvs_by_frame[segment_start_frame]
    pixel_distance = float(np.hypot(ball_uv[0] - foot_uv[0], ball_uv[1] - foot_uv[1]))
    if pixel_distance > cfg.max_pixel_distance_px:
        return None

    accel = _pixel_acceleration(ball_uvs, segment_start_frame, cfg.lookahead_frames)
    if accel < cfg.min_pixel_acceleration_px_per_frame:
        return None

    anchor_world = ankle_ray_to_pitch(
        foot_uv, K=K, R=R, t=t, plane_z=cfg.foot_anchor_z_m, distortion=distortion,
    )
    return np.asarray(anchor_world, dtype=float)
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_ball_kick_anchor.py -v`
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_kick_anchor.py tests/test_ball_kick_anchor.py
git commit -m "feat(ball): add find_kick_anchor pure function (Layer 3 core)"
```

---

## Task 9: Wire Layer 3 into `BallStage`

**Files:**
- Modify: `src/stages/ball.py`
- Modify: `config/default.yaml`
- Modify: `tests/test_ball_stage.py`

- [ ] **Step 1: Add `kick_anchor` config block**

In `config/default.yaml`, add under `ball:`:

```yaml
ball:
  kick_anchor:
    enabled: true
    max_pixel_distance_px: 30.0
    lookahead_frames: 4
    min_pixel_acceleration_px_per_frame: 6.0
    foot_anchor_z_m: 0.11
```

- [ ] **Step 2: Write the failing integration test**

Append to `tests/test_ball_stage.py`:

```python
@pytest.mark.integration
def test_kick_anchored_fit_pins_p0_to_foot(tmp_path: Path):
    """When a kp2d sidecar puts a player's ankle within 30 px of the
    ball at the flight seed frame, the parabola fit's p0 is anchored
    to the foot ray-cast position (not the unconstrained 6-param fit)."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    clip = out / "shots" / "play.mp4"
    n_frames = 50
    _write_blank_clip(clip, n=n_frames)
    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    ShotsManifest(
        clip_id="kick-test",
        shots=(Shot(id="play", clip_file="shots/play.mp4", start_frame=0, end_frame=n_frames-1),),
    ).save(out / "shots" / "shots_manifest.json")

    # True kick: p0 = (10, 5, 0.11), v0 = (3, 0.5, 12).
    p0_true = np.array([10.0, 5.0, 0.11])
    v0_true = np.array([3.0, 0.5, 12.0])
    g_vec = np.array([0.0, 0.0, -9.81])
    detections: list[tuple[float, float, float] | None] = [None] * 5
    for i in range(30):
        dt = i / 30.0
        pt = p0_true + v0_true * dt + 0.5 * g_vec * dt ** 2
        uv = _project(pt, K, R, t)
        detections.append((uv[0], uv[1], 0.85))
    detections += [None] * (n_frames - len(detections))

    # Synthesise a kp2d sidecar with the kicker's right ankle at p0_true.
    import json as _json
    hmr_dir = out / "hmr_world"
    hmr_dir.mkdir(parents=True, exist_ok=True)
    foot_uv_kick = _project(p0_true, K, R, t)
    kp_zero = [0.0, 0.0, 0.0]
    kp_payload = {
        "player_id": "P001",
        "shot_id": "play",
        "frames": [{
            "frame": 5,
            "keypoints": [kp_zero] * 15 + [list(foot_uv_kick) + [0.9], list(foot_uv_kick) + [0.9]],
        }],
    }
    (hmr_dir / "play__P001_kp2d.json").write_text(_json.dumps(kp_payload))

    stage = BallStage(
        config={
            "ball": {
                "detector": "fake",
                "ball_radius_m": 0.11,
                "max_gap_frames": 6,
                "flight_max_residual_px": 5.0,
                "tracker": {
                    "process_noise_grounded_px": 4.0,
                    "process_noise_flight_px": 12.0,
                    "measurement_noise_px": 2.0,
                    "gating_sigma": 4.0,
                    "min_flight_frames": 6,
                    "max_flight_frames": 90,
                    "initial_p_flight": 0.5,
                },
                "spin": {"enabled": False, "min_flight_seconds": 0.5, "min_residual_improvement": 0.2, "max_omega_rad_s": 200.0, "drag_k_over_m": 0.005},
                "plausibility": {"z_max_m": 50.0, "horizontal_speed_max_m_s": 40.0, "pitch_margin_m": 5.0},
                "flight_promotion": {"enabled": False, "min_run_frames": 6, "off_pitch_margin_m": 5.0, "max_ground_speed_m_s": 35.0},
                "kick_anchor": {"enabled": True, "max_pixel_distance_px": 30.0, "lookahead_frames": 4, "min_pixel_acceleration_px_per_frame": 0.0, "foot_anchor_z_m": 0.11},
            },
            "pitch": {"length_m": 105.0, "width_m": 68.0},
        },
        output_dir=out,
        ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    assert len(track.flight_segments) >= 1
    seg = track.flight_segments[0]
    p0_fit = np.array(seg.parabola["p0"])
    # With anchored fit we expect p0 to land within 0.5 m of the truth.
    assert np.linalg.norm(p0_fit - p0_true) < 0.5, (
        f"expected kick-anchored p0 ≈ {p0_true.tolist()}, got {p0_fit.tolist()}"
    )
```

- [ ] **Step 3: Run new test — expect FAIL**

Run: `pytest tests/test_ball_stage.py::test_kick_anchored_fit_pins_p0_to_foot -v`
Expected: FAIL — current implementation does not consult kp2d.

- [ ] **Step 4: Wire Layer 3 into `_run_shot`**

In `src/stages/ball.py`:

(a) Add imports:

```python
import json

from src.utils.ball_kick_anchor import KickAnchorCfg, find_kick_anchor
```

(b) Add a private helper at module scope:

```python
def _load_foot_uvs_for_shot(
    output_dir: Path, shot_id: str
) -> dict[int, list[tuple[float, float]]]:
    """Aggregate ankle pixel positions across all players for a shot.

    Reads ``output/hmr_world/<shot>__<player>_kp2d.json`` files (COCO-17
    keypoints; indices 15 = left_ankle, 16 = right_ankle). Returns a dict
    keyed by frame index with a list of ankle pixel positions, ignoring
    any with confidence below 0.3.
    """
    hmr_dir = output_dir / "hmr_world"
    if not hmr_dir.exists():
        return {}
    prefix = f"{shot_id}__" if shot_id else ""
    pattern = f"{prefix}*_kp2d.json" if shot_id else "*_kp2d.json"
    feet_by_frame: dict[int, list[tuple[float, float]]] = {}
    for path in hmr_dir.glob(pattern):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        for entry in payload.get("frames", []):
            fi = int(entry.get("frame", -1))
            if fi < 0:
                continue
            kps = entry.get("keypoints", [])
            for idx in (15, 16):
                if idx >= len(kps):
                    continue
                kp = kps[idx]
                if len(kp) < 3 or kp[2] < 0.3:
                    continue
                feet_by_frame.setdefault(fi, []).append((float(kp[0]), float(kp[1])))
    return feet_by_frame
```

(c) Inside `_run_shot`, immediately after building `tracker`, load the
feet:

```python
        feet_pixel_by_frame = _load_foot_uvs_for_shot(self.output_dir, shot_id)
        kick_cfg = KickAnchorCfg(
            enabled=bool(cfg.get("kick_anchor", {}).get("enabled", True))
                    and bool(feet_pixel_by_frame),
            max_pixel_distance_px=float(cfg.get("kick_anchor", {}).get("max_pixel_distance_px", 30.0)),
            lookahead_frames=int(cfg.get("kick_anchor", {}).get("lookahead_frames", 4)),
            min_pixel_acceleration_px_per_frame=float(cfg.get("kick_anchor", {}).get("min_pixel_acceleration_px_per_frame", 6.0)),
            foot_anchor_z_m=float(cfg.get("kick_anchor", {}).get("foot_anchor_z_m", 0.11)),
        )
        if not feet_pixel_by_frame and cfg.get("kick_anchor", {}).get("enabled", True):
            logger.warning(
                "ball stage: kick_anchor enabled but no kp2d sidecars found under %s",
                self.output_dir / "hmr_world",
            )
```

(d) Inside the flight-segment loop (after `obs = [...]` and before the
`try: fit_parabola_to_image_observations(...)` call), compute the
optional anchor:

```python
            ball_uvs_seg = {fi: uv for fi, uv in obs}
            anchor_world: np.ndarray | None = None
            if kick_cfg.enabled:
                # Pick the nearest foot per frame in the segment seed region.
                seed_feet: dict[int, tuple[float, float]] = {}
                for fi in range(a, min(a + kick_cfg.lookahead_frames + 1, b + 1)):
                    feet = feet_pixel_by_frame.get(fi, [])
                    if not feet or fi not in ball_uvs_seg:
                        continue
                    bu, bv = ball_uvs_seg[fi]
                    nearest = min(feet, key=lambda f: (f[0] - bu) ** 2 + (f[1] - bv) ** 2)
                    seed_feet[fi] = nearest
                if a in per_frame_K:
                    anchor_world = find_kick_anchor(
                        segment_start_frame=a,
                        ball_uvs=ball_uvs_seg,
                        foot_uvs_by_frame=seed_feet,
                        K=per_frame_K[a],
                        R=per_frame_R[a],
                        t=per_frame_t[a],
                        cfg=kick_cfg,
                        distortion=distortion,
                    )
```

(e) Modify the parabola fit call to pass `p0_fixed`:

```python
            try:
                p0, v0, parab_resid = fit_parabola_to_image_observations(
                    obs, Ks=Ks_seg, Rs=Rs_seg, t_world=ts_seg,
                    fps=camera.fps, distortion=distortion,
                    p0_fixed=anchor_world,
                )
            except Exception as exc:
                logger.debug("parabola fit failed on segment %d: %s", sid, exc)
                continue
```

(f) Modify the Magnus fit call to pass `p0_fixed` when anchored:

```python
                    mp0, mv0, momega, magnus_resid = fit_magnus_trajectory(
                        obs,
                        Ks=Ks_seg, Rs=Rs_seg, t_world=ts_seg,
                        fps=camera.fps,
                        drag_k_over_m=drag,
                        p0_seed=p0, v0_seed=v0,
                        p0_fixed=anchor_world,
                    )
```

- [ ] **Step 5: Run all ball tests — expect pass**

Run: `pytest tests/test_ball_stage.py tests/test_ball_kick_anchor.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add config/default.yaml src/stages/ball.py tests/test_ball_stage.py
git commit -m "feat(ball): kick-anchored parabola fit using kp2d (Layer 3)"
```

---

## Task 10: Layer 4 — `AppearanceBridge` class

**Files:**
- Create: `src/utils/ball_appearance_bridge.py`
- Create: `tests/test_ball_appearance_bridge.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ball_appearance_bridge.py`:

```python
"""Unit tests for the Layer 4 appearance bridge."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_appearance_bridge import (
    AppearanceBridge,
    AppearanceBridgeCfg,
)


def _cfg(**over) -> AppearanceBridgeCfg:
    base = dict(
        enabled=True,
        max_gap_frames=8,
        template_size_px=32,
        search_radius_px=64,
        min_ncc=0.6,
        template_max_age_frames=30,
        template_update_confidence=0.5,
    )
    base.update(over)
    return AppearanceBridgeCfg(**base)


def _frame_with_ball(uv: tuple[int, int], shape=(720, 1280)) -> np.ndarray:
    """A pitch-green frame with a white-ish ball at uv."""
    img = np.full((*shape, 3), [50, 200, 50], dtype=np.uint8)
    u, v = int(uv[0]), int(uv[1])
    for du in range(-6, 7):
        for dv in range(-6, 7):
            if du * du + dv * dv <= 36 and 0 <= v + dv < shape[0] and 0 <= u + du < shape[1]:
                img[v + dv, u + du] = [240, 240, 240]
    return img


def test_bridge_finds_ball_in_predicted_window():
    bridge = AppearanceBridge(_cfg())
    f0 = _frame_with_ball((640, 360))
    bridge.update_template(frame=0, frame_image=f0, uv=(640.0, 360.0), confidence=0.9)
    # Next frame: ball moved by (10, 5).
    f1 = _frame_with_ball((650, 365))
    result = bridge.try_bridge(frame=1, frame_image=f1, predicted_uv=(648.0, 364.0), consecutive_misses=1)
    assert result is not None
    uv, conf = result
    assert abs(uv[0] - 650.0) < 2.0
    assert abs(uv[1] - 365.0) < 2.0
    assert 0.0 < conf < 1.0


def test_bridge_returns_none_when_no_ball_in_window():
    bridge = AppearanceBridge(_cfg())
    f0 = _frame_with_ball((640, 360))
    bridge.update_template(frame=0, frame_image=f0, uv=(640.0, 360.0), confidence=0.9)
    # Plain green frame, no ball anywhere.
    green = np.full((720, 1280, 3), [50, 200, 50], dtype=np.uint8)
    result = bridge.try_bridge(frame=1, frame_image=green, predicted_uv=(648.0, 364.0), consecutive_misses=1)
    assert result is None


def test_bridge_disabled_after_max_gap():
    bridge = AppearanceBridge(_cfg(max_gap_frames=8))
    f0 = _frame_with_ball((640, 360))
    bridge.update_template(frame=0, frame_image=f0, uv=(640.0, 360.0), confidence=0.9)
    f9 = _frame_with_ball((730, 405))
    result = bridge.try_bridge(frame=9, frame_image=f9, predicted_uv=(728.0, 404.0), consecutive_misses=9)
    assert result is None


def test_bridge_disabled_when_template_stale():
    bridge = AppearanceBridge(_cfg(template_max_age_frames=5))
    f0 = _frame_with_ball((640, 360))
    bridge.update_template(frame=0, frame_image=f0, uv=(640.0, 360.0), confidence=0.9)
    f10 = _frame_with_ball((650, 365))
    result = bridge.try_bridge(frame=10, frame_image=f10, predicted_uv=(648.0, 364.0), consecutive_misses=1)
    assert result is None


def test_bridge_disabled_by_config_flag():
    bridge = AppearanceBridge(_cfg(enabled=False))
    f0 = _frame_with_ball((640, 360))
    bridge.update_template(frame=0, frame_image=f0, uv=(640.0, 360.0), confidence=0.9)
    f1 = _frame_with_ball((650, 365))
    result = bridge.try_bridge(frame=1, frame_image=f1, predicted_uv=(648.0, 364.0), consecutive_misses=1)
    assert result is None


def test_update_template_ignored_when_low_confidence():
    bridge = AppearanceBridge(_cfg(template_update_confidence=0.5))
    f0 = _frame_with_ball((640, 360))
    bridge.update_template(frame=0, frame_image=f0, uv=(640.0, 360.0), confidence=0.2)
    f1 = _frame_with_ball((650, 365))
    result = bridge.try_bridge(frame=1, frame_image=f1, predicted_uv=(648.0, 364.0), consecutive_misses=1)
    assert result is None
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `pytest tests/test_ball_appearance_bridge.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the class**

Create `src/utils/ball_appearance_bridge.py`:

```python
"""Layer 4: bridge short WASB detection gaps via normalised cross-
correlation against a rolling template.

The bridge is *guided* by the IMM tracker's prediction — it only
searches inside a ``search_radius_px`` window around the predicted
pixel. A high NCC peak there is accepted as a bridged detection (with
discounted confidence) so the IMM can continue updating instead of
gap-filling for several frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AppearanceBridgeCfg:
    enabled: bool
    max_gap_frames: int
    template_size_px: int
    search_radius_px: int
    min_ncc: float
    template_max_age_frames: int
    template_update_confidence: float


class AppearanceBridge:
    """Stateful holder for a rolling ball template plus an NCC bridger.

    Not thread-safe. One instance per shot.
    """

    def __init__(self, cfg: AppearanceBridgeCfg) -> None:
        self._cfg = cfg
        self._template: np.ndarray | None = None
        self._template_frame: int | None = None

    def update_template(
        self,
        *,
        frame: int,
        frame_image: np.ndarray,
        uv: tuple[float, float],
        confidence: float,
    ) -> None:
        if not self._cfg.enabled:
            return
        if confidence < self._cfg.template_update_confidence:
            return
        half = self._cfg.template_size_px // 2
        u, v = int(round(uv[0])), int(round(uv[1]))
        h, w = frame_image.shape[:2]
        if u - half < 0 or v - half < 0 or u + half > w or v + half > h:
            return
        crop = frame_image[v - half:v + half, u - half:u + half]
        if crop.shape[:2] != (self._cfg.template_size_px, self._cfg.template_size_px):
            return
        self._template = crop.copy()
        self._template_frame = frame

    def try_bridge(
        self,
        *,
        frame: int,
        frame_image: np.ndarray,
        predicted_uv: tuple[float, float] | None,
        consecutive_misses: int,
    ) -> tuple[tuple[float, float], float] | None:
        """Try to find the ball near ``predicted_uv``. Returns
        ``((u, v), confidence)`` on success, or ``None`` if disabled,
        out of gap budget, template stale, or NCC below threshold."""
        if not self._cfg.enabled:
            return None
        if self._template is None or self._template_frame is None:
            return None
        if consecutive_misses > self._cfg.max_gap_frames:
            return None
        if predicted_uv is None:
            return None
        age = frame - self._template_frame
        if age > self._cfg.template_max_age_frames:
            return None

        h, w = frame_image.shape[:2]
        r = self._cfg.search_radius_px
        u, v = int(round(predicted_uv[0])), int(round(predicted_uv[1]))
        u0, v0 = max(0, u - r), max(0, v - r)
        u1, v1 = min(w, u + r), min(h, v + r)
        if u1 - u0 <= self._cfg.template_size_px or v1 - v0 <= self._cfg.template_size_px:
            return None
        window = frame_image[v0:v1, u0:u1]
        result = cv2.matchTemplate(window, self._template, cv2.TM_CCOEFF_NORMED)
        _, peak, _, peak_loc = cv2.minMaxLoc(result)
        if peak < self._cfg.min_ncc:
            return None
        half = self._cfg.template_size_px // 2
        peak_u = u0 + peak_loc[0] + half
        peak_v = v0 + peak_loc[1] + half
        # Discount confidence so the IMM weighs real WASB hits higher.
        return (float(peak_u), float(peak_v)), float(peak) * 0.5
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_ball_appearance_bridge.py -v`
Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_appearance_bridge.py tests/test_ball_appearance_bridge.py
git commit -m "feat(ball): add AppearanceBridge for Layer 4 NCC bridging"
```

---

## Task 11: Wire Layer 4 into `BallStage`

**Files:**
- Modify: `src/stages/ball.py`
- Modify: `config/default.yaml`
- Modify: `tests/test_ball_stage.py`

- [ ] **Step 1: Add `appearance_bridge` config block**

In `config/default.yaml`, add under `ball:`:

```yaml
ball:
  appearance_bridge:
    enabled: true
    max_gap_frames: 8
    template_size_px: 32
    search_radius_px: 64
    min_ncc: 0.6
    template_max_age_frames: 30
    template_update_confidence: 0.5
```

- [ ] **Step 2: Write the failing integration test**

Append to `tests/test_ball_stage.py`:

```python
@pytest.mark.integration
def test_appearance_bridge_fills_short_detection_gap(tmp_path: Path):
    """When WASB returns None for 1-3 frames but a fresh template and
    the IMM prediction agree on a region containing the ball, the
    appearance bridge fills the gap (no state='missing')."""
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    clip_path = out / "shots" / "play.mp4"
    n_frames = 30
    # Write a clip where the ball is a real white circle on green; the
    # bridge will find it in the predicted window.
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720)
    )
    for i in range(n_frames):
        img = np.full((720, 1280, 3), [50, 200, 50], dtype=np.uint8)
        u, v = 200 + 5 * i, 200 + 1 * i
        cv2.circle(img, (u, v), 8, (240, 240, 240), -1)
        writer.write(img)
    writer.release()

    _save_camera_track(out / "camera" / "play_camera_track.json", K, R, t, n_frames)
    ShotsManifest(
        clip_id="bridge-test",
        shots=(Shot(id="play", clip_file="shots/play.mp4", start_frame=0, end_frame=n_frames-1),),
    ).save(out / "shots" / "shots_manifest.json")

    # Detections: present for frames 0..9, missing for 10..12 (3-frame gap), present for 13..29.
    detections: list[tuple[float, float, float] | None] = []
    for i in range(n_frames):
        if 10 <= i <= 12:
            detections.append(None)
        else:
            u, v = 200.0 + 5.0 * i, 200.0 + 1.0 * i
            detections.append((u, v, 0.85))

    stage = BallStage(
        config={
            "ball": {
                "detector": "fake",
                "ball_radius_m": 0.11,
                "max_gap_frames": 6,
                "flight_max_residual_px": 5.0,
                "tracker": {
                    "process_noise_grounded_px": 4.0,
                    "process_noise_flight_px": 12.0,
                    "measurement_noise_px": 2.0,
                    "gating_sigma": 4.0,
                    "min_flight_frames": 6,
                    "max_flight_frames": 90,
                },
                "spin": {"enabled": False, "min_flight_seconds": 0.5, "min_residual_improvement": 0.2, "max_omega_rad_s": 200.0, "drag_k_over_m": 0.005},
                "plausibility": {"z_max_m": 50.0, "horizontal_speed_max_m_s": 40.0, "pitch_margin_m": 5.0},
                "flight_promotion": {"enabled": False, "min_run_frames": 6, "off_pitch_margin_m": 5.0, "max_ground_speed_m_s": 35.0},
                "kick_anchor": {"enabled": False, "max_pixel_distance_px": 30.0, "lookahead_frames": 4, "min_pixel_acceleration_px_per_frame": 6.0, "foot_anchor_z_m": 0.11},
                "appearance_bridge": {"enabled": True, "max_gap_frames": 8, "template_size_px": 32, "search_radius_px": 64, "min_ncc": 0.6, "template_max_age_frames": 30, "template_update_confidence": 0.5},
            },
            "pitch": {"length_m": 105.0, "width_m": 68.0},
        },
        output_dir=out,
        ball_detector=FakeBallDetector(detections),
    )
    stage.run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")
    # Frames 10..12 must NOT be state="missing".
    gap_states = [f.state for f in track.frames if 10 <= f.frame <= 12]
    assert all(s != "missing" for s in gap_states), (
        f"expected bridge to fill frames 10-12; got {gap_states}"
    )
```

- [ ] **Step 3: Run new test — expect FAIL**

Run: `pytest tests/test_ball_stage.py::test_appearance_bridge_fills_short_detection_gap -v`
Expected: FAIL — frames 10..12 emit `state="missing"`.

- [ ] **Step 4: Wire Layer 4 into the detection loop**

In `src/stages/ball.py`:

(a) Add import:

```python
from src.utils.ball_appearance_bridge import (
    AppearanceBridge,
    AppearanceBridgeCfg,
)
```

(b) Inside `_run_shot`, before the `cap = cv2.VideoCapture(...)` line,
build the bridge:

```python
        bridge_cfg = AppearanceBridgeCfg(
            enabled=bool(cfg.get("appearance_bridge", {}).get("enabled", True)),
            max_gap_frames=int(cfg.get("appearance_bridge", {}).get("max_gap_frames", 8)),
            template_size_px=int(cfg.get("appearance_bridge", {}).get("template_size_px", 32)),
            search_radius_px=int(cfg.get("appearance_bridge", {}).get("search_radius_px", 64)),
            min_ncc=float(cfg.get("appearance_bridge", {}).get("min_ncc", 0.6)),
            template_max_age_frames=int(cfg.get("appearance_bridge", {}).get("template_max_age_frames", 30)),
            template_update_confidence=float(cfg.get("appearance_bridge", {}).get("template_update_confidence", 0.5)),
        )
        bridge = AppearanceBridge(bridge_cfg)
        consecutive_misses = 0
```

(c) Modify the detection loop. Replace:

```python
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                det = detector.detect(frame)
                if det is None:
                    uv: tuple[float, float] | None = None
                else:
                    uv = (float(det[0]), float(det[1]))
                    raw_confidences[frame_idx] = float(det[2])
                step = tracker.update(frame_idx, uv)
                steps.append(step)
                frame_idx += 1
```

with:

```python
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                det = detector.detect(frame)
                if det is None:
                    consecutive_misses += 1
                    bridge_result = bridge.try_bridge(
                        frame=frame_idx,
                        frame_image=frame,
                        predicted_uv=(
                            (float(steps[-1].uv[0]), float(steps[-1].uv[1]))
                            if steps and steps[-1].uv is not None else None
                        ),
                        consecutive_misses=consecutive_misses,
                    )
                    if bridge_result is None:
                        uv: tuple[float, float] | None = None
                    else:
                        uv, bridged_conf = bridge_result
                        raw_confidences[frame_idx] = bridged_conf
                else:
                    consecutive_misses = 0
                    uv = (float(det[0]), float(det[1]))
                    raw_confidences[frame_idx] = float(det[2])
                    bridge.update_template(
                        frame=frame_idx,
                        frame_image=frame,
                        uv=uv,
                        confidence=float(det[2]),
                    )
                step = tracker.update(frame_idx, uv)
                steps.append(step)
                frame_idx += 1
```

- [ ] **Step 5: Run all ball tests — expect pass**

Run: `pytest tests/test_ball_stage.py tests/test_ball_appearance_bridge.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add config/default.yaml src/stages/ball.py tests/test_ball_stage.py
git commit -m "feat(ball): bridge short detection gaps via NCC (Layer 4)"
```

---

## Task 12: End-to-end integration scenario

**Files:**
- Create: `tests/test_ball_stage_layered.py`

- [ ] **Step 1: Write a single scenario that touches all four layers**

Create `tests/test_ball_stage_layered.py`:

```python
"""End-to-end scenario reproducing the origi01 failure modes and
asserting all four layers cooperate correctly."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas.ball_track import BallTrack
from src.schemas.camera_track import CameraFrame, CameraTrack
from src.schemas.shots import Shot, ShotsManifest
from src.stages.ball import BallStage
from src.utils.ball_detector import FakeBallDetector


def _camera_pose():
    look = np.array([0.0, 64.0, -30.0]); look /= np.linalg.norm(look)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(look, right)
    R = np.array([right, down, look], dtype=float)
    t = -R @ np.array([52.5, -30.0, 30.0])
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 360.0], [0, 0, 1.0]])
    return K, R, t


def _project(p, K, R, t):
    cam = R @ p + t
    pix = K @ cam
    return float(pix[0] / pix[2]), float(pix[1] / pix[2])


def _full_cfg() -> dict:
    return {
        "ball": {
            "detector": "fake",
            "ball_radius_m": 0.11,
            "max_gap_frames": 6,
            "flight_max_residual_px": 5.0,
            "tracker": {
                "process_noise_grounded_px": 4.0,
                "process_noise_flight_px": 12.0,
                "measurement_noise_px": 2.0,
                "gating_sigma": 4.0,
                "min_flight_frames": 6,
                "max_flight_frames": 90,
            },
            "spin": {"enabled": False, "min_flight_seconds": 0.5, "min_residual_improvement": 0.2, "max_omega_rad_s": 200.0, "drag_k_over_m": 0.005},
            "plausibility": {"z_max_m": 50.0, "horizontal_speed_max_m_s": 40.0, "pitch_margin_m": 5.0},
            "flight_promotion": {"enabled": True, "min_run_frames": 6, "off_pitch_margin_m": 5.0, "max_ground_speed_m_s": 35.0},
            "kick_anchor": {"enabled": True, "max_pixel_distance_px": 30.0, "lookahead_frames": 4, "min_pixel_acceleration_px_per_frame": 0.0, "foot_anchor_z_m": 0.11},
            "appearance_bridge": {"enabled": True, "max_gap_frames": 8, "template_size_px": 32, "search_radius_px": 64, "min_ncc": 0.6, "template_max_age_frames": 30, "template_update_confidence": 0.5},
        },
        "pitch": {"length_m": 105.0, "width_m": 68.0},
    }


@pytest.mark.integration
def test_origi01_like_scenario(tmp_path: Path):
    K, R, t = _camera_pose()
    out = tmp_path / "out"
    clip_path = out / "shots" / "play.mp4"
    n_frames = 90
    clip_path.parent.mkdir(parents=True, exist_ok=True)

    # Trajectory plan:
    # 0..29   grounded (rolling)
    # 30..59  airborne kick from (10, 5, 0.11) with v0 = (4, 1, 10)
    # 60..62  detection gap (handled by Layer 4)
    # 63..89  grounded again
    p0_kick = np.array([10.0, 5.0, 0.11])
    v0_kick = np.array([4.0, 1.0, 10.0])
    g_vec = np.array([0.0, 0.0, -9.81])

    detections: list[tuple[float, float, float] | None] = []
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720))

    for i in range(n_frames):
        img = np.full((720, 1280, 3), [50, 200, 50], dtype=np.uint8)
        if i < 30:
            pt = np.array([10.0 - 0.1 * i, 5.0, 0.11])
        elif i <= 59:
            dt = (i - 30) / 30.0
            pt = p0_kick + v0_kick * dt + 0.5 * g_vec * dt ** 2
        else:
            pt = np.array([0.0, 8.0, 0.11])
        u, v = _project(pt, K, R, t)
        cv2.circle(img, (int(u), int(v)), 8, (240, 240, 240), -1)
        writer.write(img)
        if 60 <= i <= 62:
            detections.append(None)
        else:
            detections.append((u, v, 0.85))
    writer.release()

    CameraTrack(
        clip_id="origi-like", fps=30.0, image_size=(1280, 720),
        t_world=t.tolist(),
        frames=tuple(
            CameraFrame(frame=i, K=K.tolist(), R=R.tolist(), confidence=1.0, is_anchor=(i == 0))
            for i in range(n_frames)
        ),
    ).save(out / "camera" / "play_camera_track.json")

    ShotsManifest(
        clip_id="origi-like",
        shots=(Shot(id="play", clip_file="shots/play.mp4", start_frame=0, end_frame=n_frames - 1),),
    ).save(out / "shots" / "shots_manifest.json")

    # Foot kp2d sidecar pinning the kicker's ankle at the kick start.
    hmr_dir = out / "hmr_world"
    hmr_dir.mkdir(parents=True, exist_ok=True)
    foot_uv_kick = _project(p0_kick, K, R, t)
    kp_zero = [0.0, 0.0, 0.0]
    payload = {
        "player_id": "P001",
        "shot_id": "play",
        "frames": [{
            "frame": 30,
            "keypoints": [kp_zero] * 15 + [list(foot_uv_kick) + [0.9], list(foot_uv_kick) + [0.9]],
        }],
    }
    (hmr_dir / "play__P001_kp2d.json").write_text(json.dumps(payload))

    BallStage(
        config=_full_cfg(),
        output_dir=out,
        ball_detector=FakeBallDetector(detections),
    ).run()

    track = BallTrack.load(out / "ball" / "play_ball_track.json")

    # Layer 1 + 4: no z above z_max_m, no off-pitch beyond 5 m.
    for f in track.frames:
        if f.world_xyz is None:
            continue
        x, y, z = f.world_xyz
        assert abs(x) <= 105.0 / 2 + 5.0
        assert abs(y) <= 68.0 / 2 + 5.0
        assert -1.0 <= z <= 50.0

    # Layer 2/3: at least one flight segment in the 30..59 range with apex z >= 3 m.
    apex_zs = [
        max(f.world_xyz[2] for f in track.frames if 30 <= f.frame <= 59 and f.world_xyz)
    ]
    assert max(apex_zs) >= 3.0, f"expected apex >= 3 m, got {apex_zs}"

    # Layer 4: frames 60..62 should not be missing.
    gap = [f.state for f in track.frames if 60 <= f.frame <= 62]
    assert all(s != "missing" for s in gap), f"appearance bridge missed gap: {gap}"

    # All FlightSegment p0 within plausible pitch envelope.
    for seg in track.flight_segments:
        p0 = seg.parabola["p0"]
        assert abs(p0[0]) <= 105.0 / 2 + 5.0
        assert abs(p0[1]) <= 68.0 / 2 + 5.0
        assert -1.0 <= p0[2] <= 50.0
```

- [ ] **Step 2: Run scenario — expect PASS**

Run: `pytest tests/test_ball_stage_layered.py -v`
Expected: pass. If it fails on the kick anchor (because no segment was
detected as flight by the IMM), reduce the IMM seed `initial_p_flight`
to make flight more likely, then re-run.

- [ ] **Step 3: Run the full ball test suite**

Run: `pytest tests/test_ball_stage.py tests/test_ball_stage_layered.py tests/test_ball_plausibility.py tests/test_ball_kick_anchor.py tests/test_ball_appearance_bridge.py tests/test_bundle_adjust_p0_fixed.py tests/test_ball_flight.py tests/test_ball_grounded.py tests/test_ball_tracker_imm.py tests/test_ball_spin_fit.py -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ball_stage_layered.py
git commit -m "test(ball): end-to-end layered scenario"
```

---

## Task 13: Validate on real `origi01` data

**Files:** none

This task verifies the result on real data — no code changes.

- [ ] **Step 1: Re-run the ball stage on `origi01`**

Run: `python recon.py run --input clip.mp4 --output ./output/ --from-stage ball`
(Adjust `--input` to your actual clip path if different.)

- [ ] **Step 2: Inspect the new ball track**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("output/ball/origi01_ball_track.json").read_text())
print(f"flight_segments: {len(d['flight_segments'])}")
for s in d["flight_segments"]:
    p0 = s["parabola"]["p0"]
    print(f"  seg {s['id']} range={s['frame_range']} p0=({p0[0]:.1f}, {p0[1]:.1f}, {p0[2]:.2f}) resid={s['fit_residual_px']:.2f}")

counts = {"flight": 0, "grounded": 0, "missing": 0}
for f in d["frames"]:
    counts[f["state"]] = counts.get(f["state"], 0) + 1
print(f"state counts: {counts}")

# Sample the diagonal switch range.
print("\nFrames 101-191 sample (every 10):")
for f in d["frames"]:
    fi = f["frame"]
    if 101 <= fi <= 191 and (fi - 101) % 10 == 0:
        w = f.get("world_xyz")
        if w:
            print(f"  f={fi:3d} state={f['state']:<8} xyz=({w[0]:.1f}, {w[1]:.1f}, {w[2]:.2f}) seg={f.get('flight_segment_id')}")
        else:
            print(f"  f={fi:3d} state={f['state']:<8} xyz=None")
PY
```

Expected:
- No flight segments with `p0` outside ~[±60 m, ±40 m, [-1, 50] m].
- Frames 101–191 contain at least one flight segment with non-trivial z (apex ≥ 3 m).
- The frame-state mix should show fewer `missing` frames than before in the 191–442 range.

- [ ] **Step 3: Open the dashboard and visually verify**

Run: `python recon.py serve --output ./output/`

Open `http://localhost:8000/viewer` (or whatever port the server reports)
and play through the clip. The diagonal switch (frames 101–191) should
show the ball arcing over the pitch rather than skimming the ground.
The goal sequence (442+) should still nestle correctly. If anything
regressed, capture the failure as a new test case and iterate.

- [ ] **Step 4: No commit necessary** (verification only).

---

## Self-Review Notes

- **Spec coverage:** Goals 1–3 from the spec are covered by Tasks 3, 5,
  9, and 11 (with the integration in Task 12 asserting all three on
  synthetic data and Task 13 verifying real data).
- **Placeholders:** none — every code block is concrete.
- **Type consistency:**
  - `PlausibilityCfg`, `PitchDims`, `GroundedRun`, `GroundPromotionCfg`,
    `KickAnchorCfg`, `AppearanceBridgeCfg` are all `@dataclass(frozen=True)`
    and consistent across tasks.
  - `find_kick_anchor` returns `np.ndarray | None`, which Tasks 8 and 9
    both treat the same way.
  - `AppearanceBridge.try_bridge` returns `tuple[tuple[float, float], float] | None`,
    consumed identically in Task 11.
- **Risks acknowledged in spec:**
  - Pipeline-order change avoided (Layer 3 reads `hmr_world/*_kp2d.json`,
    which already ships before ball).
  - Graceful degradation when kp2d absent: `kick_cfg.enabled` is
    auto-disabled in Task 9 step 4(c) with a warning.
  - Magnus rank-deficiency mitigated by existing `spin.min_flight_seconds`
    floor.
