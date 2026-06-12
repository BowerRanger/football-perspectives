# Ball Evidence Booster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 1 of [the ball v2 design](../specs/2026-06-12-ball-v2-design.md): a second detection pass over evidence gaps, gated by a forward–backward smoothed corridor, so detector-limited clips (origi02: 44 % coverage) become solvable with zero new user input.

**Architecture:** Pass 1 (existing detect loop) is unchanged. A forward–backward IMM smoother built from pass-1 observations only predicts a per-frame corridor (mean + covariance); gap frames are revisited with low-threshold top-k candidate detection, candidates are gated by Mahalanobis distance inside the corridor and re-scored, with a zoom-crop retry when the predicted apparent ball size is small. Accepted detections merge into the observation stream (`source: "second_pass"`), the IMM re-runs over the merged stream, and downstream solving proceeds as today — except second-pass frames can never mint auto-anchors.

**Tech Stack:** Python, numpy/scipy (all new pure logic is light-venv testable), cv2 for video access, existing `BallTracker` IMM, WASB HRNet detector (torch — not unit-tested; its postprocessing is factored into a pure module).

**Conventions:** Repo rules apply — TDD, frozen dataclasses, type annotations, files < 800 lines. Run tests with `python -m pytest` from the repo root using the project venv (`.venv/bin/python -m pytest` works for all new tests; integration tests also need `cv2`, available in `.venv311`). Use whichever venv the existing `tests/test_ball_stage.py` passes in on your machine — verify with step 0 below.

**Pre-flight check (run once before Task 1):**

Run: `.venv311/bin/python -m pytest tests/test_ball_tracker_imm.py tests/test_ball_stage.py -q`
Expected: all pass. If `cv2` or `torch` import errors occur, use the venv where these pass for all later "Run" steps.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `src/utils/ball_tracker.py` | Modify | `TrackerStep.pos_cov` + blended covariance output |
| `src/utils/ball_heatmap.py` | Create | Pure heatmap→candidate-blob extraction (numpy/scipy only) |
| `src/utils/ball_detector.py` | Modify | `detect_candidates`/`reset` base API + `FakeBallDetector` support |
| `src/utils/wasb_ball_detector.py` | Modify | Implement candidate API on WASB via `ball_heatmap` |
| `src/utils/ball_second_pass.py` | Create | Cfg, Gaussian fusion, corridor prediction, gap runs, gating/scoring, apparent-size, crop mapping — pure logic, no video/torch |
| `src/stages/ball.py` | Modify | `_build_tracker` extraction, `_resmooth_observations`, `_second_pass_loop` + `_zoom_detect` (video), `_run_shot` integration, diag coverage, `_second_pass_cfg` |
| `src/utils/ball_auto_anchor.py` | Modify | `sources` param — exclude `second_pass` frames from anchor candidates |
| `src/pipeline/quality_report.py` | Modify | Pass `detection_coverage` through to the report |
| `config/default.yaml` | Modify | `ball.second_pass.*` block |
| `tests/test_ball_tracker_cov.py` | Create | Covariance emission behavior |
| `tests/test_ball_heatmap.py` | Create | Blob extraction |
| `tests/test_ball_detector_candidates.py` | Create | Base adapter + fake detector candidate API |
| `tests/test_ball_second_pass.py` | Create | Fusion, corridor, gap runs, gating, apparent size, crop mapping |
| `tests/test_ball_auto_anchor.py` | Modify | Exclusion of second-pass frames |
| `tests/test_ball_stage_second_pass.py` | Create | End-to-end stage test with a scripted detector |

---

### Task 1: Expose blended position covariance from the IMM tracker

The corridor needs per-frame uncertainty. `TrackerStep` gains an optional `pos_cov` field — `(sigma_uu, sigma_vv, sigma_uv)` of the blended 4-state posterior's position block. Default `None` keeps every existing constructor call valid.

**Files:**
- Modify: `src/utils/ball_tracker.py`
- Modify: `src/stages/ball.py:597-602` (raw-uv override constructs `TrackerStep` — must forward the new field)
- Test: `tests/test_ball_tracker_cov.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ball_tracker_cov.py`:

```python
"""TrackerStep.pos_cov: blended position covariance for corridor gating.

The second-pass corridor (ball_second_pass.py) fuses forward and
backward IMM passes; it needs each step's positional uncertainty. Pins:
covariance present once initialised, grows during a detection gap, and
shrinks again when detections resume.
"""

from __future__ import annotations

import pytest

from src.utils.ball_tracker import BallTracker, TrackerStep


@pytest.mark.unit
def test_pos_cov_none_before_first_detection():
    tracker = BallTracker()
    step = tracker.update(0, None)
    assert step.pos_cov is None


@pytest.mark.unit
def test_pos_cov_emitted_and_grows_during_gap():
    tracker = BallTracker(max_gap_frames=100)
    covs = []
    for i in range(10):
        step = tracker.update(i, (100.0 + 5.0 * i, 400.0))
        covs.append(step.pos_cov)
    assert all(c is not None for c in covs)

    gap_covs = []
    for i in range(10, 20):
        step = tracker.update(i, None)
        gap_covs.append(step.pos_cov)
    # Uncertainty grows monotonically while predicting blind.
    assert gap_covs[-1][0] > gap_covs[0][0] > covs[-1][0]
    assert gap_covs[-1][1] > gap_covs[0][1] > covs[-1][1]

    resumed = tracker.update(20, (200.0, 400.0))
    assert resumed.pos_cov[0] < gap_covs[-1][0]


@pytest.mark.unit
def test_trackerstep_pos_cov_defaults_none():
    step = TrackerStep(frame=0, uv=None, p_flight=0.1,
                       is_outlier=False, is_gap_fill=True)
    assert step.pos_cov is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python -m pytest tests/test_ball_tracker_cov.py -q`
Expected: FAIL — `TypeError`/`AttributeError` around `pos_cov`.

- [ ] **Step 3: Implement**

In `src/utils/ball_tracker.py`, add the field to `TrackerStep`:

```python
@dataclass(frozen=True)
class TrackerStep:
    """One filtered step emitted by :class:`BallTracker`."""

    frame: int
    uv: tuple[float, float] | None
    p_flight: float
    is_outlier: bool
    is_gap_fill: bool
    # Blended posterior position covariance (sigma_uu, sigma_vv, sigma_uv),
    # px². None until the filter has been initialised by a detection.
    pos_cov: tuple[float, float, float] | None = None
```

In `update()`, the cold-start branches return `pos_cov=None` (no detection yet) and `pos_cov=(self._r ** 2, self._r ** 2, 0.0)` (first detection, from `P0`):

```python
        # Cold start — wait for the first detection to seed the filter.
        if self._x[0] is None:
            if uv is None:
                return TrackerStep(
                    frame=frame, uv=None, p_flight=float(self._mu[1]),
                    is_outlier=False, is_gap_fill=True, pos_cov=None,
                )
            self._init_state(uv)
            self._consecutive_gap = 0
            return TrackerStep(
                frame=frame, uv=uv, p_flight=float(self._mu[1]),
                is_outlier=False, is_gap_fill=False,
                pos_cov=(self._r ** 2, self._r ** 2, 0.0),
            )
```

At the end of `update()`, compute the blended state once and derive both `out_uv` and `pos_cov` from it (replace the existing `blended = ...` line inside the `else` branch):

```python
        blended = self._mu[0] * x_post[0] + self._mu[1] * x_post[1]
        P_blend = np.zeros((4, 4))
        for j in range(2):
            d = x_post[j] - blended
            P_blend += self._mu[j] * (P_post[j] + np.outer(d, d))
        pos_cov = (
            float(P_blend[0, 0]), float(P_blend[1, 1]), float(P_blend[0, 1]),
        )

        if is_gap and self._consecutive_gap > self._max_gap:
            out_uv: tuple[float, float] | None = None
        else:
            out_uv = (float(blended[0]), float(blended[1]))

        return TrackerStep(
            frame=frame,
            uv=out_uv,
            p_flight=float(self._mu[1]),
            is_outlier=is_outlier,
            is_gap_fill=is_gap,
            pos_cov=pos_cov,
        )
```

In `src/stages/ball.py` (raw-uv override around line 597), forward the field:

```python
                if uv is not None and not step.is_outlier:
                    step = TrackerStep(
                        frame=step.frame, uv=uv, p_flight=step.p_flight,
                        is_outlier=step.is_outlier,
                        is_gap_fill=step.is_gap_fill,
                        pos_cov=step.pos_cov,
                    )
```

- [ ] **Step 4: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_ball_tracker_cov.py tests/test_ball_tracker_imm.py -q`
Expected: PASS (existing IMM tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_tracker.py src/stages/ball.py tests/test_ball_tracker_cov.py
git commit -m "feat(ball): expose blended position covariance from the IMM tracker"
```

---

### Task 2: Pure heatmap candidate extraction (`ball_heatmap.py`)

WASB's blob postprocessing moves into a torch-free, cv2-free module so the top-k candidate logic is unit-testable in the light venv. Note: the existing code uses `cv2.connectedComponents` (8-connectivity); `scipy.ndimage.label` defaults to 4-connectivity, so pass an explicit 8-connected structure.

**Files:**
- Create: `src/utils/ball_heatmap.py`
- Test: `tests/test_ball_heatmap.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ball_heatmap.py`:

```python
"""heatmap_candidates: top-k blob extraction from a detector heatmap."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_heatmap import heatmap_candidates


def _hm_with_blobs(*blobs: tuple[int, int, float, int]) -> np.ndarray:
    """Blobs as (y, x, peak, radius) square stamps on a 72x128 heatmap."""
    hm = np.zeros((72, 128), dtype=np.float32)
    for y, x, peak, r in blobs:
        hm[y - r:y + r + 1, x - r:x + r + 1] = peak
    return hm


@pytest.mark.unit
def test_orders_by_blob_mass_and_returns_peak():
    hm = _hm_with_blobs((20, 30, 0.9, 1), (50, 100, 0.6, 3))
    # Blob 2 has lower peak but far more mass (7x7 @ 0.6 vs 3x3 @ 0.9).
    cands = heatmap_candidates(hm, min_score=0.3, top_k=5)
    assert len(cands) == 2
    (x0, y0, p0), (x1, y1, p1) = cands
    assert (round(x0), round(y0), p0) == (100, 50, pytest.approx(0.6))
    assert (round(x1), round(y1), p1) == (30, 20, pytest.approx(0.9))


@pytest.mark.unit
def test_min_score_filters_and_top_k_truncates():
    hm = _hm_with_blobs((20, 30, 0.9, 1), (50, 100, 0.2, 3), (60, 10, 0.5, 2))
    assert len(heatmap_candidates(hm, min_score=0.3, top_k=5)) == 2
    assert len(heatmap_candidates(hm, min_score=0.3, top_k=1)) == 1
    assert heatmap_candidates(hm, min_score=0.95, top_k=5) == []


@pytest.mark.unit
def test_diagonal_pixels_are_one_blob():
    """8-connectivity parity with cv2.connectedComponents."""
    hm = np.zeros((10, 10), dtype=np.float32)
    hm[2, 2] = 0.8
    hm[3, 3] = 0.8  # diagonal neighbour
    assert len(heatmap_candidates(hm, min_score=0.5, top_k=5)) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python -m pytest tests/test_ball_heatmap.py -q`
Expected: FAIL — `ModuleNotFoundError: src.utils.ball_heatmap`.

- [ ] **Step 3: Implement**

Create `src/utils/ball_heatmap.py`:

```python
"""Pure candidate-blob extraction from ball-detector heatmaps.

Torch- and cv2-free so the candidate logic is unit-testable in the
light venv. 8-connected labelling matches the cv2.connectedComponents
behaviour previously used in wasb_ball_detector.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

_EIGHT_CONNECTED = np.ones((3, 3), dtype=int)


def heatmap_candidates(
    hm: np.ndarray,
    min_score: float,
    top_k: int,
) -> list[tuple[float, float, float]]:
    """Top-k heatmap blobs as ``(x, y, peak)`` in heatmap pixel coords.

    Blobs are connected components of ``hm >= min_score``, ordered by
    descending sum-of-weights (mass); ``(x, y)`` is the heatmap-weighted
    centroid and ``peak`` the blob's max heatmap value.
    """
    mask = hm >= min_score
    if not mask.any():
        return []
    labels, n_labels = ndimage.label(mask, structure=_EIGHT_CONNECTED)
    blobs: list[tuple[float, float, float, float]] = []
    for label in range(1, n_labels + 1):
        ys, xs = np.nonzero(labels == label)
        ws = hm[ys, xs]
        mass = float(ws.sum())
        x = float((xs * ws).sum() / mass)
        y = float((ys * ws).sum() / mass)
        blobs.append((mass, x, y, float(ws.max())))
    blobs.sort(key=lambda b: -b[0])
    return [(x, y, peak) for _, x, y, peak in blobs[:top_k]]
```

- [ ] **Step 4: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_ball_heatmap.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_heatmap.py tests/test_ball_heatmap.py
git commit -m "feat(ball): pure heatmap candidate extraction module"
```

---

### Task 3: Candidate API on the detector interface

`BallDetector` gains `detect_candidates` (default: adapt `detect()`) and `reset()` (default: no-op). WASB implements both for real via `ball_heatmap`; `FakeBallDetector` gains scripted-candidate support. The WASB class is torch-bound and not unit-tested — its `detect()` is refactored onto the same tested `heatmap_candidates` function so behavior is pinned indirectly.

**Files:**
- Modify: `src/utils/ball_detector.py`
- Modify: `src/utils/wasb_ball_detector.py`
- Test: `tests/test_ball_detector_candidates.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ball_detector_candidates.py`:

```python
"""detect_candidates / reset on the BallDetector interface."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_detector import FakeBallDetector

_FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.mark.unit
def test_default_adapter_wraps_detect():
    det = FakeBallDetector([(100.0, 200.0, 0.8), None])
    assert det.detect_candidates(_FRAME, min_score=0.5, top_k=5) == [
        (100.0, 200.0, 0.8)
    ]
    assert det.detect_candidates(_FRAME, min_score=0.5, top_k=5) == []


@pytest.mark.unit
def test_default_adapter_applies_min_score():
    det = FakeBallDetector([(100.0, 200.0, 0.2)])
    assert det.detect_candidates(_FRAME, min_score=0.5, top_k=5) == []


@pytest.mark.unit
def test_fake_scripted_candidates_filter_and_truncate():
    det = FakeBallDetector(
        [None],
        candidates=[[(10.0, 10.0, 0.9), (20.0, 20.0, 0.4), (30.0, 30.0, 0.7)]],
    )
    out = det.detect_candidates(_FRAME, min_score=0.5, top_k=2)
    assert out == [(10.0, 10.0, 0.9), (20.0, 20.0, 0.7)] or out == [
        (10.0, 10.0, 0.9), (30.0, 30.0, 0.7)
    ]
    assert len(out) == 2


@pytest.mark.unit
def test_reset_is_counted_on_fake_and_noop_on_base():
    det = FakeBallDetector([None])
    det.reset()
    det.reset()
    assert det.reset_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python -m pytest tests/test_ball_detector_candidates.py -q`
Expected: FAIL — no `detect_candidates` / `reset_count`.

- [ ] **Step 3: Implement the interface + fake**

In `src/utils/ball_detector.py`, add to `BallDetector` (below the abstract `detect`):

```python
    def detect_candidates(
        self, frame: np.ndarray, min_score: float, top_k: int = 5,
    ) -> list[tuple[float, float, float]]:
        """Low-threshold candidate detections ``[(u, v, score), ...]``.

        Default adapter wraps :meth:`detect` (a single candidate, if its
        confidence clears ``min_score``). Detectors with heatmap access
        override this with true top-k extraction.
        """
        det = self.detect(frame)
        if det is None or det[2] < min_score:
            return []
        return [det]

    def reset(self) -> None:
        """Clear any temporal state (frame buffers). Default: no-op."""
        return None
```

Replace `FakeBallDetector` with:

```python
class FakeBallDetector(BallDetector):
    """Deterministic detector for tests — cycles through pre-supplied detections.

    Each entry is either ``(u, v, confidence)`` or ``None``. Optional
    ``candidates`` (a parallel cycle of candidate lists) scripts
    :meth:`detect_candidates`; ``reset_count`` records :meth:`reset` calls.
    """

    def __init__(
        self,
        detections: list[tuple[float, float, float] | None],
        candidates: list[list[tuple[float, float, float]]] | None = None,
    ) -> None:
        self._detections = detections
        self._candidates = candidates
        self._idx = 0
        self._cand_idx = 0
        self.reset_count = 0

    def detect(self, frame: np.ndarray) -> tuple[float, float, float] | None:
        d = self._detections[self._idx % len(self._detections)]
        self._idx += 1
        return d

    def detect_candidates(
        self, frame: np.ndarray, min_score: float, top_k: int = 5,
    ) -> list[tuple[float, float, float]]:
        if self._candidates is None:
            return super().detect_candidates(frame, min_score, top_k)
        cands = self._candidates[self._cand_idx % len(self._candidates)]
        self._cand_idx += 1
        kept = [c for c in cands if c[2] >= min_score]
        kept.sort(key=lambda c: -c[2])
        return kept[:top_k]

    def reset(self) -> None:
        self.reset_count += 1
```

- [ ] **Step 4: Implement WASB candidate detection**

In `src/utils/wasb_ball_detector.py`: add `from src.utils.ball_heatmap import heatmap_candidates` to the imports, then refactor `detect()` and add the two methods. Extract the buffer+forward portion of `detect()` (everything from the buffer maintenance through `hm = hms[0, -1]`) into:

```python
    def _forward(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Update the 3-frame ring, run HRNet; returns (heatmap, trans_inv)."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame must be (H, W, 3); got {frame.shape}")
        if not self._buffer:
            self._buffer = [frame.copy(), frame.copy(), frame.copy()]
        else:
            self._buffer.append(frame.copy())
            if len(self._buffer) > self._frames_in:
                self._buffer.pop(0)
        h, w = frame.shape[:2]
        stacked, trans_inv = self._preprocess_buffer((h, w))
        inp = self._torch.from_numpy(stacked).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            out = self._model(inp)
        hms = self._torch.sigmoid(out[0]).cpu().numpy()  # (1, 3, H, W)
        return hms[0, -1], trans_inv
```

Then replace the bodies of `detect` and add the new methods (the old
connected-components block is deleted — `heatmap_candidates` is its tested
replacement):

```python
    def detect(self, frame: np.ndarray) -> tuple[float, float, float] | None:
        cands = self.detect_candidates(frame, min_score=self._confidence, top_k=1)
        return cands[0] if cands else None

    def detect_candidates(
        self, frame: np.ndarray, min_score: float, top_k: int = 5,
    ) -> list[tuple[float, float, float]]:
        hm, trans_inv = self._forward(frame)
        out: list[tuple[float, float, float]] = []
        for cx, cy, peak in heatmap_candidates(hm, min_score, top_k):
            uv = _affine_apply(np.array([cx, cy], dtype=np.float32), trans_inv)
            out.append((float(uv[0]), float(uv[1]), peak))
        return out

    def reset(self) -> None:
        self._buffer = []
```

Also give `YOLOBallDetector` (in `src/utils/ball_detector.py`) true top-k, per the design spec:

```python
    def detect_candidates(
        self, frame: np.ndarray, min_score: float, top_k: int = 5,
    ) -> list[tuple[float, float, float]]:
        results = self._model(frame, verbose=False)[0]
        out: list[tuple[float, float, float]] = []
        for box in results.boxes:
            if int(box.cls) != self._ball_class_id:
                continue
            conf = float(box.conf)
            if conf < min_score:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, conf))
        out.sort(key=lambda c: -c[2])
        return out[:top_k]
```

(YOLO is stateless per frame, so the inherited no-op `reset()` is correct. The shim `WASBBallDetector` in `ball_detector.py` delegates via `__new__`, so it needs no change.)

- [ ] **Step 5: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_ball_detector_candidates.py tests/test_ball_heatmap.py tests/test_ball_stage.py -q`
Expected: PASS (stage tests confirm `FakeBallDetector`'s changed constructor stays compatible).

- [ ] **Step 6: Commit**

```bash
git add src/utils/ball_detector.py src/utils/wasb_ball_detector.py tests/test_ball_detector_candidates.py
git commit -m "feat(ball): candidate detection API on BallDetector, WASB and the test fake"
```

---

### Task 4: Corridor prediction (`ball_second_pass.py`, part 1)

Forward and backward IMM passes over pass-1 observations, fused per frame into a corridor `(mean, cov)`. The corridor is built from pass-1 evidence only — this is the feedback-loop guard from the spec.

**Files:**
- Create: `src/utils/ball_second_pass.py`
- Test: `tests/test_ball_second_pass.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ball_second_pass.py`:

```python
"""Second-pass detection: corridor prediction, gating, gap runs."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.ball_second_pass import (
    SecondPassCfg,
    corridor_predictions,
    fuse_gaussians,
)
from src.utils.ball_tracker import BallTracker


def _tracker_factory() -> BallTracker:
    # Huge max_gap so predictions persist through long gaps.
    return BallTracker(max_gap_frames=10 ** 6)


@pytest.mark.unit
def test_fuse_gaussians_tightens_and_weights_by_precision():
    m1, c1 = np.array([0.0, 0.0]), np.eye(2) * 1.0
    m2, c2 = np.array([10.0, 0.0]), np.eye(2) * 9.0
    m, c = fuse_gaussians(m1, c1, m2, c2)
    # Precision-weighted: 9x tighter first estimate dominates.
    assert m[0] == pytest.approx(1.0)
    assert c[0, 0] == pytest.approx(0.9)
    assert c[0, 0] < min(c1[0, 0], c2[0, 0])


@pytest.mark.unit
def test_corridor_bridges_gap_near_interpolation():
    """Constant-velocity roll with a hole at frames 20-29: the fused
    forward/backward corridor must stay near the true line, far closer
    than either causal pass alone could drift."""
    n = 50
    truth = {f: (100.0 + 8.0 * f, 400.0) for f in range(n)}
    obs: dict[int, tuple[float, float] | None] = {
        f: (truth[f] if not (20 <= f < 30) else None) for f in range(n)
    }
    corridors = corridor_predictions(obs, n, _tracker_factory)
    for f in range(20, 30):
        mean, cov = corridors[f]
        du = abs(mean[0] - truth[f][0])
        dv = abs(mean[1] - truth[f][1])
        assert du < 12.0 and dv < 6.0, (f, du, dv)
        assert cov.shape == (2, 2)
        # Mid-gap uncertainty exceeds observed-frame uncertainty.
        assert cov[0, 0] > corridors[5][1][0, 0]


@pytest.mark.unit
def test_corridor_empty_when_no_observations():
    assert corridor_predictions({0: None, 1: None}, 2, _tracker_factory) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python -m pytest tests/test_ball_second_pass.py -q`
Expected: FAIL — `ModuleNotFoundError: src.utils.ball_second_pass`.

- [ ] **Step 3: Implement**

Create `src/utils/ball_second_pass.py`:

```python
"""Second-pass ball detection: corridor prediction + candidate gating.

Pass 1 (the streaming detect loop) misses frames; this module predicts
where the ball should be on those frames (a forward/backward IMM fusion
over pass-1 observations ONLY — second-pass output never steers its own
corridor) and gates low-threshold detector candidates against that
corridor. Pure logic: no video access, no torch (the stage owns I/O).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.utils.ball_tracker import BallTracker
from src.utils.camera_projection import pixel_ray

# Covariance floor (px²) so a corridor next to a confident observation
# still admits detector-sized localisation error.
_COV_FLOOR_PX2 = 4.0 ** 2


@dataclass(frozen=True)
class SecondPassCfg:
    enabled: bool = True
    candidate_min_score: float = 0.05
    top_k: int = 5
    corridor_sigma: float = 3.0
    accept_min: float = 0.25
    zoom_min_ball_px: float = 8.0
    zoom_crop_px: int = 320


@dataclass(frozen=True)
class SecondPassDetection:
    frame: int
    uv: tuple[float, float]
    combined_score: float
    used_zoom: bool


def fuse_gaussians(
    m1: np.ndarray, c1: np.ndarray, m2: np.ndarray, c2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Product of two 2D Gaussians (precision-weighted mean, fused cov)."""
    i1, i2 = np.linalg.inv(c1), np.linalg.inv(c2)
    cov = np.linalg.inv(i1 + i2)
    return cov @ (i1 @ m1 + i2 @ m2), cov


def _cov_matrix(pos_cov: tuple[float, float, float]) -> np.ndarray:
    suu, svv, suv = pos_cov
    return np.array([[suu, suv], [suv, svv]], dtype=float)


def _run_pass(
    per_frame_uv: dict[int, tuple[float, float] | None],
    order: range,
    tracker_factory: Callable[[], BallTracker],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    tracker = tracker_factory()
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, f in enumerate(order):
        step = tracker.update(i, per_frame_uv.get(f))
        if step.uv is not None and step.pos_cov is not None:
            out[f] = (np.array(step.uv, dtype=float), _cov_matrix(step.pos_cov))
    return out


def corridor_predictions(
    per_frame_uv: dict[int, tuple[float, float] | None],
    n_frames: int,
    tracker_factory: Callable[[], BallTracker],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-frame (mean, cov) search corridor from pass-1 observations.

    Forward and backward IMM passes (the constant-velocity model is
    time-symmetric), fused where both are initialised.
    """
    fwd = _run_pass(per_frame_uv, range(n_frames), tracker_factory)
    bwd = _run_pass(per_frame_uv, range(n_frames - 1, -1, -1), tracker_factory)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for f in range(n_frames):
        a, b = fwd.get(f), bwd.get(f)
        if a is not None and b is not None:
            out[f] = fuse_gaussians(a[0], a[1], b[0], b[1])
        elif a is not None or b is not None:
            out[f] = a if a is not None else b
    return out
```

- [ ] **Step 4: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_ball_second_pass.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_second_pass.py tests/test_ball_second_pass.py
git commit -m "feat(ball): forward-backward corridor prediction for second-pass detection"
```

---

### Task 5: Gap runs, candidate gating, apparent size, crop mapping (part 2)

**Files:**
- Modify: `src/utils/ball_second_pass.py`
- Test: `tests/test_ball_second_pass.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ball_second_pass.py`:

```python
from src.utils.ball_second_pass import (  # noqa: E402
    apparent_ball_px,
    best_gated_candidate,
    find_gap_runs,
    map_crop_candidates,
)


@pytest.mark.unit
def test_find_gap_runs_groups_missing_and_outlier_frames():
    sources = {0: "detector", 1: "detector", 5: "bridge", 6: "anchor"}
    runs = find_gap_runs(sources, outlier_frames={5}, n_frames=8)
    assert runs == [(2, 5), (7, 7)]


@pytest.mark.unit
def test_gate_rejects_decoy_outside_corridor_accepts_inside():
    cfg = SecondPassCfg()
    mean, cov = np.array([500.0, 300.0]), np.eye(2) * 25.0
    decoy = (900.0, 300.0, 0.95)          # high score, far outside
    true_cand = (505.0, 302.0, 0.6)       # modest score, inside
    best = best_gated_candidate([decoy, true_cand], mean, cov, cfg)
    assert best is not None
    (u, v), combined = best
    assert (u, v) == (505.0, 302.0)
    assert 0.0 < combined <= 0.6


@pytest.mark.unit
def test_gate_enforces_accept_min():
    cfg = SecondPassCfg(accept_min=0.5)
    mean, cov = np.array([500.0, 300.0]), np.eye(2) * 25.0
    assert best_gated_candidate([(505.0, 302.0, 0.3)], mean, cov, cfg) is None


@pytest.mark.unit
def test_gate_is_deterministic():
    cfg = SecondPassCfg()
    mean, cov = np.array([500.0, 300.0]), np.eye(2) * 25.0
    cands = [(505.0, 302.0, 0.6), (498.0, 297.0, 0.6)]
    assert best_gated_candidate(cands, mean, cov, cfg) == best_gated_candidate(
        cands, mean, cov, cfg
    )


@pytest.mark.unit
def test_apparent_ball_px_scales_inverse_with_depth():
    # Camera 20 m above pitch looking straight down: depth ~ 20 m.
    K = np.array([[2000.0, 0, 640.0], [0, 2000.0, 360.0], [0, 0, 1.0]])
    R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # z_cam = -z_world
    t = -R @ np.array([0.0, 0.0, 20.0])
    size = apparent_ball_px(K, R, t, (640.0, 360.0), ball_radius_m=0.11)
    # f * d / depth = 2000 * 0.22 / (20 - 0.11) ≈ 22.1 px
    assert size == pytest.approx(22.1, abs=0.5)


@pytest.mark.unit
def test_apparent_ball_px_none_when_ray_misses_pitch():
    K = np.array([[2000.0, 0, 640.0], [0, 2000.0, 360.0], [0, 0, 1.0]])
    R = np.eye(3)  # looking along +z_world (up): never reaches the pitch
    t = -R @ np.array([0.0, 0.0, 20.0])
    assert apparent_ball_px(K, R, t, (640.0, 360.0), ball_radius_m=0.11) is None


@pytest.mark.unit
def test_map_crop_candidates_offsets_back_to_full_frame():
    assert map_crop_candidates([(10.0, 20.0, 0.7)], x0=300, y0=100) == [
        (310.0, 120.0, 0.7)
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv311/bin/python -m pytest tests/test_ball_second_pass.py -q`
Expected: new tests FAIL with ImportError; Task-4 tests still pass.

- [ ] **Step 3: Implement**

Append to `src/utils/ball_second_pass.py`:

```python
# Sources that mean "pass 1 accepted evidence on this frame".
PASS1_SOURCES = ("detector", "anchor", "bridge")


def find_gap_runs(
    sources: dict[int, str],
    outlier_frames: set[int],
    n_frames: int,
) -> list[tuple[int, int]]:
    """Consecutive runs of frames with no accepted pass-1 detection."""
    gap = [
        f for f in range(n_frames)
        if sources.get(f) not in PASS1_SOURCES or f in outlier_frames
    ]
    runs: list[tuple[int, int]] = []
    for f in gap:
        if runs and f == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], f)
        else:
            runs.append((f, f))
    return runs


def best_gated_candidate(
    candidates: list[tuple[float, float, float]],
    mean: np.ndarray,
    cov: np.ndarray,
    cfg: SecondPassCfg,
) -> tuple[tuple[float, float], float] | None:
    """Best corridor-gated candidate as ``((u, v), combined_score)``.

    Gate: Mahalanobis² <= corridor_sigma². Score:
    ``candidate_score * exp(-0.5 * d² / corridor_sigma²)``, accepted when
    it clears ``accept_min``.
    """
    cov_f = cov + _COV_FLOOR_PX2 * np.eye(2)
    inv = np.linalg.inv(cov_f)
    best: tuple[tuple[float, float], float] | None = None
    for u, v, score in candidates:
        d = np.array([u, v], dtype=float) - mean
        d2 = float(d @ inv @ d)
        if d2 > cfg.corridor_sigma ** 2:
            continue
        combined = float(score) * math.exp(-0.5 * d2 / cfg.corridor_sigma ** 2)
        if best is None or combined > best[1]:
            best = ((float(u), float(v)), combined)
    if best is None or best[1] < cfg.accept_min:
        return None
    return best


def apparent_ball_px(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    uv: tuple[float, float],
    ball_radius_m: float,
    distortion: tuple[float, float] = (0.0, 0.0),
) -> float | None:
    """Predicted apparent ball diameter (px) at a pixel's ground depth.

    Ray-casts the pixel to the ball-centre plane ``z = ball_radius_m``;
    None when the ray never reaches it (above-horizon prediction). A
    ground-depth approximation — airborne balls are nearer the camera
    and look bigger, so this under-zooms, never over-zooms.
    """
    C, d_hat = pixel_ray(uv, K, R, t, distortion)
    dz = float(d_hat[2])
    if abs(dz) < 1e-9:
        return None
    s = (ball_radius_m - float(C[2])) / dz
    if s <= 0:
        return None
    return float(K[0][0]) * (2.0 * ball_radius_m) / s


def map_crop_candidates(
    candidates: list[tuple[float, float, float]],
    x0: int,
    y0: int,
) -> list[tuple[float, float, float]]:
    """Translate crop-space candidates back into full-frame pixels."""
    return [(u + x0, v + y0, s) for u, v, s in candidates]
```

(`K[0][0]` works for both nested lists and arrays; cameras here are arrays.)

- [ ] **Step 4: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_ball_second_pass.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_second_pass.py tests/test_ball_second_pass.py
git commit -m "feat(ball): gap runs, corridor gating and zoom helpers for the second pass"
```

---

### Task 6: Stage plumbing — `_build_tracker` and `_resmooth_observations`

Extract tracker construction from `_detect_loop` (it will be needed three times: pass 1, corridor factory, re-smooth) and add the merged-observation re-smoother. Pure refactor + one new function; no behavior change to pass 1.

**Files:**
- Modify: `src/stages/ball.py`
- Test: `tests/test_ball_stage_second_pass.py` (created here with the re-smooth test; grows in Task 8)

- [ ] **Step 1: Write the failing test**

Create `tests/test_ball_stage_second_pass.py`:

```python
"""Second-pass integration: re-smoothing and the BallStage end-to-end run."""

from __future__ import annotations

import numpy as np
import pytest

from src.stages.ball import _build_tracker, _resmooth_observations


@pytest.mark.unit
def test_resmooth_keeps_raw_uv_and_fills_gaps():
    n = 30
    uv = {f: (100.0 + 5.0 * f, 400.0) for f in range(n)}
    uv[10] = None
    uv[11] = None
    steps = _resmooth_observations(uv, n, cfg={})
    assert len(steps) == n
    # Raw observations pass through exactly (raw-uv override rule).
    assert steps[5].uv == (125.0, 400.0)
    # Short gap is IMM-filled near the constant-velocity line.
    assert steps[10].uv is not None
    assert abs(steps[10].uv[0] - 150.0) < 5.0
    assert steps[10].is_gap_fill


@pytest.mark.unit
def test_build_tracker_honours_max_gap_override():
    tracker = _build_tracker({}, max_gap_frames=10 ** 6)
    for i in range(5):
        tracker.update(i, (100.0 + i, 400.0))
    last = None
    for i in range(5, 105):
        last = tracker.update(i, None)
    assert last.uv is not None  # would be None with the default max_gap
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python -m pytest tests/test_ball_stage_second_pass.py -q`
Expected: FAIL — ImportError on `_build_tracker` / `_resmooth_observations`.

- [ ] **Step 3: Implement**

In `src/stages/ball.py`, add module-level functions (near the other `_`-helpers, e.g. below `_write_observations_sidecar`):

```python
def _build_tracker(cfg: dict, max_gap_frames: int | None = None) -> BallTracker:
    """IMM tracker from the ball config; optional max-gap override for
    corridor prediction (predictions must persist through long gaps)."""
    tracker_cfg = cfg.get("tracker", {})
    return BallTracker(
        process_noise_grounded_px=float(tracker_cfg.get("process_noise_grounded_px", 4.0)),
        process_noise_flight_px=float(tracker_cfg.get("process_noise_flight_px", 12.0)),
        measurement_noise_px=float(tracker_cfg.get("measurement_noise_px", 2.0)),
        gating_sigma=float(tracker_cfg.get("gating_sigma", 4.0)),
        max_gap_frames=(
            int(cfg.get("max_gap_frames", 6))
            if max_gap_frames is None else int(max_gap_frames)
        ),
        initial_p_flight=float(tracker_cfg.get("initial_p_flight", 0.1)),
    )


def _resmooth_observations(
    per_frame_uv: dict[int, tuple[float, float] | None],
    n_frames: int,
    cfg: dict,
) -> list[TrackerStep]:
    """Fresh IMM pass over a merged observation set (no video access).

    Applies the same raw-uv override rule as the streaming detect loop:
    fits must see raw measurements, the tracker only bridges misses.
    """
    tracker = _build_tracker(cfg)
    steps: list[TrackerStep] = []
    for f in range(n_frames):
        uv = per_frame_uv.get(f)
        step = tracker.update(f, uv)
        if uv is not None and not step.is_outlier:
            step = TrackerStep(
                frame=step.frame, uv=uv, p_flight=step.p_flight,
                is_outlier=step.is_outlier, is_gap_fill=step.is_gap_fill,
                pos_cov=step.pos_cov,
            )
        steps.append(step)
    return steps
```

In `_detect_loop`, replace the inline `tracker = BallTracker(...)` block (lines 516–524) with:

```python
        tracker = _build_tracker(cfg)
```

- [ ] **Step 4: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_ball_stage_second_pass.py tests/test_ball_stage.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stages/ball.py tests/test_ball_stage_second_pass.py
git commit -m "refactor(ball): extract tracker construction; add merged-observation re-smoother"
```

---### Task 7: Auto-anchor exclusion of second-pass frames

Second-pass observations densify evidence but never mint constraints.

**Files:**
- Modify: `src/utils/ball_auto_anchor.py` (`generate_auto_anchors`)
- Modify: `src/stages/ball.py` (call site — pass `sources`)
- Test: `tests/test_ball_auto_anchor.py` (append)

- [ ] **Step 1: Write the failing test**

Open `tests/test_ball_auto_anchor.py`, find an existing test that calls `generate_auto_anchors` with grounded sampling (there is one exercising `_grounded_candidates` via confidently-grounded steps — reuse its fixture style for `steps`/`confidences`/camera mappings), and append:

```python
@pytest.mark.unit
def test_second_pass_frames_never_become_anchors():
    """Frames whose observation source is 'second_pass' are excluded from
    anchor candidacy even when their confidence clears every gate."""
    # Build the same confidently-grounded scenario as the grounded-sampling
    # test above (copy its steps/confidences/camera setup verbatim), then:
    sources = {s.frame: "second_pass" for s in steps}
    anchors = generate_auto_anchors(
        events=[],
        steps=steps,
        confidences=confidences,
        player_ctx=player_ctx,
        per_frame_K=per_frame_K,
        per_frame_R=per_frame_R,
        per_frame_t=per_frame_t,
        distortion=(0.0, 0.0),
        fps=30.0,
        pitch_cfg=pitch_cfg,
        cfg=cfg,
        sources=sources,
    )
    assert anchors == ()

    # Same call with detector-sourced frames still yields anchors.
    sources_ok = {s.frame: "detector" for s in steps}
    anchors_ok = generate_auto_anchors(
        events=[],
        steps=steps,
        confidences=confidences,
        player_ctx=player_ctx,
        per_frame_K=per_frame_K,
        per_frame_R=per_frame_R,
        per_frame_t=per_frame_t,
        distortion=(0.0, 0.0),
        fps=30.0,
        pitch_cfg=pitch_cfg,
        cfg=cfg,
        sources=sources_ok,
    )
    assert len(anchors_ok) > 0
```

Bind `steps`, `confidences`, `player_ctx`, `per_frame_K/R/t`, `pitch_cfg`, `cfg` exactly as the neighbouring grounded-sampling test does (copy its arrange block — do not import across tests). If `tests/test_ball_auto_anchor.py` has no grounded-sampling test to copy from, the fixture pattern lives in `tests/test_auto_anchor_generate.py` — copy from there instead and put this test next to its siblings in that file.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python -m pytest tests/test_ball_auto_anchor.py -q`
Expected: new test FAILS — `generate_auto_anchors() got an unexpected keyword argument 'sources'`.

- [ ] **Step 3: Implement**

In `src/utils/ball_auto_anchor.py`, add the keyword to `generate_auto_anchors` (after `cfg`):

```python
def generate_auto_anchors(
    *,
    events: Sequence[BallEvent],
    steps,
    confidences: Mapping[int, float],
    player_ctx,
    per_frame_K: Mapping[int, np.ndarray],
    per_frame_R: Mapping[int, np.ndarray],
    per_frame_t: Mapping[int, np.ndarray],
    distortion: tuple[float, float],
    fps: float,
    pitch_cfg: Mapping[str, float],
    cfg: AutoAnchorCfg | None = None,
    sources: Mapping[int, str] | None = None,
) -> tuple[BallAnchor, ...]:
```

and, immediately after `candidates.extend(_grounded_candidates(...))`:

```python
    if sources is not None:
        # Second-pass detections densify solver evidence but never mint
        # constraints (ball v2 design, Phase 1).
        candidates = [
            c for c in candidates
            if sources.get(c.anchor.frame) != "second_pass"
        ]
```

In `src/stages/ball.py`, find the `generate_auto_anchors(` call inside `_run_shot` (around line 684) and add `sources=sources,` to its keyword arguments.

- [ ] **Step 4: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_ball_auto_anchor.py tests/test_auto_anchor_generate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/ball_auto_anchor.py src/stages/ball.py tests/test_ball_auto_anchor.py
git commit -m "feat(ball): exclude second-pass observations from auto-anchor generation"
```

---

### Task 8: Second-pass loop + `_run_shot` integration + config

The video-touching orchestration: run-grouped full-frame second pass, zoom retry, merge, re-smooth, sidecar/diag updates, config block. Verified by an end-to-end stage test with a scripted detector.

**Files:**
- Modify: `src/stages/ball.py`
- Modify: `config/default.yaml`
- Test: `tests/test_ball_stage_second_pass.py` (append)

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_ball_stage_second_pass.py` (helpers `_camera_pose`, `_save_camera_track`, `_write_blank_clip`, `_project` — copy them verbatim from `tests/test_ball_stage.py` into this file's top section first; they are module-private there):

```python
import json
from collections import deque
from pathlib import Path

from src.schemas.ball_track import BallTrack
from src.utils.ball_detector import FakeBallDetector


class ScriptedDetector(FakeBallDetector):
    """Pass-1 detections by call order; second-pass candidate lists are
    served FIFO across detect_candidates calls (the second pass visits
    frames in a deterministic order: prime frames first, then the gap)."""

    def __init__(self, detections, second_pass_cands):
        super().__init__(detections)
        self._sp = deque(second_pass_cands)

    def detect_candidates(self, frame, min_score, top_k=5):
        if not self._sp:
            return []
        cands = self._sp.popleft()
        kept = [c for c in cands if c[2] >= min_score]
        kept.sort(key=lambda c: -c[2])
        return kept[:top_k]


@pytest.mark.integration
def test_second_pass_fills_gap_and_never_anchors(tmp_path: Path):
    from src.stages.ball import BallStage

    n = 60
    fps = 30.0
    K, R, t = _camera_pose()
    _save_camera_track(tmp_path / "camera" / "camera_track.json", K, R, t, n, fps=fps)
    _write_blank_clip(tmp_path / "shots" / "play.mp4", n, fps=fps)

    truth = {i: np.array([30.0 + 0.2 * i, 34.0, 0.11]) for i in range(n)}
    uv_truth = {i: _project(truth[i], K, R, t) for i in range(n)}

    # Pass 1: detections everywhere except frames 20-24.
    detections = []
    for i in range(n):
        if 20 <= i <= 24:
            detections.append(None)
        else:
            detections.append((uv_truth[i][0], uv_truth[i][1], 0.9))

    # Second pass visits frames 18..24 (prime offset 2, then the gap).
    # Prime frames return nothing; gap frames offer the true ball at a
    # weak score plus a strong decoy far outside the corridor.
    sp_cands = [[], []]
    for i in range(20, 25):
        sp_cands.append([
            (uv_truth[i][0] + 1.0, uv_truth[i][1] - 1.0, 0.55),
            (uv_truth[i][0] + 400.0, uv_truth[i][1] + 200.0, 0.95),  # decoy
        ])

    stage = BallStage(
        config={"ball": {
            "detector": "fake",
            # appearance bridge would gap-fill 20-24 itself; isolate the
            # second pass by disabling it.
            "appearance_bridge": {"enabled": False},
            "second_pass": {"enabled": True, "zoom_min_ball_px": 0.0},
            "auto_anchors": {"enabled": True, "grounded_interval": 8},
        }},
        output_dir=tmp_path,
        ball_detector=ScriptedDetector(detections, sp_cands),
    )
    stage.run()

    obs = json.loads((tmp_path / "ball" / "ball_observations.json").read_text())
    by_frame = {f["frame"]: f for f in obs["frames"]}
    for i in range(20, 25):
        assert by_frame[i]["source"] == "second_pass"
        assert abs(by_frame[i]["uv"][0] - uv_truth[i][0]) < 3.0  # decoy rejected
        assert by_frame[i]["confidence"] > 0.0

    diag = json.loads((tmp_path / "ball" / "ball_diag.json").read_text())
    cov = diag["detection_coverage"]
    assert cov["second_pass"] > 0.0
    assert cov["total"] == pytest.approx(cov["pass1"] + cov["second_pass"])
    assert cov["total"] > cov["pass1"]

    # Second-pass frames never become anchors.
    anchors_path = tmp_path / "ball" / "ball_anchors_auto.json"
    if anchors_path.exists():
        anchors = json.loads(anchors_path.read_text())
        frames = [a["frame"] for a in anchors.get("anchors", [])]
        assert not any(20 <= f <= 24 for f in frames)

    # Track is continuous through the gap (no missing state inside it).
    track = BallTrack.load(tmp_path / "ball" / "ball_track.json")
    states = {f.frame: f.state for f in track.frames}
    assert all(states[i] != "missing" for i in range(20, 25))


@pytest.mark.integration
def test_second_pass_disabled_is_noop(tmp_path: Path):
    from src.stages.ball import BallStage

    n = 30
    fps = 30.0
    K, R, t = _camera_pose()
    _save_camera_track(tmp_path / "camera" / "camera_track.json", K, R, t, n, fps=fps)
    _write_blank_clip(tmp_path / "shots" / "play.mp4", n, fps=fps)
    detections = []
    for i in range(n):
        p = np.array([30.0 + 0.2 * i, 34.0, 0.11])
        u, v = _project(p, K, R, t)
        detections.append((u, v, 0.9))

    stage = BallStage(
        config={"ball": {"detector": "fake", "second_pass": {"enabled": False}}},
        output_dir=tmp_path,
        ball_detector=FakeBallDetector(detections),
    )
    stage.run()
    diag = json.loads((tmp_path / "ball" / "ball_diag.json").read_text())
    assert diag["detection_coverage"]["second_pass"] == 0.0
```

Note the exact sidecar/diag filenames in the legacy single-shot path: confirm them by reading how `_run_shot` derives them from `ball_out_path` (`ball_track` → `ball_observations` / `ball_diag`); the legacy track file is `ball/ball_track.json`, so the sidecars are `ball/ball_observations.json` and `ball/ball_diag.json`. The auto-anchors filename follows the same `_load_ball_anchors` convention — check `_persist_auto_anchors`/equivalent in `src/stages/ball.py` and adjust `anchors_path` in the test if it differs.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python -m pytest tests/test_ball_stage_second_pass.py -q`
Expected: integration tests FAIL — no second pass exists, frames 20–24 are `missing`/`none`, no `detection_coverage` key.

- [ ] **Step 3: Implement config parsing and the loop**

In `src/stages/ball.py`:

a) Imports — extend the existing `ball_second_pass` import (or add it):

```python
from src.utils.ball_second_pass import (
    SecondPassCfg,
    SecondPassDetection,
    apparent_ball_px,
    best_gated_candidate,
    corridor_predictions,
    find_gap_runs,
    map_crop_candidates,
)
```

b) Config helper (next to `_auto_event_cfg`):

```python
def _second_pass_cfg(cfg: dict) -> SecondPassCfg:
    sp = cfg.get("second_pass", {})
    base = SecondPassCfg()
    return SecondPassCfg(
        enabled=bool(sp.get("enabled", base.enabled)),
        candidate_min_score=float(sp.get("candidate_min_score", base.candidate_min_score)),
        top_k=int(sp.get("top_k", base.top_k)),
        corridor_sigma=float(sp.get("corridor_sigma", base.corridor_sigma)),
        accept_min=float(sp.get("accept_min", base.accept_min)),
        zoom_min_ball_px=float(sp.get("zoom_min_ball_px", base.zoom_min_ball_px)),
        zoom_crop_px=int(sp.get("zoom_crop_px", base.zoom_crop_px)),
    )
```

c) Methods on `BallStage` (below `_detect_loop`):

```python
    def _second_pass_loop(
        self,
        clip_path: Path,
        gap_runs: list[tuple[int, int]],
        corridors: dict[int, tuple[np.ndarray, np.ndarray]],
        per_frame_K: dict[int, np.ndarray],
        per_frame_R: dict[int, np.ndarray],
        per_frame_t: dict[int, np.ndarray],
        distortion: tuple[float, float],
        detector: BallDetector,
        sp_cfg: SecondPassCfg,
        ball_radius: float,
    ) -> list[SecondPassDetection]:
        """Revisit evidence gaps with corridor-gated candidate detection.

        Full-frame pass first (run-grouped so the detector's temporal
        buffer is primed once per run), then a zoom retry on frames where
        nothing cleared the gate and the predicted ball is small.
        """
        accepted: list[SecondPassDetection] = []
        zoom_targets: list[int] = []
        prime_offset = getattr(detector, "_frames_in", 3) - 1
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open clip: {clip_path}")
        try:
            for start, end in gap_runs:
                prime = max(0, start - prime_offset)
                cap.set(cv2.CAP_PROP_POS_FRAMES, prime)
                detector.reset()
                for f in range(prime, end + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cands = detector.detect_candidates(
                        frame, sp_cfg.candidate_min_score, sp_cfg.top_k,
                    )
                    if f < start or f not in corridors:
                        continue
                    mean, cov = corridors[f]
                    best = best_gated_candidate(cands, mean, cov, sp_cfg)
                    if best is not None:
                        accepted.append(SecondPassDetection(
                            frame=f, uv=best[0],
                            combined_score=best[1], used_zoom=False,
                        ))
                        continue
                    K = per_frame_K.get(f)
                    if K is None:
                        continue
                    size = apparent_ball_px(
                        K, per_frame_R[f], per_frame_t[f],
                        (float(mean[0]), float(mean[1])),
                        ball_radius, distortion,
                    )
                    if size is not None and size < sp_cfg.zoom_min_ball_px:
                        zoom_targets.append(f)
            for f in zoom_targets:
                mean, cov = corridors[f]
                best = self._zoom_detect(cap, f, mean, cov, detector, sp_cfg)
                if best is not None:
                    accepted.append(SecondPassDetection(
                        frame=f, uv=best[0],
                        combined_score=best[1], used_zoom=True,
                    ))
        finally:
            detector.reset()
            cap.release()
        accepted.sort(key=lambda d: d.frame)
        return accepted

    def _zoom_detect(
        self,
        cap: "cv2.VideoCapture",
        frame_idx: int,
        mean: np.ndarray,
        cov: np.ndarray,
        detector: BallDetector,
        sp_cfg: SecondPassCfg,
    ) -> tuple[tuple[float, float], float] | None:
        """Crop around the corridor and re-detect; the detector's own
        letterbox upscales the crop, magnifying a small ball."""
        half = sp_cfg.zoom_crop_px // 2
        prime_offset = getattr(detector, "_frames_in", 3) - 1
        prime = max(0, frame_idx - prime_offset)
        cap.set(cv2.CAP_PROP_POS_FRAMES, prime)
        detector.reset()
        best: tuple[tuple[float, float], float] | None = None
        for f in range(prime, frame_idx + 1):
            ret, frame = cap.read()
            if not ret:
                return None
            h, w = frame.shape[:2]
            x0 = int(np.clip(mean[0] - half, 0, max(0, w - sp_cfg.zoom_crop_px)))
            y0 = int(np.clip(mean[1] - half, 0, max(0, h - sp_cfg.zoom_crop_px)))
            crop = frame[y0:y0 + sp_cfg.zoom_crop_px, x0:x0 + sp_cfg.zoom_crop_px]
            if crop.size == 0:
                return None
            cands = detector.detect_candidates(
                crop, sp_cfg.candidate_min_score, sp_cfg.top_k,
            )
            if f == frame_idx:
                best = best_gated_candidate(
                    map_crop_candidates(cands, x0, y0), mean, cov, sp_cfg,
                )
        detector.reset()
        return best
```

d) Integration in `_run_shot` — insert between the detect loop and the observations-sidecar write (i.e. after `n_frames = max(n_frames, steps[-1].frame + 1)` and before the `try: _write_observations_sidecar(...)` block), so the sidecar records the merged stream:

```python
        # --- 1b. Second pass over evidence gaps -------------------------
        sp_cfg = _second_pass_cfg(cfg)
        n_clip = steps[-1].frame + 1
        if sp_cfg.enabled:
            outliers = {s.frame for s in steps if s.is_outlier}
            # Pass-1 raw observations ONLY (feedback-loop guard): the
            # corridor that admits a second-pass detection is never built
            # from second-pass output.
            pass1_uv: dict[int, tuple[float, float] | None] = {
                s.frame: (
                    s.uv if (s.frame in sources and not s.is_outlier) else None
                )
                for s in steps
            }
            gap_runs = find_gap_runs(sources, outliers, n_clip)
            if gap_runs:
                corridors = corridor_predictions(
                    pass1_uv, n_clip,
                    tracker_factory=lambda: _build_tracker(
                        cfg, max_gap_frames=10 ** 6),
                )
                sp_dets = self._second_pass_loop(
                    clip_path, gap_runs, corridors, per_frame_K, per_frame_R,
                    per_frame_t, distortion, detector, sp_cfg, ball_radius,
                )
                if sp_dets:
                    logger.info(
                        "ball: second pass recovered %d/%d gap frames for %s",
                        len(sp_dets),
                        sum(e - s + 1 for s, e in gap_runs),
                        shot_id or "(legacy)",
                    )
                    merged_uv = dict(pass1_uv)
                    for d in sp_dets:
                        merged_uv[d.frame] = d.uv
                        raw_confidences[d.frame] = d.combined_score
                        sources[d.frame] = "second_pass"
                    steps = _resmooth_observations(merged_uv, n_clip, cfg)

        n_pass1 = sum(
            1 for s in steps
            if sources.get(s.frame) in ("detector", "anchor", "bridge")
        )
        n_pass2 = sum(1 for s in steps if sources.get(s.frame) == "second_pass")
        detection_coverage = {
            "pass1": n_pass1 / n_clip,
            "second_pass": n_pass2 / n_clip,
            "total": (n_pass1 + n_pass2) / n_clip,
        }
```

e) Diag — in the `diag_path.write_text(json.dumps({...}))` payload at the end of `_run_shot`, add one entry:

```python
            "detection_coverage": detection_coverage,
```

f) Config — in `config/default.yaml`, inside the `ball:` block immediately after the `ray_faithful_tolerance_px` entry, add:

```yaml
  # Second-pass detection: frames with no accepted pass-1 detection are
  # revisited with low-threshold candidate detection, gated by a corridor
  # predicted from forward/backward IMM smoothing of pass-1 observations
  # only (second-pass output never steers its own corridor, and runs
  # exactly once). Accepted frames carry source="second_pass" in the
  # observations sidecar; they densify solver evidence but never mint
  # auto-anchors. zoom: when nothing clears the gate and the predicted
  # apparent ball diameter is below zoom_min_ball_px, a crop around the
  # corridor is re-detected (the detector's letterbox upscales it).
  second_pass:
    enabled: true
    candidate_min_score: 0.05
    top_k: 5
    corridor_sigma: 3.0
    accept_min: 0.25
    zoom_min_ball_px: 8.0
    zoom_crop_px: 320
```

- [ ] **Step 4: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_ball_stage_second_pass.py tests/test_ball_stage.py tests/test_ball_stage_anchors.py tests/test_ball_stage_keyframes.py tests/test_ball_stage_layered.py -q`
Expected: PASS — including all pre-existing stage tests (second pass on a fully-detected clip finds no gap runs and is a no-op; on gap-y stage tests the `FakeBallDetector` default adapter returns its scripted detections, which the corridor either accepts or rejects — if any pre-existing stage test changes behavior, set `"second_pass": {"enabled": False}` in that test's config and note it in the commit message).

- [ ] **Step 5: Commit**

```bash
git add src/stages/ball.py config/default.yaml tests/test_ball_stage_second_pass.py
git commit -m "feat(ball): corridor-gated second detection pass with zoom retry and coverage diagnostics"
```

---

### Task 9: Quality-report passthrough

**Files:**
- Modify: `src/pipeline/quality_report.py` (`_ball_shot_entry`)
- Test: covered by extending the Task 8 integration test

- [ ] **Step 1: Write the failing test**

Append to the end of `test_second_pass_fills_gap_and_never_anchors` in `tests/test_ball_stage_second_pass.py`:

```python
    # Quality report surfaces coverage.
    from src.pipeline.quality_report import _ball_shot_entry
    entry = _ball_shot_entry(tmp_path / "ball" / "ball_track.json", "")
    assert entry["detection_coverage"]["total"] > entry["detection_coverage"]["pass1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python -m pytest tests/test_ball_stage_second_pass.py -q`
Expected: FAIL — `KeyError: 'detection_coverage'`.

- [ ] **Step 3: Implement**

In `src/pipeline/quality_report.py`, inside `_ball_shot_entry`'s `entry.update({...})` block (the one guarded by `if diag_path.exists()`), add:

```python
            "detection_coverage": diag.get("detection_coverage"),
```

- [ ] **Step 4: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_ball_stage_second_pass.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/quality_report.py tests/test_ball_stage_second_pass.py
git commit -m "feat(quality): surface ball detection coverage in the quality report"
```

---

### Task 10: Full-suite check + real-clip validation

- [ ] **Step 1: Run the full ball test suite**

Run: `.venv311/bin/python -m pytest tests/ -q -k "ball or anchor or tracker"`
Expected: PASS, including `tests/test_ball_anchor_accuracy.py` (the anchor-accuracy harness — an acceptance criterion).

- [ ] **Step 2: Real-clip validation (needs the WASB checkpoint + torch venv; GPU optional)**

Re-run the ball stage on the validation outputs (the original input file is whatever was used to produce each output dir — ask the operator if not on disk):

```bash
python recon.py run --input <origi-input>.mp4 --output ./output-origi/ --from-stage ball
python recon.py run --input <kroupi-input>.mp4 --output ./output-kroupi/ --from-stage ball
```

- [ ] **Step 3: Check acceptance criteria** (from the design spec, Phase 1)

```bash
python3 - <<'EOF'
import json
for clip, path in [
    ("origi01", "output-origi/ball/origi01_ball_diag.json"),
    ("origi02", "output-origi/ball/origi02_ball_diag.json"),
    ("kroupi01", "output-kroupi/ball/kroupi01_ball_diag.json"),
]:
    d = json.load(open(path))
    cov = d.get("detection_coverage", {})
    spans = d.get("underconstrained_spans", [])
    residuals = [s.get("residual_px") for s in d.get("segments", []) if s.get("residual_px")]
    print(f"{clip}: coverage={cov}  underconstrained={len(spans)}  max_residual={max(residuals or [0]):.1f}")
EOF
```

Acceptance:
- origi02 `detection_coverage.total >= 0.75` (was 0.44).
- kroupi01 / origi01: no segment-residual regressions vs the values in the design spec's §1 (kroupi01 max accepted 5.1 px), no new jumps > 2 m.
- Anchor-accuracy harness green (step 1).

- [ ] **Step 4: Record results**

Append a `## Phase 1 validation results` section with the metrics table to `docs/superpowers/specs/2026-06-12-ball-v2-design.md` and commit:

```bash
git add docs/superpowers/specs/2026-06-12-ball-v2-design.md
git commit -m "docs: ball evidence booster validation results"
```

If origi02 misses the 0.75 bar, tune `second_pass.accept_min` (down) and `corridor_sigma` (up) and re-run — record the final values in the results section. If false positives corrupt the track (residual regressions), raise `accept_min`. Both knobs are config; no code changes expected.
