# Landmark-free Camera Tracking Implementation Plan (reuse-first revision)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate manual landmark placement in the camera stage by reviving the proven (but shelved) PnLCalib integration as an "auto-anchor generator" whose keypoint detections become standard `Anchor` objects that feed the existing sub-pixel line solver unchanged.

**Architecture:** Recover the proven PnLCalib code from git history (`PnLCalibrator` + keypoint→world bridge + plausibility/MAD-consensus helpers, removed in the May-4 reset), adapt it, and add thin new glue that turns per-keyframe keypoint detections into an `AnchorSet`. The existing joint solve → static-camera C-profile → sub-pixel line solver then runs verbatim. Opt-in via `camera.auto_anchors.enabled`; falls back to manual anchoring when unavailable.

**Tech Stack:** Python 3.11+, PyTorch (auto device: CUDA on the Linux box, CPU/MPS on Mac), OpenCV, NumPy, SciPy, pytest. PnLCalib vendored at `third_party/PnLCalib` (GPL-2.0, in-process import via the proven sys.path shim).

**Design doc:** `docs/superpowers/specs/2026-06-05-landmark-free-camera-design.md`

---

## Why this revises the original plan

Investigation during execution found PnLCalib was **already integrated** (commit `200762d`, Apr 2026: *"replace landmark PnP with PnLCalib neural solver"*, 0.26–0.45 m accuracy vs manual anchors) and removed in the May-4 architectural reset (`262d08a`) — not because it failed. The user confirmed: revive and reuse the proven code. The original plan's from-scratch subprocess wrapper + hand-transcribed bridge are replaced by recovery of:

- `src/utils/neural_calibrator.py` (commit `98e9cbd`) — `PnLCalibrator` with `extract_keypoints_pixels()`, `calibrate()`, the in-process shim `_import_pnlcalib_modules()`, `convert_pnlcalib_to_ours()`, and weight auto-download to `data/models/pnlcalib/`.
- `src/utils/fixed_position_solver.py` (commit `98e9cbd`) — `_load_pnlcalib_keypoint_table()` + `_world_coords_for()` (imports the vendored keypoint table; goalpost tops {12,15,16,19} at z=−2.44).
- `src/stages/calibration.py` (commit `98e9cbd`) — helpers `_is_plausible`, `_median_absolute_deviation`, `_robust_median_position`, `_compute_keyframes`.

**Confirmed bridge (authoritative):** PnLCalib's keypoint table is corner-origin x∈[0,105], y∈[0,68], but its y runs OPPOSITE to this project (PnLCalib y=0 = image-top = our far touchline y=68), and goalpost tops are z=−2.44 (z-down). A table point converts to THIS project's frame as:

```
x_ours = x_table
y_ours = 68 - y_table
z_ours = -z_table
```

(The original plan's identity bridge mirrored the whole pitch — this is the bug the recovered code avoids.)

**Recovery command pattern** (used in several tasks): `git show 98e9cbd:<path> > <path>`.

---

## File Structure

**Recovered (from git history, then adapted):**
- `src/utils/neural_calibrator.py`
- `src/utils/pnlcalib_pitch_map.py` — NEW home for the recovered `_load_pnlcalib_keypoint_table` + `_world_coords_for` + a new `keypoint_world_xyz_ours()` (so we don't recover the whole `fixed_position_solver.py`, most of which the hybrid doesn't need).

**New:**
- `src/utils/auto_anchor.py` — keyframe sampling, keypoints→Anchor, plausibility/MAD consensus, `generate()` → `AnchorSet`.
- `scripts/pnlcalib_smoke.py` — one-frame manual validation CLI.
- `scripts/eval_auto_anchor.py` + `tests/test_auto_anchor_e2e.py` — opt-in parity check.
- Tests under `tests/` mirroring each module.

**Modified:**
- `.gitignore` — `data/models/pnlcalib/` (where weights auto-download). Reconcile/replace the Task-1 lowercase `third_party/pnlcalib/weights/` entry.
- `scripts/fetch_pnlcalib_weights.sh` — repoint `DEST` to `data/models/pnlcalib` (or delete; auto-download covers it).
- `config/default.yaml` — `camera.auto_anchors` block.
- `src/stages/camera.py` — `_ensure_anchors` pre-step hook.

**Already done (Task 1, commit `1d78a1e`):** submodule vendored at `third_party/PnLCalib`; weights fetched locally (gitignored).

---

## Phase 0 — Reconcile weights path

### Task A: Fix weights location to match the recovered code

The recovered `neural_calibrator.py` auto-downloads weights to `data/models/pnlcalib/`. Task 1 committed a fetch script + gitignore pointing at `third_party/pnlcalib/weights/` (wrong case, wrong location).

**Files:**
- Modify: `.gitignore`, `scripts/fetch_pnlcalib_weights.sh`, `tests/test_pnlcalib_vendor.py`

- [ ] **Step 1: Update `.gitignore`** — replace the line `third_party/pnlcalib/weights/` with:

```
data/models/
```

- [ ] **Step 2: Repoint the fetch script** — in `scripts/fetch_pnlcalib_weights.sh`, change `DEST="third_party/pnlcalib/weights"` to `DEST="data/models/pnlcalib"`.

- [ ] **Step 3: Update the vendor test** — in `tests/test_pnlcalib_vendor.py::test_pnlcalib_submodule_present`, change the base path from `third_party / "pnlcalib"` to `third_party / "PnLCalib"` (the actual, case-sensitive submodule directory).

- [ ] **Step 4: Run the test**

Run: `pytest tests/test_pnlcalib_vendor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .gitignore scripts/fetch_pnlcalib_weights.sh tests/test_pnlcalib_vendor.py
git commit -m "chore: point PnLCalib weights at data/models/pnlcalib (match recovered code)"
```

---

## Phase 1 — Recover the proven PnLCalib core

### Task B: Recover `neural_calibrator.py`

**Files:**
- Create (via recovery): `src/utils/neural_calibrator.py`
- Test: `tests/test_neural_calibrator.py`

- [ ] **Step 1: Recover the file verbatim**

```bash
git show 98e9cbd:src/utils/neural_calibrator.py > src/utils/neural_calibrator.py
```

- [ ] **Step 2: Sanity-check imports against the current tree**

Run: `python -c "import ast; ast.parse(open('src/utils/neural_calibrator.py').read()); print('parse ok')"`
Then confirm it has no imports of removed modules:
Run: `grep -nE "^from src\.|^import src\." src/utils/neural_calibrator.py`
Expected: no matches (the module only uses stdlib + cv2/numpy, with torch/yaml/PIL imported lazily). If any `from src.X` import appears that no longer exists, report DONE_WITH_CONCERNS with the specifics — do not invent replacements.

- [ ] **Step 3: Write the failing test (pure-math conversion, no torch)**

Create `tests/test_neural_calibrator.py`:

```python
import numpy as np
import pytest

from src.utils.neural_calibrator import convert_pnlcalib_to_ours


def test_convert_identity_rotation_centres_to_corner_origin():
    """A camera at PnLCalib origin (pitch centre) maps to our pitch centre
    (52.5, 34, 0) with z flipped."""
    R = np.eye(3)
    C_pnl = np.array([0.0, 0.0, 0.0])
    rvec, tvec, C_ours = convert_pnlcalib_to_ours(R, C_pnl)
    assert C_ours == pytest.approx([52.5, 34.0, 0.0])


def test_convert_flips_y_and_z():
    R = np.eye(3)
    C_pnl = np.array([10.0, 5.0, -15.0])   # z-down 15 m up
    _, _, C_ours = convert_pnlcalib_to_ours(R, C_pnl)
    # x+=52.5, y=34-y, z=-z
    assert C_ours == pytest.approx([62.5, 29.0, 15.0])


def test_convert_preserves_projection_consistency():
    """R_ours = R_pnl @ diag(1,-1,-1) and t_ours = -R_ours @ C_ours."""
    rng = np.random.default_rng(0)
    R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    C_pnl = np.array([3.0, -2.0, -20.0])
    rvec, tvec, C_ours = convert_pnlcalib_to_ours(R, C_pnl)
    import cv2
    R_ours, _ = cv2.Rodrigues(rvec)
    assert tvec == pytest.approx((-R_ours @ C_ours))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_neural_calibrator.py -v`
Expected: PASS. (If it fails, the failure is in recovered math — read `convert_pnlcalib_to_ours` and reconcile the test to the proven implementation, not vice-versa, unless the math is provably wrong.)

- [ ] **Step 5: Commit**

```bash
git add src/utils/neural_calibrator.py tests/test_neural_calibrator.py
git commit -m "feat: recover PnLCalib neural calibrator (extract_keypoints_pixels, calibrate)"
```

---

### Task C: Keypoint→world bridge in our frame

**Files:**
- Create: `src/utils/pnlcalib_pitch_map.py`
- Test: `tests/test_pnlcalib_pitch_map.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pnlcalib_pitch_map.py`:

```python
import pytest

from src.utils.pnlcalib_pitch_map import (
    keypoint_world_xyz_ours,
    NUM_KEYPOINTS,
)


def test_total_keypoint_count():
    assert NUM_KEYPOINTS == 73  # 57 main + 16 aux


def test_far_left_corner_keypoint_is_y_flipped():
    """PnLCalib kp1 = table (0,0) = far-left corner; in our frame y=68."""
    assert keypoint_world_xyz_ours(1) == pytest.approx((0.0, 68.0, 0.0))


def test_ground_keypoint_y_flips():
    """Table kp4 = (0, 13.84) -> ours (0, 68-13.84, 0) = (0, 54.16, 0)."""
    assert keypoint_world_xyz_ours(4) == pytest.approx((0.0, 54.16, 0.0))


def test_goalpost_top_is_z_up_244():
    """Table kp16 = (0, 37.66, -2.44) -> ours (0, 30.34, +2.44)."""
    assert keypoint_world_xyz_ours(16) == pytest.approx((0.0, 30.34, 2.44))


def test_unknown_keypoint_returns_none():
    assert keypoint_world_xyz_ours(999) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pnlcalib_pitch_map.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Recover the table loader and wrap it**

Recover the proven loader as a private helper, then add the our-frame conversion. Create `src/utils/pnlcalib_pitch_map.py`:

```python
"""Map PnLCalib keypoint IDs to world coordinates in THIS project's frame.

Recovers the proven keypoint-table loader from the (shelved) calibration
work: it imports PnLCalib's own ``keypoint_world_coords_2D`` /
``keypoint_aux_world_coords_2D`` tables via a temporary sys.path swap, so the
table is never hand-transcribed. PnLCalib's table is corner-origin
x in [0,105], y in [0,68] with y pointing toward the image top and goalpost
tops at z=-2.44. This project is near-left-corner, y toward the far touchline,
z-up, so a table point converts as: x_ours = x, y_ours = 68 - y, z_ours = -z.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PNLCALIB_ROOT = _REPO_ROOT / "third_party" / "PnLCalib"
_PITCH_WIDTH = 68.0

# PnLCalib assigns z = -2.44 to the four goalpost-top keypoints (crossbars).
_GOALPOST_TOP_KEYS = {12, 15, 16, 19}
_GOALPOST_Z = -2.44


def _load_pnlcalib_keypoint_table() -> dict[int, tuple[float, float, float]]:
    """Import PnLCalib's keypoint world tables; return {kp_id: (x, y, z)} in
    PnLCalib's own (table) frame. kp_id is 1-based: 1..57 main, 58..73 aux."""
    if not _PNLCALIB_ROOT.exists():
        raise RuntimeError(
            f"PnLCalib submodule missing at {_PNLCALIB_ROOT}. "
            "Run `git submodule update --init --recursive`."
        )
    src_path = str(_REPO_ROOT / "src")
    removed = [p for p in sys.path if p == src_path or p.rstrip("/") == src_path]
    for p in removed:
        sys.path.remove(p)
    sys.path.insert(0, str(_PNLCALIB_ROOT))
    try:
        from utils.utils_calib import (  # type: ignore
            keypoint_aux_world_coords_2D,
            keypoint_world_coords_2D,
        )
    finally:
        try:
            sys.path.remove(str(_PNLCALIB_ROOT))
        except ValueError:
            pass
        for p in removed:
            sys.path.append(p)

    table: dict[int, tuple[float, float, float]] = {}
    for idx, (xw, yw) in enumerate(keypoint_world_coords_2D):
        kp_id = idx + 1
        zw = _GOALPOST_Z if kp_id in _GOALPOST_TOP_KEYS else 0.0
        table[kp_id] = (float(xw), float(yw), float(zw))
    for idx, (xw, yw) in enumerate(keypoint_aux_world_coords_2D):
        table[idx + 1 + 57] = (float(xw), float(yw), 0.0)
    return table


_TABLE = _load_pnlcalib_keypoint_table()
NUM_KEYPOINTS = len(_TABLE)


def keypoint_world_xyz_ours(kp_id: int) -> tuple[float, float, float] | None:
    """World (x, y, z) in THIS project's pitch frame for 1-based PnLCalib
    keypoint ``kp_id``, or None if unknown."""
    entry = _TABLE.get(kp_id)
    if entry is None:
        return None
    x, y, z = entry
    return (x, _PITCH_WIDTH - y, -z)
```

> **Note:** the import name is `from utils.utils_calib import keypoint_world_coords_2D, keypoint_aux_world_coords_2D`. If those symbols are defined in `utils.utils_keypoints` instead in the pinned submodule commit, adjust the import line (verify with `grep -rn "keypoint_world_coords_2D" third_party/PnLCalib/utils/`). The proven recovered code imported them from `utils.utils_calib`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pnlcalib_pitch_map.py -v`
Expected: PASS (requires the submodule present; it is).

- [ ] **Step 5: Commit**

```bash
git add src/utils/pnlcalib_pitch_map.py tests/test_pnlcalib_pitch_map.py
git commit -m "feat: PnLCalib keypoint->world bridge (our frame, y/z flip)"
```

---

## Phase 2 — Auto-anchor generation

### Task D: Keypoints → Anchor

**Files:**
- Create: `src/utils/auto_anchor.py`
- Test: `tests/test_auto_anchor_to_anchor.py`

- [ ] **Step 1: Read the live schema** — read `src/schemas/anchor.py` for the exact `Anchor` and `LandmarkObservation` constructors (field names). Use them precisely below.

- [ ] **Step 2: Write the failing test**

Create `tests/test_auto_anchor_to_anchor.py`:

```python
from src.schemas.anchor import Anchor
from src.utils.auto_anchor import keypoints_to_anchor


def test_keypoints_to_anchor_maps_world_coords():
    pixels = {
        1: (100.0, 900.0),     # far-left corner -> (0, 68, 0)
        4: (200.0, 880.0),     # -> (0, 54.16, 0)
        16: (950.0, 220.0),    # goalpost top -> (0, 30.34, 2.44) (non-coplanar)
        5: (500.0, 700.0),
        7: (1700.0, 880.0),
        9: (600.0, 690.0),
    }
    anchor = keypoints_to_anchor(pixels, frame=0, min_points=4)
    assert isinstance(anchor, Anchor)
    assert anchor.frame == 0
    assert len(anchor.landmarks) == 6
    lm16 = next(l for l in anchor.landmarks if l.image_xy == (950.0, 220.0))
    assert lm16.world_xyz[2] == 2.44   # non-coplanar point present


def test_keypoints_to_anchor_returns_none_below_min_points():
    anchor = keypoints_to_anchor({1: (1.0, 2.0)}, frame=0, min_points=4)
    assert anchor is None


def test_keypoints_to_anchor_skips_unknown_ids():
    anchor = keypoints_to_anchor(
        {1: (1.0, 2.0), 999: (3.0, 4.0), 4: (5.0, 6.0),
         5: (7.0, 8.0), 7: (9.0, 10.0)},
        frame=3, min_points=4,
    )
    assert len(anchor.landmarks) == 4  # 999 dropped
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_auto_anchor_to_anchor.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 4: Implement**

Create `src/utils/auto_anchor.py` (adapt `LandmarkObservation`/`Anchor` field names to the live schema from Step 1):

```python
"""Generate camera-stage anchors automatically from PnLCalib keypoints.

The learned model replaces the human clicker: per keyframe, PnLCalib's
keypoint detections become point LandmarkObservations (world coords in our
frame), an Anchor per keyframe, and the existing camera solver does the rest.
"""

from __future__ import annotations

from src.schemas.anchor import Anchor, LandmarkObservation
from src.utils.pnlcalib_pitch_map import keypoint_world_xyz_ours


def keypoints_to_anchor(
    pixels: dict[int, tuple[float, float]],
    frame: int,
    *,
    min_points: int,
) -> Anchor | None:
    """Build a point-only Anchor from PnLCalib keypoint pixels.

    Names are synthesised (``pnl_kp_<id>``); the solver consumes world_xyz,
    not the name. Returns None if fewer than ``min_points`` known keypoints.
    """
    landmarks = []
    for kp_id, image_xy in pixels.items():
        world = keypoint_world_xyz_ours(kp_id)
        if world is None:
            continue
        landmarks.append(
            LandmarkObservation(
                name=f"pnl_kp_{kp_id}",
                image_xy=(float(image_xy[0]), float(image_xy[1])),
                world_xyz=world,
            )
        )
    if len(landmarks) < min_points:
        return None
    return Anchor(frame=frame, landmarks=tuple(landmarks), lines=())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_auto_anchor_to_anchor.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/utils/auto_anchor.py tests/test_auto_anchor_to_anchor.py
git commit -m "feat: convert PnLCalib keypoints to a camera Anchor"
```

---

### Task E: Keyframe sampling + plausibility/MAD consensus (recovered)

**Files:**
- Modify: `src/utils/auto_anchor.py`
- Test: `tests/test_auto_anchor_consensus.py`

These helpers are recovered from `src/stages/calibration.py` (commit `98e9cbd`): `_compute_keyframes`, `_median_absolute_deviation`, `_robust_median_position`, and a plausibility gate adapted from `_is_plausible` to operate on the recovered `NeuralCalibration`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_auto_anchor_consensus.py`:

```python
import numpy as np

from src.utils.auto_anchor import (
    compute_keyframes,
    robust_median_position,
    is_plausible_position,
)


def test_compute_keyframes_caps_count():
    got = compute_keyframes(total_frames=1000, keyframe_interval=10, max_keyframes=5)
    assert len(got) <= 5
    assert got[0] == 0


def test_compute_keyframes_uses_interval_for_short_shots():
    assert compute_keyframes(total_frames=100, keyframe_interval=30, max_keyframes=10) == [0, 30, 60, 90]


def test_robust_median_drops_outliers():
    positions = [
        np.array([52.0, -30.0, 15.0]),
        np.array([52.5, -30.2, 15.1]),
        np.array([52.3, -29.8, 14.9]),
        np.array([200.0, 500.0, 90.0]),   # gross outlier
    ]
    med = robust_median_position(positions)
    assert med == None or abs(med[0] - 52.3) < 1.0  # outlier excluded


def test_is_plausible_rejects_underground_camera():
    bounds = {"x": (-30, 135), "y": (-60, 130), "z": (3, 80)}
    assert is_plausible_position(np.array([52.5, -30.0, 15.0]), bounds)
    assert not is_plausible_position(np.array([52.5, -30.0, -5.0]), bounds)
    assert not is_plausible_position(np.array([300.0, -30.0, 15.0]), bounds)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_auto_anchor_consensus.py -v`
Expected: FAIL (functions undefined).

- [ ] **Step 3: Add the recovered helpers**

Append to `src/utils/auto_anchor.py` (add `import numpy as np` at top):

```python
def compute_keyframes(
    total_frames: int, keyframe_interval: int, max_keyframes: int
) -> list[int]:
    """Sample keyframes from [0, total_frames). Interval is at least
    ``keyframe_interval`` but never yields more than ``max_keyframes``.
    Recovered from the shelved calibration stage."""
    if total_frames <= 0:
        return []
    effective = max(keyframe_interval, 1)
    if max_keyframes > 0:
        interval_from_cap = -(-total_frames // max_keyframes)  # ceil div
        effective = max(effective, interval_from_cap)
    return list(range(0, total_frames, effective))


def _median_absolute_deviation(arr: np.ndarray) -> np.ndarray:
    med = np.median(arr, axis=0)
    return np.median(np.abs(arr - med), axis=0)


def robust_median_position(positions: list[np.ndarray]) -> np.ndarray | None:
    """Median camera world position across keyframes, dropping samples >3 MAD
    from the median on any axis. Recovered from the shelved calibration stage."""
    if not positions:
        return None
    arr = np.asarray(positions, dtype=np.float64)
    if len(arr) <= 2:
        return np.median(arr, axis=0)
    med = np.median(arr, axis=0)
    mad = _median_absolute_deviation(arr)
    mad_clipped = np.where(mad > 1e-6, mad, 1e-6)
    deviation = np.abs(arr - med) / mad_clipped
    mask = np.all(deviation < 3.0, axis=1)
    if not np.any(mask):
        return med
    return np.median(arr[mask], axis=0)


def is_plausible_position(
    position: np.ndarray, bounds: dict[str, tuple[float, float]]
) -> bool:
    """Bounds check on a camera world position (our frame). Adapted from the
    recovered _is_plausible (position component only; the full solver's
    optical-axis check is applied where a NeuralCalibration is available)."""
    x_lo, x_hi = bounds["x"]
    y_lo, y_hi = bounds["y"]
    z_lo, z_hi = bounds["z"]
    return (
        x_lo <= position[0] <= x_hi
        and y_lo <= position[1] <= y_hi
        and z_lo <= position[2] <= z_hi
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_auto_anchor_consensus.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/auto_anchor.py tests/test_auto_anchor_consensus.py
git commit -m "feat: recover keyframe sampling + MAD consensus helpers"
```

---

### Task F: `generate()` orchestration → AnchorSet

**Files:**
- Modify: `src/utils/auto_anchor.py`
- Test: `tests/test_auto_anchor_generate.py`

The orchestration runs PnLCalib on each keyframe. Per keyframe it gets BOTH a full `calibrate()` (→ `world_position`, for the plausibility + MAD gate) AND `extract_keypoints_pixels()` (→ anchor correspondences). Keyframes that are implausible or MAD-outliers in camera position are dropped (this is the static-camera consensus that also kills left/right flips). Survivors become anchors.

- [ ] **Step 1: Read the live `AnchorSet`** — read `src/schemas/anchor.py` for the `AnchorSet` constructor (fields `clip_id`, `image_size`, `anchors`).

- [ ] **Step 2: Write the failing test (fake provider, no torch)**

Create `tests/test_auto_anchor_generate.py`:

```python
import numpy as np

from src.schemas.anchor import AnchorSet
from src.utils.neural_calibrator import NeuralCalibration
from src.utils.auto_anchor import generate


class _FakeCalibrator:
    """Stands in for PnLCalibrator. Returns a plausible calibration + a rich
    keypoint set for the listed good frames; None/empty otherwise."""

    def __init__(self, good_frames, flipped_frames=()):
        self.good = set(good_frames)
        self.flipped = set(flipped_frames)
        self._frame_cursor = []

    def calibrate(self, frame_bgr):
        f = int(frame_bgr[0, 0, 0])  # frame index smuggled in pixel 0
        if f in self.flipped:
            pos = np.array([52.5, 200.0, 15.0])  # far-side flip -> MAD outlier
        elif f in self.good:
            pos = np.array([52.5, -30.0, 15.0])
        else:
            return None
        K = np.array([[3000.0, 0, 960], [0, 3000.0, 540], [0, 0, 1]])
        return NeuralCalibration(K=K, rvec=np.zeros(3), tvec=np.zeros(3),
                                 world_position=pos)

    def extract_keypoints_pixels(self, frame_bgr):
        f = int(frame_bgr[0, 0, 0])
        if f not in (self.good | self.flipped):
            return {}
        return {i: (float(100 + i), float(900 - i)) for i in (1, 4, 5, 7, 9, 16)}


def _frames_reader(indices, image_size=(1920, 1080)):
    """Yield (idx, fake_bgr) where pixel 0 encodes the frame index."""
    out = {}
    for idx in indices:
        fr = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        fr[0, 0, 0] = idx
        out[idx] = fr
    return out


def _cfg(**over):
    base = dict(
        keyframe_interval=30, max_keyframes=12, min_points_per_anchor=4,
        plausibility_bounds={"x": (-30, 135), "y": (-60, 130), "z": (3, 80)},
        consensus_max_position_mad=3.0,
    )
    base.update(over)
    return base


def test_generate_builds_anchorset_and_drops_flips():
    cal = _FakeCalibrator(good_frames=[0, 30, 60], flipped_frames=[90])
    anchor_set = generate(
        calibrator=cal, clip_id="gberch", n_frames=120, image_size=(1920, 1080),
        cfg=_cfg(), frames_reader=_frames_reader,
    )
    assert isinstance(anchor_set, AnchorSet)
    assert {a.frame for a in anchor_set.anchors} == {0, 30, 60}  # 90 flip dropped


def test_generate_returns_none_when_nothing_plausible():
    cal = _FakeCalibrator(good_frames=[])
    assert generate(
        calibrator=cal, clip_id="x", n_frames=120, image_size=(1920, 1080),
        cfg=_cfg(), frames_reader=_frames_reader,
    ) is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_auto_anchor_generate.py -v`
Expected: FAIL (`generate` undefined).

- [ ] **Step 4: Implement `generate`**

Append to `src/utils/auto_anchor.py` (add `from src.schemas.anchor import AnchorSet` up top):

```python
def generate(
    *,
    calibrator,
    clip_id: str,
    n_frames: int,
    image_size: tuple[int, int],
    cfg: dict,
    frames_reader,
) -> AnchorSet | None:
    """Auto-anchor pipeline: sample keyframes -> PnLCalib per keyframe ->
    plausibility + MAD-consensus gate on camera position -> keypoint anchors.

    ``calibrator`` is a PnLCalibrator (or a stand-in with ``calibrate`` and
    ``extract_keypoints_pixels``). ``frames_reader(indices, image_size)``
    returns ``{idx: bgr_frame}`` (injected so tests need no video/torch).
    Returns an AnchorSet, or None if no keyframe survives.
    """
    keyframes = compute_keyframes(
        total_frames=n_frames,
        keyframe_interval=int(cfg.get("keyframe_interval", 30)),
        max_keyframes=int(cfg.get("max_keyframes", 12)),
    )
    if not keyframes:
        return None
    bounds = cfg.get("plausibility_bounds", {
        "x": (-30.0, 135.0), "y": (-60.0, 130.0), "z": (3.0, 80.0),
    })
    frames = frames_reader(keyframes, image_size)

    # Pass 1: calibrate each keyframe, keep plausible camera positions.
    plausible: dict[int, np.ndarray] = {}
    for idx in keyframes:
        frame = frames.get(idx)
        if frame is None:
            continue
        calib = calibrator.calibrate(frame)
        if calib is None:
            continue
        if is_plausible_position(np.asarray(calib.world_position), bounds):
            plausible[idx] = np.asarray(calib.world_position, dtype=np.float64)
    if not plausible:
        return None

    # Pass 2: MAD consensus on position -> keep keyframes near the robust median.
    median = robust_median_position(list(plausible.values()))
    max_mad = float(cfg.get("consensus_max_position_mad", 3.0))
    arr = np.asarray(list(plausible.values()))
    mad = _median_absolute_deviation(arr)
    mad_clipped = np.where(mad > 1e-6, mad, 1e-6)
    kept = [
        idx for idx, pos in plausible.items()
        if np.all(np.abs(pos - median) / mad_clipped < max_mad)
    ]
    if not kept:
        return None

    # Pass 3: keypoint anchors for the survivors.
    min_points = int(cfg.get("min_points_per_anchor", 4))
    anchors = []
    for idx in sorted(kept):
        pixels = calibrator.extract_keypoints_pixels(frames[idx])
        anchor = keypoints_to_anchor(pixels, frame=idx, min_points=min_points)
        if anchor is not None:
            anchors.append(anchor)
    if not anchors:
        return None
    return AnchorSet(clip_id=clip_id, image_size=image_size, anchors=tuple(anchors))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_auto_anchor_generate.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/utils/auto_anchor.py tests/test_auto_anchor_generate.py
git commit -m "feat: auto_anchor.generate orchestration with consensus gate"
```

---

## Phase 3 — Camera-stage integration

### Task G: Config block

**Files:**
- Modify: `config/default.yaml`
- Test: `tests/test_auto_anchor_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_auto_anchor_config.py`:

```python
import yaml
from pathlib import Path


def test_default_config_has_auto_anchors_block():
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "default.yaml").read_text()
    )
    aa = cfg["camera"]["auto_anchors"]
    assert aa["enabled"] is False
    assert aa["mode"] == "replace_when_empty"
    assert aa["keyframe_interval"] == 30
    assert "min_points_per_anchor" in aa
    assert aa["model"]["device"] == "auto"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_auto_anchor_config.py -v`
Expected: FAIL (`KeyError`).

- [ ] **Step 3: Add the config block**

In `config/default.yaml`, inside `camera:` (after `pitch_line_consistency_max_px`):

```yaml
  # Landmark-free cold-start via the vendored PnLCalib model. When enabled,
  # the camera stage auto-generates anchors and writes {shot}_anchors.json
  # before the normal solve. mode:
  #   replace_when_empty - only auto-generate when no manual anchors exist
  #   augment            - union auto anchors with existing manual ones
  #   force              - always regenerate (overwrites manual)
  auto_anchors:
    enabled: false
    mode: replace_when_empty
    keyframe_interval: 30
    max_keyframes: 12
    min_points_per_anchor: 4
    consensus_max_position_mad: 3.0
    plausibility_bounds:
      x: [-30.0, 135.0]
      y: [-60.0, 130.0]
      z: [3.0, 80.0]
    model:
      device: auto          # auto -> cuda on the Linux box, cpu/mps on Mac
      kp_threshold: 0.3434
      line_threshold: 0.7867
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_auto_anchor_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config/default.yaml tests/test_auto_anchor_config.py
git commit -m "feat: camera.auto_anchors config block (opt-in, off by default)"
```

---

### Task H: Camera-stage pre-step hook

**Files:**
- Modify: `src/stages/camera.py`
- Test: `tests/test_camera_stage_auto_anchor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_camera_stage_auto_anchor.py`:

```python
from src.schemas.anchor import Anchor, AnchorSet, LandmarkObservation
from src.stages.camera import CameraStage


def _anchor_set():
    lms = tuple(
        LandmarkObservation(name=f"pnl_kp_{i}", image_xy=(float(i), float(i)),
                            world_xyz=(float(i), float(i), 0.0))
        for i in range(1, 7)
    )
    return AnchorSet(clip_id="s1", image_size=(1920, 1080),
                     anchors=(Anchor(frame=0, landmarks=lms, lines=()),))


def test_ensure_anchors_writes_generated_set(tmp_path, monkeypatch):
    stage = CameraStage.__new__(CameraStage)
    stage.output_dir = tmp_path
    anchors_path = tmp_path / "camera" / "s1_anchors.json"
    cfg = {"auto_anchors": {"enabled": True, "mode": "replace_when_empty"}}
    monkeypatch.setattr(
        "src.stages.camera._generate_auto_anchors",
        lambda shot_id, clip_path, cfg: _anchor_set(),
    )
    stage._ensure_anchors("s1", anchors_path, tmp_path / "s1.mp4", cfg)
    assert anchors_path.exists()
    assert len(AnchorSet.load(anchors_path).anchors) == 1


def test_ensure_anchors_skips_when_manual_exists(tmp_path):
    stage = CameraStage.__new__(CameraStage)
    stage.output_dir = tmp_path
    anchors_path = tmp_path / "camera" / "s1_anchors.json"
    anchors_path.parent.mkdir(parents=True)
    _anchor_set().save(anchors_path)
    before = anchors_path.read_text()
    cfg = {"auto_anchors": {"enabled": True, "mode": "replace_when_empty"}}
    stage._ensure_anchors("s1", anchors_path, tmp_path / "s1.mp4", cfg)
    assert anchors_path.read_text() == before


def test_ensure_anchors_noop_when_disabled(tmp_path):
    stage = CameraStage.__new__(CameraStage)
    stage.output_dir = tmp_path
    anchors_path = tmp_path / "camera" / "s1_anchors.json"
    cfg = {"auto_anchors": {"enabled": False}}
    stage._ensure_anchors("s1", anchors_path, tmp_path / "s1.mp4", cfg)
    assert not anchors_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_camera_stage_auto_anchor.py -v`
Expected: FAIL (`_ensure_anchors` undefined).

- [ ] **Step 3: Implement the hook**

In `src/stages/camera.py`, add a module-level helper (local imports keep torch out of the default path):

```python
def _generate_auto_anchors(shot_id, clip_path, cfg):
    """Run the PnLCalib auto-anchor pipeline for one shot. Returns an
    AnchorSet or None. Heavy imports are local so the camera stage has no
    hard torch dependency unless auto-anchors are actually used."""
    import cv2

    from src.utils.auto_anchor import generate
    from src.utils.neural_calibrator import PnLCalibrator

    aa = cfg.get("auto_anchors", {})
    model_cfg = aa.get("model", {})
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    calibrator = PnLCalibrator(
        device=model_cfg.get("device", "auto"),
        kp_threshold=float(model_cfg.get("kp_threshold", 0.3434)),
        line_threshold=float(model_cfg.get("line_threshold", 0.7867)),
    )

    def _frames_reader(indices, image_size):
        cap = cv2.VideoCapture(str(clip_path))
        out = {}
        try:
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok:
                    out[idx] = frame
        finally:
            cap.release()
        return out

    return generate(
        calibrator=calibrator, clip_id=shot_id, n_frames=n_frames,
        image_size=(w, h), cfg=aa, frames_reader=_frames_reader,
    )
```

Add this method to `CameraStage`:

```python
    def _ensure_anchors(self, shot_id, anchors_path, clip_path, cfg):
        """Auto-generate anchors when enabled and appropriate. On any failure,
        leave the file as-is so existing manual path/warnings apply."""
        aa = cfg.get("auto_anchors", {})
        if not aa.get("enabled", False):
            return
        mode = aa.get("mode", "replace_when_empty")
        if anchors_path.exists() and mode == "replace_when_empty":
            return
        try:
            generated = _generate_auto_anchors(shot_id, clip_path, cfg)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning(
                "auto_anchors: generation failed for shot %s (%s); "
                "falling back to manual anchors", shot_id, exc,
            )
            return
        if generated is None or not generated.anchors:
            logger.warning(
                "auto_anchors: no usable anchors for shot %s; "
                "falling back to manual anchors", shot_id,
            )
            return
        if mode == "augment" and anchors_path.exists():
            existing = AnchorSet.load(anchors_path)
            seen = {e.frame for e in existing.anchors}
            merged = existing.anchors + tuple(
                a for a in generated.anchors if a.frame not in seen
            )
            generated = AnchorSet(
                clip_id=generated.clip_id, image_size=generated.image_size,
                anchors=merged,
            )
        anchors_path.parent.mkdir(parents=True, exist_ok=True)
        generated.save(anchors_path)
        logger.info(
            "auto_anchors: wrote %d generated anchors for shot %s to %s",
            len(generated.anchors), shot_id, anchors_path,
        )
```

Then, in `_run_shot`, replace the opening `anchors = AnchorSet.load(anchors_path)` with:

```python
        self._ensure_anchors(shot_id, anchors_path, clip_path, cfg)
        anchors = AnchorSet.load(anchors_path)
```

And in `run()`, change the no-anchors skip guard so an enabled auto path still enters `_run_shot`:

```python
            if not anchors_path.exists() and not cfg.get("auto_anchors", {}).get("enabled", False):
                logger.warning(
                    "camera stage skipping shot %s — no anchors at %s. Open "
                    "the anchor editor and place keyframes before re-running.",
                    shot.id, anchors_path,
                )
                continue
```

> The `_run_shot` signature must have `clip_path` available before the anchor load (it does — see the current method). Confirm `AnchorSet` is imported at module top in `camera.py` (it is).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_camera_stage_auto_anchor.py -v`
Expected: PASS.

- [ ] **Step 5: Regression — existing camera tests**

Run: `pytest tests/test_camera_stage.py tests/test_camera_stage_static_line.py -v`
Expected: PASS (auto path off by default).

- [ ] **Step 6: Commit**

```bash
git add src/stages/camera.py tests/test_camera_stage_auto_anchor.py
git commit -m "feat: camera stage auto-anchor pre-step with manual fallback"
```

---

## Phase 4 — Validation

### Task I: One-frame smoke CLI (manual gate)

**Files:**
- Create: `scripts/pnlcalib_smoke.py`
- Test: `tests/test_pnlcalib_smoke_importable.py`

- [ ] **Step 1: Write the failing import test**

Create `tests/test_pnlcalib_smoke_importable.py`:

```python
import importlib.util
from pathlib import Path


def test_smoke_script_imports():
    path = Path(__file__).resolve().parents[1] / "scripts" / "pnlcalib_smoke.py"
    spec = importlib.util.spec_from_file_location("pnlcalib_smoke", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")
```

- [ ] **Step 2: Run it (fails: missing script)**

Run: `pytest tests/test_pnlcalib_smoke_importable.py -v`
Expected: FAIL.

- [ ] **Step 3: Write the smoke CLI**

Create `scripts/pnlcalib_smoke.py`:

```python
"""Register ONE real frame and print mapped (image -> world) correspondences
in OUR pitch frame. Manual validation gate for the bridge + provider.

  python scripts/pnlcalib_smoke.py --video CLIP --frame 0
Eyeball that goal-post keypoints (12-19) land on the posts and the four
crossbar tops (12,15,16,19) report z = +2.44.
"""

from __future__ import annotations

import argparse

import cv2

from src.utils.neural_calibrator import PnLCalibrator
from src.utils.pnlcalib_pitch_map import keypoint_world_xyz_ours


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("could not read frame")
        return

    calibrator = PnLCalibrator(device=args.device)
    pixels = calibrator.extract_keypoints_pixels(frame)
    print(f"frame {args.frame}: {len(pixels)} keypoints")
    for kp_id in sorted(pixels):
        px, py = pixels[kp_id]
        world = keypoint_world_xyz_ours(kp_id)
        if world is None:
            continue
        wx, wy, wz = world
        print(f"  kp{kp_id:>2}  px=({px:7.1f},{py:7.1f})  "
              f"world=({wx:6.2f},{wy:6.2f},{wz:4.2f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run import test (passes)**

Run: `pytest tests/test_pnlcalib_smoke_importable.py -v`
Expected: PASS.

- [ ] **Step 5: MANUAL gate (requires weights + clip)**

Run: `python scripts/pnlcalib_smoke.py --video "test-media/<gberch>.mp4" --frame 0`
Confirm keypoint pixels are sane and crossbar tops report z=+2.44. This validates the full recovered path before the e2e parity run.

- [ ] **Step 6: Commit**

```bash
git add scripts/pnlcalib_smoke.py tests/test_pnlcalib_smoke_importable.py
git commit -m "feat: PnLCalib one-frame smoke CLI (bridge validation gate)"
```

---

### Task J: Zero-click parity e2e (opt-in)

**Files:**
- Create: `tests/test_auto_anchor_e2e.py`, `scripts/eval_auto_anchor.py`

- [ ] **Step 1: Write the eval script**

Create `scripts/eval_auto_anchor.py`:

```python
"""Inspect a camera track produced with auto_anchors (zero manual clicks)
and print confidence + quality-report camera diagnostics for comparison
against the manual baseline (~0.95 px mean line RMS on gberch).

  python scripts/eval_auto_anchor.py --output ./output-autotest --shot s1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.schemas.camera_track import CameraTrack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--shot", default="s1")
    args = ap.parse_args()
    out = Path(args.output)
    track = CameraTrack.load(out / "camera" / f"{args.shot}_camera_track.json")
    confs = [f.confidence for f in track.frames]
    print(f"frames={len(track.frames)} mean_conf={sum(confs)/len(confs):.3f}")
    qr = out / "quality_report.json"
    if qr.exists():
        print(json.dumps(json.loads(qr.read_text()).get("camera", {}), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the opt-in e2e test**

Create `tests/test_auto_anchor_e2e.py`:

```python
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_PNLCALIB_E2E"),
    reason="set RUN_PNLCALIB_E2E=1 and provide PNLCALIB_E2E_CLIP to run",
)


def test_zero_click_generates_rich_anchorset():
    """With zero manual clicks, the auto pipeline yields an AnchorSet with at
    least one rich (>=6 non-coplanar pts) anchor on a real clip."""
    import cv2
    from src.utils.auto_anchor import generate
    from src.utils.neural_calibrator import PnLCalibrator
    from src.utils.anchor_solver import _is_rich

    clip = os.environ["PNLCALIB_E2E_CLIP"]
    cap = cv2.VideoCapture(clip)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    def reader(indices, image_size):
        cap = cv2.VideoCapture(clip)
        out = {}
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if ok:
                out[i] = fr
        cap.release()
        return out

    cal = PnLCalibrator(device=os.environ.get("PNLCALIB_DEVICE", "auto"))
    cfg = dict(keyframe_interval=30, max_keyframes=12, min_points_per_anchor=4,
               consensus_max_position_mad=3.0)
    anchor_set = generate(calibrator=cal, clip_id="gberch", n_frames=n,
                          image_size=(w, h), cfg=cfg, frames_reader=reader)
    assert anchor_set is not None
    assert any(_is_rich(a) for a in anchor_set.anchors), "no rich anchor"
```

- [ ] **Step 3: Run (auto-skips without env)**

Run: `pytest tests/test_auto_anchor_e2e.py -v`
Expected: SKIPPED.

- [ ] **Step 4: MANUAL e2e (GPU box or patient CPU)**

```bash
RUN_PNLCALIB_E2E=1 PNLCALIB_E2E_CLIP="test-media/<gberch>.mp4" \
  pytest tests/test_auto_anchor_e2e.py -v
```
Then run the full camera stage with `auto_anchors.enabled: true` and **no** `{shot}_anchors.json`, and compare line-RMS in the logs / quality report against the 0.95 px manual baseline. Tune `keyframe_interval`, `min_points_per_anchor`, `consensus_max_position_mad` until mean ≤ 1.0 px.

- [ ] **Step 5: Commit**

```bash
git add tests/test_auto_anchor_e2e.py scripts/eval_auto_anchor.py
git commit -m "test: zero-click auto-anchor e2e (opt-in) + eval script"
```

---

## Self-review

**Spec coverage:** auto-anchor generator (Tasks D, F, H) ✓; writes anchors JSON / editor override (H) ✓; components — recovered calibrator (B), bridge (C), auto_anchor (D/E/F), config (G), hook (H) ✓; intrinsic association (C, via keypoint identity) + symmetry consensus (E/F, MAD on camera position) ✓; error handling/fallback (H: try/except, mode, disabled no-op) ✓; testing incl. headline e2e (J) ✓; YAGNI (recover only the bridge primitives, not all of fixed_position_solver; one calibrator) ✓.

**Placeholder scan:** none. Two scoped "verify against live source" notes (Task C import path; Task D/F schema field names) name the exact file and what to adjust — not deferred work.

**Type consistency:** `NeuralCalibration` (B) used in E/F tests + `generate`. `keypoint_world_xyz_ours` (C) used in D + smoke (I). `keypoints_to_anchor`/`compute_keyframes`/`robust_median_position`/`is_plausible_position`/`generate` all defined in `auto_anchor.py` (D/E/F) and consumed consistently in H/J. `PnLCalibrator.calibrate`/`extract_keypoints_pixels` signatures match between B, the fake in F, and H/J.

**Live touchpoints to verify during execution (not gaps):** `src/schemas/anchor.py` constructors (D, F, H read it first); PnLCalib table import path `utils.utils_calib` vs `utils.utils_keypoints` (C); `_is_rich` import in `anchor_solver.py` (J); recovered `neural_calibrator.py` has no stale `from src.*` imports (B Step 2).
```
