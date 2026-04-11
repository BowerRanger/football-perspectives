# Camera Calibration Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix camera calibration so that camera positions are correct (above the pitch, temporally stable) by adding off-plane landmarks, multi-solution PnP scoring, player-height disambiguation, and temporal continuity.

**Architecture:** Layered approach — off-plane landmarks (goal crossbar, corner flags) break PnP coplanarity degeneracy; hard constraints (camera above pitch, looking down) reject bad solutions; player bounding box heights disambiguate remaining cases; temporal seeding ensures frame-to-frame consistency on panning shots.

**Tech Stack:** Python 3.11, OpenCV (`cv2.solvePnPGeneric`, `SOLVEPNP_IPPE`), NumPy, pytest

**Spec:** `docs/superpowers/specs/2026-04-11-calibration-improvements-design.md`

---

### Task 1: Rename and fix FIFA_LANDMARKS

Rewrite the landmark dictionary with near/far naming and corrected y-coordinates. All `_top` → `_far` (y swaps to far side), all `_bottom` → `_near` (y swaps to near side). Remove old goalpost landmarks, add goal structure (crossbar at z=2.44) and corner flag tops (z=1.5).

**Files:**
- Modify: `src/utils/pitch.py`
- Modify: `tests/test_calibration.py` (landmark name references)
- Modify: `tests/test_pitch_detector.py` (landmark name references)
- Modify: `tests/test_schemas.py` (landmark name in calibration round-trip test)

- [ ] **Step 1: Rewrite `src/utils/pitch.py`**

Replace the entire `FIFA_LANDMARKS` dict. The y-axis convention changes: near = y=0, far = y=68. Old `_top` had y closer to 0; new `_far` has y closer to 68 (and vice versa).

```python
import numpy as np

PITCH_LENGTH = 105.0  # metres (FIFA standard)
PITCH_WIDTH = 68.0    # metres
# Origin at near-left corner; x along length (0→105), y along width (0→68), z up
# "near" = y=0 (near side, bottom of broadcast view)
# "far"  = y=68 (far side, top of broadcast view)

_GOAL_HALF = 7.32 / 2.0  # 3.66m — half goal width
_GOAL_HEIGHT = 2.44       # metres — crossbar height
_FLAG_HEIGHT = 1.5        # metres — corner flag height

_CIRCLE_R = 9.15  # centre circle radius

# Penalty arc: radius 9.15m from penalty spot, intersects 18-yard line
_LEFT_D_DY = (9.15 ** 2 - (16.5 - 11.0) ** 2) ** 0.5   # ≈ 7.312
_RIGHT_D_DY = _LEFT_D_DY

FIFA_LANDMARKS: dict[str, np.ndarray] = {
    # Corners (near = y=0, far = y=68)
    "corner_near_left":             np.array([0.0,   0.0,  0.0]),
    "corner_near_right":            np.array([105.0, 0.0,  0.0]),
    "corner_far_left":              np.array([0.0,   68.0, 0.0]),
    "corner_far_right":             np.array([105.0, 68.0, 0.0]),
    # Halfway line
    "halfway_near":                 np.array([52.5,  0.0,  0.0]),
    "halfway_far":                  np.array([52.5,  68.0, 0.0]),
    "center_spot":                  np.array([52.5,  34.0, 0.0]),
    # Centre circle intersections with halfway line
    "centre_circle_near":           np.array([52.5,  34.0 - _CIRCLE_R, 0.0]),
    "centre_circle_far":            np.array([52.5,  34.0 + _CIRCLE_R, 0.0]),
    # Penalty spots
    "left_penalty_spot":            np.array([11.0,  34.0, 0.0]),
    "right_penalty_spot":           np.array([94.0,  34.0, 0.0]),
    # Left 6-yard box (near = lower y, far = higher y)
    "left_6yard_near_left":         np.array([0.0,   24.84, 0.0]),
    "left_6yard_near_right":        np.array([5.5,   24.84, 0.0]),
    "left_6yard_far_right":         np.array([5.5,   43.16, 0.0]),
    "left_6yard_far_left":          np.array([0.0,   43.16, 0.0]),
    # Right 6-yard box
    "right_6yard_near_left":        np.array([99.5,  24.84, 0.0]),
    "right_6yard_near_right":       np.array([105.0, 24.84, 0.0]),
    "right_6yard_far_right":        np.array([105.0, 43.16, 0.0]),
    "right_6yard_far_left":         np.array([99.5,  43.16, 0.0]),
    # Left 18-yard box
    "left_18yard_near_left":        np.array([0.0,   13.84, 0.0]),
    "left_18yard_near_right":       np.array([16.5,  13.84, 0.0]),
    "left_18yard_far_right":        np.array([16.5,  54.16, 0.0]),
    "left_18yard_far_left":         np.array([0.0,   54.16, 0.0]),
    # Right 18-yard box
    "right_18yard_near_left":       np.array([88.5,  13.84, 0.0]),
    "right_18yard_near_right":      np.array([105.0, 13.84, 0.0]),
    "right_18yard_far_right":       np.array([105.0, 54.16, 0.0]),
    "right_18yard_far_left":        np.array([88.5,  54.16, 0.0]),
    # Penalty arc / 18-yard line intersections ("D" shape)
    "left_18yard_d_near":           np.array([16.5,  34.0 - _LEFT_D_DY, 0.0]),
    "left_18yard_d_far":            np.array([16.5,  34.0 + _LEFT_D_DY, 0.0]),
    "right_18yard_d_near":          np.array([88.5,  34.0 - _RIGHT_D_DY, 0.0]),
    "right_18yard_d_far":           np.array([88.5,  34.0 + _RIGHT_D_DY, 0.0]),
    # Goal structure — left goal (x=0)
    "left_goal_near_post_base":     np.array([0.0,   34.0 - _GOAL_HALF, 0.0]),
    "left_goal_near_post_top":      np.array([0.0,   34.0 - _GOAL_HALF, _GOAL_HEIGHT]),
    "left_goal_far_post_base":      np.array([0.0,   34.0 + _GOAL_HALF, 0.0]),
    "left_goal_far_post_top":       np.array([0.0,   34.0 + _GOAL_HALF, _GOAL_HEIGHT]),
    # Goal structure — right goal (x=105)
    "right_goal_near_post_base":    np.array([105.0, 34.0 - _GOAL_HALF, 0.0]),
    "right_goal_near_post_top":     np.array([105.0, 34.0 - _GOAL_HALF, _GOAL_HEIGHT]),
    "right_goal_far_post_base":     np.array([105.0, 34.0 + _GOAL_HALF, 0.0]),
    "right_goal_far_post_top":      np.array([105.0, 34.0 + _GOAL_HALF, _GOAL_HEIGHT]),
    # Corner flag tops
    "corner_near_left_flag_top":    np.array([0.0,   0.0,  _FLAG_HEIGHT]),
    "corner_near_right_flag_top":   np.array([105.0, 0.0,  _FLAG_HEIGHT]),
    "corner_far_left_flag_top":     np.array([0.0,   68.0, _FLAG_HEIGHT]),
    "corner_far_right_flag_top":    np.array([105.0, 68.0, _FLAG_HEIGHT]),
}
```

- [ ] **Step 2: Update landmark name references in `tests/test_calibration.py`**

In `_make_synthetic_correspondences()` (line 49), update the landmark names list:

```python
landmark_names = [
    "corner_near_left", "corner_near_right", "corner_far_left",
    "corner_far_right", "center_spot", "left_penalty_spot",
    "right_penalty_spot", "halfway_near", "halfway_far",
]
```

In `test_pitch_constants()` (line 14), update assertions:

```python
def test_pitch_constants():
    assert PITCH_LENGTH == 105.0
    assert PITCH_WIDTH == 68.0
    assert "corner_near_left" in FIFA_LANDMARKS
    assert "center_spot" in FIFA_LANDMARKS
    pt = FIFA_LANDMARKS["corner_near_left"]
    assert pt[2] == 0.0  # z=0, pitch is ground plane
```

- [ ] **Step 3: Update landmark name references in `tests/test_pitch_detector.py`**

In `test_manual_json_detector_filters_invalid_names_and_low_confidence` (line 42), change `"halfway_bottom"` to `"halfway_far"` in both the fixture data and the assertion (line 64).

In `test_manual_json_detector_accepts_confidence_boundaries` (line 84), change `"halfway_top"` to `"halfway_near"` in fixture data.

In `test_manual_json_detector_invalid_payloads_are_ignored` (line 106), change `"halfway_top"` to `"halfway_near"` and `"halfway_bottom"` to `"halfway_far"` in fixture data.

In `test_heuristic_detector_detects_core_landmarks` (line 139), change assertions:

```python
assert "halfway_near" in detections
assert "halfway_far" in detections
assert "center_spot" in detections
```

- [ ] **Step 4: Update `tests/test_schemas.py`**

In `test_calibration_round_trip` (line 45), change `"halfway_top"` to `"halfway_near"` in the `tracked_landmark_types` list (line 54) and the assertion (line 61).

- [ ] **Step 5: Run tests to verify landmark rename is consistent**

Run: `.venv311/bin/python -m pytest tests/test_calibration.py tests/test_pitch_detector.py tests/test_schemas.py -v`
Expected: All tests PASS

- [ ] **Step 6: Add tests for new off-plane landmarks**

Add to `tests/test_calibration.py`:

```python
def test_goal_crossbar_landmarks_are_off_plane():
    for name in ["left_goal_near_post_top", "left_goal_far_post_top",
                 "right_goal_near_post_top", "right_goal_far_post_top"]:
        assert name in FIFA_LANDMARKS
        assert FIFA_LANDMARKS[name][2] == 2.44  # crossbar height

def test_corner_flag_landmarks_are_off_plane():
    for name in ["corner_near_left_flag_top", "corner_near_right_flag_top",
                 "corner_far_left_flag_top", "corner_far_right_flag_top"]:
        assert name in FIFA_LANDMARKS
        assert FIFA_LANDMARKS[name][2] == 1.5  # flag height

def test_near_landmarks_have_lower_y_than_far():
    assert FIFA_LANDMARKS["corner_near_left"][1] < FIFA_LANDMARKS["corner_far_left"][1]
    assert FIFA_LANDMARKS["halfway_near"][1] < FIFA_LANDMARKS["halfway_far"][1]
    assert FIFA_LANDMARKS["centre_circle_near"][1] < FIFA_LANDMARKS["centre_circle_far"][1]
    assert FIFA_LANDMARKS["left_goal_near_post_base"][1] < FIFA_LANDMARKS["left_goal_far_post_base"][1]
```

Run: `.venv311/bin/python -m pytest tests/test_calibration.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/utils/pitch.py tests/test_calibration.py tests/test_pitch_detector.py tests/test_schemas.py
git commit -m "refactor: rename FIFA landmarks to near/far, fix y-axis, add off-plane landmarks"
```

---

### Task 2: Update HeuristicPitchDetector landmark names

The heuristic detector hardcodes landmark names in its output. Update to match the new naming convention.

**Files:**
- Modify: `src/utils/pitch_detector.py:225-267` (HeuristicPitchDetector.detect candidates dict)

- [ ] **Step 1: Update hardcoded landmark names in `HeuristicPitchDetector.detect`**

In `src/utils/pitch_detector.py`, update the `candidates` dict (around line 225):

```python
candidates: dict[str, LandmarkDetection] = {
    "halfway_near": LandmarkDetection(
        uv=np.array([x_center, y_top], dtype=np.float32),
        confidence=base_conf,
        source="heuristic",
    ),
    "halfway_far": LandmarkDetection(
        uv=np.array([x_center, y_bottom], dtype=np.float32),
        confidence=base_conf,
        source="heuristic",
    ),
    "center_spot": LandmarkDetection(
        uv=np.array([x_center, y_center], dtype=np.float32),
        confidence=max(self.min_confidence, min(0.8, base_conf - 0.05)),
        source="heuristic",
    ),
}
```

And the corner landmarks block (around line 243):

```python
if len(v_mids_x) >= 2:
    x_left = float(np.percentile(v_mids_x, 10))
    x_right = float(np.percentile(v_mids_x, 90))
    edge_conf = max(self.min_confidence, min(0.75, base_conf - 0.1))
    candidates["corner_near_left"] = LandmarkDetection(
        uv=np.array([x_left, y_top], dtype=np.float32),
        confidence=edge_conf,
        source="heuristic",
    )
    candidates["corner_near_right"] = LandmarkDetection(
        uv=np.array([x_right, y_top], dtype=np.float32),
        confidence=edge_conf,
        source="heuristic",
    )
    candidates["corner_far_left"] = LandmarkDetection(
        uv=np.array([x_left, y_bottom], dtype=np.float32),
        confidence=edge_conf,
        source="heuristic",
    )
    candidates["corner_far_right"] = LandmarkDetection(
        uv=np.array([x_right, y_bottom], dtype=np.float32),
        confidence=edge_conf,
        source="heuristic",
    )
```

Note: `y_top` in the detector refers to the top of the image (near side), `y_bottom` to the bottom of the image (far side). The naming now matches.

- [ ] **Step 2: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_pitch_detector.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add src/utils/pitch_detector.py
git commit -m "refactor: update HeuristicPitchDetector to use near/far landmark names"
```

---

### Task 3: Camera validation utilities

Add helper functions to compute camera world position and validate that a PnP solution is physically plausible (camera above pitch, looking downward).

**Files:**
- Modify: `src/utils/camera.py`
- Modify: `tests/test_calibration.py` (add new tests)

- [ ] **Step 1: Write failing tests for camera validation**

Add to `tests/test_calibration.py`:

```python
from src.utils.camera import camera_world_position, is_camera_valid

def test_camera_world_position_synthetic():
    K, rvec, tvec = _synthetic_camera()
    pos = camera_world_position(rvec, tvec)
    # Synthetic camera: tvec=(-52.5, -34, 60) with small rotation
    # World position should be roughly above pitch centre, z > 0
    assert pos[2] > 0  # camera above pitch
    assert pos.shape == (3,)

def test_is_camera_valid_accepts_normal_broadcast_camera():
    K, rvec, tvec = _synthetic_camera()
    assert is_camera_valid(rvec, tvec, min_height=3.0, max_height=80.0)

def test_is_camera_valid_rejects_camera_below_pitch():
    rvec = np.array([0.05, 0.15, 0.0], dtype=np.float32)
    tvec = np.array([-52.5, -34.0, -60.0], dtype=np.float32)  # negative z
    # This will place camera below pitch depending on rotation
    # Use identity-ish rotation so camera pos ≈ (52.5, 34, -60)
    rvec_id = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    tvec_below = np.array([0.0, 0.0, -10.0], dtype=np.float32)
    assert not is_camera_valid(rvec_id, tvec_below, min_height=3.0, max_height=80.0)

def test_is_camera_valid_rejects_camera_looking_up():
    # Camera at height 30m but pointing upward (away from pitch)
    rvec_up = np.array([np.pi, 0.0, 0.0], dtype=np.float32)  # flipped 180 around x
    tvec = np.array([0.0, 0.0, 30.0], dtype=np.float32)
    assert not is_camera_valid(rvec_up, tvec, min_height=3.0, max_height=80.0)
```

Run: `.venv311/bin/python -m pytest tests/test_calibration.py::test_camera_world_position_synthetic tests/test_calibration.py::test_is_camera_valid_accepts_normal_broadcast_camera tests/test_calibration.py::test_is_camera_valid_rejects_camera_below_pitch tests/test_calibration.py::test_is_camera_valid_rejects_camera_looking_up -v`
Expected: FAIL (functions don't exist)

- [ ] **Step 2: Implement camera validation functions in `src/utils/camera.py`**

Add to `src/utils/camera.py`:

```python
def camera_world_position(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Compute camera position in world coordinates: C = -R^T @ t."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    return (-R.T @ t).astype(np.float64)


def is_camera_valid(
    rvec: np.ndarray,
    tvec: np.ndarray,
    min_height: float = 3.0,
    max_height: float = 80.0,
) -> bool:
    """Check if a PnP solution is physically plausible for a broadcast camera.

    Rejects solutions where:
    - Camera is below min_height (not above the pitch)
    - Camera is above max_height (implausibly high)
    - Camera optical axis points upward (away from pitch)
    """
    pos = camera_world_position(rvec, tvec)
    if pos[2] < min_height or pos[2] > max_height:
        return False
    # Check optical axis direction: camera z-axis in world coords
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    # The camera looks along its +z axis. In world coords, that's R^T @ [0,0,1] = R[:,2]
    # For the camera to look downward toward the pitch (z=0), the world-z component
    # of the optical axis must be negative.
    optical_axis_world = R[:, 2]  # third column of R = camera z-axis in world frame
    if optical_axis_world[2] > 0:
        return False  # camera looking upward
    return True
```

- [ ] **Step 3: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_calibration.py::test_camera_world_position_synthetic tests/test_calibration.py::test_is_camera_valid_accepts_normal_broadcast_camera tests/test_calibration.py::test_is_camera_valid_rejects_camera_below_pitch tests/test_calibration.py::test_is_camera_valid_rejects_camera_looking_up -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/utils/camera.py tests/test_calibration.py
git commit -m "feat: add camera_world_position and is_camera_valid utilities"
```

---

### Task 4: Multi-solution PnP solver

Replace the current single-best `calibrate_frame` with a multi-solution approach: generate candidates from multiple PnP methods, score with hard constraints and height plausibility.

**Files:**
- Modify: `src/stages/calibration.py:76-178` (rewrite `calibrate_frame`)
- Modify: `tests/test_calibration.py` (update and add calibration tests)

- [ ] **Step 1: Write failing test for camera-above-pitch constraint**

Add to `tests/test_calibration.py`:

```python
def test_calibrate_frame_produces_camera_above_pitch():
    """The solver must place the camera above the pitch (z > 0)."""
    correspondences, _ = _make_synthetic_correspondences()
    result = calibrate_frame(
        correspondences=correspondences,
        landmarks_3d=FIFA_LANDMARKS,
        image_shape=(1080, 1920),
    )
    assert result is not None
    from src.utils.camera import camera_world_position
    pos = camera_world_position(
        np.array(result.rotation_vector),
        np.array(result.translation_vector),
    )
    assert pos[2] > 0, f"Camera placed below pitch at z={pos[2]:.1f}"
```

Run: `.venv311/bin/python -m pytest tests/test_calibration.py::test_calibrate_frame_produces_camera_above_pitch -v`
Expected: Likely FAIL (current solver often places camera below pitch)

- [ ] **Step 2: Rewrite `calibrate_frame` in `src/stages/calibration.py`**

Replace `calibrate_frame` (lines 76-178) and `_validate_calibration` (lines 64-73) with the multi-solution approach:

```python
def _validate_calibration(tvec: np.ndarray, fx: float) -> bool:
    """Reject calibrations with unreasonable camera distance or focal length."""
    dist = float(np.linalg.norm(tvec))
    if dist < 5.0 or dist > 200.0:
        logging.debug("Rejected calibration: camera distance %.1fm (expected 5-200m)", dist)
        return False
    if fx < 200.0 or fx > 10000.0:
        logging.debug("Rejected calibration: focal length %.0fpx (expected 200-10000)", fx)
        return False
    return True


@dataclass(frozen=True)
class _PnPCandidate:
    """A candidate PnP solution with its scoring metadata."""
    rvec: np.ndarray
    tvec: np.ndarray
    K: np.ndarray
    inlier_indices: np.ndarray
    reprojection_error: float
    camera_height: float


def _generate_pnp_candidates(
    pts_3d: np.ndarray,
    pts_2d: np.ndarray,
    focal_candidates: list[float],
    cx: float,
    cy: float,
    ransac_reproj_threshold: float,
    min_height: float,
    max_height: float,
) -> list[_PnPCandidate]:
    """Generate multiple PnP candidates using different methods and focal lengths."""
    from src.utils.camera import camera_world_position, is_camera_valid

    candidates: list[_PnPCandidate] = []

    for fx in focal_candidates:
        K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)

        # Method 1: RANSAC (robust to outliers)
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d.astype(np.float64),
                pts_2d.astype(np.float64),
                K, None,
                reprojectionError=ransac_reproj_threshold,
                confidence=0.99,
                iterationsCount=5000,
            )
            if success and inliers is not None and len(inliers) >= 4:
                idx = inliers.flatten()
                if _validate_calibration(tvec, fx) and is_camera_valid(rvec, tvec, min_height, max_height):
                    err = reprojection_error(pts_3d[idx], pts_2d[idx], K.astype(np.float32), rvec, tvec)
                    pos = camera_world_position(rvec, tvec)
                    candidates.append(_PnPCandidate(
                        rvec=rvec, tvec=tvec, K=K.astype(np.float32),
                        inlier_indices=idx, reprojection_error=err,
                        camera_height=float(pos[2]),
                    ))
        except cv2.error:
            pass

        # Method 2: IPPE (planar — returns both ambiguous solutions)
        if len(pts_3d) >= 4:
            try:
                n_solutions, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(
                    pts_3d.astype(np.float64),
                    pts_2d.astype(np.float64),
                    K, None,
                    flags=cv2.SOLVEPNP_IPPE,
                )
                for i in range(n_solutions):
                    rv, tv = rvecs[i], tvecs[i]
                    if not _validate_calibration(tv, fx):
                        continue
                    if not is_camera_valid(rv, tv, min_height, max_height):
                        continue
                    err = reprojection_error(pts_3d, pts_2d, K.astype(np.float32), rv, tv)
                    pos = camera_world_position(rv, tv)
                    candidates.append(_PnPCandidate(
                        rvec=rv, tvec=tv, K=K.astype(np.float32),
                        inlier_indices=np.arange(len(pts_3d)),
                        reprojection_error=err,
                        camera_height=float(pos[2]),
                    ))
            except cv2.error:
                pass

        # Method 3: EPNP (non-iterative, good starting point)
        if len(pts_3d) >= 4:
            try:
                success, rvec, tvec = cv2.solvePnP(
                    pts_3d.astype(np.float64),
                    pts_2d.astype(np.float64),
                    K, None,
                    flags=cv2.SOLVEPNP_EPNP,
                )
                if success and _validate_calibration(tvec, fx) and is_camera_valid(rvec, tvec, min_height, max_height):
                    err = reprojection_error(pts_3d, pts_2d, K.astype(np.float32), rvec, tvec)
                    pos = camera_world_position(rvec, tvec)
                    candidates.append(_PnPCandidate(
                        rvec=rvec, tvec=tvec, K=K.astype(np.float32),
                        inlier_indices=np.arange(len(pts_3d)),
                        reprojection_error=err,
                        camera_height=float(pos[2]),
                    ))
            except cv2.error:
                pass

    return candidates


def _score_candidate(c: _PnPCandidate, preferred_height_range: tuple[float, float] = (5.0, 80.0)) -> float:
    """Score a PnP candidate. Lower is better.

    Combines reprojection error with a soft penalty for implausible camera heights.
    """
    height_lo, height_hi = preferred_height_range
    height_penalty = 0.0
    if c.camera_height < height_lo:
        height_penalty = (height_lo - c.camera_height) * 2.0
    elif c.camera_height > height_hi:
        height_penalty = (c.camera_height - height_hi) * 0.5
    return c.reprojection_error + height_penalty


def calibrate_frame(
    correspondences: dict[str, np.ndarray] | dict[str, LandmarkDetection],
    landmarks_3d: dict[str, np.ndarray],
    image_shape: tuple[int, int],
    frame_idx: int = 0,
    max_reprojection_error: float = 15.0,
    ransac_reproj_threshold: float = 40.0,
    min_camera_height: float = 3.0,
    max_camera_height: float = 80.0,
    initial_rvec: np.ndarray | None = None,
    initial_tvec: np.ndarray | None = None,
    initial_fx: float | None = None,
    focal_length_tolerance: float = 0.2,
) -> CameraFrame | None:
    """Solve camera pose from 2D-3D pitch correspondences.

    Uses multiple PnP methods to generate candidate solutions, then scores
    them by reprojection error and camera height plausibility. Returns None
    if fewer than 4 common points or no valid solution found.
    """
    normalized = _normalize_correspondences(correspondences)
    common = [k for k in normalized if k in landmarks_3d]
    if len(common) < 4:
        return None

    pts_2d = np.array([normalized[k].uv for k in common], dtype=np.float32)
    pts_3d = np.array([landmarks_3d[k] for k in common], dtype=np.float32)

    h, w = image_shape
    cx, cy = w / 2.0, h / 2.0
    diagonal = float(np.sqrt(h ** 2 + w ** 2))

    # Build focal length candidates
    if initial_fx is not None:
        # Constrain to nearby focal lengths
        tol = focal_length_tolerance
        focal_candidates = [initial_fx * s for s in (1.0 - tol, 1.0, 1.0 + tol)]
    else:
        focal_candidates = [diagonal * s for s in (0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.4)]

    candidates = _generate_pnp_candidates(
        pts_3d, pts_2d, focal_candidates, cx, cy,
        ransac_reproj_threshold, min_camera_height, max_camera_height,
    )

    # If we have an initial guess, also try seeded iterative solve
    if initial_rvec is not None and initial_tvec is not None:
        from src.utils.camera import camera_world_position, is_camera_valid
        for fx in focal_candidates:
            K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)
            try:
                success, rvec, tvec = cv2.solvePnP(
                    pts_3d.astype(np.float64),
                    pts_2d.astype(np.float64),
                    K, None,
                    rvec=initial_rvec.copy().reshape(3, 1),
                    tvec=initial_tvec.copy().reshape(3, 1),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if success and _validate_calibration(tvec, fx) and is_camera_valid(rvec, tvec, min_camera_height, max_camera_height):
                    err = reprojection_error(pts_3d, pts_2d, K.astype(np.float32), rvec, tvec)
                    pos = camera_world_position(rvec, tvec)
                    candidates.append(_PnPCandidate(
                        rvec=rvec, tvec=tvec, K=K.astype(np.float32),
                        inlier_indices=np.arange(len(common)),
                        reprojection_error=err,
                        camera_height=float(pos[2]),
                    ))
            except cv2.error:
                pass

    if not candidates:
        return None

    # Select best candidate by combined score
    best = min(candidates, key=_score_candidate)
    idx = best.inlier_indices
    err = best.reprojection_error
    K = best.K

    tracked_landmark_types = [common[int(i)] for i in idx]
    frame_confidence = float(np.mean([normalized[k].confidence for k in common]))
    confidence = float(max(0.0, 1.0 - err / max_reprojection_error) * frame_confidence)

    return CameraFrame(
        frame=frame_idx,
        intrinsic_matrix=K.tolist(),
        rotation_vector=best.rvec.flatten().tolist(),
        translation_vector=best.tvec.flatten().tolist(),
        reprojection_error=float(err),
        num_correspondences=int(len(idx)),
        confidence=confidence,
        tracked_landmark_types=tracked_landmark_types,
    )
```

- [ ] **Step 3: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_calibration.py -v`
Expected: All PASS (including the new camera-above-pitch test and the existing tests)

- [ ] **Step 4: Commit**

```bash
git add src/stages/calibration.py tests/test_calibration.py
git commit -m "feat: multi-solution PnP solver with camera height validation"
```

---

### Task 5: Player height disambiguation

Add a function that scores PnP candidates by checking whether detected players have plausible physical heights (1.5–2.1m) when projected through each candidate's camera model.

**Files:**
- Create: `src/utils/player_height.py`
- Modify: `tests/test_calibration.py` (add tests)
- Modify: `src/stages/calibration.py` (integrate into `_calibrate_shot`)

- [ ] **Step 1: Write failing tests for player height scoring**

Add to `tests/test_calibration.py`:

```python
from src.utils.player_height import score_player_heights

def test_score_player_heights_correct_solution_scores_higher():
    """A camera solution that produces ~1.8m player heights should score better
    than one that produces absurd heights."""
    K, rvec_good, tvec_good = _synthetic_camera()

    # Simulate player bounding boxes: foot at pitch level, head ~1.8m up
    # Project a standing player at (30, 20) on the pitch
    foot_3d = np.array([[30.0, 20.0, 0.0]], dtype=np.float32)
    head_3d = np.array([[30.0, 20.0, 1.8]], dtype=np.float32)
    foot_2d, _ = cv2.projectPoints(foot_3d, rvec_good, tvec_good, K, None)
    head_2d, _ = cv2.projectPoints(head_3d, rvec_good, tvec_good, K, None)

    bboxes = [[
        float(head_2d[0, 0, 0]) - 20, float(head_2d[0, 0, 1]),  # x1, y1 (top)
        float(foot_2d[0, 0, 0]) + 20, float(foot_2d[0, 0, 1]),  # x2, y2 (bottom)
    ]]

    score_good = score_player_heights(
        bboxes, K, rvec_good, tvec_good, height_range=(1.5, 2.1),
    )
    assert score_good > 0.5  # most players should have plausible height

def test_score_player_heights_empty_bboxes_returns_zero():
    K, rvec, tvec = _synthetic_camera()
    score = score_player_heights([], K, rvec, tvec)
    assert score == 0.0
```

Run: `.venv311/bin/python -m pytest tests/test_calibration.py::test_score_player_heights_correct_solution_scores_higher tests/test_calibration.py::test_score_player_heights_empty_bboxes_returns_zero -v`
Expected: FAIL (module doesn't exist)

- [ ] **Step 2: Implement `src/utils/player_height.py`**

```python
"""Score PnP candidates by checking implied player heights from bounding boxes."""

import numpy as np
import cv2

from src.utils.camera import project_to_pitch


def score_player_heights(
    bboxes: list[list[float]],
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    height_range: tuple[float, float] = (1.5, 2.1),
) -> float:
    """Score a PnP solution by how many player bboxes imply plausible heights.

    For each bounding box, projects the foot (bottom-centre) and head (top-centre)
    onto the pitch plane and computes the implied 3D height. Returns the fraction
    of players with heights in the plausible range.

    Args:
        bboxes: list of [x1, y1, x2, y2] bounding boxes in pixel space
        K: 3x3 intrinsic matrix
        rvec: rotation vector (3,)
        tvec: translation vector (3,)
        height_range: (min_height, max_height) in metres

    Returns:
        fraction of players with plausible heights (0.0 to 1.0)
    """
    if not bboxes:
        return 0.0

    K_f64 = np.asarray(K, dtype=np.float64)
    rv = np.asarray(rvec, dtype=np.float64).reshape(3)
    tv = np.asarray(tvec, dtype=np.float64).reshape(3)
    R, _ = cv2.Rodrigues(rv)

    min_h, max_h = height_range
    plausible = 0

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        foot_px = np.array([(x1 + x2) / 2.0, y2], dtype=np.float64)
        head_px = np.array([(x1 + x2) / 2.0, y1], dtype=np.float64)

        # Project foot onto pitch plane (z=0) to get pitch position
        foot_pitch = project_to_pitch(foot_px, K_f64, rv, tv)

        # For the head, we need to find the 3D point along the camera ray
        # that is directly above the foot position. The head ray intersects
        # the vertical line at (foot_x, foot_y, z) for some z > 0.
        # Solve: K @ [R | t] @ [foot_x, foot_y, z, 1]^T = s * [head_u, head_v, 1]^T
        foot_x, foot_y = float(foot_pitch[0]), float(foot_pitch[1])

        # Project the foot position at various heights to find which height
        # matches the head pixel position
        pt_ground = np.array([[foot_x, foot_y, 0.0]], dtype=np.float64)
        pt_high = np.array([[foot_x, foot_y, 3.0]], dtype=np.float64)  # 3m reference

        proj_ground, _ = cv2.projectPoints(pt_ground, rv, tv, K_f64, None)
        proj_high, _ = cv2.projectPoints(pt_high, rv, tv, K_f64, None)

        ground_v = float(proj_ground[0, 0, 1])
        high_v = float(proj_high[0, 0, 1])

        dv = ground_v - high_v  # pixels per 3.0m of height
        if abs(dv) < 1.0:
            continue  # degenerate projection

        head_v = float(head_px[1])
        implied_height = 3.0 * (ground_v - head_v) / dv

        if min_h <= implied_height <= max_h:
            plausible += 1

    return float(plausible) / float(len(bboxes))
```

- [ ] **Step 3: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_calibration.py::test_score_player_heights_correct_solution_scores_higher tests/test_calibration.py::test_score_player_heights_empty_bboxes_returns_zero -v`
Expected: All PASS

- [ ] **Step 4: Integrate player height scoring into `_calibrate_shot`**

In `src/stages/calibration.py`, modify the `CameraCalibrationStage._calibrate_shot` method. After the existing landmark-based calibration, if the result has low confidence or multiple candidates were close, load track bounding boxes and re-score.

Add to the imports at the top of `src/stages/calibration.py`:

```python
from src.utils.player_height import score_player_heights
from src.schemas.tracks import TracksResult
```

Add a helper method to `CameraCalibrationStage`:

```python
def _load_track_bboxes(self, shot_id: str, frame_idx: int) -> list[list[float]]:
    """Load player bounding boxes for a specific frame from tracking output."""
    tracks_path = self.output_dir / "tracks" / f"{shot_id}_tracks.json"
    if not tracks_path.exists():
        return []
    tracks_result = TracksResult.load(tracks_path)
    bboxes: list[list[float]] = []
    for track in tracks_result.tracks:
        if track.class_name not in ("player", "goalkeeper"):
            continue
        for tf in track.frames:
            if tf.frame == frame_idx:
                bboxes.append(tf.bbox)
                break
    return bboxes
```

This will be used in Task 6 when we rewrite `_calibrate_shot` with temporal continuity.

- [ ] **Step 5: Commit**

```bash
git add src/utils/player_height.py tests/test_calibration.py src/stages/calibration.py
git commit -m "feat: add player height disambiguation for PnP scoring"
```

---

### Task 6: Temporal continuity for panning shots

Rewrite `_calibrate_shot` to process frames in optimal order (start from best-landmarks frame), seed subsequent frames from previous solutions, reject position jumps, and enforce focal length continuity.

**Files:**
- Modify: `src/stages/calibration.py:418-501` (rewrite `_calibrate_shot`)
- Modify: `tests/test_calibration.py` (add temporal continuity tests)

- [ ] **Step 1: Write failing test for temporal continuity**

Add to `tests/test_calibration.py`:

```python
def test_panning_shot_produces_consistent_camera_positions(tmp_path):
    """Camera positions should not jump wildly between frames of a panning shot."""
    # Create manual landmarks for a panning camera showing the left goal area
    # across 3 frames, simulated by projecting with a camera that pans slightly
    K = np.array([[2000, 0, 960], [0, 2000, 540], [0, 0, 1]], dtype=np.float32)
    base_rvec = np.array([1.3, -0.2, 0.0], dtype=np.float32)
    base_tvec = np.array([-10.0, -34.0, 40.0], dtype=np.float32)

    landmarks_to_use = [
        "left_penalty_spot", "left_6yard_near_left", "left_6yard_near_right",
        "left_6yard_far_right", "left_18yard_near_right", "left_18yard_d_near",
    ]

    import json
    manual_dir = tmp_path / "calibration" / "manual_landmarks"
    manual_dir.mkdir(parents=True)

    frames_data = {}
    for fidx, pan_offset in [(0, 0.0), (50, 0.02), (100, 0.04)]:
        rvec = base_rvec.copy()
        rvec[1] += pan_offset
        pts_3d = np.array([FIFA_LANDMARKS[n] for n in landmarks_to_use], dtype=np.float32)
        pts_2d, _ = cv2.projectPoints(pts_3d, rvec, base_tvec, K, None)
        pts_2d = pts_2d.reshape(-1, 2)
        frame_pts = {}
        for i, name in enumerate(landmarks_to_use):
            frame_pts[name] = {"u": float(pts_2d[i, 0]), "v": float(pts_2d[i, 1]), "confidence": 1.0}
        frames_data[str(fidx)] = frame_pts

    (manual_dir / "shot_001.json").write_text(json.dumps({"frames": frames_data}))

    # Create a dummy clip
    from tests.test_calibration import _create_dummy_clip
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    _create_dummy_clip(shots_dir / "shot_001.mp4", fps=25.0, frames=120)

    stage = CameraCalibrationStage(config={
        "calibration": {"temporal_max_jump": 5.0, "focal_length_tolerance": 0.2}
    }, output_dir=tmp_path, detector=None)
    result = stage._calibrate_shot("shot_001", "shots/shot_001.mp4", 5, 50.0)

    assert len(result.frames) >= 2
    # Check camera positions don't jump more than 5m between consecutive frames
    from src.utils.camera import camera_world_position
    positions = [
        camera_world_position(np.array(f.rotation_vector), np.array(f.translation_vector))
        for f in result.frames
    ]
    for i in range(1, len(positions)):
        jump = float(np.linalg.norm(positions[i] - positions[i - 1]))
        assert jump < 5.0, f"Camera jumped {jump:.1f}m between frames {result.frames[i-1].frame} and {result.frames[i].frame}"
```

Run: `.venv311/bin/python -m pytest tests/test_calibration.py::test_panning_shot_produces_consistent_camera_positions -v`
Expected: FAIL

- [ ] **Step 2: Rewrite `_calibrate_shot` with temporal continuity**

Replace `CameraCalibrationStage._calibrate_shot` (lines 418-501) in `src/stages/calibration.py`:

```python
def _calibrate_shot(
    self, shot_id: str, clip_file: str, keyframe_interval: int, max_err: float, ransac_thresh: float = 40.0,
) -> CalibrationResult:
    clip_path = self.output_dir / clip_file
    cfg = self.config.get("calibration", {})
    min_cam_height = float(cfg.get("min_camera_height", 3.0))
    max_cam_height = float(cfg.get("max_camera_height", 80.0))
    temporal_max_jump = float(cfg.get("temporal_max_jump", 5.0))
    focal_tol = float(cfg.get("focal_length_tolerance", 0.2))
    player_height_range = cfg.get("player_height_range", [1.5, 2.1])
    player_height_range = (float(player_height_range[0]), float(player_height_range[1]))

    manual_path = self.output_dir / "calibration" / "manual_landmarks" / f"{shot_id}.json"
    if manual_path.exists():
        return self._calibrate_shot_from_manual(
            shot_id, clip_path, manual_path, max_err, ransac_thresh,
            min_cam_height, max_cam_height, temporal_max_jump, focal_tol,
            player_height_range,
        )

    # Fallback: per-frame detection
    if self.detector is None:
        return CalibrationResult(shot_id=shot_id, camera_type="static", frames=[])

    cap = cv2.VideoCapture(str(clip_path))
    try:
        if not cap.isOpened():
            logging.warning("Failed to open clip for calibration: %s", clip_path)
            return CalibrationResult(shot_id=shot_id, camera_type="static", frames=[])
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        frames: list[CameraFrame] = []
        prev_rvec: np.ndarray | None = None
        prev_tvec: np.ndarray | None = None
        prev_fx: float | None = None
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % keyframe_interval == 0:
                correspondences = self.detector.detect(
                    frame, frame_idx=frame_idx, shot_id=shot_id,
                )
                cf = calibrate_frame(
                    correspondences, FIFA_LANDMARKS, (h, w), frame_idx,
                    max_reprojection_error=max_err,
                    ransac_reproj_threshold=ransac_thresh,
                    min_camera_height=min_cam_height,
                    max_camera_height=max_cam_height,
                    initial_rvec=prev_rvec,
                    initial_tvec=prev_tvec,
                    initial_fx=prev_fx,
                    focal_length_tolerance=focal_tol,
                )
                if cf is not None and cf.reprojection_error <= max_err:
                    if self._passes_temporal_check(cf, frames, temporal_max_jump):
                        frames.append(cf)
                        prev_rvec = np.array(cf.rotation_vector, dtype=np.float64)
                        prev_tvec = np.array(cf.translation_vector, dtype=np.float64)
                        prev_fx = float(cf.intrinsic_matrix[0][0])
            frame_idx += 1
    finally:
        cap.release()

    camera_type = "tracking" if len(frames) > 1 else "static"
    return CalibrationResult(shot_id=shot_id, camera_type=camera_type, frames=frames)


def _calibrate_shot_from_manual(
    self, shot_id: str, clip_path: Path, manual_path: Path,
    max_err: float, ransac_thresh: float,
    min_cam_height: float, max_cam_height: float,
    temporal_max_jump: float, focal_tol: float,
    player_height_range: tuple[float, float],
) -> CalibrationResult:
    """Calibrate a shot using manual landmark annotations with temporal continuity."""
    import json as _json
    data = _json.loads(manual_path.read_text())
    frame_annotations = data.get("frames", {})

    if not frame_annotations:
        return CalibrationResult(shot_id=shot_id, camera_type="static", frames=[])

    cap = cv2.VideoCapture(str(clip_path))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 1080
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 1920
    cap.release()
    image_shape = (h, w)

    # Build correspondences per frame
    frame_corrs: dict[int, dict[str, LandmarkDetection]] = {}
    for fid_str, pts in frame_annotations.items():
        fid = int(fid_str)
        corrs: dict[str, LandmarkDetection] = {}
        for name, pt in pts.items():
            if name not in FIFA_LANDMARKS:
                continue
            corrs[name] = LandmarkDetection(
                uv=np.array([float(pt["u"]), float(pt["v"])], dtype=np.float32),
                confidence=float(pt.get("confidence", 1.0)),
                source="manual_json",
            )
        if len(corrs) >= 4:
            frame_corrs[fid] = corrs

    if not frame_corrs:
        return CalibrationResult(shot_id=shot_id, camera_type="static", frames=[])

    # Start from the frame with the most correspondences (best chance of correct solve)
    seed_frame = max(frame_corrs, key=lambda fid: len(frame_corrs[fid]))
    sorted_frames = sorted(frame_corrs.keys())

    # Calibrate seed frame first
    seed_cf = calibrate_frame(
        frame_corrs[seed_frame], FIFA_LANDMARKS, image_shape,
        frame_idx=seed_frame,
        max_reprojection_error=max_err,
        ransac_reproj_threshold=ransac_thresh,
        min_camera_height=min_cam_height,
        max_camera_height=max_cam_height,
    )

    if seed_cf is None:
        # Try player height disambiguation on seed frame
        seed_cf = self._try_with_player_heights(
            shot_id, seed_frame, frame_corrs[seed_frame], image_shape,
            max_err, ransac_thresh, min_cam_height, max_cam_height,
            player_height_range,
        )

    if seed_cf is None:
        logging.warning("%s: seed frame %d calibration failed", shot_id, seed_frame)
        return CalibrationResult(shot_id=shot_id, camera_type="static", frames=[])

    # Propagate from seed frame outward (forward then backward)
    results: dict[int, CameraFrame] = {seed_frame: seed_cf}
    seed_idx = sorted_frames.index(seed_frame)

    # Forward from seed
    prev_cf = seed_cf
    for fid in sorted_frames[seed_idx + 1:]:
        cf = self._calibrate_frame_with_continuity(
            shot_id, fid, frame_corrs[fid], image_shape, max_err, ransac_thresh,
            prev_cf, min_cam_height, max_cam_height, focal_tol, temporal_max_jump,
            player_height_range,
        )
        if cf is not None:
            results[fid] = cf
            prev_cf = cf

    # Backward from seed
    prev_cf = seed_cf
    for fid in reversed(sorted_frames[:seed_idx]):
        cf = self._calibrate_frame_with_continuity(
            shot_id, fid, frame_corrs[fid], image_shape, max_err, ransac_thresh,
            prev_cf, min_cam_height, max_cam_height, focal_tol, temporal_max_jump,
            player_height_range,
        )
        if cf is not None:
            results[fid] = cf
            prev_cf = cf

    frames = [results[fid] for fid in sorted(results.keys())]
    camera_type = "tracking" if len(frames) > 1 else "static"

    if len(frames) > 0:
        panning = self._is_panning(manual_path) if len(frame_annotations) > 1 else False
        if panning:
            camera_type = "tracking"

    print(f"     {camera_type} camera: {len(frames)}/{len(frame_corrs)} frames calibrated")
    return CalibrationResult(shot_id=shot_id, camera_type=camera_type, frames=frames)


def _calibrate_frame_with_continuity(
    self, shot_id: str, frame_idx: int,
    correspondences: dict[str, LandmarkDetection],
    image_shape: tuple[int, int],
    max_err: float, ransac_thresh: float,
    prev_cf: CameraFrame,
    min_cam_height: float, max_cam_height: float,
    focal_tol: float, temporal_max_jump: float,
    player_height_range: tuple[float, float],
) -> CameraFrame | None:
    """Calibrate a frame using temporal continuity from the previous frame."""
    prev_rvec = np.array(prev_cf.rotation_vector, dtype=np.float64)
    prev_tvec = np.array(prev_cf.translation_vector, dtype=np.float64)
    prev_fx = float(prev_cf.intrinsic_matrix[0][0])

    cf = calibrate_frame(
        correspondences, FIFA_LANDMARKS, image_shape,
        frame_idx=frame_idx,
        max_reprojection_error=max_err,
        ransac_reproj_threshold=ransac_thresh,
        min_camera_height=min_cam_height,
        max_camera_height=max_cam_height,
        initial_rvec=prev_rvec,
        initial_tvec=prev_tvec,
        initial_fx=prev_fx,
        focal_length_tolerance=focal_tol,
    )

    if cf is not None and self._passes_temporal_check(cf, [prev_cf], temporal_max_jump):
        return cf

    # Fallback: try without temporal seeding but with player height disambiguation
    cf = self._try_with_player_heights(
        shot_id, frame_idx, correspondences, image_shape,
        max_err, ransac_thresh, min_cam_height, max_cam_height,
        player_height_range,
    )
    if cf is not None and self._passes_temporal_check(cf, [prev_cf], temporal_max_jump):
        return cf

    return None


@staticmethod
def _passes_temporal_check(
    cf: CameraFrame, previous_frames: list[CameraFrame], max_jump: float,
) -> bool:
    """Check if a calibration result is temporally consistent with previous frames."""
    if not previous_frames:
        return True
    from src.utils.camera import camera_world_position
    pos = camera_world_position(
        np.array(cf.rotation_vector), np.array(cf.translation_vector),
    )
    prev_pos = camera_world_position(
        np.array(previous_frames[-1].rotation_vector),
        np.array(previous_frames[-1].translation_vector),
    )
    jump = float(np.linalg.norm(pos - prev_pos))
    if jump > max_jump:
        logging.debug("Rejected frame %d: camera jumped %.1fm (max %.1fm)", cf.frame, jump, max_jump)
        return False
    return True


def _try_with_player_heights(
    self, shot_id: str, frame_idx: int,
    correspondences: dict[str, LandmarkDetection],
    image_shape: tuple[int, int],
    max_err: float, ransac_thresh: float,
    min_cam_height: float, max_cam_height: float,
    player_height_range: tuple[float, float],
) -> CameraFrame | None:
    """Try calibrating with player height scoring as additional disambiguation."""
    bboxes = self._load_track_bboxes(shot_id, frame_idx)
    if not bboxes:
        return None

    # Generate candidates by trying without hard height constraints first
    # (wider range) then scoring with player heights
    cf = calibrate_frame(
        correspondences, FIFA_LANDMARKS, image_shape,
        frame_idx=frame_idx,
        max_reprojection_error=max_err,
        ransac_reproj_threshold=ransac_thresh,
        min_camera_height=min_cam_height,
        max_camera_height=max_cam_height,
    )
    if cf is None:
        return None

    K = np.array(cf.intrinsic_matrix, dtype=np.float64)
    rvec = np.array(cf.rotation_vector, dtype=np.float64)
    tvec = np.array(cf.translation_vector, dtype=np.float64)

    height_score = score_player_heights(bboxes, K, rvec, tvec, player_height_range)
    if height_score >= 0.3:  # at least 30% of players have plausible heights
        return cf

    return None
```

- [ ] **Step 3: Run tests**

Run: `.venv311/bin/python -m pytest tests/test_calibration.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/stages/calibration.py tests/test_calibration.py
git commit -m "feat: temporal continuity and player height disambiguation for panning shots"
```

---

### Task 7: Pipeline reorder and config update

Move tracking before calibration in the stage order, update numeric aliases, and add new calibration config keys.

**Files:**
- Modify: `src/pipeline/runner.py:19-39` (STAGE_ORDER, _ALIASES)
- Modify: `config/default.yaml:22-31` (add new calibration keys)
- Modify: `tests/test_runner.py` (update expected stage ordering)

- [ ] **Step 1: Update `STAGE_ORDER` and `_ALIASES` in `src/pipeline/runner.py`**

```python
STAGE_ORDER: list[tuple[str, type[BaseStage]]] = [
    ("segmentation", ShotSegmentationStage),
    ("tracking", PlayerTrackingStage),
    ("calibration", CameraCalibrationStage),
    ("sync", TemporalSyncStage),
    ("pose", PoseEstimationStage),
    ("triangulation", TriangulationStage),
    ("smpl_fitting", SmplFittingStage),
    ("export", ExportStage),
]

_ALIASES: dict[str, str] = {
    "1": "segmentation",
    "2": "tracking",
    "3": "calibration",
    "4": "sync",
    "5": "pose",
    "6": "triangulation",
    "7": "smpl_fitting",
    "8": "export",
}
```

- [ ] **Step 2: Update `config/default.yaml`**

Replace the calibration section:

```yaml
calibration:
  pitch_model: fifa_standard
  detector_type: hybrid
  manual_landmarks_dir: null
  min_point_confidence: 0.3
  max_reprojection_error: 50.0
  ransac_reproj_threshold: 40.0
  static_ransac_threshold: 80.0
  keyframe_interval: 5
  require_detector: false
  min_camera_height: 3.0
  max_camera_height: 80.0
  player_height_range: [1.5, 2.1]
  temporal_max_jump: 5.0
  focal_length_tolerance: 0.2
```

- [ ] **Step 3: Update `tests/test_runner.py`**

Update expected stage orderings in all tests:

`test_resolve_stages_from` (line 22):
```python
def test_resolve_stages_from():
    names = resolve_stages("all", from_stage="calibration")
    assert names == ["calibration", "sync", "pose", "triangulation", "smpl_fitting", "export"]
```

`test_resolve_stages_explicit` (line 26):
```python
def test_resolve_stages_explicit():
    names = resolve_stages("1,2", from_stage=None)
    assert names == ["segmentation", "tracking"]
```

`test_resolve_stages_from_numeric_alias` (line 31):
```python
def test_resolve_stages_from_numeric_alias():
    names_numeric = resolve_stages("all", from_stage="3")
    names_canonical = resolve_stages("all", from_stage="calibration")
    assert names_numeric == names_canonical
```

`test_aliases_include_stages_3_to_6` (line 45):
```python
def test_aliases_include_stages_3_to_6():
    from src.pipeline.runner import _ALIASES
    assert _ALIASES["2"] == "tracking"
    assert _ALIASES["3"] == "calibration"
    assert _ALIASES["4"] == "sync"
    assert _ALIASES["5"] == "pose"
```

`test_resolve_stages_from_tracking` (line 53):
```python
def test_resolve_stages_from_tracking():
    names = resolve_stages("all", from_stage="tracking")
    assert names == ["tracking", "calibration", "sync", "pose", "triangulation", "smpl_fitting", "export"]
```

`test_resolve_stages_explicit_3_4` (line 58):
```python
def test_resolve_stages_explicit_3_4():
    names = resolve_stages("3,4", from_stage=None)
    assert names == ["calibration", "sync"]
```

`test_aliases_remapped_to_new_order` (line 63):
```python
def test_aliases_remapped_to_new_order():
    from src.pipeline.runner import _ALIASES
    assert _ALIASES["2"] == "tracking"
    assert _ALIASES["3"] == "calibration"
    assert _ALIASES["4"] == "sync"
    assert _ALIASES["5"] == "pose"
```

- [ ] **Step 4: Run all tests**

Run: `.venv311/bin/python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/runner.py config/default.yaml tests/test_runner.py
git commit -m "feat: reorder pipeline — tracking before calibration; add new calibration config keys"
```

---

### Task 8: Delete stale calibration output and update CLAUDE.md

Clean up the existing (now-invalid) calibration output and update documentation to reflect the new pipeline order and landmark naming.

**Files:**
- Modify: `CLAUDE.md` (pipeline table, landmark naming)

- [ ] **Step 1: Delete stale calibration results**

The existing calibration JSON files use old landmark names and have wrong camera positions. Delete them so they don't confuse future runs.

```bash
rm -f output/calibration/origi*_calibration.json
```

- [ ] **Step 2: Update pipeline table in `CLAUDE.md`**

Update the pipeline table to reflect the new ordering:

```markdown
| Stage | Name | Input | Output |
|-------|------|-------|--------|
| 1 | Shot Segmentation | video file | `shots/shots_manifest.json` + per-shot `.mp4` clips |
| 2 | Player Detection & Tracking | shots | `tracks/shot_XXX_tracks.json` |
| 3 | Camera Calibration | shots + tracks | `calibration/shot_XXX_calibration.json` |
| 4 | Temporal Synchronisation | shots + calibration | `sync/sync_map.json` |
| 5 | 2D Pose Estimation | shots + tracks | `poses/shot_XXX_poses.json` |
| 6 | Cross-View Player Matching | tracks + sync | `matching/player_matches.json` |
| 7 | 3D Triangulation | poses + calibration + matches | `triangulated/PXXX_3d_joints.npz` |
| 8 | SMPL Fitting | triangulated joints | `smpl/PXXX_smpl.npz` |
| 9 | Export | SMPL params | `export/fbx/`, `export/gltf/` |
```

Update the "Pitch coordinate system" section:

```markdown
**Pitch coordinate system**: The football pitch is the ground plane (z=0), 105m × 68m (FIFA standard). All 3D positions are in pitch-metres. Near side = y=0 (bottom of broadcast view), far side = y=68 (top of broadcast view). Camera calibration maps pixel space → pitch space.
```

Update the "Keypoint format" section to add:

```markdown
**Landmark naming**: Pitch landmarks use near/far convention (near = y=0, far = y=68). Goal landmarks include crossbar points at z=2.44m. Corner flag tops at z=1.5m.
```

- [ ] **Step 3: Run full test suite one final time**

Run: `.venv311/bin/python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for new pipeline order and landmark naming"
```
