# Camera Calibration Improvements

## Problem

The camera calibration stage produces incorrect camera positions for every shot. All 37 FIFA pitch landmarks lie on the ground plane (z=0), creating a degenerate configuration for PnP that causes:

1. **4-fold pose ambiguity** — solvePnP cannot distinguish between mirrored solutions when all 3D points are coplanar. Every calibration result currently places the camera below the pitch (z < 0).
2. **Frame-to-frame instability** — on panning shots, the solver lands on different ambiguous solutions per frame, causing the camera position to jump wildly (e.g., from (33, 70) to (-18, -14) between adjacent frames).
3. **Focal length instability** — independently guessed focal lengths vary from 881px to 3084px within the same shot.
4. **Inverted y-axis** — the current landmark coordinate system maps `_top` to y=0 and `_bottom` to y=68, which is inverted relative to broadcast convention and causes the pitch to appear flipped in the web preview.

## Design

### 1. Landmark System Overhaul

#### 1a. Fix y-axis convention

Adopt near/far naming where near = y=0 (near side of pitch) and far = y=68 (far side).

All existing `_top` landmarks become `_far` with y=68, all `_bottom` become `_near` with y=0. The current code has these inverted (`_top` at y=0, `_bottom` at y=68), so both names and y-coordinates change:

| Old name | Old y | New name | New y |
|----------|-------|----------|-------|
| `corner_top_left` | 0 | `corner_far_left` | 68 |
| `corner_bottom_left` | 68 | `corner_near_left` | 0 |
| `halfway_top` | 0 | `halfway_far` | 68 |
| `halfway_bottom` | 68 | `halfway_near` | 0 |
| `centre_circle_top` | 24.85 | `centre_circle_far` | 43.15 |
| `centre_circle_bottom` | 43.15 | `centre_circle_near` | 24.85 |

The same pattern applies to all `_top_left`, `_top_right`, `_bottom_left`, `_bottom_right` variants in the 6-yard, 18-yard, and penalty arc landmarks. The `_top` component becomes `_far`, the `_bottom` component becomes `_near`. Y-coordinates swap so that near-side landmarks have lower y values and far-side landmarks have higher y values.

#### 1b. Goal landmarks with 3D structure

Remove the 4 ambiguous goalpost landmarks (`left_goalpost_top`, `left_goalpost_bottom`, `right_goalpost_top`, `right_goalpost_bottom`) and replace with 8 landmarks that capture the full goal structure:

**Left goal (x=0):**
| Name | 3D position |
|------|-------------|
| `left_goal_near_post_base` | (0, 30.34, 0) |
| `left_goal_near_post_top` | (0, 30.34, 2.44) |
| `left_goal_far_post_base` | (0, 37.66, 0) |
| `left_goal_far_post_top` | (0, 37.66, 2.44) |

**Right goal (x=105):**
| Name | 3D position |
|------|-------------|
| `right_goal_near_post_base` | (105, 30.34, 0) |
| `right_goal_near_post_top` | (105, 30.34, 2.44) |
| `right_goal_far_post_base` | (105, 37.66, 0) |
| `right_goal_far_post_top` | (105, 37.66, 2.44) |

The crossbar top landmarks at z=2.44 break the coplanarity degeneracy and directly resolve the PnP sign ambiguity.

#### 1c. Corner flag landmarks

Add 4 corner flag top landmarks at z=1.5 (standard corner flag height):

| Name | 3D position |
|------|-------------|
| `corner_near_left_flag_top` | (0, 0, 1.5) |
| `corner_near_right_flag_top` | (105, 0, 1.5) |
| `corner_far_left_flag_top` | (0, 68, 1.5) |
| `corner_far_right_flag_top` | (105, 68, 1.5) |

These provide secondary off-plane constraints in wide shots where goalposts are not visible.

#### 1d. Manual re-annotation

Existing manual landmark annotations will be discarded and re-annotated from scratch using the new naming scheme. No migration utility needed.

### 2. Pipeline Reorder

Move tracking before calibration so that bounding box data is available for player height disambiguation:

| Stage | Name | Input |
|-------|------|-------|
| 1 | Shot Segmentation | video |
| 2 | Player Detection & Tracking | shots |
| 3 | Camera Calibration | shots + tracks |
| 4 | Temporal Sync | shots + calibration |
| 5 | 2D Pose Estimation | shots + tracks |
| 6 | Cross-View Player Matching | tracks + sync |
| 7 | 3D Triangulation | poses + calibration + matches |
| 8 | SMPL Fitting | triangulated joints |
| 9 | Export | SMPL params |

### 3. PnP Solver Improvements

#### 3a. Hard constraints

After solving PnP, compute the camera world position `C = -R^T @ t` and reject any solution where:
- Camera height `C[2] < min_camera_height` (default 3.0m) — cameras are always above the pitch
- Camera optical axis is not pointing downward — check that the z-column of the rotation matrix has a negative z-component in world coordinates

#### 3b. Multi-solution candidate generation

Replace the current single-best approach. For each focal length candidate:
1. Run `cv2.SOLVEPNP_IPPE` (explicitly returns both planar ambiguity solutions)
2. Run `cv2.SOLVEPNP_P3P` and `cv2.SOLVEPNP_EPNP` to generate additional candidates
3. Score all valid candidates (those passing hard constraints) by:
   - Reprojection error (lower is better)
   - Camera height plausibility — soft preference for the `min_camera_height` to `max_camera_height` range (default 3–80m)
4. Select the highest-scoring candidate

#### 3c. Player height disambiguation

When only coplanar landmarks are available and scoring from 3b doesn't produce a confident winner:

1. Read player bounding boxes from the tracking stage output (`tracks/shot_XXX_tracks.json`)
2. For each PnP candidate solution, project foot (bottom-centre of bbox) and head (top-centre of bbox) points through the candidate's camera model
3. Compute the implied 3D height difference between head and foot for each player
4. Score the candidate by how many players have plausible heights (configurable, default 1.5–2.1m)
5. A solution producing ~1.8m players is preferred; one producing 0.3m or 5m players is rejected

### 4. Temporal Continuity for Panning Shots

#### 4a. Processing order

For panning shots with multiple annotated frames, start from the frame with the most landmark correspondences (best chance of correct initial solve), then propagate forward and backward chronologically.

#### 4b. Extrinsic seeding

After successfully calibrating frame N, pass its rvec/tvec as `useExtrinsicGuess=True` to frame N+1's solvePnP. This biases the solver toward temporally consistent solutions.

#### 4c. Jump rejection

If the camera world position for frame N+1 is more than `temporal_max_jump` (default 5m) from frame N, reject the solution and carry forward frame N's calibration. A panning camera rotates but stays in roughly the same world position.

#### 4d. Focal length continuity

After the first successfully calibrated frame in a shot, constrain subsequent frames to use the same focal length within `focal_length_tolerance` (default ± 20%). This prevents the solver from jumping between 881px and 3084px within the same shot.

### 5. Configuration

New keys in `config/default.yaml` under `calibration:`:

```yaml
calibration:
  # existing keys unchanged...
  min_camera_height: 3.0          # hard reject below this (metres)
  max_camera_height: 80.0         # soft penalty above this (metres)
  player_height_range: [1.5, 2.1] # plausible player height for disambiguation (metres)
  temporal_max_jump: 5.0          # max camera position change between frames (metres)
  focal_length_tolerance: 0.2     # max focal length change within a shot (fraction)
```

### 6. What Does Not Change

- **Output schema:** `CalibrationResult` and `CameraFrame` dataclasses remain unchanged — same fields, better values.
- **Bundle adjustment:** `bundle_adjust.py` continues to work as-is on the improved calibration output.
- **Downstream stages:** Triangulation, SMPL fitting, and export consume the same calibration format.
- **Heuristic detector:** Not modified in this change. Can be improved separately.
