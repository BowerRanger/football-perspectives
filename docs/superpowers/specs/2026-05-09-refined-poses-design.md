# Refined Poses Stage — Cross-Shot Fusion of Player Animations

**Date:** 2026-05-09
**Status:** Draft (auto-mode; user to review)
**Scope:** New pipeline stage (`refined_poses`) that fuses each player's per-shot SMPL reconstructions (one per shot the player appears in) into a single per-player track on the shared reference timeline. Adds robust per-frame outlier rejection across views, post-fusion temporal smoothing, and per-player diagnostics surfaced via the quality report. Updates `export` to consume fused tracks.

## 1. Background

The multi-shot plumbing spec (`2026-05-08-multi-shot-plumbing-design.md`) explicitly deferred cross-shot HMR fusion to a follow-up. After that work, each shot's reconstruction is independent: `output/hmr_world/{shot_id}__{player_id}_smpl_world.npz` holds one SMPL track per `(shot_id, player_id)` pair, and `export` emits one GLB per shot using only that shot's tracks.

When the same physical player appears in multiple shots (annotated with the same `player_id` during tracking), the per-shot SMPL reconstructions are noisy estimates of the same underlying motion, with errors driven by:
- monocular HMR depth ambiguity that varies with the player's screen-space size and camera angle,
- camera-track residuals (per-shot, per-frame) propagating into pitch-coordinate errors via the foot anchor,
- brief occlusions / ByteTrack ID flickers that GVHMR's confidence captures imperfectly.

A second view (or third, fourth) of the same player at the same wall-clock instant gives independent observations that can be fused into a more accurate single estimate. This stage does that fusion, plus post-fusion smoothing and per-frame disagreement diagnostics.

## 2. Goals

1. One fused `RefinedPose` per `player_id` covering every reference-timeline frame any contributing shot reconstructed.
2. Confidence-weighted per-frame fusion of `root_t`, `root_R`, and `thetas`, with robust outlier rejection (drop a view's contribution to a frame when it disagrees with the median by more than a configurable threshold).
3. Single shared `betas` per player, fused from all contributing shots' betas.
4. Post-fusion temporal smoothing — Savitzky-Golay on `root_t`, lie-algebra Savgol on rotations.
5. Per-player diagnostics JSON listing per-frame contributing shots, disagreement metrics, dropped views, and frame-level flags (low coverage, high disagreement). Aggregated counters surfaced via `quality_report.json`.
6. `export` consumes fused tracks: each shot's GLB still emits, but each player's pose at shot-local frame `f` comes from the fused track at reference frame `f - sync_map.offset_for(shot_id)`.
7. The stage degrades gracefully: a player seen in only one shot, a shot missing from `sync_map`, or unannotated tracks (track-id-only pids) all pass through without erroring.

### Non-goals (this work)

- Re-solving SMPL from 2D keypoints with multi-view bundle adjustment. (Architectural option B vs C earlier in design.)
- Sync-map refinement. `sync_map.json` is treated as authoritative.
- Auto-matching across shots. Cross-shot identity is the user-assigned `player_id` annotation only; no automatic re-id.
- Kalman / physics-model gap-filling across all-views-blind frames. Frames where no shot saw the player are dropped from the fused track, not interpolated.
- Rebuilding `output/matching/player_matches.json`. That artefact is dead — the pipeline does not write or read it.
- New dashboard pages. Surface counters via the existing Multi-shot status panel only.

## 3. Architecture

### 3.1 Pipeline placement

`refined_poses` is inserted between `ball` and `export` in `_STAGE_NAMES`:

```
prepare_shots → tracking → camera → hmr_world → ball → refined_poses → export
```

Rationale: `refined_poses` consumes only `hmr_world` outputs + `sync_map.json` + per-shot camera tracks (the last only as a future hook for camera-confidence weighting; not used in v1). `ball` doesn't depend on it. Putting `refined_poses` last among per-player processing makes it the single point where per-shot SMPL becomes per-player fused.

### 3.2 File layout

```
output/
├── hmr_world/                                    (unchanged — intermediate)
│   ├── {shot_id}__{player_id}_smpl_world.npz
│   └── {shot_id}__{player_id}_kp2d.json
└── refined_poses/                                (new)
    ├── {player_id}_refined.npz                   # RefinedPose track on the reference timeline
    ├── {player_id}_diagnostics.json              # per-frame diagnostics
    └── refined_poses_summary.json                # aggregate counters; folded into quality_report
```

### 3.3 Identity model

Cross-shot fusion is keyed solely on the `player_id` annotation that the operator assigns during tracking. Two NPZs `origi01__P002_smpl_world.npz` and `origi02__P002_smpl_world.npz` are by definition the same player. A track without an annotation gets the fallback pid `{shot_id}_T{track_id}` from `hmr_world`, so its NPZ filename starts with the shot id and never collides with another shot's fallback. Such pids are treated as their own single-shot players — passthrough + smoothing only, no fusion possible.

### 3.4 Reference timeline

Each contributing shot has a `frame_offset` in `sync_map.json` (sign convention: `frame_offset = matched_frame_in_this_shot − matched_frame_in_reference`). To convert a shot-local frame `f_local` to the reference frame `f_ref`:

```
f_ref = f_local − offset
```

A player's fused track spans `[min(f_ref) … max(f_ref)]` across all contributing shots. Reference frames where no contributing shot reconstructed the player are dropped (`frames` is sparse, not zero-filled).

### 3.5 Sync trust

`sync_map.json` is treated as authoritative. If a shot is missing from `sync_map.alignments`, the stage logs a warning and treats its offset as 0. The diagnostics file flags such shots so the operator can fix the sync map.

### 3.6 Re-run semantics (`is_complete`)

`is_complete` returns `True` iff for every distinct `player_id` discoverable by globbing `output/hmr_world/*_smpl_world.npz`, a `output/refined_poses/{player_id}_refined.npz` exists. Adding a shot mid-project re-runs `hmr_world` for the new shot, then `refined_poses` re-fuses every player whose contributing-shot set changed. (For v1 simplicity, a re-run of `refined_poses` re-fuses every player; the per-player check is the cache-skip granularity, not the per-frame work.)

The runner already supports `--from-stage refined_poses` and `--stages refined_poses` via the resolver.

## 4. Components

### 4.1 `src/schemas/refined_pose.py` (new, ~120 lines)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np


@dataclass(frozen=True)
class RefinedPose:
    """Per-player SMPL track on the shared reference timeline.

    Pitch-world frame matches SmplWorldTrack: z-up, x along nearside
    touchline, y toward far side, root_t in pitch metres.
    """
    player_id: str
    frames: np.ndarray              # (N,) reference-timeline frame indices
    betas: np.ndarray               # (10,) shared across the whole track
    thetas: np.ndarray              # (N, 24, 3) axis-angle
    root_R: np.ndarray              # (N, 3, 3)
    root_t: np.ndarray              # (N, 3) pitch metres
    confidence: np.ndarray          # (N,) fused confidence (sum of contributing weights)
    view_count: np.ndarray          # (N,) int — how many shots contributed to this frame
    contributing_shots: tuple[str, ...]   # union across all frames

    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "RefinedPose": ...


@dataclass(frozen=True)
class FrameDiagnostic:
    frame: int                          # reference-timeline frame index
    contributing_shots: tuple[str, ...] # shots that retained their contribution at this frame
    dropped_shots: tuple[str, ...]      # shots whose view was dropped by the outlier check
    pos_disagreement_m: float           # max pairwise L2 distance among contributing root_t
    rot_disagreement_rad: float         # max pairwise geodesic distance among contributing root_R
    low_coverage: bool                  # view_count < min_contributing_views (after drops)
    high_disagreement: bool             # pos_disagreement_m or rot_disagreement_rad above thresholds


@dataclass(frozen=True)
class RefinedPoseDiagnostics:
    player_id: str
    frames: tuple[FrameDiagnostic, ...]
    contributing_shots: tuple[str, ...]
    summary: dict                       # counts: total_frames, single_view_frames, high_disagreement_frames

    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "RefinedPoseDiagnostics": ...
```

Persistence: `RefinedPose` saves as NPZ (same pattern as `SmplWorldTrack`); `RefinedPoseDiagnostics` saves as JSON.

### 4.2 `src/utils/pose_fusion.py` (new, ~250 lines)

Pure math, no I/O. Tested in isolation.

```python
def so3_chordal_mean(rotations: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted chordal mean of SO(3) matrices (project weighted-average to SO(3))."""

def so3_geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angular distance in radians via arccos((trace(R1.T @ R2) - 1) / 2)."""

def robust_translation_fuse(
    positions: np.ndarray,         # (V, 3)
    weights: np.ndarray,           # (V,)
    k_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted geometric-median fuse with MAD-based outlier drop.

    Returns (fused_position, kept_mask). With ≤2 views, no outlier drop —
    returns weighted mean and all-True mask.
    """

def robust_rotation_fuse(
    rotations: np.ndarray,         # (V, 3, 3)
    weights: np.ndarray,           # (V,)
    k_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """SO(3) chordal mean with geodesic-distance outlier drop. Same shape contract."""

def savgol_translations(
    positions: np.ndarray,         # (N, 3)
    window: int,
    polyorder: int,
) -> np.ndarray:
    """Savitzky-Golay along axis 0 per-component. Window clamped to len(N) odd."""

def savgol_rotations_lie(
    rotations: np.ndarray,         # (N, 3, 3)
    window: int,
    polyorder: int,
) -> np.ndarray:
    """log() to (N, 3) axis-angle, Savgol per-component, exp() back to SO(3).

    Per-frame log assumes consecutive rotations stay in the same sheet
    of the cover. For football-clip joint rates this holds; the unit
    test asserts it for a 30 fps sprint sample and warns otherwise.
    """
```

### 4.3 `src/stages/refined_poses.py` (new, ~300 lines)

```python
class RefinedPosesStage(BaseStage):
    name = "refined_poses"

    def is_complete(self) -> bool:
        # True iff every player_id in hmr_world has a matching refined NPZ.
        ...

    def run(self) -> None:
        sync_map = SyncMap.load(self.output_dir / "shots" / "sync_map.json")
        # 1. discover (shot_id, player_id) pairs from hmr_world/
        # 2. group by player_id
        # 3. for each player_id: fuse, smooth, write outputs + diagnostics
        # 4. write refined_poses_summary.json
        ...

    def _fuse_player(
        self,
        player_id: str,
        contributions: list[tuple[str, SmplWorldTrack]],   # (shot_id, track)
        sync_map: SyncMap,
        cfg: dict,
    ) -> tuple[RefinedPose, RefinedPoseDiagnostics]:
        # a. shift each track's frames onto the reference timeline
        # b. compute the union of reference frames
        # c. for each ref frame, gather contributing views and run robust fuse
        # d. fuse betas (confidence-weighted mean across all contributing tracks)
        # e. apply Savgol smoothing on root_t and (optionally) rotations
        # f. populate diagnostics
        ...
```

### 4.4 Updates to existing files

**`src/pipeline/runner.py`**
- Add `"refined_poses"` to `_STAGE_NAMES` between `"ball"` and `"export"`.
- Add the corresponding lazy import branch in `_stage_class()`.

**`src/stages/export.py`**
- For each shot in the manifest, replace the existing per-shot SmplWorldTrack list with: load every `output/refined_poses/*_refined.npz`, and for each track, slice the frames within `[shot.start_frame_local … shot.end_frame_local]` after reverse-shifting `f_ref → f_local = f_ref + offset`. Skip players with no overlap with the shot.
- Backwards-compat: if `output/refined_poses/` is empty (e.g. user runs `--stages export` after `hmr_world` without `refined_poses`), fall back to the legacy per-shot SmplWorldTrack path with a one-line info log.

**`src/pipeline/quality_report.py`**
- Add a `refined_poses` section that reads `output/refined_poses/refined_poses_summary.json` (counts of players refined, total frames, single-view frames, high-disagreement frames, shots missing from sync map).

**`src/web/static/index.html` (Multi-shot status panel)**
- Add a column or sub-row showing the per-player refined-poses status (✓ if `_refined.npz` exists; flag count for high-disagreement frames, sourced from quality_report). v1 keeps the change minimal — no new page, no new endpoint.

**`config/default.yaml`**

```yaml
refined_poses:
  # Outlier rejection (per frame, per view)
  outlier_k_sigma: 3.0
  min_contributing_views: 1   # frames with fewer kept views are flagged but not dropped
  high_disagreement_pos_m: 0.5
  high_disagreement_rot_rad: 0.5
  # Smoothing
  savgol_window: 7
  savgol_poly: 3
  smooth_rotations: true
  # Beta fusion
  beta_aggregation: weighted_mean   # alternatives kept simple for v1
```

## 5. Data flow (per player)

```
1. Load SyncMap once (stage-level).
2. For each player_id:
   a. Collect contributing tracks: list[(shot_id, SmplWorldTrack)].
   b. Map each track's frames onto the reference timeline:
        ref_frames = local_frames − sync_map.offset_for(shot_id)
   c. Build the union of reference frames across all contributors.
   d. For each ref frame f in the union:
        - Gather views (per-shot) that have an entry at this f_ref.
        - If 0: skip (unreachable in normal cases — union construction).
        - If 1: passthrough; view_count=1; no outlier drop.
        - If 2: weighted mean (no outlier drop possible with V<3).
        - If ≥3:
            * robust_translation_fuse(positions, weights, k_sigma) → (fused_t, kept_mask)
            * so3_chordal_mean(root_R[kept_mask], weights[kept_mask]) → fused_root_R
              (rotations inherit the position-derived kept_mask; no further outlier check on rotations.
               Rationale: pitch position is the dominant cue and the most reliable outlier signal —
               adding a second rotational outlier check on top would over-trim and hurt rotation
               recovery in fast-rotation frames.)
            * for each joint j: thetas_per_view → SO(3) via Rodrigues → so3_chordal_mean over
              kept_mask views → log map back to axis-angle → fused theta[j]
        - Compute pos_disagreement_m and rot_disagreement_rad (max pairwise among kept views).
        - Flag low_coverage if kept-view count < min_contributing_views.
        - Flag high_disagreement if either disagreement exceeds its threshold.
   e. Fuse betas: confidence-weighted mean across all contributing tracks' betas, single (10,) for the whole track. (Justification: betas should be invariant per player; weighted mean is a defensible v1, with median as a future option.)
   f. Apply Savitzky-Golay to root_t (savgol_window, savgol_poly) along the time axis.
   g. If smooth_rotations: apply lie-algebra Savgol to root_R and per-joint rotations.
   h. Build RefinedPose, write {player_id}_refined.npz.
   i. Build RefinedPoseDiagnostics with summary counters, write {player_id}_diagnostics.json.
3. Aggregate per-player summaries into refined_poses_summary.json.
```

### 5.1 SO(3) details

- **Chordal mean**: `R̂ = U V^T` where `U Σ V^T = SVD(Σ_v w_v R_v)`; sign-flip if `det < 0`.
- **Per-joint thetas**: convert axis-angle → rotation matrix via Rodrigues, fuse on SO(3), convert back via log map.
- **Outlier drop on rotations**: compute candidate mean, then geodesic distances; drop views above `k_sigma * MAD(distances)`. Recompute mean once. (Single iteration is enough for the typical 2–6 view case; not worth full IRLS.)

### 5.2 Joint-rotation outlier handling

For v1, joints inherit the kept_mask from the root rotation outlier check (one mask per frame). Per-joint outlier checks are deferred — the root is the dominant cue and per-joint rejection adds 24× the cost with marginal benefit on broadcast clips. Flagged in the diagnostic summary as a known limitation.

### 5.3 Confidence weighting

`SmplWorldTrack.confidence` per frame is used directly as the per-view weight. Views whose weight at a frame is zero are excluded from that frame's contribution list before fusion (treated as a non-contribution, not as a zero-position outlier). Camera-track confidence and per-keypoint kp2d confidence are not consumed in v1; the design leaves space for them in the weight calculation but the v1 implementation passes only `track.confidence`.

## 6. Edge cases and error handling

| Case | Behaviour |
|---|---|
| Player in only one shot | Passthrough fuse (view_count=1 throughout); smoothing applied; written as RefinedPose. |
| Player with unannotated pid (`{shot_id}_T{track_id}` fallback) | Treated as a single-shot player; its NPZ filename is `{shot_id}_T{track_id}_refined.npz`. |
| Shot missing from sync_map | Warn, treat offset as 0; record in diagnostics under `shots_missing_sync` summary key. |
| All contributing views drop out at a frame | Frame removed from fused track entirely (no Kalman fill in v1). |
| Two views, both flagged as outliers | With V=2, no outlier drop runs — both kept and weighted-meaned. Disagreement is reported via diagnostics. |
| `output/refined_poses/` empty when running export | Export falls back to legacy per-shot SmplWorldTracks with an info log. |
| Player NPZs differ in `betas` substantially | Weighted-mean still computed; if max pairwise beta L2 > beta_disagreement_warn (config default 0.3), warn and surface in summary. |
| Single contributing track has length < savgol_window | Savgol window auto-clamps to the largest odd value ≤ track length; if length < polyorder + 2, smoothing is skipped for that track. |
| `sync_map.json` missing entirely | Warn; treat all offsets as 0 (single-shot reconstructions still produce valid refined NPZs). |

## 7. Validation and testing

### 7.1 Unit tests

| Test | Module | Asserts |
|---|---|---|
| `test_so3_chordal_mean_rotational_invariant` | `tests/test_pose_fusion.py` (new) | Mean of a rotation and its conjugate equals identity (within float eps); mean of a rotation with itself equals itself. |
| `test_so3_geodesic_distance_known_angles` | same | Distance between identity and a 90° z-rotation is π/2. |
| `test_robust_translation_fuse_drops_outlier` | same | Three positions where one is 10 m offset → kept_mask flags it out; fused position equals weighted mean of the other two. |
| `test_robust_rotation_fuse_drops_outlier` | same | Same idea for rotations. |
| `test_robust_fuse_passthrough_when_v_lt_3` | same | V=2 returns weighted mean and all-kept mask. |
| `test_savgol_translations_smooths_noise` | same | Synthetic noisy parabola → smoothed RMS error vs ground truth lower than raw. |
| `test_savgol_rotations_lie_recovers_known_motion` | same | Slerp-interpolated rotation sequence with added noise → smoothed result close to ground truth. |
| `test_refined_pose_round_trip` | `tests/test_refined_pose_schema.py` (new) | Save/load preserves all fields and dtypes. |
| `test_refined_pose_diagnostics_round_trip` | same | JSON save/load preserves frame list and summary. |

### 7.2 Stage tests

| Test | Module | Asserts |
|---|---|---|
| `test_refined_poses_single_shot_passthrough` | `tests/test_refined_poses_stage.py` (new) | Player in only one shot → fused frames equal raw frames after smoothing; view_count all 1. |
| `test_refined_poses_two_shots_fuses_root_t` | same | Two shots agree on root_t to within noise → fused root_t closer to ground truth than either input. |
| `test_refined_poses_outlier_view_dropped` | same | Three shots, one with a 5 m offset → diagnostics report that shot as dropped at the affected frames; fused root_t matches the agreeing two. |
| `test_refined_poses_handles_missing_sync` | same | Sync map missing one shot → that shot's offset is 0; warning logged; refined NPZ still written. |
| `test_refined_poses_unannotated_pid_passthrough` | same | A `{shot_id}_T{track_id}` pid is treated as a single-shot player; refined NPZ filename matches the pid. |
| `test_refined_poses_high_disagreement_flagged` | same | Two shots disagree on root_t by 1 m → diagnostics flags `high_disagreement` for those frames. |
| `test_refined_poses_is_complete` | same | After running on a 2-player, 2-shot fixture, `is_complete()` returns True; deleting one refined NPZ flips it to False. |
| `test_export_consumes_refined_poses` | `tests/test_export_stage.py` (existing) | When `output/refined_poses/` exists, the per-shot GLB pulls the player's pose from the fused track at `f_local − offset`, not from the raw hmr_world NPZ. |
| `test_export_fallback_to_hmr_world` | same | When `output/refined_poses/` is empty, export falls back to the legacy SmplWorldTrack path and emits per-shot GLBs as before. |
| `test_runner_includes_refined_poses_stage` | `tests/test_runner.py` (existing) | `_STAGE_NAMES` contains `"refined_poses"` between `"ball"` and `"export"`; `--from-stage refined_poses` resolves correctly. |

### 7.3 Integration test

End-to-end on a synthetic two-shot fixture (re-uses the existing fixture from the multi-shot plumbing tests, with a player annotated in both shots). Asserts:
- `output/refined_poses/{player_id}_refined.npz` exists.
- The fused `root_t` for a frame visible in both shots is the weighted mean of the two raw `root_t` values within 1 mm tolerance after smoothing is bypassed (test runs with `savgol_window: 1`).
- The two per-shot GLBs both reflect the fused pose at their respective local frames.
- `quality_report.json` contains a `refined_poses` section with the expected counters.

## 8. Quality report integration

`output/quality_report.json` gains a `refined_poses` key:

```json
"refined_poses": {
  "players_refined": 23,
  "single_shot_players": 4,
  "multi_shot_players": 19,
  "total_fused_frames": 4821,
  "single_view_frames": 612,
  "high_disagreement_frames": 17,
  "shots_missing_sync": [],
  "beta_disagreement_warnings": []
}
```

The Multi-shot status panel in the dashboard reads these counters to show "Refined N/M players, K flagged frames" alongside the existing per-shot status.

## 9. Configuration summary

```yaml
refined_poses:
  outlier_k_sigma: 3.0
  min_contributing_views: 1
  high_disagreement_pos_m: 0.5
  high_disagreement_rot_rad: 0.5
  savgol_window: 7
  savgol_poly: 3
  smooth_rotations: true
  beta_aggregation: weighted_mean
  beta_disagreement_warn: 0.3
```

## 10. Open questions / decisions logged

1. **Per-joint outlier drop**: deferred (uses root's kept_mask). Revisit if joint-jitter complaints arise.
2. **Camera-confidence weighting**: deferred. The code path leaves a hook so the weight expression can be extended without touching the fusion math.
3. **Beta fusion strategy**: weighted_mean for v1; median is a one-line alternative if outliers in betas become a problem.
4. **Kalman fill across all-views-blind frames**: out of scope (option C from brainstorming). Fused track stays sparse where no shot saw the player.
5. **Within-shot fragmented tracks** (same player_id, multiple ByteTrack ids in one shot): not handled at this layer — `hmr_world` already produces one NPZ per `(shot_id, player_id)` so this stage sees one track per shot per player. If `hmr_world` ever changes that contract, this stage would need a per-shot pre-merge.
