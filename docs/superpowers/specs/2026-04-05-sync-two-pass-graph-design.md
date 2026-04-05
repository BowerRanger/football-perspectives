# Sync Stage: Two-Pass Graph Alignment Design

**Date:** 2026-04-05  
**Stage:** Stage 3 — Temporal Synchronisation (`src/stages/sync.py`)

---

## Problem

The current sync stage uses a **star topology**: every non-reference clip is compared only to `shots[0]`. This breaks down when a clip shares little visible content with the reference (e.g. a close-up replay showing 4–6 players while the reference wide shot shows 12+). Aggregate signals fire on different player populations and produce wrong or low-confidence offsets.

By comparing non-reference clips **against each other** as well, we create an over-determined offset graph. Even if clip A is hard to sync to the reference, it may sync cleanly to clip B — and that edge, combined with B's reference alignment, constrains A's final offset.

---

## Architecture

Three phases inside `TemporalSyncStage.run()`:

```
Pass 1 (existing)
  for each non-reference clip i:
    run signal stack (audio → celebration → reid → fallback) vs reference
    → AlignmentEstimate(offset_i, confidence_i)

Pass 2 (new)
  for each pair (i, j) of non-reference clips:
    run signal stack with narrowed search window
      window = [expected ± pass2_search_margin_frames]
      expected = offset_j_pass1 − offset_i_pass1
      if either pass-1 confidence < min_confidence → full search
    → AlignmentEstimate(offset_ij, confidence_ij)

Graph solve (_solve_offset_graph)
  inputs: all pass-1 and pass-2 estimates as weighted edges
  output: globally consistent integer offset per clip
  method: weighted least-squares (numpy.linalg.lstsq)

Assembly
  final Alignment objects from solved offsets + method tags + residuals
```

---

## Solver: `_solve_offset_graph()`

**Variables:** `x = [o_1, ..., o_n]` — offset of each non-reference clip relative to the reference. Reference is pinned at 0 (not a variable).

**Sign convention:** `frame_in_reference = frame_in_shot + offset`, so for a pair `(i, j)` where `i` is the "reference" side: `o_j − o_i = e_ij`.

**Linear system:** Each edge becomes one row in matrix `A` with weight `w`:

| Edge type | Row in A | rhs |
|-----------|----------|-----|
| reference → clip_i | `A[k, i] = +1` | `e_0i` |
| clip_i → clip_j | `A[k, i] = −1`, `A[k, j] = +1` | `e_ij` |

**Solve:**

```python
x, _, _, _ = np.linalg.lstsq(
    np.sqrt(W)[:, None] * A,
    np.sqrt(W) * b,
    rcond=None,
)
offsets = np.round(x).astype(int)
```

`W = diag(confidences)`. Zero-weight edges are ignored. Rank-deficient cases (isolated clips with all-zero-confidence edges) return the minimum-norm solution (0.0) for that clip's variable.

**Residuals:** After solving, `graph_residual_frames_i = |solved_offset_i − pass1_offset_i|`. Written to `Alignment.graph_residual_frames`. Clips with residual above `sync.graph_residual_flag_threshold_frames` (default 15) get `method` tagged `"graph_refined"` to surface significant corrections for inspection.

**Confidence propagation:**

```
confidence_final = confidence_pass1 * α + mean(incident_edge_confidences) * (1 − α)
```

Default `α = 0.4` (configurable as `sync.graph_confidence_alpha`).

---

## Pass 2 Signal Stack

Runs the full existing signal stack — audio → celebration → reid → visual/formation fallback — in the same priority order as pass 1. The "reference" side of each pair is whichever clip has the lower shot index (canonical ordering).

**Search window constraint** applied per signal:

- `_cross_correlate_signals()` — already has `max_lag`; formation and celebration cross-correlation go through this.
- `_align_audio()` — add optional `max_lag_frames` param that clips the lag array before peak-picking.
- `_align_visual()` / `_align_formation_spatial()` — pass `max_lag` to `_visual_similarity_profile()` to skip out-of-window offsets.
- `_align_player_reid()` — goes through `_correlate_with_speed_sweep()` → `_cross_correlate_signals()`, already windowed.
- `_align_by_matched_pitch_trajectories()` — already takes `coarse_offset`; pass expected relative offset.

Falls back to full search (no window) when either pass-1 confidence is below `sync.min_confidence`.

---

## New Functions

```python
def _collect_pairwise_estimates(
    shots: list[Shot],
    clips: dict[str, Path],
    tracks_by_shot: dict[str, TracksResult],
    n_frames_by_shot: dict[str, int],
    pass1_offsets: dict[str, AlignmentEstimate],
    fps: float,
    cfg: dict,
) -> list[tuple[int, int, AlignmentEstimate]]:
    """Returns list of (i, j, estimate) for all non-reference pairs.
    i and j are indices into shots list (0 = reference)."""

def _solve_offset_graph(
    n_clips: int,
    edges: list[tuple[int, int, float, float]],  # (i, j, offset, weight)
) -> np.ndarray:
    """Weighted least-squares solve. Returns float array length n_clips.
    Index 0 is the reference (always 0.0). Indices 1..n are non-reference clips."""
```

`run()` calls pass 1 (existing loop), then `_collect_pairwise_estimates`, then builds the combined edge list, then `_solve_offset_graph`, then assembles `Alignment` objects. For clips where the graph solved to 0.0 (isolated — all incident edges had zero confidence), the assembly step falls back to the pass-1 offset and method unchanged.

---

## Schema Changes

**`src/schemas/sync_map.py`** — add one optional field to `Alignment`:

```python
@dataclass
class Alignment:
    shot_id: str
    frame_offset: int
    confidence: float
    method: str
    overlap_frames: list[int]
    graph_residual_frames: float | None = None
```

Backwards-compatible: existing `sync_map.json` files load fine (field defaults to `None`).

**`config/default.yaml`** — add two keys under `sync:`:

```yaml
sync:
  pass2_search_margin_frames: 30
  graph_confidence_alpha: 0.4
  graph_residual_flag_threshold_frames: 15
```

No changes to `ShotsManifest`, `TracksResult`, or any downstream stage schemas.

---

## Out of Scope

- Changing which signals are used (same stack, no new signals).
- Multi-reference or iterative refinement beyond two passes.
- Surfacing residuals in the quality report (follow-on work).
