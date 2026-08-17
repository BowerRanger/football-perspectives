# Ball W2: Touch Geometry + Evidence-Gated Auto Events — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans.
> Executor: same autonomous session that wrote it.

**Goal:** Fix baseline findings 2 and 4 — (W2a) touch keyframes can resolve
below the pitch; (W2b) phantom auto touches born from synthetic pixel
evidence become hard body-pinned keyframes metres off the true path.

**Spec:** `docs/superpowers/specs/2026-08-17-ball-sub20cm-accuracy-design.md`
(§5, updated by the baseline's finding 2 — see
`docs/superpowers/notes/ball-accuracy/2026-08-17-baseline.md`).

**Architecture:** W2a lives in `ball_event_resolver._resolve_touch_world`
(events mode only; piecewise stays frozen): clamp resolved touches to
z ≥ ball_radius, preferring the on-ray solution (ray ∩ z=r) when a pixel
exists. W2b lives in `ball_auto_anchor`: touch/bounce/goal event candidates
require ≥1 frame of HARD detector evidence (`detector`/`second_pass`/
`foot_guided`) within a window; grounded sampling additionally accepts
`bridge` (real on-image template evidence). Both changes are config-gated,
default ON, and validated by the eval harness + gberch touch-recall parity.

## Global Constraints

- `.venv311/bin/python` for everything; tests marked `unit` unless data-bound.
- Count-preserving `touch_attribution` untouched; operator anchors always win.
- Auto sidecar keeps being written (authoring value); gating changes what is
  *minted*, with dropped candidates logged and counted in diagnostics.
- Scoped commits only (unrelated user edits exist in the tree).

### Task 1: W2a ground clamp in the event resolver

**Files:** Modify `src/utils/ball_event_resolver.py`; test
`tests/test_ball_event_resolver.py` (append).

**Interfaces:** `_resolve_touch_world` keeps its signature; returns a world
whose z ≥ ball_radius. `resolve_events` diagnostics gains
`"touch_ground_clamped": int`.

Steps: failing tests (joint below ground + pixel → resolved point on the
clicked ray at z == ball_radius; no-pixel → joint xy kept, z == ball_radius;
above-ground touch unchanged; diagnostics counter increments) → implement
(pixel branch: if resolved z < r, replace with `ray_plane_z`-style ray ∩ z=r
when a forward intersection exists, else vertical lift; no-pixel branch:
vertical lift) → green → commit.

### Task 2: W2b evidence gate in auto-anchor generation

**Files:** Modify `src/utils/ball_auto_anchor.py` (AutoAnchorCfg fields +
`_event_candidates` + `_grounded_candidates` + `generate_auto_anchors`
pass-through), `src/stages/ball.py` (cfg parsing), `config/default.yaml`
(keys + comments); test `tests/test_ball_auto_anchor.py` (append).

**Interfaces:** `AutoAnchorCfg` gains
`require_event_evidence: bool = True`,
`event_evidence_window: int = 3`,
`event_evidence_sources: tuple = ("detector", "second_pass", "foot_guided")`,
`grounded_evidence_sources: tuple = (..., "bridge")`.
`_event_candidates(..., sources)` and `_grounded_candidates(..., sources)`
take the per-frame source map (None → gate inert, legacy behaviour).
`generate_auto_anchors` return unchanged; diagnostics not touched here (the
stage logs counts).

Steps: failing tests (touch event whose ±window sources are only
`anchor`/`bridge`/absent → not minted; same event with one `detector` frame
in window → minted; `require_event_evidence=False` restores legacy; grounded
sample on a `bridge` frame minted, on a source-less gap-fill frame not;
sources=None keeps legacy) → implement → green → config keys + stage parsing
→ scoped `tests/test_auto_anchor_*.py tests/test_ball_auto_anchor.py` run →
commit.

### Task 3: Re-measure + recall parity

Run the four noop hold-out evals + the wasb gberch hold-out; expect phantom
outliers gone (gberch fold-1 max 8.40 → ≤ pure-interp 6.00) with p50 not
worse than pure-interp; gberch wasb auto-touch set before/after compared for
recall parity (foot_guided/kinematic touches near detections must survive).
Record in the baseline note (delta section) and commit JSONs.
