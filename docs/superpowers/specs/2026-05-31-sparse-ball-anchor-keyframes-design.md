# Sparse anchor-driven ball, engine-shaped — design

**Date:** 2026-05-31
**Status:** Approved (brainstorm), pending implementation plan

## Problem & intent

Today `BallStage` produces a *dense* `ball_track.json` with a `world_xyz` for
every frame in the camera span (monocular ground projection + fitted parabolas
+ interpolation between anchors). The UE side reads that dense track and keys a
`MovieScene3DTransformTrack` on `BP_BallActor` at **every** frame
(`_add_ball_spawnable`, one key per frame).

We want to try the opposite division of labour: let the **engine / art pass do
the heavy lifting**. The pipeline should export only the *trustworthy ground
truth* — the manual anchors — as sparse 3D keyframes carrying their semantic
state. The engine then interpolates between them, shapes trajectories, adds
spin, and lets an artist correct placement.

Key realisation that shaped this design: the UE "Load Reconstruction" path does
**not** read `ball.fbx`. It reads per-frame `world_xyz` from the JSON track
pointed to by the manifest (`export.py` BallEntry, `load_reconstruction.py
_load_ball_keys`). So the real lever is the **JSON payload**, not the FBX. And
because UE keys a transform track from that JSON, feeding it *sparse* keys makes
**Sequencer's own keyframe interpolation** the trajectory-tweening mechanism —
exactly the artist-facing behaviour we want.

## Decisions (from brainstorm)

1. **Sparse anchor keyframes only** — export `world_xyz` only at frames that
   have a manual anchor, plus each anchor's state/metadata. No
   ground-projection / parabola / interpolated frames in the new artifact.
2. **Airborne depth: export both** — for `airborne_*` anchors, emit the
   pipeline-resolved `world_xyz` (physics depth, the existing C2/C4 "10 cm"
   work) **and** the clicked camera ray, so engine tools default to the
   physics depth but can re-snap to the ray when an artist reshapes the arc.
3. **New sidecar artifact** — write a new `ball_keyframes.json`; leave the
   dense `ball_track.json` untouched so the web viewer (`.glb`) and the
   `tests/test_ball_anchor_accuracy.py` harness keep their dense input. Clean
   A/B; nothing existing breaks.
4. **Scope: both pipeline + UE tools**, split into two implementation phases
   joined by the `ball_keyframes.json` contract.

## Architecture & data flow

```
BallStage (this repo)
  ├─ resolves each manual anchor → 3D (EXISTING machinery in ball.py:
  │   ground∩ray, player-bone FK, goal geometry, airborne ray+physics depth)
  ├─ writes ball_track.json        (DENSE — unchanged: web viewer + 10cm harness)
  └─ writes ball_keyframes.json    (NEW — SPARSE: one entry per manual anchor)

export stage → ue_manifest.json    (ball entry gains `keyframes_json` pointer)

UE (separate repo: "FootballPerspectives 5.8/Content/Python")
  ├─ load_reconstruction reads ball_keyframes.json (prefers it; falls back to dense)
  ├─ build_sequence keys transform track ONLY at anchor frames → Sequencer tweens
  ├─ spin driven from anchor spin/touch_type metadata
  └─ artist tools: re-snap a moved airborne key onto its exported ray; reshape arc
```

The seam between the two codebases is the `ball_keyframes.json` contract.
Nothing in the dense path changes.

## The `ball_keyframes.json` contract

Frame numbers mirror the existing `ball_track.json` frame space exactly, so
UE's clip-relative mapping is unchanged. Lives at
`output/ball/{shot}_ball_keyframes.json` (and unprefixed
`output/ball/ball_keyframes.json` for the single-shot legacy layout), mirroring
the `ball_track.json` naming.

**Header:** `clip_id`, `fps`, `image_size`.

**Each keyframe:**

| field | when | meaning |
|---|---|---|
| `frame` | always | same frame space as `ball_track.json` |
| `state` | always | `kick` / `bounce` / `grounded` / `airborne_low` / `airborne_mid` / `airborne_high` / `catch` / `header` / `volley` / `chest` / `player_touch` / `goal_impact` / `off_screen_flight` |
| `world_xyz` | always except `off_screen_flight` | resolved 3D — the artist's default key position (pitch metres) |
| `image_xy` | when clicked (i.e. not `off_screen_flight`) | the authoritative pixel |
| `ray` | `airborne_*` | `{origin: [x,y,z], dir: [x,y,z]}` camera ray (unit dir) for re-snapping after an artist moves the key |
| `depth_source` | always | `ground` \| `ray_physics` \| `player_bone` \| `goal_geometry` — how depth was obtained, signals trustworthiness |
| `player_id`, `bone` | `player_touch` | drives the ball through that bone |
| `goal_element` | `goal_impact` | `post` / `crossbar` / `back_net` / `side_net` |
| `touch_type`, `spin` | `shot` / `volley` touches | spin preset → engine spin |
| `confidence` | always | per-anchor confidence (carried from resolution) |

Every field above is already computed inside `BallStage` during anchor
resolution; the new code path *collects* this rather than discarding it after
densification.

`depth_source` mapping:
- ground-level hard knots (`grounded`, `kick`, `bounce`, and `player_touch`
  classified ground-level) → `ground`
- `player_touch` resolved to a bone (airborne) → `player_bone`
- `goal_impact` → `goal_geometry`
- `airborne_*` → `ray_physics`

## Schema design (this repo)

New module `src/schemas/ball_keyframes.py` following the existing
frozen-dataclass + `save`/`load` pattern of `ball_track.py` and
`ball_anchor.py`:

- `BallKeyframe` (frozen dataclass) — the per-anchor fields above, with
  `ray`, `player_id`, `bone`, `goal_element`, `touch_type`, `spin`,
  `image_xy`, `world_xyz` all `| None` defaulting to `None`.
- `BallKeyframeSet` (frozen dataclass) — `clip_id`, `fps`, `image_size`,
  `keyframes: tuple[BallKeyframe, ...]`, with `save(path)` / `load(path)`.

Validation at load mirrors `BallAnchorSet.load` (state whitelist, conditional
required fields), so a hand-edited file fails fast.

## Phase A — pipeline (this repo)

Fully unit-testable; TDD throughout.

1. **`src/schemas/ball_keyframes.py`** — the schema above. Tests:
   round-trip save/load, validation errors, optional-field handling.
2. **`BallStage` emits the sidecar.** During the existing anchor-resolution
   pass, accumulate a `BallKeyframe` per resolved anchor (reusing
   `_resolve_anchor_world`, the hard-knot overrides, the ray projection
   helpers `_project_point_onto_pixel_ray` / `_snap_world_onto_pixel_ray`
   already in `ball.py`, and the camera for ray origin/dir). Write
   `{shot}_ball_keyframes.json` next to `{shot}_ball_track.json`. The dense
   path is unchanged. Tests: a fixture anchor set of each state resolves to
   the expected keyframe payload (world_xyz, depth_source, ray presence).
3. **`export.py` manifest wiring.** `BallEntry` gains a `keyframes_json`
   relative path; populate it when `{shot}_ball_keyframes.json` exists.
   `track_json` stays for back-compat. Tests: manifest round-trips the new
   field; absent sidecar → field empty.
4. **`src/schemas/ue_manifest.py` + UE `manifest.py`** (the unreal-free,
   unit-tested mirror) gain `keyframes_json` on the ball entry. Tests in both
   `tests/` and `Content/Python/tests/`.

**Acceptance for Phase A:** running the ball + export stages on an existing
output produces a `ball_keyframes.json` whose entries match the manual anchors,
with `world_xyz` equal (within tolerance) to the dense track's value at those
frames, and a manifest pointing at it. The 10 cm harness still passes
unchanged.

## Phase B — engine (separate UE repo)

Verified in-editor via `_smoke/probe_*.py` scripts (no `import unreal` in CI).
This phase will get its own focused spec once Phase A lands and the contract is
exercised against real data; sketched here so the contract is unambiguous.

1. **`manifest.py`** already gains `keyframes_json` in Phase A (unreal-free).
2. **`load_reconstruction._load_ball_keyframes`** — parse the richer keyframes
   into structs carrying frame, world_xyz, state, ray, spin metadata. Prefer
   `keyframes_json`; fall back to dense `track_json` when absent (old runs).
3. **`build_sequence._add_ball_spawnable`** — key the transform track **only
   at anchor frames**, setting per-state interpolation/tangents: cubic auto
   for `airborne_*` arcs, linear or constant for ground-level contacts.
   Sequencer tweens between keys; the artist grabs keys in the viewport to
   correct placement.
4. **Spin** — drive `BP_BallActor` rotation from the `spin` / `touch_type`
   metadata on `kick` / `volley` keyframes (map preset → angular velocity),
   keyed from the launch anchor.
5. **Trajectory-correction helper** — a Python/EUW tool that, given a moved
   `airborne_*` key, re-snaps it onto the exported `ray` (the click stays
   authoritative for the line of sight) while letting the artist set depth.

**Acceptance for Phase B:** loading a reconstruction places the ball with
sparse keys an artist can manipulate; airborne keys re-snap to their ray;
spin is visible on struck balls.

## Out of scope / YAGNI

- No change to the dense reconstruction (`ball_track.json`), the web viewer, or
  the `test_ball_anchor_accuracy.py` harness.
- The `ball.fbx` Blender bake is left as-is (already unused by the UE load
  path); not stripped in this work to avoid touching a back-compat artifact.
- Phase B's full spin model and trajectory-correction UX are deferred to a
  Phase-B spec; this design only fixes the contract they consume.

## Testing strategy

- Phase A: pytest unit tests on the new schema, the `BallStage` keyframe
  emission (per-state fixtures), and the manifest field on both the pipeline
  and UE-mirror sides. Run with `pytest --cov=src`.
- Cross-check: for an existing `output*/`, assert each `ball_keyframes.json`
  entry's `world_xyz` matches the dense `ball_track.json` at that frame within
  tolerance, confirming we reuse the same resolution.
- Phase B: in-editor probe scripts (`_smoke/probe_ball_keyframes.py`).
```
