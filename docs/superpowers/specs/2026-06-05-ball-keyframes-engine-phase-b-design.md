# Engine-shaped ball from sparse keyframes — Phase B design

**Date:** 2026-06-05
**Status:** Approved (brainstorm), pending implementation plan
**Depends on:** Phase A (merged) — `docs/superpowers/specs/2026-05-31-sparse-ball-anchor-keyframes-design.md`

## Problem & intent

Phase A made the pipeline emit a sparse `ball_keyframes.json` sidecar (one entry
per manual ball anchor: `world_xyz`, the clicked camera `ray` for airborne
anchors, a `depth_source` tag, and anchor metadata incl. `spin`/`touch_type`)
and pointed the UE manifest's ball entry at it via `keyframes_json`.

Phase B is the **engine consumer**: make the UE "Load Reconstruction" path read
those sparse keyframes and let the engine/art pass do the heavy lifting —
key the ball's transform track only at anchor frames (Sequencer tweens the
trajectory), and give the ball motion-driven spin authored in `BP_BallActor` so
an artist can tweak it.

This work lives in the **separate, non-git** UE project
`/Users/joebower/workplace/FootballPerspectives 5.8/Content/Python/` plus the
`BP_BallActor` asset. The UE MCP server is connected, so Blueprint graph
authoring and in-editor visual verification are done via MCP tools.

## Decisions (from brainstorm)

1. **Scope: all three pieces** — A (sparse keying + per-state interpolation),
   B (spin in `BP_BallActor`), C (ray handling at load).
2. **Interpolation: per-state** — airborne-context keys get cubic/auto tangents
   (smooth flight arcs); pure ground-to-ground keys stay linear (rolls don't
   bow below the pitch).
3. **Spin: rolling + preset curl, authored in `BP_BallActor` (not Sequencer).**
   The Blueprint derives spin from the ball's own motion each Tick — robust to
   artist edits — and adds a curl from the kick/volley `spin` preset.
4. **Piece C: auto-snap on load (no interactive tool).** Use the exported
   `world_xyz` directly (already ray-faithful at physics depth from Phase A).
   For `airborne_*` keyframes with `world_xyz == null`, place the key on the
   exported `ray` at a fallback depth. No Sequencer-selection tooling.

## Current state (verified via MCP / source)

- `BP_BallActor` (`/Game/Players/BP_BallActor`): parent `Actor`; components
  `DefaultSceneRoot` + `MeshComp` (the ball mesh); **no member variables**;
  `EventGraph` and `UserConstructionScript` empty.
- `build_sequence._add_ball_spawnable` keys a `MovieScene3DTransformTrack` at
  **every** frame from dense `(frame, x, y, z)` tuples, applying the axis-swap
  `(x,y)→(y,x)` + pitch-centre offset; rotation/scale channels held at identity.
- `load_reconstruction._load_ball_keys` reads `m.ball.track_json` →
  `(frame, x_m, y_m, z_m)` tuples (dense, all frames).
- `manifest.py` `BallEntry` already carries `keyframes_json` (Phase A, Task 6).
- Spin presets (`src/utils/ball_spin_presets.py`, pipeline side):
  `none`, `knuckle` → no spin; `instep_curl_{left,right}`,
  `outside_curl_{left,right}` → vertical (±z) axis side-spin;
  `topspin`/`backspin` → axis ⊥ travel (Magnus down / up). The keyframes carry
  only the preset **string** + `touch_type`; the ω vector is NOT exported, so
  the preset→axis mapping is re-implemented UE-side.

## Architecture & data flow

```
ue_manifest.json (ball.keyframes_json)
   │
   ▼
ball_keyframes.py   (NEW, unreal-free, unit-tested)  ── parse sidecar → BallKeyframe structs
ball_motion.py      (NEW, unreal-free, unit-tested)  ── pure helpers:
     • resolved_position_m(kf)   → world_xyz, or ray-fallback depth when null (pitch m)
     • key_interp_modes(kfs)     → per-key CUBIC (airborne-context) | LINEAR (ground)
     • flight_curl(preset, v0)   → (curl_axis_world, curl_strength) for the BP seed
   │
   ▼
load_reconstruction.py (MODIFY _load_ball_keys)  ── prefer keyframes_json; build a
     richer BallMotion struct (positions + per-key interp + curl seed); fall back to
     dense track_json (linear, no curl) when keyframes_json is absent
   │
   ▼
build_sequence.py (MODIFY _add_ball_spawnable)  ── key the transform track ONLY at
     anchor frames; set per-key position interpolation; seed the ball spawnable's
     CurlAxis / CurlStrength variables. Keeps the existing axis-swap + offset.
   │
   ▼
BP_BallActor (Blueprint, authored via MCP)  ── Tick: motion-derived spin
     • airborne (Z > radius): spin MeshComp about CurlAxis at CurlStrength
     • grounded: roll MeshComp (axis ⊥ velocity, rate = speed / ball_radius)
```

`camera_math` (unreal-free, already tested) supplies pitch↔UE conversion where
the ray-fallback needs it. The seam to Phase A is unchanged: the
`ball_keyframes.json` contract.

## Component detail

### A — Sparse keying + per-state interpolation (`build_sequence`)

`_add_ball_spawnable` is generalised to accept the richer ball data:
- Key the position channels (`[0],[1],[2]`) **only at anchor frames**, applying
  the same axis-swap + offset as today. Sequencer tweens between keys.
- Per-key interpolation from `ball_motion.key_interp_modes`: a key is **cubic**
  if its own state is `airborne_*` or either neighbouring keyframe is
  `airborne_*`; otherwise **linear**. (So a flight's launch / apex / landing
  ease through a smooth arc, and pure ground rolls stay flat.)
- Rotation/scale channels stay at identity here — rotation is owned by the BP.

### B — Spin in `BP_BallActor` (authored via MCP BlueprintTools)

Member variables added (instance-editable so the artist can tweak):
- `PrevLocation` (Vector, transient) — last Tick world location.
- `BallRadiusCm` (Float, default `11.0`).
- `SpinEnabled` (Bool, default `true`).
- `RollMultiplier` (Float, default `1.0`) — artist scale on rolling rate.
- `CurlAxis` (Vector, default `(0,0,1)`) — world-space curl axis, seeded per shot.
- `CurlStrengthDegPerSec` (Float, default `0.0`) — seeded from the preset.

`EventTick` logic (guard on `SpinEnabled` and `DeltaSeconds > 0`):
1. `loc = GetActorLocation()`; `vel = (loc - PrevLocation) / DeltaSeconds`;
   `speed = VectorLength(vel)`.
2. **Airborne** (`loc.Z > BallRadiusCm * 1.5`): rotate `MeshComp` by
   `CurlAxis * CurlStrengthDegPerSec * DeltaSeconds` (world rotation).
3. **Grounded** (else, and `speed > epsilon`): `axis = normalize(cross(WorldUp,
   vel))`; `rollDeg = degrees(speed / BallRadiusCm) * RollMultiplier *
   DeltaSeconds`; rotate `MeshComp` about `axis` by `rollDeg`.
4. `PrevLocation = loc`.

Gating spin mode on the ball's own height (not on span metadata) keeps the BP
self-contained and keeps spin correct when the artist re-positions keys.
`build_sequence` seeds `CurlAxis` + `CurlStrengthDegPerSec` on the ball
spawnable from `ball_motion.flight_curl(preset, v0)` for the clip's
representative flight (the first `player_touch` shot/volley anchor carrying a
spin preset). **Known limitation (documented):** a single curl per ball this
increment; per-flight curl variation (keying the variables) is a future
refinement.

### C — Ray handling at load (`ball_motion.resolved_position_m`)

Per keyframe: if `world_xyz` is present, use it (already ray-faithful at physics
depth). If `world_xyz is None` (airborne under-determined depth), place the key
on the exported `ray` at a fallback depth = ray ∩ the state's canonical-height
plane (`airborne_low/mid/high` → low/mid/high canonical z); the result is a
pitch-metre xyz the existing axis-swap maps to UE. No interactive tool.

### Fallback (back-compat)

When `m.ball.keyframes_json` is empty/absent (pre-Phase-A runs),
`_load_ball_keys` returns today's dense per-frame linear tuples with no curl —
current behaviour is unchanged, and `BP_BallActor` rolling spin still applies
(it is motion-derived, so dense playback rolls too).

## Module boundaries & testability

- **Unreal-free, unit-tested with the pipeline venv** (the logic core):
  `ball_keyframes.py` (parse round-trip; `ray` array `[[origin],[dir]]`;
  null-world handling) and `ball_motion.py` (`key_interp_modes` over mixed
  sequences; `resolved_position_m` incl. ray-fallback; `flight_curl` preset
  mapping incl. side-curl vertical axis and top/back ⊥-travel). Run:
  `cd "<UE>/Content/Python" && <pipeline>/.venv/bin/python -m pytest tests -q`.
- **In-editor only, MCP-verified** (thin glue + asset): `build_sequence`
  keying changes, `load_reconstruction` wiring, and the `BP_BallActor` graph.
  Verified by `_smoke/probe_ball_keyframes.py` (assert the ball binding has
  sparse keys, expected per-key interpolation, and seeded curl vars) plus MCP
  visual checks (`CaptureEditorImage` / `CaptureAssetImage`, `GetLogEntries`).

## Risks to validate early (plan spikes)

1. **Spawnable Tick during Sequencer eval** — confirm `BP_BallActor`'s Tick runs
   during editor playback / Movie Render Queue so motion-derived spin is visible.
   If spawnable tick is suppressed, fall back to keying rotation channels for
   curl and reconsider rolling.
2. **Seeding variables on a spawnable's object template** — confirm
   `build_sequence` can set `CurlAxis`/`CurlStrengthDegPerSec` on the ball
   spawnable's template so the values persist at spawn.
3. **Cubic-tangent API** — confirm the MovieScene scripting API exposes per-key
   interpolation/tangent mode on the position channels from Python.

Each risk gets a small spike at the top of the plan before the dependent work.

## Out of scope / YAGNI

- Per-flight curl variation (keyed spin vars) — single curl per ball this pass.
- Interactive re-snap-to-ray Sequencer tool — replaced by load-time auto-snap.
- Any change to Phase A's pipeline contract or the dense `ball_track.json`.
- Ball material / VFX polish (trails, etc.).

## Acceptance

- Loading a clip whose manifest carries `keyframes_json` produces a ball with
  **sparse** transform keys (one per anchor), per-state interpolation, and
  seeded curl variables; visually the ball tweens a smooth flight arc and rolls
  with motion-derived spin, curling on a preset-tagged shot.
- A clip without `keyframes_json` still loads via the dense fallback unchanged.
- The unreal-free modules pass their unit tests; the probe script asserts the
  binding/interp/curl-seed state in-editor.
