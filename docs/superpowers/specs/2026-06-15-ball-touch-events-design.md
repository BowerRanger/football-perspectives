# Ball Touch-Event Tracking — Design

- **Date:** 2026-06-15
- **Status:** Draft (awaiting review)
- **Branch context:** builds on `ball-auto-physics`
- **Supersedes (as default):** the dense trajectory solve of [`2026-06-12-ball-v2-design.md`](2026-06-12-ball-v2-design.md) / [`2026-06-14-phase2-mode-sequence-solve.md`](2026-06-14-phase2-mode-sequence-solve.md) — that solver is retained behind a config flag, not deleted.
- **Related:** [`2026-05-31-sparse-ball-anchor-keyframes-design.md`](2026-05-31-sparse-ball-anchor-keyframes-design.md), [`2026-06-05-ball-keyframes-engine-phase-b-design.md`](2026-06-05-ball-keyframes-engine-phase-b-design.md), [`2026-05-11-ball-anchor-editor-design.md`](2026-05-11-ball-anchor-editor-design.md)

---

## 1. Summary

Today the ball stage **detects events** (player touches, bounces, goal impacts) and then **solves a dense, physically-correct trajectory** (`ball_track.json`, one `BallFrame` per frame) constrained by those events. The expensive, fragile half of that pipeline is the dense monocular solve: ball depth is underconstrained, detection is sparse, and a single mis-segmented span corrupts a whole flight.

This design **inverts the architecture**. The ball stage's product becomes a **sparse, editable set of 3-D ball events** — every player touch (with player + body part), plus the physics waypoints that bend the ball's path (bounces, post/crossbar/net impacts, rest, possession/carry). The **3-D position of a touch is pinned to the contacting player's body joint**, which we already reconstruct in 3-D, sidestepping monocular ball-depth ambiguity entirely. **Interpolation between events moves to Unreal Engine** (with a Python reference interpolator so the web viewer and quality reports still get a dense ball).

The detection + attribution machinery (`ball_auto_events.py`, `ball_player_context.py`) and the editable-anchor + sparse-keyframe + UE-import paths already exist. The work is: (C) refactor the ball stage into a shared *evidence + events* core with two pluggable resolvers; a new **EventResolver** (default) that resolves events to keyframes directly; a **touch/event list** editor in the web viewer; an export/UE path driven by the sparse set; and UE-side interpolation tooling (stretch).

---

## 2. Goals & non-goals

**Goals**
- Detect **every player touch** of the ball: frame, player id, body part (SMPL bone), and touch character (pass/control vs shot/volley/header).
- Detect the **non-touch physics waypoints** needed for faithful interpolation: ground bounce, goal/post/crossbar/net impact, ball-at-rest, and possession/carry spans.
- Resolve each event to a 3-D pitch position: **touches pinned to the body joint**; waypoints via ball-pixel ray ∩ ground/goal geometry (the existing ray-faithful rules).
- Make the auto-detected event list **visible and editable in the web viewer**, exactly in the spirit of anchors today (add / edit / delete; operator edits win).
- **Export the sparse event set to UE5** and interpolate there; keep a derived dense ball for glTF/web/quality.
- Keep the existing dense trajectory solvers available behind `ball.solver=piecewise` / `global`.

**Non-goals (v1)**
- No new ML model. Reuse WASB/IMM detection, GVHMR-derived FK, camera track.
- No change to player/camera/refined-pose stages.
- No attempt to recover finer foot regions than SMPL provides (instep/laces) from geometry alone — spin character stays an *inferred* attribute (`touch_type`/`spin`), not a new joint.
- No real-time/online operation; this remains an offline per-shot batch stage.

---

## 3. Design decisions (locked)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | What the keyframe set contains | **Touches + physics waypoints** (bounce, goal impact, rest, carry) | A ball that bounces or hits the net between two touches must not interpolate as a straight line. |
| D2 | Relationship to the dense solver | **New default (`ball.solver=events`); existing trajectory solvers stay reachable via `ball.solver=piecewise` / `global`** | Preserves prior investment; lowest migration risk; lets us A/B. The config key today only has `piecewise`/`global`; we add `events` and make it the default. |
| D3 | 3-D position of a touch | **Pin to contacting body joint** (+ ball radius offset along the camera ray) | Touch depth becomes as good as the player reconstruction; removes the dominant monocular ball-depth failure mode. |
| D4 | Code structure | **Approach C** — shared *evidence+events* core + pluggable `EventResolver` / `TrajectorySolver` | "New default, solver behind a flag" falls out for free; no duplication; small, testable units. |
| D5 | Waypoint depth (bounce/goal/rest) | **Ball-pixel ray ∩ geometry** (ground plane `z=ball_radius`, or goal plane) | Waypoints are not on a body; this reuses the existing `_resolve_anchor_world` rules and the ray-faithful principle. |
| D6 | Where interpolation runs | **UE5 (authoritative for the engine product)** + a **Python reference interpolator** for glTF/web/quality | Matches "we interpolate in UE"; keeps the web viewer and quality report working without a second authoring tool. |

---

## 4. Architecture (Approach C)

Refactor [`src/stages/ball.py`](../../../src/stages/ball.py) so the stage is a thin orchestrator over a **shared core** and a **resolver strategy**.

```
                          ┌─────────────────────────────────────────────┐
                          │            Ball stage (orchestrator)          │
                          └─────────────────────────────────────────────┘
                                            │
              ┌─────────────────────────────┴──────────────────────────────┐
              │  SHARED EVIDENCE + EVENTS CORE  (reused by both resolvers)   │
              │  1. Detect ball pixels (WASB/YOLO) → IMM smooth → steps      │
              │  2. PlayerContext: per-frame FK contact joints (world+pixel) │
              │  3. detect_events(): velocity breaks → touch/bounce/goal/    │
              │     stationary  (ball_auto_events.py)                        │
              │  4. carry/possession spans  (NEW: ball_possession.py)        │
              │  5. merge with manual anchors (operator wins)                │
              └─────────────────────────────┬──────────────────────────────┘
                                            │  (resolved events)
                       ┌────────────────────┴─────────────────────┐
            ball.solver=events (DEFAULT)          ball.solver=piecewise|global (flag)
                       │                                          │
        ┌──────────────▼───────────────┐         ┌────────────────▼────────────────┐
        │   EventResolver (NEW)          │         │  TrajectorySolver (EXISTING)     │
        │   - touch → body-joint 3-D     │         │  piecewise / mode-search →       │
        │   - waypoint → ray∩geometry    │         │  dense BallTrack                 │
        │   - segment classification     │         └────────────────┬────────────────┘
        │   → BallKeyframeSet (sparse)   │                          │
        └──────────────┬─────────────────┘                          │
                       │                                            │
        ┌──────────────▼───────────────┐                           │
        │ Python reference interpolator │                           │
        │ (ball_interpolate.py, NEW)    │                           │
        │ → derived dense BallTrack     │                           │
        └──────────────┬───────────────┘                           │
                       └───────────────┬───────────────────────────┘
                                       ▼
        Outputs:  BallKeyframeSet (authoritative sparse, both modes)
                  BallTrack (authoritative in piecewise/global mode; derived in events mode)
                  *_ball_anchors_auto.json, *_ball_observations.json, *_ball_diag.json
```

Functions that move into the shared core today live across `ball.py`, `ball_auto_events.py`, `ball_player_context.py`, `ball_auto_anchor.py`. The new `EventResolver` reuses `_resolve_anchor_world` (already in `ball.py`) but **changes the touch branch to body-pin** (§7) and **skips `solve_piecewise`**.

**Why C over A/B:** `ball.py` is already ~1700 lines; A bolts a second control path onto it; B duplicates the detection/context/projection wiring. C extracts the front half once and makes the back half a strategy — the codebase shrinks per file and each resolver is independently testable. (If minimizing the diff becomes the priority, A is a strict subset of C's data model and UE work, so we can fall back without rework.)

---

## 5. Event taxonomy

Every product is a list of **ball events**, each resolved to a 3-D keyframe. Kinds (mapped onto existing `BallAnchorState` where possible):

| Kind | `state` | Player? | Depth source | Notes |
|------|---------|---------|--------------|-------|
| **touch** | `player_touch` | yes (id + bone) | `player_bone` (NEW behaviour: joint 3-D) | The core event. `touch_type` ∈ {none=pass/control, shot, volley, header}. |
| **bounce** | `bounce` | no | `ground` (ray ∩ z=ball_radius) | Ground rebound; restitution hint for interpolation. |
| **goal impact** | `goal_impact` | no | `goal_geometry` | `goal_element` ∈ {post, crossbar, back_net, side_net}. |
| **rest** | `grounded` | no | `ground` | Ball stationary; ends a roll. |
| **carry / possession** | `carry` (NEW) | yes | `player_bone` (foot/ground) | A *span* (start+end frame), not a point: ball travels with a dribbling player. |
| **out of view** | `off_screen_flight` | no | none (`ray=None`) | Ball leaves frame between events; interpolation is a free arc. |

Touches and carry are body-anchored (depth-stable). Bounce / goal / rest depend on the ball pixel + geometry. The taxonomy is a superset of what `detect_events()` already emits; only `carry` is new (§6.3).

---

## 6. Detection & attribution

### 6.1 Touches (reuse, largely unchanged)
`detect_events()` in [`ball_auto_events.py`](../../../src/utils/ball_auto_events.py) already:
- finds pixel-velocity breaks (direction change ≥ `min_direction_change_deg`, speed delta ≥ `min_break_speed_px`, both sides moving),
- probes frame ±1, calls `PlayerContext.joints_near_pixel(frame, uv, radius_px=touch_max_px)`,
- attributes the nearest contact joint as `BallEvent(kind='touch', player_id, bone)` (ties prefer feet),
- distinguishes keeper saves (hand/arm bones) from outfield touches.

Body parts come from `BONE_TO_SMPL_INDEX` (10 bones: `l_foot`, `r_foot`, `l_knee`, `r_knee`, `chest`, `head`, `l_shoulder`, `r_shoulder`, `l_hand`, `r_hand`). This set is sufficient for v1; finer character (instep curl, backspin, header) is the inferred `touch_type`/`spin`, seeded geometrically (commit `f7db86d`).

### 6.2 Physics waypoints (reuse)
Bounce, goal-impact, and stationary classification already exist in `detect_events()` and the goal-geometry path. They feed the EventResolver the same way they fed the solver.

### 6.3 Carry / possession spans (NEW — `src/utils/ball_possession.py`)
A dribble produces many micro-touches; sparse linear interpolation would make the ball float in straight lines between them. Detect a **carry span** when, over a window:
- consecutive touches share the same `player_id`,
- the ball pixel stays within a possession radius of that player's foot joints, and
- ball displacement relative to the player is small (ball moves *with* the player, not away).

Emit one `carry` event covering `[start_frame, end_frame]` with the owning `player_id`. UE/reference interpolation for a carry span follows the player's ground/foot path rather than a straight line (§10). *This sub-feature is isolated; it can ship in a later phase without blocking the core (§17).*

### 6.4 Attribution confidence
Each event carries `confidence` derived from: pixel distance joint↔ball, break sharpness, joint FK confidence, and the contact-gap (perpendicular distance from joint 3-D to the ball pixel ray, the existing `contact_max_gap_m` gate). The score `BallEvent.score` is **propagated into `BallAnchor.confidence`** (today it is computed and discarded — `_event_candidates`/`_grounded_candidates` in [`ball_auto_anchor.py`](../../../src/utils/ball_auto_anchor.py) must pass it through); manual anchors default to `1.0`. Low-confidence auto events are surfaced in the editor as *suggestions* for the operator to confirm/dismiss.

### 6.5 Tricky attribution cases (explicit rules)
- **Simultaneous two-foot / multi-part touches:** the *auto* detector emits **one touch per velocity break** — the single best joint (ties prefer feet). The data model permits more than one event at a frame (keyed by `(player_id, bone, frame)`, not just `frame`), so the operator can **add a second simultaneous touch** in the editor when both feet/parts genuinely contact. We do not auto-emit multiple touches per break in v1 (avoids double-counting from FK jitter).
- **Unintentional deflection / ricochet (incl. own-goal off a defender):** any ball contact with a player body part is a **touch** (body-pinned), regardless of intent; `touch_type` stays `none` (not `shot`). Intent is not modelled.
- **Player track ID switch / gap mid-touch:** attribution uses the joint nearest the ball pixel at the break (probing frame ±1). If the track ID flips across that boundary the attributed `player_id` may be the post-switch ID; the event still resolves geometrically. The joint FK confidence is folded into the score so low-confidence (gappy) frames rank below clean ones, and the editor surfaces the touch for operator correction. No automatic ID-stitching in v1.

---

## 7. 3-D resolution (the key change)

`EventResolver` resolves each merged event to a `BallKeyframe.world_xyz`:

- **touch / carry (body-pinned, D3):** take the contacting joint's 3-D world position from FK (`PlayerContext.joint_world(frame, player_id, bone)`), then offset by the ball radius along the camera ray toward the camera (so the ball surface, not the joint centre, sits on the sight-line). `depth_source='player_bone'`. This is depth-stable: error = player-reconstruction error, not monocular ball-depth error.
  - *Lateral refinement (optional):* if a confident ball pixel exists at the frame, project the joint onto the ball-pixel ray (existing `project_point_onto_pixel_ray`) so the click/detection stays authoritative for lateral XY while the joint supplies depth. This preserves the **C1 ray-faithful invariant** from [`project_ball_ray_faithful_anchoring`]. Default: body-joint position; ray-refine when a high-confidence ball pixel is present.
  - *Occlusion robustness (key win):* when the ball pixel is **missing/occluded at the contact frame** (no `image_xy`), the touch still resolves — straight to the body-joint 3-D position. This is a robustness advantage over the dense solver, which needs a detection at the knot. The EventResolver therefore handles `player_touch` with `image_xy=None` explicitly rather than falling through to a ground ray-cast.
- **bounce / rest (D5):** ball-pixel ray ∩ ground plane at `z=ball_radius`. `depth_source='ground'`.
- **goal impact:** ball-pixel ray ∩ goal-element plane. `depth_source='goal_geometry'`.
- **out of view:** no `world_xyz`, `ray=None`; interpolation treats it as a free knot (gravity arc determined by the two bracketing 3-D events).

This is exactly `_resolve_anchor_world` in [`ball.py`](../../../src/stages/ball.py) **minus** the trajectory dependency, with the touch branch switched from "project bone onto clicked ray (depth from bone)" to "use bone world position (depth from bone), ray-refine lateral if available." The resolver never calls `solve_piecewise`.

---

## 8. Data model

We **reuse the existing schemas** so the editor, persistence, and UE import keep working, with small additions.

### 8.1 Editable layer = anchors (unchanged persistence)
The editable event list **is** `BallAnchorSet` ([`src/schemas/ball_anchor.py`](../../../src/schemas/ball_anchor.py)), persisted to `*_ball_anchors.json` (manual) and `*_ball_anchors_auto.json` (auto), merged operator-wins (`merge_anchors`, suppress radius 3 frames). Additions:
- **Phase 1 (now):** `BallAnchor` gains optional `confidence: float = 1.0` (auto events surface their score; manual = 1.0) and `end_frame: int | None = None`. Both default-valued and parsed with backward-compat in `load()` (missing key → default), so existing `*_ball_anchors.json` load unchanged.
- **Phase 3 (with carry):** `BallAnchorState` gains `carry`; validation then requires a `carry` anchor to have `player_id`, `bone` (a foot), and `end_frame > frame`. Until then `end_frame` is unused by validation.

### 8.2 Authoritative product = `BallKeyframeSet` (extended)
`BallKeyframe` ([`src/schemas/ball_keyframes.py`](../../../src/schemas/ball_keyframes.py)) already carries `frame, state, depth_source, world_xyz, image_xy, ray, player_id, bone, goal_element, touch_type, spin, confidence, omega_rad_s`. Additions:
- `end_frame: int | None` (for `carry`).
- A new **segments** list on `BallKeyframeSet`: `BallSegment(start_frame, end_frame, kind, hints)` where `kind ∈ {ballistic, roll, carry, rest, free_flight}` and `hints` carries the interpolation contract (§10): launch velocity direction, gravity, restitution, spin preset / `omega_rad_s`, apex (if known), owning `player_id` for carry. Segments are *derived* from consecutive keyframes by the resolver, so UE and the reference interpolator don't each re-derive them (and can't disagree).

### 8.3 Derived dense track (compat)
In `events` mode, `ball_track.json` (`BallTrack`/`BallFrame`) is **produced by the Python reference interpolator** (§11) from the keyframe set, tagged in `_ball_diag.json` as `derived: true`. In `piecewise`/`global` mode it is the authoritative solve as today. Either way, glTF/FBX export consumes a dense `BallTrack` unchanged.

---

## 9. Web viewer — touch/event list editor

Extend the existing ball anchor editor ([`src/web/static/ball_anchor_editor.html`](../../../src/web/static/ball_anchor_editor.html)) rather than build a new page (consistent UX, shared video/overlay/scrub/save code). New **Event List panel** (right column, replacing/augmenting the flat anchor list):

- **List view:** chronological events (auto ∪ manual), each row showing `time · kind · player (team-coloured dot) · body part · touch_type · confidence`. Auto events render with a dashed/badge style and a *confirm* / *dismiss* affordance; manual events render solid. Clicking a row seeks to its frame.
- **Add a touch (anchor-style):** pick a player + body part (dropdown populated from the shot's tracks + `VALID_BONES`) and a kind, scrub to the frame, click the ball in the video. Clicking near the ball calls `joints_near_pixel` (new read endpoint) to **suggest** the player+bone under the cursor, so the operator usually just accepts the suggestion. One touch per (player, frame); simultaneous L+R foot allowed as two rows.
- **Edit:** change player / body part / kind / touch_type / spin; nudge frame; set carry `end_frame`.
- **Delete / dismiss:** remove a manual event; dismiss an auto event (persists a suppression so it doesn't reappear on re-run).
- **Persistence:** unchanged — `POST /ball-anchors/{shot_id}` → `BallAnchorSet.save()`. Auto events served read-only from `GET /ball-anchors/{shot_id}/auto`. Add `GET /joints-near?shot=&frame=&u=&v=&r=` for the click-suggest.
- **Re-resolve button:** "Re-resolve ball" runs the ball stage in `events` mode for the shot (mirrors the existing "Rerun camera" job flow) and refreshes the list + viewer.

Read-only mirror in [`viewer.html`](../../../src/web/static/viewer.html): a "Ball touches" section listing events for the current frame, click-to-jump, with the touching player highlighted in the 3-D scene.

---

## 10. Interpolation contract (the shared spec)

A **single, precisely specified** per-segment interpolation, implemented twice (Python reference for glTF/web; UE for the engine product) so both agree. Segment `kind` and `hints` come from `BallKeyframeSet.segments`.

- **ballistic** (between a launch event — kick/shot/header/volley — and the next contact): gravity parabola through both 3-D endpoints with `g` from hints; **two body-pinned endpoints + gravity fully determine the arc** (the same identifiability the old solver relied on, but the endpoints are now reliable). Optional Magnus curl from `spin`/`omega_rad_s`.
- **roll** (between two grounded events): along-ground path with bounded deceleration (`rolling_decel`), ease to rest if the next event is `rest`.
- **carry** (a `carry` span): ball position = owning player's foot/ground path offset by ball radius; the ball "sticks" near the dribbler rather than cutting straight lines.
- **rest:** constant position.
- **free_flight** (a bracket includes `off_screen_flight`): gravity arc between the bracketing 3-D events; lateral unconstrained beyond the endpoints.

**Clip-boundary / open segments.** Segment derivation walks consecutive keyframes; the first and last events leave one open end:
- *Open start* (clip frame 0 precedes the first event and the ball is already moving): hold the first keyframe's position back to frame 0 if grounded; if airborne, infer launch velocity from the first two keyframes and extend the ballistic arc backward. Tag `hints.inferred=true`.
- *Open end* (a launch — shot/clearance — has no succeeding contact, e.g. ball ends in the net, comes to rest off-screen, or the clip cuts): bound the ballistic/roll segment to `N_frames-1`. If a `goal_impact` or `off_screen_flight` event was detected it provides the real endpoint; otherwise the arc is extrapolated under gravity and flagged `hints.open_ended=true`. Zero-duration trailing segments (launch on the final frame) are omitted.

Rotation: keep the existing model — `omega_rad_s` per ballistic segment (Magnus), rolling spin `|v|/r` on roll segments, integrated to a quaternion. The reference interpolator reuses [`ball_orientation.py`](../../../src/utils/ball_orientation.py); UE keeps deriving curl at runtime from `CurlStrengthDegPerSec` (no double-spin — see §13 gotcha).

---

## 11. Python reference interpolator (`src/utils/ball_interpolate.py`, NEW)

`interpolate_events(keyframe_set) -> BallTrack`: walks `segments`, evaluates the §10 spec per frame, returns a dense `BallTrack` (with `quat_wxyz` via `ball_orientation.integrate_orientation`). Deterministic, dependency-light (numpy/scipy only — testable in the light venv per [`project_refined_poses_dev_env`]). Purposes: (1) feed glTF/FBX so the web viewer shows a moving ball that matches UE; (2) feed the confidence/quality viz; (3) be the executable reference UE is validated against. It is explicitly **replaceable** — UE may exceed it.

---

## 12. Export & UE consumption

- **Export stage** ([`src/stages/export.py`](../../../src/stages/export.py)): unchanged contract — writes `ue_manifest.json` `BallEntry` with `track_json` + `keyframes_json`. In `events` mode `track_json` is the derived dense track; `keyframes_json` is the authoritative sparse set (now including `segments`).
- **glTF/FBX:** unchanged — built from the (derived) dense `BallTrack`. No re-architecture of the builders needed for v1 because the dense track still exists; a later optimization can emit sparse CUBICSPLINE keys.
- **UE5:** already prefers `ball_keyframes.json` ([`load_reconstruction.py::_load_ball_motion`], [`build_sequence.py::_add_ball_spawnable`]). Extend the UE-side reader ([`ball_keyframes.py`], [`ball_motion.py`]) to parse `segments`, and have `key_interp_modes()` / `_key_ball_curl()` honour segment `kind` + `hints`. This is the seam the stretch tooling plugs into.

---

## 13. UE5 interpolation tooling (stretch)

An **Editor Utility** ("Ball Interpolation Tools") operating on the ball binding in the Level Sequence, segment-aware:

- **Per-segment inspector:** select a segment (ballistic/roll/carry/rest); see its endpoints, kind, and hints.
- **Ballistic controls:** apex height, gravity, launch direction; **Magnus curl** toggle with strength + axis (seeded from `spin`/`omega_rad_s`); live viewport preview of the arc.
- **Shot preset:** flat-fast + late-dip (topspin) shaping for `touch_type=shot`.
- **Spin:** per-flight `CurlStrengthDegPerSec` curve (the track already exists; tooling exposes it).
- **Re-key:** writes computed sub-frame samples into the ball `MovieScene3DTransformTrack` (densify the segment) or sets cubic tangents.

**Gotchas to honour (from the UE map):** coordinate transform is `(x,y)→(y,x)` + 90° yaw + cm scaling + corner-origin offsets; rotation channels are **not** keyed (curl is applied at runtime — keying rotation double-spins); `ray` is populated only for airborne keyframes; UE ignores `BallEntry.fbx` for animation (JSON only).

### First implementation target (further stretch)
Port the §11 ballistic+spin segment evaluator into a UE editor Python script in `Content/Python/football_perspectives/` that reads `keyframes_json` + `segments`, computes per-segment dense samples for `ballistic` (gravity + Magnus) and `roll`, and writes them into the ball transform track via the existing `build_sequence` hooks. This is the highest-value, lowest-risk slice: the reader, the curl track, and the transform-keying hooks already exist.

---

## 14. Validation & quality report

- **Unit tests** (light venv): EventResolver body-pin math (joint + radius along ray); ray-refine lateral; bounce/goal ray∩geometry; `interpolate_events` per segment kind; schema round-trips (`BallAnchor.end_frame`, `carry`, `segments`); web API payloads.
- **Touch detection accuracy:** precision/recall of auto touches vs hand-labelled touch lists on origi / kroupi / the Liverpool reel ([`project_highlights_ingestion`]). Report per-shot in the quality report.
- **Reprojection residual:** project each resolved keyframe back to image, compare to the ball pixel where one exists (px). Body-pinned touches should reproject near the ball; large residuals flag bad attribution. Reuse the no-op-detector harness idea (`tests/test_ball_anchor_accuracy.py`, [`project_ball_ray_faithful_anchoring`]).
- **Quality report** (`output/quality_report.json`): ball section reports `mode` (events/piecewise/global), event counts by kind, auto-vs-manual, mean touch reprojection residual, carry spans, and `derived` flag for the dense track.
- **Existing-test migration:** flipping the default touches solver-assertion tests. `test_default_solver_matches_explicit_piecewise` (in `tests/test_ball_stage_global_solver.py`) must assert the new default (`events`) and pin the old behaviour under an explicit `ball.solver=piecewise`. `BallTrack`-shape tests pass unchanged in events mode (the derived track has the same structure); only solver-name and residual-exactness assertions need updating. Add an `events`-mode integration test alongside `tests/test_ball_stage.py`.
- **Visual:** web viewer + UE playback plausibility against the broadcast.

---

## 15. Migration & backward compatibility

- Default flips to `ball.solver=events`; existing clips re-run produce the sparse-first outputs. `ball.solver=piecewise` (or `global`) reproduces today's behaviour exactly (no resolver change on those paths). The config key today only accepts `piecewise`/`global`; the stage must add `events`, make it the default, and raise on unknown values.
- All sidecars keep their filenames and (super-set) schemas; old `*_ball_anchors.json` load unchanged (new fields optional/defaulted).
- `ball_track.json` still exists (derived), so glTF/FBX/UE manifest and the current viewer keep working with no change required on day one.
- UE additions (`segments`) are optional; an older UE build ignores them and falls back to today's cubic/linear behaviour.

---

## 16. Scope & phasing

1. **Phase 1 — Core (events mode).** Approach-C refactor (shared core + resolver strategy); `EventResolver` with body-pinned touches + ray∩geometry waypoints; reuse auto-events for touch/bounce/goal/rest; extend schemas (`carry` deferred to P3, but `end_frame`/`confidence`/`segments` land now); `ball_interpolate.py`; derived dense track; `ball.solver` flag; quality-report fields. **Ship criterion:** events-mode output renders a plausible ball in the existing viewer for origi/kroupi.
2. **Phase 2 — Viewer.** Event-list panel (list, add/edit/delete, confirm/dismiss auto, player/bone picker + click-suggest); `GET /joints-near`; viewer read-only mirror; re-resolve job.
3. **Phase 3 — Carry/possession.** `ball_possession.py`; `carry` event + segment; carry interpolation in reference + UE.
4. **Stretch — UE tooling.** Segment-aware Editor Utility (§13).
5. **Further stretch — UE impl.** Ballistic+spin segment evaluator ported into UE editor Python (§13 target).

---

## 17. Risks & open questions

- **Body-pin lateral drift:** the ball isn't exactly at the joint centre (boot tip, header off the brow). Mitigation: ray-refine lateral when a confident ball pixel exists (§7); editor override. *Open:* default radius-offset direction when no ball pixel is present.
- **Attribution errors in crowds:** nearest-joint within radius can pick the wrong player when bodies overlap. Mitigation: contact-gap gate + confidence + editor confirm. *Open:* should we add a velocity-consistency check (did the ball leave along a vector consistent with that joint's motion)?
- **Bounce lateral remains monocular:** waypoints still need the ball pixel; sparse detection can miss a bounce. Mitigation: operator adds it; flagged as an under-evidenced span. (This is the residual of the known detection-sparsity bottleneck, now confined to non-touch waypoints.)
- **Two interpolators diverging:** Python reference vs UE. Mitigation: the contract in §10 + `segments` shipped from one source; UE validated against the reference in tests.
- **Carry detection false spans:** could glue together genuinely separate touches. Mitigation: conservative thresholds; it's isolated and deferrable.

---

## 18. Out of scope (YAGNI)

- Bidirectional UE→Python confidence feedback loop (noted as a future idea, not built).
- Learned 2D→3D ball lifter (deferred idea 6 from ball-v2-ideas).
- Finer foot-region joints / contact meshes.
- Real-time operation.
