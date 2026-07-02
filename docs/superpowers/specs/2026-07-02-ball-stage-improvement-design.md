# Ball Stage Improvement — Auto Tracking, Manual Landmark Layer, Shot Chains

Date: 2026-07-02
Status: draft for review
Supersedes nothing; extends `2026-06-15-ball-touch-events-design.md` and
`2026-06-27-body-kinematics-touch-proposer-design.md`. Idea inventory referenced from
`2026-06-12-ball-v2-ideas.md`.

## 1. Review of the current ball stage

### 1.1 What exists (post touch-events merge, main @ 0582b71)

Three-pass architecture in `src/stages/ball.py` (`run()` ~line 681):

1. **Detect** — WASB per frame, appearance bridging, corridor-gated second pass
   (`ball.second_pass.*`, on), foot-guided zoom (`ball.foot_guided.enabled: false` —
   validated mechanism, over-fires without a better detector). Raw observations persisted.
2. **Triangulate** — cross-replay 3-D fixes for synced multi-shot groups
   (`ball.cross_replay.*`), `*_ball_fixes.json`.
3. **Solve** — default `ball.solver: events`: sparse 3-D events (touches body-pinned to
   the contacting SMPL joint, bounce/goal/rest waypoints, carry spans) resolved by
   `ball_event_resolver.py`; segments classified by `ball_segments.py`; dense track derived
   by the reference interpolator (`ball_interpolate.py`, byte-parity with the UE evaluator).
   Piecewise/global dense solvers remain behind the flag.

Touch candidates come from two unioned sources: pixel-track direction-change
segmentation (`ball_traj_segment.py`) attributed to bodies (`ball_pose_touch.py`), and the
body-kinematics proposer (`ball_kinematic_touch.py`, on by default) which triggers on limb
motion and uses the ball only as a confidence modifier. Auto events become auto anchors
(`generate_auto_anchors`), same `BallAnchorSet` schema as the manual editor; manual wins
within `suppress_radius_frames`.

Goal geometry is already fully modelled (`src/utils/goal_geometry.py`): posts and crossbar
as 3-D lines, back/side nets as planes, ray-intersection resolution to a world point;
`_classify_goal_line` / `_classify_net` emit `goal_impact` events with
`goal_element ∈ {post, crossbar, back_net, side_net}`.

The manual editor (`ball_anchor_editor.html`) authors 13 states including `player_touch`
(player + bone, `/joints-near` click-suggest, `touch_type="shot"`, spin presets) and
`goal_impact` (element dropdown). Promote-auto-to-manual exists. Every manual save appends
to the fine-tune label corpus (`ball_label_corpus.py` → `output/ball_finetune/`).

### 1.2 What's proven broken or missing

- **The empirical wall is detector quality at touch moments.** On gberch (59 manual
  anchors as ground truth) break-only auto touch recall is 0/8 despite correct
  segmentation and excellent pose (named player's foot 7–22 px from the true ball pixel at
  every touch). WASB emits *confident* false positives 800–1650 px wrong; ~30 % of frames
  are IMM gap-fill extrapolations. Cleaning removes wrong detections but cannot add right
  ones. Conclusion on record: fine-tuning the detector is the unlock; foot-guided zoom and
  the recall already built are gated on it.
- **The kinematic proposer is merged but unvalidated on GPU** (Phase 3 of the 2026-06-27
  spec): the two-config recall report on gberch has not been run; thresholds untuned; no
  stage-level integration test proves the proposer actually fires inside `BallStage.run`.
- **No ball quality timeline.** The camera anchor editor has a per-frame confidence strip;
  the ball editor and viewer surface nothing — the operator cannot see unexplained spans,
  detection coverage, or "annotate here next" cues, even though `*_ball_diag.json` and the
  quality report already carry them.
- **Span events have no UI** (carry needs hand-edited JSON `end_frame`); the event-list
  panel with confirm/dismiss from the touch-events spec was deferred.
- **Shot-at-goal is not a first-class flow.** All the parts exist (strike touch, goal
  geometry, two-knot gravity arcs that fully determine monocular depth) but nothing
  composes them: no guided authoring, no chained deflection solve, no auto pairing of
  strike → impact.
- Known lesser gaps: carry interpolates linearly; `ball_motion_flow.py` is built but
  unwired; spin bounce-coupling is monocular-degenerate (off pending 3-D fixes).

## 2. Goals and non-goals

**Goals**

1. Raise automatic touch/track recall using the source footage plus outputs of earlier
   stages (tracking boxes, refined poses, camera), attacking the proven bottleneck.
2. Give the operator a fast, guided manual layer: see where the reconstruction is weak,
   fix it with the fewest clicks, and let clicks include *pitch-landmark-coincident* ball
   fixes that resolve monocular depth for free.
3. Make goal-mouth shots a first-class primitive: mark the strike and the terminal impact
   (post / crossbar / net / goalkeeper), solve the motion between them exactly.

**Non-goals**

- Replacing the events-mode architecture (it is the right product shape; D3 stands).
- Learned 2D→3D lifting, learned contact classifier, UE realism layer (net cloth, post
  ping) — stay deferred per prior specs.
- Real-time / live operation.

## 3. Approaches considered

**A. Incremental hardening of the current architecture (recommended).** Keep events mode;
execute the already-identified unlocks (proposer validation, detector fine-tune loop,
context-prior rejection), add the missing manual UX, and compose existing primitives into
a shot-chain solver. Low risk; every piece lands on validated machinery; the label
flywheel means the manual layer continuously improves the auto layer.

**B. Detection-first rebuild.** Swap/augment WASB with a newer joint player+ball tracker
and offline global data association over the whole clip. Attacks the root cause hardest,
but high effort and risk, and it discards a working three-pass pipeline before the cheap
levers (fine-tune + priors) have been tried. Revisit only if A's fine-tune plateaus.

**C. Annotation-first.** Accept detector limits; optimise for a ~2-minute guided manual
pass per shot, auto layer only proposes. Lowest ML risk but caps the product at operator
throughput and abandons proven-recoverable recall.

**Decision: A, absorbing C's UX** (guided annotation is needed by A's flywheel anyway).
B deferred.

## 4. Layer 1 — Automatic tracking improvements

### 4.1 Finish what's merged: kinematic proposer validation (first, cheap)

Run the ball stage on gberch twice (`ball.kinematic_touch.enabled` false/true), then
`python scripts/ball_touch_recall_report.py <manual> <break_only> <union>`. Tune
`contact_gap_m`, `kin_min_foot_speed`, `min_emit_score` toward union recall materially
above break-only with post-NMS precision ≥ 0.5 (the 2026-06-27 acceptance bar). Add a
stage-level integration test asserting the proposer fires inside `BallStage.run` (its
try/except injection can silently disable it today).

### 4.2 Context-prior false-positive suppression (inference-only, uses prior stages)

`ball_track_clean` removes off-image and teleport spikes but not the stable, confident
wrong-object detections (frame-top blobs, scoreboard). Add a plausibility prior computed
per detection from earlier-stage outputs, applied as a *score modifier* (never a hard
gate, preserving recall):

- **Pitch prior**: ray-cast the detection through the solved camera; distance of the
  ground intersection from the pitch polygon (+ margin) penalises crowd/stand blobs.
- **Player-context prior**: distance to the nearest tracked player box / projected joint —
  the ball far from every player for many consecutive frames is rarely real in highlights.
- **Static-in-image persistence**: detections whose *pixel* position is near-constant
  while the camera pans (i.e. glued to the image, not the world) are overlays/scoreboard.

New pure module `src/utils/ball_context_prior.py`, wired into `_detect_shot` scoring and
the second-pass `accept_min` comparison; config `ball.context_prior.*` (on by default,
weights small). Acceptance: gberch's known top-of-frame false positives drop below
`accept_min` while origi01/origi02/kroupi `detection_coverage` does not regress.

**Measured outcome (2026-07-02, Phase 3 execution):** the score-modifier design was
revised to a **factor veto** after measurement. Multiplying penalties into absolute
confidence regressed origi02 coverage 0.512→0.225 (a genuine long ball is simultaneously
"far from players" and "ground-ray off-pitch", and origi02's real detections have median
confidence 0.449, far below the arithmetic's 0.8 assumption). Shipped semantics: a
detection is dropped only when the prior's *factor* falls to `drop_below` — reachable only
by signal pairs involving the static-under-pan overlay signature — and confidences are
never scaled. Result: origi01/origi02 coverage bit-identical to baseline, static-overlay
suppression proven by stage test; gberch's real-clip FP count unchanged (its residual FPs
are moving crowd blobs, not static overlays — Phase 4 territory).

### 4.3 Execute the detector fine-tune loop (the unlock)

The scaffold exists (`ball_finetune_export.py`, corpus flywheel, README). Remaining work
is operational, in order:

1. **Densify the corpus**: exporter variant that also emits high-confidence *solved-track*
   frames (accepted evidence, not gap-fill) as weak labels around each gold anchor — WASB
   needs the centre frame + 2 neighbours per sample; one clip's ~60 gold anchors alone is
   too thin. Optionally mix in public soccer ball data (SoccerNet) for regularisation.
2. **Author `train.yaml`** for the vendored WASB (mirror eval.yaml + loss/optimiser +
   `runner: train_and_test`), init from `wasb_soccer_best.pth.tar`.
3. **Train on GPU; evaluate** with the existing harnesses: `detection_coverage` per clip,
   touch recall report on gberch, `tests/test_ball_anchor_accuracy.py` unchanged.
4. On success, swap `ball.wasb.checkpoint` and **re-enable `ball.foot_guided`** — its
   mechanism is validated (recovered 4/8 real touches) and was disabled only because the
   stock detector cannot disambiguate the ball near feet.

Acceptance: union touch recall on gberch > 0 (target ≥ 4/8) at precision ≥ 0.5;
`detection_coverage.total` improves on origi02 (worst clip, 0.58) without regressing
origi01/kroupi; anchor-accuracy harness stays green.

### 4.4 Touch bone-attribution refinement (added 2026-07-02 from Phase-1 validation data)

The gberch recall run showed the strict-recall gap is mostly *attribution*, not detection:
with the same ±2-frame tolerance, ignoring the bone claim takes recall from 0.25 to 0.50
in both configs — half the touch moments are found but pinned to the wrong body part
(l_foot vs r_foot, knee vs foot). Add a post-pass that re-picks each emitted touch event's
(player, bone) by minimising the 3-D bone↔ball-ray gap over a small window around the
event frame (the `ray_gap_series` machinery from the kinematic proposer), keeping the
original attribution when the refinement margin is ambiguous. Config
`ball.touch_attribution.*`. Acceptance: gberch strict union recall rises
from 2/8 toward the 4/8 loose ceiling with no new false touches (refinement never adds or
removes events, only relabels).

**Measured outcome (2026-07-02, Phase 3 execution):** shipped **default-off**. Three
measurement waves on gberch showed relabelling by bone↔ball-ray gap trusts exactly the
evidence that is broken on detector-limited clips — the ball pixel at the touch moment —
and overwrites the kinematic proposer's body-derived correct labels (union recall halved
to 1/8 with it on, isolated from the context prior by an ablation run). The mechanism is
unit-proven and stays available per-clip; its payoff is gated on Phase 4 detector quality,
after which the default flips back on with re-measurement.

### 4.5 Deliberately not doing (now)

Motion-flow candidate source stays unwired until 4.3 lands (same disambiguation problem);
global mode-sequence solver stays opt-in; spin bounce-coupling stays off until cross-replay
fixes make it identifiable.

## 5. Layer 2 — Manual layer: guided annotation + landmark fixes

### 5.1 Ball quality timeline in the anchor editor

Mirror the camera editor's confidence strip under the ball editor's seek bar, fed from
existing sidecars (`*_ball_diag.json`, observations, keyframes):

- per-frame detection confidence / coverage band (pass1 vs second-pass vs gap);
- event markers (auto = dashed, manual = solid) and segment kinds;
- **red spans = "annotate here next"**: underconstrained flight spans (< 2 hard knots),
  unexplained solver spans, long detection gaps — ranked, click-to-seek.

Server: one read-only `GET /ball-quality/{shot_id}` aggregating what the diag sidecar and
quality report already compute. No new stage work.

### 5.2 Event-list panel + span editing

Land the deferred touch-events UI: chronological auto ∪ manual event list with
confirm (promote) / dismiss per row, and drag handles on the timeline for `end_frame`
span events (carry). Dismissals persist (a `dismissed_auto` list in the manual
`BallAnchorSet` sidecar) so re-runs don't resurrect rejected suggestions.

This panel is also the **exhaustive touch-annotation workflow** (added 2026-07-02 from
Phase-1 validation data): the gberch "precision" numbers are a lower bound because the
59-anchor GT is a trajectory-constraint set, not an exhaustive touch inventory — the
biggest FP cluster (P003, frames 193–271) is a dribble where several auto touches are
plausibly real but unannotated. Walking the event list once, confirming real touches and
dismissing false ones, yields a trustworthy precision denominator (confirmed = GT,
dismissed = true FP) and every confirmation feeds the fine-tune corpus flywheel. The
recall report gains an FP breakdown (dismissed vs unreviewed) so tuning is done against
reviewed data only.

### 5.3 Landmark-coincidence ball fixes (the new "manual landmark annotations")

A new, very cheap class of hard 3-D knots: the operator marks a frame where the ball
visibly coincides with a known pitch feature — sitting on the penalty spot, crossing a
line, on the corner arc, kick-off. The pitch landmark/line catalogue
(`pitch_landmarks.py`, `pitch_lines_catalogue.py`) already carries exact world
coordinates, and the camera editor already snaps clicks to lines.

- UI: in the ball editor, a "pitch fix" mode — click the ball, pick the feature from the
  same palette the camera editor uses (or accept the nearest-feature auto-suggestion from
  the projected overlay).
- Resolution: for a *point* landmark, world x,y is the landmark's (z = ball radius); for a
  *line*, intersect the clicked-pixel ground ray with the line (1-D snap). Stored as a
  `grounded` anchor with a new optional `landmark: str` field on `BallAnchor` (schema
  addition, backward compatible) so provenance survives.
- Why it matters: these are exact hard knots that bracket flights — the C2 two-knot
  machinery then resolves monocular depth for the whole arc. One click can fix a
  metre-scale depth error; it is the manual counterpart of a cross-replay fix.

### 5.4 Flywheel stays the engine

Every manual click (anchors, pitch fixes, shot chains below) continues appending to
`output/ball_finetune/` — the manual layer is also the training set for Layer 1's
fine-tune. Add the densified exporter from 4.3(1) so each click yields a full 3-frame
WASB sample.

## 6. Layer 3 — Shot chains (goal-mouth system)

A **shot chain** is a first-class composite: *strike → [deflections…] → terminal state*,
each node a hard 3-D knot, each segment a ballistic arc fully determined by its two
endpoint knots plus gravity (the "IK" of the user's framing: endpoints constrain the
in-between motion exactly; no iterative solve is needed for a single arc, and chains
subdivide into arcs with impulse-consistency checks at the joints).

### 6.1 Node types (all resolvable today)

| Node | 3-D authority | Mechanism (exists) |
|---|---|---|
| Strike | contacting player joint | `player_touch` + `touch_type="shot"`, body-pinned, ray-refined |
| Post / crossbar hit | ray ∩ 3-D line | `goal_impact` + `resolve_goal_impact_world` |
| Net hit | ray ∩ net plane | `goal_impact` (back_net/side_net) |
| Keeper save | keeper joint (hand/body) | `player_touch` on keeper bone via `/joints-near` |
| Ground bounce (deflection) | z = ball radius | `bounce` |
| Rest / out | grounded / off_screen | `grounded`, `off_screen_flight` |

### 6.2 Authoring flow (manual, guided)

New "Shot" quick action in the ball editor:

1. Operator clicks the strike frame on the striker's foot → `player_touch(shot)` prefilled
   by `/joints-near` (exists).
2. Operator clicks the terminal frame: on the goal frame the click resolves via
   `resolve_goal_impact_world` and auto-selects the element (nearest of
   post/crossbar/net by ray residual — today the operator picks the element manually;
   add ray-based auto-suggest); on the keeper it resolves via `/joints-near` restricted
   to the keeper's track (hands first).
3. Optional intermediate clicks (crossbar-then-line bounce, post-then-net) insert
   deflection nodes.
4. Save produces ordinary anchors plus a `shot_chain` grouping record (list of member
   anchor frames) in the manual sidecar — no new solve path is *required* for the anchors
   to work, the chain adds validation + segment typing.

### 6.3 Solve + validation semantics

For each consecutive knot pair the events resolver already yields a ballistic segment;
the chain adds:

- **Time-of-flight sanity**: implied launch speed from (distance, frame gap); warn outside
  a shot envelope (default warn band 8–45 m/s, config `ball.shot_chain.*`) — catches a
  mis-clicked frame immediately, in the editor via `/ball-anchors/{shot}/preview`.
- **Impulse consistency at deflections**: post/crossbar reflection with restitution inside
  the existing envelope; net capture requires speed collapse (reuse
  `goal_net_speed_drop_ratio`); keeper deflection starts a new segment from the hand knot.
- **Spin**: the existing `spin` preset on the strike seeds Magnus refinement over the
  chain's first segment (bounded, per Phase-3 spin rules).
- **After the terminal node**: net/keeper-catch → decay to `rest` waypoint inside the goal
  (existing rest semantics); deflections continue as normal events.

Implementation: `ball_shot_chain.py` (pure: chain validation + segment typing hints) +
schema addition (`shot_chains` list in the manual sidecar, member frames only) + editor
quick action + preview warnings. The interpolator and UE evaluator are untouched — chains
compile down to existing keyframes/segments (parity tests stay byte-equal).

### 6.4 Auto-proposal

`detect_events` already emits `goal_impact` events. Add a pairing pass: for each
goal_impact, find the last preceding touch within a window whose ball direction points
goalward → propose a shot chain (dashed in the event list, confirm/dismiss). This is pure
composition over existing events; no new detection.

## 7. Phasing

| Phase | Contents | Exit criterion |
|---|---|---|
| 1 | 4.1 proposer validation + integration test; 5.1 quality timeline | recall report published; timeline usable on gberch |
| 2 | 6 shot chains (authoring, validation, auto-proposal); 5.3 landmark fixes | a goal-mouth shot on origi01/gberch reconstructs from 2 clicks with preview warnings working |
| 3 | 4.2 context prior; 4.4 attribution refinement; 5.2 event-list/span UI + exhaustive-annotation workflow | gberch strict union recall ≥ 4/8 via relabelling; frame-top FPs suppressed, no coverage regression; event list usable for a full review pass — superseded by §4.2/§4.4 measured outcomes: recall-via-relabelling moves to Phase 4; shipped bars were baseline-parity, met bit-identically. |
| 4 | 4.3 fine-tune loop end-to-end; re-enable foot_guided | recall ≥ 4/8 @ precision ≥ 0.5 on gberch; coverage lift on origi02 |

Phases 1–2 are GPU-light and land the user-facing systems; Phase 4 is the ML lift and is
deliberately last so the corpus (grown by Phases 2–3 usage) is thicker.

## 8. Risks / open questions

- **Fine-tune data volume**: even densified, one operator's corpus may be thin; SoccerNet
  mixing helps but domain gap (broadcast zoom levels) is unmeasured. Mitigation: eval
  strictly on held-out project clips; keep stock checkpoint as fallback via config.
- **Context prior on unusual clips** (ball genuinely far from players, e.g. long goal
  kicks): weights must stay small and modifier-only; validated against origi01 long arcs.
- **Keeper pose quality**: GVHMR struggles with diving keepers (known deferred item);
  keeper-save knots inherit that error. The ray-refinement (lateral from click) bounds it;
  full fix stays out of scope.
- **Element auto-suggest ambiguity** near post/net junction: fall back to manual dropdown
  (already exists) when residuals are within a tie band.
