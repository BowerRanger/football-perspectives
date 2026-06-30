# Body-Kinematics Touch Proposer — Design

- **Date:** 2026-06-27
- **Status:** Draft (awaiting review)
- **Branch:** `ball-kinematic-touch` (worktree off `ball-touch-events`)
- **Builds on:** [`2026-06-15-ball-touch-events-design.md`](2026-06-15-ball-touch-events-design.md) (sparse body-pinned events) and [`2026-06-15-ball-detection-direction-changes-design.md`](2026-06-15-ball-detection-direction-changes-design.md) (ball-break detection + pose attribution).
- **Goal in one line:** Make the reconstructed player skeleton — not ball detection — the **primary trigger** for ball touches, so we detect contacts the ball detector drops at the moment of contact.

---

## 1. Summary

The ball-touch-events architecture already makes the player body the **authority for a touch's 3-D position**: a `player_touch` event is pinned to the contacting bone's world position via SMPL forward kinematics, sidestepping monocular ball-depth ambiguity. But it still makes ball detection the **trigger for whether a touch exists at all**: `detect_events()` finds *ball-pixel* velocity breaks, and `ball_pose_touch.classify_touch()` only *attributes* those existing breaks to a body part. The empirical wall recorded on prior clips is exactly here — when the ball detector drops the ball *through* the contact (motion blur, occlusion by the leg, ball hidden against the foot), there is **no break to attribute**, so the touch is missed. `ball_pose_touch`'s own docstring states the ceiling: "22 direction-change breaks but only 1 became a touch."

This design adds **one new component** — a **body-kinematics touch proposer** — that generates candidate touches **from limb motion itself**, independent of whether a ball-pixel break exists. It detects the geometric moment a bone is closest to the ball's sight-line (depth-robust), gates that moment on a kinematic contact signature (a foot *kicking*, a head *heading*), and uses the ball only to **confirm and associate** — crucially, *without penalising* candidates where the ball is occluded through the contact. The proposer is **purely additive recall**: its candidates flow into the same merge → body-pin → `BallKeyframeSet` path as today's touches and never delete or override a confirmed one.

Everything downstream of detection is reused unchanged: merge/NMS, `EventResolver` body-pin (D3/D7 of the touch-events design), the sparse `BallKeyframeSet` + derived dense `BallTrack`, the web touch editor's confirm/dismiss UX, the UE interpolator, and the `match_touches` recall harness.

---

## 2. Goals & non-goals

**Goals**
- Detect touches whose **ball pixel is missing at the contact frame**, using the body as the trigger.
- Stay **body-primary**: the trigger is a closest-approach minimum + a kinematic signature; the ball is a modifier, never a gate.
- Be **purely additive**: union with the existing ball-break detector + manual anchors, operator always wins.
- Tune **high-recall**; let the editor prune false positives (consistent with the existing philosophy).
- Be **measurable** on the existing `match_touches` recall harness with no new labelling.

**Non-goals (v1)**
- No new ML model. Reuse `PlayerContext` (GVHMR-derived FK), the smoothed ball track, and the camera track.
- No change to player / camera / refined-pose / hmr_world stages.
- No replacement of the existing ball-break detector or `ball_pose_touch` attribution — the proposer runs alongside them.
- No change to the 3-D resolution rule (body-pin), the keyframe schema, the editor, or the UE interpolator beyond what already exists on the branch.
- No intent modelling (deflection vs deliberate); any body contact is a touch, as today.

---

## 3. Decisions (locked via brainstorming)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| K1 | Authority for ball position | **Body** (skeleton FK), per existing D3 | Touch depth = player-reconstruction error, not monocular ball-depth error. |
| K2 | Product shape | **Sparse events only** (`BallKeyframeSet`), consumer interpolates | Confirms the touch-events default; no dense solve drives output. |
| K3 | Primary touch **trigger** | **Body kinematics** (closest-approach + kinematic gate); ball confirms/associates | Only option that raises recall past the ball-break ceiling. |
| K4 | Proposer algorithm | **Closest-approach (3-D bone↔ball-ray gap) minima + kinematic gate** | Explainable; reuses `contact_max_gap_m` + `ball_pose_touch` scoring; depth-robust. |
| K5 | Baseline | **Build on `ball-touch-events`** (worktree) | main + 105 commits; all sparse-events infra + recall harness already present. |
| K6 | Relationship to existing detectors | **Additive recall** (union + NMS), operator/manual wins | Never deletes a confirmed touch; lowest-risk integration. |
| K7 | Recall vs precision | **High-recall**, editor prunes; precision floor measured after baseline | Matches existing suggestion/confirm UX. |

---

## 4. Architecture & placement

The proposer is a new module, `src/utils/ball_kinematic_touch.py`, invoked from the shared **evidence + events core** of the ball stage (Approach C in the touch-events design), *after* `PlayerContext` is built and the ball track is smoothed, *alongside* the existing `detect_events()` ball-break path.

```
        SHARED EVIDENCE + EVENTS CORE
        1. Detect ball pixels (WASB/YOLO) → IMM smooth → steps
        2. PlayerContext: per-frame FK joints (world + uv + conf)
        3a. detect_events()  ── ball-pixel breaks ──► ball_pose_touch.classify_touch ─┐
        3b. propose_touches() ── body kinematics ──────────────────────────────────► ├─► merge + NMS
        4. carry/possession spans                                                    │   (operator wins)
        5. manual anchors ──────────────────────────────────────────────────────────┘
                                            │ (merged events)
                                            ▼
                                   EventResolver (body-pin, D3/D7)
                                            ▼
                          BallKeyframeSet (sparse)  +  derived dense BallTrack
```

Only box **3b** and the merge wiring are new. Boxes 3a, 4, 5, the resolver, and both outputs are unchanged.

---

## 5. The proposer algorithm (`propose_touches()`)

**Inputs** (all already available in the core):
- `player_ctx: PlayerContext` — per-frame, per-player, per-bone 3-D world position + reprojected pixel `uv` + FK confidence (10 bones from `BONE_TO_SMPL_INDEX`).
- `ball_uv: dict[int, (u, v, conf)]` — smoothed ball pixel track, detection gaps flagged.
- Camera `K, R, t` per frame (ray construction).
- `breaks: list[_Break]` — the existing ball-pixel velocity breaks (used only for confirmation in step 5).
- `cfg: AutoEventCfg` — new fields (§7).

**Step 1 — Ball sight-line per frame.**
For each frame with a ball pixel, build the camera ray through it. Where the ball is missing, **interpolate the ball pixel** across the gap up to `max_ball_gap_frames` (linear in pixel space between the bracketing detections is sufficient for short gaps); flag interpolated frames as low-confidence. Gaps longer than the cap yield no ray (we do not invent contacts across long blackouts).

**Step 2 — Per-(player, bone) gap series.**
For each player `p` and each contact bone `b`, for each frame `f` where the bone FK exists with confidence ≥ `min_fk_conf` **and** a ball ray exists, compute:
- `gap3d(f)` = perpendicular distance from the bone's 3-D world point to the ball ray. **Depth-robust** contact measure; this is `contact_max_gap_m` evaluated every frame.
- `pixgap(f)` = pixel distance between bone `uv` and ball `uv` (association sanity).

**Step 3 — Closest-approach minima.**
Find local minima of `gap3d` over `f` where `gap3d_min ≤ contact_gap_m` (≈ ball radius + slack) **and** `pixgap ≤ touch_relaxed_px` at the minimum. Each surviving minimum is a candidate contact `(p, b, f*)`.

**Step 4 — Kinematic gate** (makes the trigger body-*primary*):
- **Feet / knees:** require a foot pixel-speed (central finite difference, reuse `ball_pose_touch.joint_pixel_velocity`) peak ≥ `kin_min_foot_speed` within ±`kin_window` of `f*`. A planted/standing foot at a gap minimum is **rejected** — that is the ball passing a stationary leg, not a kick.
- **Head:** the gap minimum itself is the heading signature; require a small head 3-D speed (head moving *into* the ball) to reject the ball grazing a still head.
- **Hands (keeper) / shoulders / chest:** relaxed — a hand/arm on the ball line is a save/block; no speed requirement, lower contact threshold weight.

**Step 5 — Ball-confirmation (a *modifier*, never a gate).**
- A ball-pixel `_Break` within ±`confirm_window` of `f*` → **strong confidence boost** (both signals agree → a "double-confirmed" touch).
- Ball **fully occluded** through the candidate (only interpolated pixels in the window) → **no penalty**. This is the occlusion-rescue case the proposer exists for; the kinematic gate carries it.
- Ball **clearly detected through the candidate but its motion does not change** → **downweight** (foot passed near but did not touch — the main precision lever).

**Step 6 — Score & emit.**
```
score = w_gap     * (1 - gap3d_min / contact_gap_m)
      + w_kin     * kin_strength            # normalised foot/head speed
      + w_confirm * ball_confirm            # +boost / 0 / -downweight from step 5
      + w_fk      * fk_conf
      - w_interp  * interp_penalty          # ball pixel was interpolated at f*
```
Clipped to [0, 1]. Emit `BallEvent(kind='touch', frame=f*, player_id=p, bone=b, score=...)` for every candidate at or above a low `min_emit_score` (high-recall default). The blend mirrors `ball_pose_touch`'s existing weighting so the two detectors produce comparable scores.

**Output:** `list[BallEvent]` — handed to the same merge path as the ball-break-attributed touches.

---

## 6. Merge, dedup & confidence

Three touch sources combine into one anchor set:
1. **Manual anchors** (`*_ball_anchors.json`) — always win.
2. **Ball-break-attributed touches** (`ball_pose_touch.classify_touch`, existing).
3. **Kinematic-proposer touches** (new, §5).

Pipeline:
- Union auto sources (2) + (3); **temporal NMS** keyed by `(player_id, bone)` within `nms_window` frames — higher `score` wins (so a double-confirmed touch beats a kinematic-only one at the same contact).
- Merge with manual via the existing `merge_anchors` (operator wins, suppress radius 3 frames).
- **Carry-span** detection (existing) collapses dribble micro-touch storms into one `carry` event.

Net effect: the proposer **adds** the touches the ball-break path missed; it never deletes or overrides a confirmed one. Each surviving auto touch carries its blended `score` into `BallAnchor.confidence`; the web editor already renders auto events as dashed **suggestions** with confirm/dismiss, and dismissals persist as suppressions so re-runs do not resurrect them.

**3-D resolution is unchanged:** every surviving touch resolves through the existing `EventResolver` body-pin path (bone world FK + ball-radius offset along the ray; lateral ray-refine when a confident ball pixel exists) → `BallKeyframeSet` (authoritative sparse) + derived dense `BallTrack` (compat for glTF / web / quality).

---

## 7. Config & flag

New `AutoEventCfg` fields, all gated behind `ball.auto_events.use_kinematic_proposer` (default **on** in `events` solver mode; A/B-able off). Defaults added to `config/default.yaml`, refined after the first baseline run (§8):

| Field | Meaning | Initial default |
|-------|---------|-----------------|
| `use_kinematic_proposer` | master flag | `true` |
| `contact_gap_m` | max bone↔ray 3-D gap for a contact | `0.30` |
| `touch_relaxed_px` | max bone↔ball pixel distance at the minimum | reuse existing `60.0` |
| `max_ball_gap_frames` | longest ball-pixel gap we interpolate across | `6` |
| `min_fk_conf` | min bone FK confidence to consider | `0.3` |
| `kin_window` | ± frames to find the kinematic peak around `f*` | `2` |
| `kin_min_foot_speed` | min foot pixel-speed (px/frame) for a kick | `8.0` |
| `confirm_window` | ± frames to look for a ball `_Break` | `3` |
| `nms_window` | temporal NMS half-width (frames) | `2` |
| `w_gap`, `w_kin`, `w_confirm`, `w_fk`, `w_interp` | score weights | `0.35, 0.3, 0.25, 0.1, 0.15` |
| `min_emit_score` | high-recall emission floor | `0.25` |

---

## 8. Validation & success metric

Use the existing `src/utils/ball_touch_recall.py` `match_touches` harness against gberch's 59 hand-placed anchors as pseudo-ground-truth, at `frame_tol=2`, `require_bone=True`. Report recall/precision for three configurations to isolate the proposer's contribution:
1. **Ball-break only** (proposer off) — the current ceiling.
2. **Proposer only.**
3. **Union + NMS** (shipping config).

**Success gates:**
- **Primary:** union recall rises **materially** over the ball-break-only baseline (baseline is near-floor per the `ball_pose_touch` docstring).
- **Guardrail:** post-NMS, pre-editor precision stays usable — provisional floor **≥ 0.5**, finalised once the baseline is measured (high-recall/operator-prune is the design intent, so recall is weighted over precision).
- **Overfit check:** validate on a **second labelled clip** if gberch alone risks tuning to one match; we avoided the learned-classifier approach partly to limit this risk, but threshold tuning can still overfit.

A small CLI/test wrapper runs all three configs and prints the table so re-tuning is one command.

---

## 9. Testing plan (TDD)

Unit tests for `ball_kinematic_touch` (pure, torch-free — feed synthetic `PlayerContext` + ball-pixel dicts):
- **Ball present through contact:** foot passes through the ball with detections at every frame → touch detected at the gap minimum, correct bone/player.
- **Ball-detection gap at contact (headline case):** ball pixels present approaching and leaving but **missing at `f*`** → touch still detected via interpolated ray + kinematic gate.
- **Stationary foot, ball grazes past:** planted foot at a gap minimum, low foot speed, ball motion unchanged → **rejected** (precision).
- **Header / keeper-hand variants:** head moving into ball → detected; hand on ball line with no speed → detected (relaxed gate).
- **Long blackout:** gap > `max_ball_gap_frames` → no candidate invented.
- **Merge precedence / NMS:** manual beats auto; higher-score auto beats lower at the same `(player, bone)` within `nms_window`; double-confirmed beats kinematic-only.

Integration: the `match_touches` recall harness on gberch (§8) wired as a test that asserts union recall ≥ baseline.

---

## 10. Phasing

1. **Proposer module + unit tests** — feet first, then head/hands. (Pure, no stage wiring; fully TDD-able on the Mac without torch.)
2. **Wire into the evidence+events core** behind `use_kinematic_proposer`; implement merge + NMS.
3. **Measure** on the recall harness (three configs); tune thresholds; set the precision floor; add the second-clip overfit check.
4. **Editor / carry-span interaction check** — mostly reused; verify dashed-suggestion rendering and that carry-span collapse still behaves with the denser proposer output.

Phase 1 is independently valuable and low-risk; phases 2–4 integrate and tune.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| **Precision collapse** from dribble scrums (many feet near the ball) | Kinematic gate rejects planted feet; carry-span collapses micro-touches; high-recall + editor prune is the intended workflow. |
| **Interpolated ray is wrong** across a gap (ball curved/bounced inside the gap) | Cap `max_ball_gap_frames` small; flag interpolated frames with `interp_penalty`; bounces inside a gap are rare at touch timescales. |
| **FK jitter** creates phantom gap minima | `min_fk_conf` gate; require a kinematic peak (jitter is high-frequency, not a sustained approach); NMS de-dups. |
| **Overfitting thresholds to gberch** | Second-clip overfit check; explainable hand-tuned weights (not a learned model) keep behaviour inspectable. |
| **Double-counting** vs the ball-break detector | NMS keyed by `(player, bone)` within `nms_window`; double-confirmed simply scores highest. |

---

## 12. Out of scope / future

- Learned contact classifier (Approach 3) — deferred; the recall harness makes it a drop-in future experiment if hand-tuned weights plateau.
- Impulse-direction modelling (Approach 2) — could later sharpen precision by checking the ball's post-contact direction against foot motion; folded in as a future `w_confirm` refinement, not v1.
- Finer foot regions (instep/laces) — remains the inferred `touch_type`/`spin`, unchanged.
