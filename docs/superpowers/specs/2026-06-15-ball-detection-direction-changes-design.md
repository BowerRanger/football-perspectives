# Ball Detection & Direction-Change Touch Detection — Design

- **Date:** 2026-06-15
- **Status:** Draft awaiting review
- **Branch:** `ball-touch-events`
- **Relates to:** [`2026-06-15-ball-touch-events-design.md`](2026-06-15-ball-touch-events-design.md) (the events-mode resolver this feeds), [`2026-06-12-ball-auto-physics-design.md`](2026-06-12-ball-auto-physics-design.md), [`2026-06-14-phase2-mode-sequence-solve.md`](2026-06-14-phase2-mode-sequence-solve.md).

## 1. Problem

Events-mode ball tracking is only as good as its automatic **touch detection**, and that is the weak link. On gberch (a complete reconstruction with 59 hand-placed manual anchors), a pure-auto events run found **2 anchors (1 touch)**.

### 1.1 Diagnosis (measured)

- The ball is detected *confidently* in only **~25 %** of frames (`detection_coverage`: pass1 17.7 % + second_pass 7.5 %). Counting every low-confidence guess, ~62 % of frames carry a position but at **median confidence 0.35**, and there is a **46-frame (1.5 s) run with no ball at all**.
- **Root cause:** WASB runs on the frame **letterboxed to 512×288**. A broadcast ball is ~5–10 px at full HD → ~2–4 px at 512×288 — barely detectable. The detector is starved by resolution.
- Touch detection then needs a clean *velocity break* (≥25° turn or ≥4 px speed change) **and** a player joint within 25 px. With a track this fragmented, only 1 of 23 velocity-break candidates survived.

### 1.2 Reframe

Today: *detect a dense pixel track → find local velocity breaks → match a joint → touch.* New, **direction-change-first**:

> **maximize ball evidence (especially at fast/turn frames) → robustly segment the trajectory so its corners are the direction changes → attribute each corner to a player using dense-pose priors → emit high-recall touch events** (operator prunes in the editor).

## 2. Decisions

| # | Decision | Choice | Why |
|---|----------|--------|-----|
| D1 | Error preference for auto touches | **High recall** (catch all, prune in editor) | A missed touch costs more than a spurious one the operator deletes. |
| D2 | Model strategy | **Inference-only now** (pretrained WASB/YOLO + classical CV + dense poses); fine-tuning is a documented future lever | No labeling effort; fastest path to a result. |
| D3 | Shot count | **Single-shot first**; cross-replay stays purely additive | "Good resolve from one shot; extra shots help, not required." |
| D4 | Scope | **A + B + C**, phased | Each is independently shippable and measurable against gberch pseudo-GT. |

## 3. Architecture & data flow

```
            video + camera track + refined_poses
                          │
   ┌──────────────────────┴───────────────────────┐
   │  A. Detection recall (denser, higher-conf)    │
   │    coarse WASB → high-res ZOOM refine          │  ball_highres_detect.py
   │    cold-start/gap → TILED relocate             │
   └──────────────────────┬───────────────────────┘
                          │  fused per-frame candidates (+velocity)
   ┌──────────────────────┴───────────────────────┐
   │  B1. Motion channel (camera-comp optical flow │  ball_motion_flow.py
   │      / frame-diff) → velocity-native blobs     │
   │  B2. Robust piecewise segmentation             │  ball_traj_segment.py
   │      (rolling/parabolic, RANSAC) → CORNERS     │
   └──────────────────────┬───────────────────────┘
                          │  direction-change breakpoints (_Break)
   ┌──────────────────────┴───────────────────────┐
   │  C. Pose-anchored attribution + hypotheses     │  ball_pose_touch.py
   │     foot kinematics + corridor → touch events  │
   └──────────────────────┬───────────────────────┘
                          │  high-recall BallEvent touches
        generate_auto_anchors → EventResolver (body-pinned) → keyframes
```

All inference-only, all per-shot. The output is the same `BallEvent`/`BallAnchor` stream the existing `generate_auto_anchors` + events-mode `EventResolver` consume — no schema change downstream.

## 4. Phase A — high-resolution detection

New `src/utils/ball_highres_detect.py` wrapping the base `BallDetector`, reusing the proven zoom code in `ball_second_pass` (`_zoom_detect`: crop → letterbox-upscale → `detect_candidates` → `map_crop_candidates`).

- **Primary zoom refine (per frame, when locatable):** after the coarse 512×288 `detect_candidates`, if there is an IMM corridor prediction *or* a coarse hit, run a full-res zoom pass on a `zoom_crop_px` (~320) window centred there. Ball ~3 px → ~10–15 px. Fires whenever `apparent_ball_px < trigger_min_ball_px` **or** coarse confidence `< trigger_max_conf` (recall-max: effectively always-zoom once located). Today this only fires in gaps.
- **Tiled relocate (cold start / after a long gap):** with no corridor, sweep the frame as overlapping full-res tiles, run WASB per tile, merge candidates back to frame coords — re-acquires the ball after it is lost (the current pipeline has no recovery, hence the 46-frame hole).
- **Integration:** inserted in `_detect_loop` between the coarse detect and the IMM update; returns the best gated candidate (feeds IMM as today) **and** retains top-K (for Phase B fusion). Sources tagged `detector_hires` / `tile`. No downstream/schema change. Handles WASB's 3-frame temporal buffer via the existing prime-and-seek.
- **Cost:** one extra ~320² WASB forward per refined frame; tiling only on gaps (bounded). Offline — acceptable.

**Config:** `ball.highres.{enabled, zoom_crop_px, trigger_min_ball_px, trigger_max_conf, always_zoom_when_located, tile_on_gap_frames, tile_grid, tile_overlap_px, top_k}`.

## 5. Phase B — motion channel + direction-change segmentation

### 5.1 Motion channel (`src/utils/ball_motion_flow.py`)

The broadcast rig is fixed-translation PTZ (`t` constant; `R`, focal vary), so consecutive frames differ by a homography `H = K₁R₁R₀ᵀK₀⁻¹` from the camera track. Warp *f→f+1* by `H` and difference → camera pan/tilt/zoom cancels, leaving real motion. The ball is the small, fast, roughly-round blob moving distinctly from larger/slower player blobs. Gate candidate blobs by apparent size (`apparent_ball_px`), speed, non-coincidence with a large player-body blob, and the IMM corridor when available. Output: per-frame motion candidates `[(u,v,score)]` **plus the flow vector** (velocity-native → fires on the blurred turn frames WASB drops). Fuse with Phase-A candidates in the detect loop (agreement → high confidence; motion-only → corridor-gated accept; WASB-only → as today).

### 5.2 Direction-change segmentation (`src/utils/ball_traj_segment.py`)

Replace the fragile local `_raw_break_candidates` with a global robust fit. Over the fused track, fit piecewise segments — rolling (linear in the ground plane) and flight (parabola) — via RANSAC + DP/greedy breakpoint search minimizing `residual + k·(#segments)` (same parsimony idea as `run_beam`'s cost). A segment needs only a few inlier points, so gaps are spanned and corners are found globally, not from noisy 3-frame velocity windows. Each corner emits the existing `_Break` shape (frame, strength, dir_change_deg, dspeed_px, speed_before/after, vy_before/after), so the downstream `_classify_touch`/`_classify_bounce` cascade, the permissive soft-NMS, and `run_beam` (as `Breakpoint`s) consume it unchanged.

`ball_traj_segment.segment_track(...) -> list[_Break]` becomes the breakpoint source inside `detect_events` (and `detect_event_candidates`).

**Config:** `ball.motion.{enabled, flow_method, max_ball_px, min_speed_px, player_exclude_px, corridor_gate}`, `ball.segment.{enabled, min_segment_frames, rolling_residual_px, flight_residual_px, ransac_iters, breakpoint_penalty}`.

## 6. Phase C — pose-anchored touches (`src/utils/ball_pose_touch.py`)

- **Foot/body kinematics:** finite-difference `PlayerContext.joint_world(f±1)` per contact bone → per-frame world velocity + acceleration, projected to pixels. Widen the attribution bone set beyond `(l_foot, r_foot)` to knee/head/chest/hands (the 10 bones already in `BONE_TO_SMPL_INDEX`) so headers, chest controls and keeper saves are catchable.
- **Touch hypotheses:** a contact is likely when a body part (a) sits within the ball corridor (IMM/segment-predicted ball position) and (b) shows a kinematic signature — a foot accelerating/decelerating into the ball, or its path crossing the corridor. Fires **even with no ball detection at that frame** when the surrounding track shows a direction change bracketing the contact window.
- **Fusion with Phase-B corners** (the high-recall attributor that replaces strict `_classify_touch`):
  - *Each corner →* nearest active joint with a **relaxed radius + kinematic-alignment bonus** (foot velocity aligned with the ball's outbound direction boosts score past the old 25 px gate).
  - *Each pose hypothesis with no corner →* emit a touch candidate when the track is consistent with a contact.
  - De-dup to one touch per `(player, frame)` window; rank by `geometric + kinematic + detection-confidence`. `contact_max_gap_m` stays as a relaxed sanity gate; events-mode body-pin uses the joint regardless.
- **Integration:** after `ball_traj_segment` yields corners, `ball_pose_touch.attribute_and_fuse(...)` produces the touch `BallEvent`s consumed by `generate_auto_anchors`. High-recall via permissive `AutoEventCfg` (kinematic alignment replaces the hard pixel gate) + relaxed `AutoAnchorCfg` (`min_event_score`, `contact_max_gap_m`).

**Config:** `ball.pose_touch.{enabled, contact_bones, foot_accel_thresh, corridor_radius_px, kinematic_bonus_weight, hypothesis_min_score}`.

## 7. Validation (no labels)

- **Pseudo-ground-truth:** gberch's 59 manual anchors. New `touch_recall_vs_manual` diagnostic — recall/precision of auto touches vs the manual set, matched by frame proximity (±2) + bone agreement — surfaced in `_ball_diag.json` + the quality report (only when a manual anchor set exists). `detection_coverage` already exists; add motion/segment/pose contribution counts.
- **Unit tests** (light venv, torch-free where possible):
  - **A:** zoom/tile crop→map-back coordinate correctness; tiled relocate finds a planted blob; trigger logic (small/low-conf → zoom).
  - **B-motion:** homography camera-comp maps a static world point to the same pixel post-warp (pan cancelled); blob gate prefers small-fast over large-slow; corridor gating.
  - **B-segment:** a synthetic roll→kick→parabola→bounce track recovers corners within ±k frames, robust to injected gaps + pixel noise; deterministic (fixed RANSAC seed per shot).
  - **C:** finite-diff kinematics correctness; hypothesis fires when a foot accelerates into the corridor with a bracketing direction change; alignment bonus rescues a just-outside-25 px touch.
  - **Integration:** events-mode ball on a synthetic shot (fake detector + fake poses) yields touches at planted contacts.
- **Real-clip:** re-run events-mode ball on gberch (single shot) after each phase; read `touch_recall_vs_manual`. Spot-check origi01/kroupi.

## 8. Phasing & success criteria

- **Phasing:** A → B → C. Each independently shippable and measured on gberch; proceed only when recall rises without precision collapse.
- **Success (single gberch shot):** detection coverage **25 % → >60 %**; recover **≥70–80 %** of manual `player_touch` anchors (±2 frames) at precision where false positives are ≤ ~1.5× true touches (cheap to prune). Multi-shot strictly improves, never required.

## 9. File structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/utils/ball_highres_detect.py` | zoom-refine + tiled-relocate high-res detection (Phase A) | Create |
| `src/utils/ball_motion_flow.py` | camera-compensated motion/optical-flow ball candidates (Phase B1) | Create |
| `src/utils/ball_traj_segment.py` | robust piecewise fit → direction-change `_Break`s (Phase B2) | Create |
| `src/utils/ball_pose_touch.py` | foot kinematics + touch hypotheses + high-recall attribution (Phase C) | Create |
| `src/stages/ball.py` `_detect_loop`/`_detect_shot` | wire high-res + motion fusion | Modify |
| `src/utils/ball_auto_events.py` | breakpoint source = `ball_traj_segment`; attributor = `ball_pose_touch` | Modify |
| `src/utils/ball_auto_anchor.py` | high-recall gate tuning | Modify |
| `src/pipeline/quality_report.py` + diag | `touch_recall_vs_manual`, contribution counts | Modify |
| `config/default.yaml` | `ball.highres/motion/segment/pose_touch.*` | Modify |

Keep each module focused (<400 lines).

## 10. Out of scope / future

- **Detector fine-tuning** (label frames, train a soccer-ball detector) — the D2 future lever if A+B+C plateaus below target.
- **Learned 2D→3D lifting** of the ball (a separate idea in `ball-v2-ideas.md`).
- Changes to the events-mode resolver, UE interpolation, or web editor (covered by the touch-events design).
