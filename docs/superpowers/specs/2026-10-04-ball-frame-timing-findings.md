# Ball frame timing — findings from authoring the first Ball Studio truth

Date: 2026-10-04 · Branch: `worktree-gberch-shorts` · Status: findings + proposed fixes (fixes need a user decision)

Authoring the origi01 demo truth in Ball Studio (two synced angles, exact
frame decodes) exposed three timing problems that sit underneath the ball
stage. They were measured, not inferred. Probe scripts and raw outputs live
in `docs/superpowers/notes/ball-frame-timing/`.

## 1. Legacy editors label the previous frame (operator anchors one frame late)

`frontend/src/pages/ball-anchor-editor/use-frame-player.ts` (and the camera
anchor editor's copy) seeks to `currentTime = f / fps` and reads the frame
back as `Math.round(currentTime * fps)`.

- Verified in Google Chrome (`seek_probe.js`): seeking to `416 / 30`
  displayed decoded frame **415** (float time lands just below the frame's
  PTS). Seeking to `(f + 0.5) / fps` displayed the right frame every time.
- Pausing playback mid-frame and rounding labels the frame shown as the
  *next* one.
- Measured on discriminable moving-ball anchors (`q4*.py`): origi01 14/19
  (74 %) match exact frame N−1, gberch 11/18 (61 %), kroupi01 2/4. None
  match N±2. Examples (origi01): anchor "415" = exact f414, "417" = f416,
  "445" = f444, "454" = f453.
- Ball Studio already uses the mid-frame seek + `floor` convention.

## 2. The fine-tuned WASB detector learned the lag

`scripts/build_finetune_corpus.py` takes gold labels from those anchors
(frame label as clicked) and weak labels from a track anchored to them, so
the corpus pairs image N with the ball of image N−1. Running the detector
directly on exact frames (`q1.py`, origi01 f414–f454, 11 hand-verified ball
pixels):

| checkpoint / heatmap | median error at lag 0 | best lag |
|---|---|---|
| fine-tuned v1, hm2 (what the stage uses) | 17.1 px | +1 (5.5 px) on 9/11 frames |
| fine-tuned v2, hm2 | 30.1 px | +1/+2 |
| stock `wasb_soccer_best` | no lock in this window | — |

The ball stage itself indexes frames correctly (sequential reads, exact
seeks in the second pass; `src/stages/ball.py` ~1443–1467, 1558–1570). The
lag is entirely in the weights.

## 3. All 30-fps test clips are 25-fps content (pulldown)

`dup_scan.py`: gberch, gberch-2, origi01, origi02 and kroupi01 repeat one
display frame in six (16 %); the 25-fps saka shots have none. Display frame
`f` shows content up to ~27 ms away from `f / fps` (a sawtooth), and a
repeated frame is a full frame stale. Two angles of one moment can be out
of phase (origi01 vs origi02 before ref 419).

Ball Studio now solves on content time (`src/utils/frame_cadence.py`,
commit 7168597): repeated frames are detected per video, keys/soft
observations/projections use each view's true instant, and `/triangulate`
warns when two views show instants > 0.3 frame apart.

## The origi sync offset holds (fresh picks)

Ball Studio's sync probe re-solves a *held* click at the other camera's
neighbouring frames; on k440 it fell monotonically toward −140. Re-picking
the ball fresh in origi02 frames 272–276 against exact origi01 frames
(`sync_check.py`) shows the stored −142 is right: 1.4–2.6 px with the ball
on the grass (z ≈ 0) at −142, 7–14 px and z off by 0.3–0.8 m one frame
either side. A held click mostly measures camera motion, so the probe now
says so and offers no verdict.

## Effect on the origi01 demo truth

Same clicks, three solver/pick regimes (max soft-observation residual on
the 411→454 span):

| regime | key residuals | soft outliers (> 12 px) |
|---|---|---|
| operator anchors as pixels, uniform time | ≤ 7.4 px | 415 (18 px), 418 (15 px) |
| exact-decode picks on frames fresh in both views, uniform time | ≤ 3.7 px | 414 (17 px), 450 (21 px), 452 (16 px) |
| + content time (pulldown-aware) + exact strike pixel at 440 | ≤ 3.3 px | none (max 7.6 px) |

## First score of the pipeline against authored truth (origi01 group)

`scripts/eval_ball_truth.py --output output-origi-shorts --group origi01`
(44 frames, ref 411–454):

| pipeline track | 3-D p50 | 3-D p95 | ≤ 0.2 m | reproj p50 own / other view | events |
|---|---|---|---|---|---|
| origi01 | 0.70 m | 1.75 m | 2 % | 18 / 29 px | 5/5 matched; line-cross point 0.19 m, 0.24 frame |
| origi02 | 1.80 m | 7.35 m | 0 % | 59 / 47 px | 2/5 matched; no line cross |

Shifting the origi01 pipeline track by one frame (pipeline N+1 vs truth N)
only moves p50 0.70 → 0.64 m: the detector lag is real but most of the
error is monocular depth, which is what multi-angle triangulation fixes.

## Proposed fixes (need a decision)

1. **Editor frame reading** — use the Ball Studio convention in both legacy
   editors (being done on this branch; existing anchors on disk are NOT
   relabelled).
2. **Relabel existing anchors** — per anchor, not a blanket −1: pick N−1 /
   N / N+1 by the ball's position on exact frames (`q4b.py`'s blob finder or
   v1 hm2 at N+1). Touch events / keyframes keyed to those frames follow.
3. **Re-fine-tune as v3** on corrected labels (keep v1 for the existing
   gates/caches; key the detection cache on checkpoint + frame index, not
   only image content). Pass bar: hm2 median ≤ 5 px at lag 0, best lag 0 on
   ≥ 9/11 frames (`q1.py`).
4. **Pulldown in the ball stage** — reuse `frame_cadence` so the physics fit
   uses content time and repeated frames aren't treated as fresh evidence.
5. **Verify** with `eval_ball_truth.py` on every authored group, the ball
   regression gate, and touch-recall validation — before and after.
