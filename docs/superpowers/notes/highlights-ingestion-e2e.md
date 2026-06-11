# Highlights ingestion — Liverpool 4-0 Barcelona E2E notes

**Date:** 2026-06-11
**Input:** `test-media/Liverpool vs Barcelona (4-0) … Highlights.mp4`
(441 s, 1080p25) → `output-highlights/`, default config, `--stages prepare_shots`.

## Iteration 1 (as-implemented defaults)

32 shots; 8 reaction + 2 "transition" excluded; 5 groups. **Bug found:**
the fade rule classified two long gameplay spans as transitions —
s002 (4–48 s, the *entire first-goal sequence*) and s019 (27 s) — because
one sampled frame dipped below the brightness threshold. The deleted
legacy stage had a fade duration cap that the port dropped.

**Fix:** `classify.transition_max_duration_s: 2.0` — the fade rule only
fires on spans ≤ 2 s (hard-black spans of any length still count).
Covered by `test_long_span_with_dark_dip_is_not_a_transition`.

## Iteration 2 (with fix)

- 32 shots: 24 gameplay, 8 reaction (intro montage, crowd cuts,
  full-time celebrations) — all correct on thumbnail inspection.
- 5 groups, each anchored by a wide real-time shot followed by replays:
  - g01 4–48 s (the long single-shot opening sequence)
  - g02 50–205 s (live + 6 replays), g03 205–301 s, g04 301–408 s,
    g05 412–427 s
- Alignment: most members `motion_profile` at 0.53–0.97 confidence;
  4 `low_confidence` align-ends fallbacks. Quality report flags
  g02/g04 in `low_confidence_groups`.

## Dashboard (port 8001, headless-chromium verified)

Groups board, ingest drop-zone, dropped tray (8 restorable), per-group
sync tabs (singleton groups correctly excluded), scale/REPLAY/confidence
badges, timeline with staggered blocks + cursor — all render with zero
console errors. Live-server smoke of `PATCH /api/shots/bulk`
(discard/restore/move) and `POST /api/sync/auto` against the real
output dir behaved as the tests promise.

## Iteration 3 (2026-06-11, cut tightening + celebration close-ups)

**Missed-cut diagnosis** (`scripts/diagnose_cuts.py` +
`scripts/eval_cut_detectors.py`): the frame-diff spike analysis found
36 violent cuts; the default AdaptiveDetector caught 28. All 8 misses
were visually confirmed real (two inside the 44 s s002, four in the
345–352 s celebration sequence). Root cause: AdaptiveDetector
normalises by a 2-frame *mean* window, which continuous fast action
inflates. Detector sweep results (hits/36, extras): adaptive 3.0 =
28/3, adaptive 2.0 = 32/5, content 27 = 30–32/55–104. Fix: **spike
rescue** — union the detector's cuts with outliers of the diff curve
against a 25-frame median/MAD window (`diff_spike_cuts`). Adaptive 3.0
∪ rescue = **36/36 with 3 extras**. After the fix all 8 previously
missed frames are manifest boundaries.

**Close-up discard**: pitch ratio keeps player close-ups (grass behind
them). YOLO person-height measurement on the sampled frames separates
perfectly on this reel — wide gameplay max-person-height ≤ 0.17,
close-ups ≥ 0.51 — so shots above `closeup_max_person_height: 0.5`
are excluded as `closeup`. Ball-presence as a "keep in-play close-ups"
discriminator was tested and **rejected**: yolov8n misses the ball in
true wide shots (0/5 frames on s013/s018) and false-fires during
celebrations.

**Result**: 40 shots (was 32), 11 kept gameplay / 21 closeup / 8
reaction. Visual review: 10/11 kept are clean wide gameplay; all 29
drops correct except one borderline (person-height 0.51, restorable).

## Iteration 4 (2026-06-11, dissolve splitting + non-gameplay drops)

**Dissolve splitting** (`dissolve_cuts`): cross-fades change the whole
frame without anything moving. Two gates: spatial uniformity
(block-median diff ≥ 10 — true dissolves ≥ 9.5 p10, static-camera
action ≤ 7.3 p50, the confound that a frame-mean threshold could not
reject: first attempt produced 16 false splits inside gameplay) and
median LK flow ≤ 1.25 (rejects pans, which change every block but
carry motion; flow with < 15 tracked corners is treated as 0 — flat
fade endpoints emit garbage LK). 27 dissolve cuts on the reel, all
spot-verified as real transitions (montage fades, replay wipes,
graphic dissolves).

**Ball-visibility gate — tested and rejected (again, properly)**: with
the purpose-built WASB detector, every kept gameplay shot shows the
ball (2–5/5 sampled frames) but so do 23/29 correctly-dropped shots
(balls in the net during celebrations + false fires). Not a usable
discard signal at shot level.

**Result**: 56 shots, 12 kept gameplay / 32 closeup / 12 reaction, 7
groups. The pre-match montage now splits and auto-drops: tunnel walk
(person-height 0.94), lineup (0.81), handshakes/anthem (0.57). Kept
sheet: 12/12 wide/medium gameplay.

## Known limitations (operator-correctable in the UI)

- The opening broadcast take (17.9–40.7 s) starts on crowd/bench and
  pans down to kickoff — one continuous camera shot, so it stays one
  span (correctly, but its first seconds aren't gameplay).
- Two dissolve cuts (151.0 s, 292.8 s) are uncertain — both sit inside
  content that is either gameplay→gameplay (benign split) or dropped
  celebrations.

- Speed-factor estimation is noisy on calm wide shots — a couple of
  real-time shots carry REPLAY badges (sf ≈ 1.3–1.5) and get retimed.
  The badge tooltip shows the factor; restore-and-retrim manually if it
  matters. A future refinement could gate retiming on scale.
- Grouping boundaries between goals 3/4 (g04 spans 107 s) need the
  merge/split controls more than the other groups — exactly what the
  boundary-confidence chips surface.
- `align_curves` offsets for heavily-zoomed replays can exceed the
  plausible window; they arrive labelled `low_confidence`, amber on the
  timeline.
