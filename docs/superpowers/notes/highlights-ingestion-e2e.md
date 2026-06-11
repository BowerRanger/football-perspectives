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

## Known limitations (operator-correctable in the UI)

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
