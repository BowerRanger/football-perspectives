# Feature Ideas

Backlog of post-MVP enhancements. Promote to a design doc / plan when ready to build.

## Future work using the stadium registry

A per-stadium registry now exists at `config/stadiums.yaml` (loaded by
`src/utils/stadium_config.py`). It currently only consumes a `mowing`
block to generate position-known mow-stripe entries for the anchor
editor. The schema reserves space for additional per-pitch overrides:

- **Per-stadium override for advertising-board height + offset** —
  current static catalogue uses nominal 1.0 m height, 2 m touchline
  offset, 4 m goal-line offset. Different stadiums use different LED
  ribbon heights and runoff distances. Wire an `ad_boards` block on the
  stadium entry into `LINE_CATALOGUE` (or a generator analogous to
  `mow_stripe_lines`) so ad-board residuals can be reduced for the
  selected stadium.
- **Known-pitch dimensions** for grounds with non-FIFA pitch sizes — store
  on the stadium entry, override `_PITCH_LEN`/`_PITCH_WID` constants in
  the catalogue / `pitch_landmarks.py` when active.

## Solver / annotation enhancements

- **Grid-intersection point landmarks** — when a stadium is genuinely
  cross-mowed, treat stripe intersections as position-known POINT
  landmarks at known `(x, y)`. Strongest constraint type but only
  helpful on actual checker-mowed pitches; needs additional UI for the
  user to specify `(i, j)` indices when clicking an intersection.
- **Auto-detect mow seams from video** — Hough on luminance gradients
  in the green-mask region, project candidate world stripes through the
  current camera, and identify seams automatically.
- **Curved or fan-shaped mow patterns** — extend `MowingPattern` beyond
  straight parallel/grid layouts.
