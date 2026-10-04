# Ball Studio — multi-angle 3-D ball-track authoring (design)

Date: 2026-10-04 · Branch: `worktree-gberch-shorts` · Status: design → build

## Why

The ball stage is still the weakest part of the reconstruction. Across the
four test clips it needed operator help at the finish every time a ray was
ambiguous (gberch frames 388–394, kroupi 133–136), and we have no ground
truth to score it against beyond sparse 2-D anchors. The plan:

1. **Author** a dense, physically consistent 3-D ball track per test clip by
   hand, using every synced camera angle at once (two views = triangulation,
   so depth stops being a guess).
2. Freeze those tracks as **ground truth** (`ball_truth`).
3. Develop the automated ball model against them (per-frame 3-D error,
   per-view reprojection, event timing, line-cross), then run it on new
   clips.

This document covers step 1–2 (the tool + the truth format + the scorer).

## Scope / constraints

- Lives in the dashboard SPA as a new page `/ball-studio?group=<group>` (one
  shell, DESIGN.md tokens, shadcn components). Requires restoring
  `frontend/src/lib/` (never committed; root `.gitignore` swallowed it).
- Multi-angle: a **group** = the shots linked by `shots/sync_map.json`
  (reference shot + `frame_offset` per member: reference frame
  `r = f_shot − frame_offset`). Test groups: gberch (`gberch` + spidercam
  `gberch-2`), origi (`origi01` + `origi02`), saka (`s009` + replay `s011`),
  kroupi (single angle). Every member already has a solved camera track in
  the shared pitch frame — that is what makes triangulation possible.
- Operator data always wins; the truth file is operator data. The tool never
  edits pipeline outputs (ball tracks, anchors) — it only reads them as
  reference overlays.
- Large UX addition → impeccable audit before, finish review after
  (CLAUDE.md).

## Data model — `<output>/ball_truth/<group>_ball_truth.json`

```jsonc
{
  "version": 1,
  "group_id": "origi", "reference_shot": "origi01", "fps": 30,
  "shots": [{"shot_id": "origi01", "frame_offset": 0}, {"shot_id": "origi02", "frame_offset": -142}],
  "outcome": "goal",            // goal | no_goal | unknown — drives the goal-only line-cross check
  "keys": [                     // 3-D control points on the REFERENCE timeline
    {"id": "k7", "frame": 440,
     "xyz": [2.1, 31.0, 0.11],
     "source": "triangulated",  // triangulated | ray_ground | ray_height | ray_plane | ray_depth | player | manual
     "constraint": {"height_m": null, "player_id": null, "bone": null},
     "observations": [{"shot_id": "origi01", "shot_frame": 440, "uv": [812.0, 604.5]},
                      {"shot_id": "origi02", "shot_frame": 298, "uv": [1203.1, 455.0]}],
     "residual_px": {"origi01": 0.8, "origi02": 1.4}}
  ],
  "segments": [                 // between consecutive keys (by frame)
    {"from": "k6", "to": "k7", "kind": "roll"}   // flight | roll | carried | linear | static
  ],
  "observations": [             // extra 2-D clicks that are NOT keys: soft constraints for the segment fit
    {"shot_id": "origi01", "shot_frame": 444, "uv": [790.0, 600.0]}
  ],
  "events": [                   // touch | bounce | post | crossbar | net | line_cross | out | keeper_save
    {"frame": 440, "kind": "touch", "player_id": "P023", "bone": "r_foot"}
  ],
  "meta": {"authored_by": "operator", "updated_at": "…", "notes": ""}
}
```

The **dense track** is derived (never hand-edited) by the solver and written
next to it as `<group>_ball_truth_dense.json`: per reference frame `xyz`,
`segment`, and per-shot projected `uv` + residual where observed.

## Solver (`src/utils/ball_truth_solver.py`, pure, unit-tested)

- **Triangulation**: N ≥ 2 observations at the same reference instant →
  least-squares closest point to all rays (linear midpoint, then
  Gauss-Newton on reprojection), residual per view in px; reject if any
  view > 15 px (shown, not silently accepted).
- **Single-view keys**: ray ∩ constraint — ground plane (z = 0.11 m),
  fixed height, a named plane (goal line x = 0/105), explicit depth along
  the ray, or a player joint (SMPL FK from `refined_poses`, e.g. foot at a
  touch).
- **Segments** between consecutive keys:
  - `flight`: gravity + quadratic drag through both keys (shooting on the
    launch velocity); if soft observations exist in the span, also fit a
    bounded Magnus (curl) term to them.
  - `roll`: on the ground, constant deceleration through both keys.
  - `carried`: follows a player joint (dribble / keeper hold).
  - `linear` / `static`: escape hatches.
- Output per frame + reprojection residual of every observation (keys and
  soft) in every view; physics sanity flags (z < 0, speed > 45 m/s,
  discontinuity, flight span that needs > 15 m/s² of curl).

## Server (`src/web/ball_studio.py`, router mounted by `server.py`)

| Route | Purpose |
|---|---|
| `GET /api/ball-studio/groups` | groups with member shots, offsets, frame counts, fps, has-truth |
| `GET /api/ball-studio/groups/{g}/scene` | per-shot camera tracks (K, R, t, distortion per frame), goal geometry, player root + key joints per reference frame (from refined_poses), pipeline ball track(s) for overlay |
| `GET /api/ball-studio/groups/{g}/truth` | the truth file (or an empty skeleton) |
| `PUT /api/ball-studio/groups/{g}/truth` | validate + atomic write; previous version kept in `ball_truth/.history/` |
| `POST /api/ball-studio/groups/{g}/solve` | keys + segments + observations → dense track, projections, residuals, flags (stateless, ms) |
| `POST /api/ball-studio/groups/{g}/triangulate` | observations → xyz + residuals (used live while clicking) |

Frames come from the existing frame/video endpoints used by the ball anchor
editor.

## UI (`frontend/src/pages/ball-studio/`)

```
┌─ group ▾  frame 440 / 506  ◀ ▶  ⏯  outcome: goal ▾   save · solve status ─┐
│ ┌── origi01 (ref) ───────┐ ┌── origi02 (−142) ──────┐ ┌──── 3-D ────────┐ │
│ │ video frame            │ │ video frame            │ │ pitch, goals,   │ │
│ │ + projected track      │ │ + epipolar line of the │ │ players, camera │ │
│ │ + keys / clicks        │ │   other view's click   │ │ frusta, rays,   │ │
│ │ + pipeline track (dash)│ │ + projected track      │ │ track, keys     │ │
│ └────────────────────────┘ └────────────────────────┘ └─────────────────┘ │
│ timeline: keys ◆  events ▲  segments (colour = kind)  residual sparkline   │
│ inspector: selected key / segment / event — source, constraint, residuals │
└────────────────────────────────────────────────────────────────────────────┘
```

Core loop: scrub → click the ball in view A (ray appears in 3-D, epipolar
line appears in view B) → click it in view B (snaps near the line) →
triangulated key with per-view residual. With only one view visible at that
instant: pick a constraint (ground / height / goal line / player joint /
drag along the ray in 3-D). Keys auto-propose a segment kind (both on the
ground → roll; else flight), editable. Every edit re-solves; the solved
track is projected into every view so it can be checked against the real
ball in all angles at once. Extra clicks without a key become soft
observations for the segment fit. Keyboard: `,`/`.` frame step,
`Shift+,/.` ±10, `K` key at cursor, `O` observation, `E` event menu,
`Del`, `Ctrl+Z`/`Ctrl+Shift+Z` undo/redo, `S` save.

## Scoring (`scripts/eval_ball_truth.py`, `src/utils/ball_truth_eval.py`)

Pipeline ball track (per shot, mapped to the reference timeline) vs the
dense truth: per-frame 3-D error p50/p95, % within 0.2/0.5 m, per-view
reprojection error, event timing error (touches/bounces/impacts), line-cross
point error (goals only). Later: a `-m regression` gate over all authored
groups, mirroring the camera/ball gates.

## Out of scope (next)

The automated model that resolves to the authored tracks (multi-view
WASB fusion + triangulation + physics segmentation), and the regression
gate over authored truth — designed after the tool exists and tracks are
authored.
