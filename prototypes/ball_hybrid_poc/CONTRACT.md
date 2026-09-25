**SUPERSEDED** by the production ball hybrid trajectory
(`src/utils/ball_hybrid_*`, `scripts/run_ball_bench.py`); kept for the
spike record.

# Ball hybrid-extraction PoC — shared contract (SPIKE, throwaway-labelled)

Goal: prove (or disprove) that a **hybrid extractor** — broadcast-faithful where
detections are confident, physically simulated (gravity + quadratic drag +
bounded Magnus) across gaps/noisy spans — produces a visibly more realistic and
more 3-D-accurate ball than the current ball stage, measured against:

1. **Synthetic 3-D truth** seeded ONLY from operator data (manual anchors) +
   reconstructed players — never from ball-stage output (`*_ball_track.json`,
   `*_ball_anchors_auto.json`, `*_ball_keyframes.json` are the thing under
   test and must not seed truth).
2. **Real-footage checks** (reality test): held-out manual anchors, and on
   origi01 the 31 cross-replay triangulated fixes (real absolute 3-D).

Target consumer is **Blender** (render stage reads the dense per-frame
`BallTrack`). Judgement is perceptual: contacts, ground float/sink, natural
motion, and error as seen from virtual cameras (depth errors show there).

## Paths

- Main repo (READ-ONLY data): `M=/Users/joebower/workplace/football-perspectives`
- Worktree (all code): `$M/.claude/worktrees/ball-poc-hybrid`
- Python: `$M/.venv311/bin/python` (run from the worktree so `src` imports
  resolve to the worktree copy).
- PoC code: `prototypes/ball_hybrid_poc/` (package, `__init__.py`); tests in
  `prototypes/ball_hybrid_poc/tests/`. Do NOT modify `src/` — import from it.
- PoC outputs (gitignored scratch): `$M/output-ball-poc/<clip>/...`
  NEVER write into `$M/output*` dirs other than `output-ball-poc`.

## Clips (all have manual anchors + real detections + camera + refined_poses)

| clip | output dir | notes |
|---|---|---|
| gberch | `$M/output` | 59 manual anchors, 30 fps, 1920×1080; camera may be moving (spidercam) — use the camera track per frame |
| origi01 | `$M/output-origi-global` | + `ball/origi01_ball_fixes.json` (31 real 3-D fixes from replay origi02) |
| kroupi01 | `$M/output-kroupi` | aerial, sparse anchors |
| s013 | `$M/output-japan` | highlights-reel shot |

Real detections: `ball/<shot>_ball_observations.json` → `frames[]` with
`frame, uv, confidence, p_flight, gap_fill, source`. Only sources in
`{"detector","second_pass","foot_guided","strike_window"}` are real detector
evidence; `anchor`/gap-fill entries are NOT evidence.

## Shared data types (JSON on disk, dataclasses in `types.py`)

`Observation`: `{frame:int, uv:[u,v], conf:float, source:str}`

`TruthTrack` (`truth_<scenario>.json`):
```
{clip_id, scenario, fps, frames:[{frame, xyz:[x,y,z], state:"ground"|"air"|"contact"}],
 events:[{frame, kind:"touch"|"bounce"|"net"|"post"|"rest", player_id?, bone?, xyz}],
 seed_anchor_frames:[int], physics:{drag_cd, magnus, restitution, ...}}
```

`SynthRun` (`synth_obs_<scenario>.json`): synthetic detector stream derived from
a TruthTrack — `{observations:[Observation], anchors:[BallAnchor-like dict],
noise_model:{...}}`. Anchors = same frames/states as the real manual anchors,
`image_xy` = truth projection + N(0, 1.5 px).

`Track` (any method's dense output, `track_<method>_<scenario>.json`):
```
{clip_id, method:"current"|"hybrid"|..., frames:[{frame, xyz|null, mode:"faithful"|"simulated"|"anchor"|..., conf}]}
```

`Results` (`results.json`, consumed by the viewer): per clip →
```
{clip_id, fps, image_size, pitch:{length:105, width:68},
 camera:{centre_xyz_per_frame:[[x,y,z]...]},        # for drawing the broadcast camera
 scenarios:{<name>:{truth:TruthTrack, tracks:{<method>:Track},
                    metrics:{<method>:{...}}, per_frame_err:{<method>:[float|null]}}},
 real:{tracks:{<method>:Track}, metrics:{<method>:{...}},
       anchors_heldout:[{frame, xyz_gt}], fixes:[{frame, xyz}]},
 players:{frames:[{frame, pid:[x,y]...}]}           # optional, ground positions for context
}
```

## Metrics (per method, per scenario)

- 3-D error vs truth: p50 / p95 / max, `pct_le_20cm` (all frames with truth).
- Split by truth state (ground / air / contact).
- Contact gap at truth touch frames (m).
- Ground float/sink: mean |z − r| on truth-ground frames (r = 0.11 m).
- Broadcast reprojection error (px) of the method's xyz vs truth pixel.
- Virtual-camera screen error (px) — project into a fixed side camera
  (e.g. 30 m side-on, 1920×1080, 50° FOV) looking at the action centroid.
- Naturalness violations (reuse `src.utils.ball_eval.naturalness_violations`).
- Real-footage: held-out-anchor 3-D error (`src.utils.ball_eval.anchor_gt_world`
  semantics) and origi01 fix error.
