# Replay playback speed — detect it, sync on it, retime to real time

Date: 2026-10-04 · Branch: `worktree-gberch-shorts` · Status: design → build

## Why

Highlight groups pair a live broadcast shot with replays of the same moment.
Every multi-shot consumer (refined_poses fusion, cross-replay ball fixes, Ball
Studio triangulation, the Shorts compositor) assumes a replay runs at the live
shot's real-time speed, offset by a whole number of frames. That breaks on
slow-motion replays, and the one existing speed signal is unusable:

- `prepare_shots` records a `speed_factor` from a zoom-invariant corner-flow
  rate compared against a reel-wide percentile. On the three test reels it
  reads 4.0x (the clamp) on most *live* shots; retiming on it was switched off
  after it compressed a 46 s live shot 4x. Clips therefore keep native timing
  and a slow replay is synced with an offset only, which can't be right.
- The motion-energy NCC offsets are themselves often badly wrong on these
  reels (hand check: saka s023 −24 vs −152 stored, mancity s024 −90 vs +102).

## Ground truth (hand-labelled, `docs/superpowers/notes/replay-speed/`)

24 matched replays across the Liverpool 4-0 Barcelona, Bournemouth 1-1 Man
City and Arsenal 3-0 Coventry reels, two to four shared events each (ball
contacts, net/keeper impacts), rate = Δlive / Δreplay frames:

- Most "replays" are **real-time alternate angles**: 15 of 24 at 0.91–1.02.
- Slow motion runs 0.11x–0.65x (s017 0.11, s037 0.175, s012 0.31 with a
  0.27→0.41 **speed ramp**, s043 0.34, saka s019 0.36, s015 0.38, s021 0.5,
  s017 0.53, s019 0.65); one replay measures 1.24x (sped up, or a soft event).
- None of the reel replays are frame-repeat or frame-blend slow motion (no
  duplicate/blend cadence): they are native high-frame-rate footage, so the
  speed has to come from the content.

## Signals evaluated

| Signal | Idea | Result on the GT |
|---|---|---|
| reel `speed_factor` | corner flow / image gradient vs reel percentile | clamps on live shots; unusable |
| player tempo, pairwise | camera-compensated player motion in body-heights/s, replay ÷ live at the matched moment | 20–27 % within 20 %: framing (one sprinter in close-up vs a whole team wide) dominates the level |
| stride cadence (ViTPose ankles) | step rate is near-universal (≈2.5–4 Hz); replay rate = apparent / real | right on clear cases (s021 0.44 vs 0.5, s012 0.42 vs 0.31, real-time 2.6–4.1 Hz) but ~half fail: 2 s slow replays hold one or two strides |
| **pitch-position matching** | project every player's feet onto the pitch in both shots; search `live = offset + rate · replay` | **every trusted sync reproduced to ~1–2 frames**: saka s011 0.989 (GT 0.994, offset 34.5 vs 33.5±1), origi02 0.991 (fresh-pick verified −142 at the goal), gberch-2 spidercam 1.033 (manual −228 at the goal); <1 s per pair |

Pixel-only cues can say "probably real time" or "probably heavy slow
motion"; only geometry gives a rate precise enough to sync on. It needs
`tracking` + `camera` on both shots — which every active shot gets anyway.

## Estimator (`src/utils/replay_speed.py`, committed a5b3f4d)

- Feet on pitch: box-bottom pixel → camera ray → z = 0 (off-pitch dropped).
- Cost of a time map: mean over replay frames of each replay player's
  distance to the nearest live player at the mapped live instant (linear
  interpolation between live frames), truncated at 3 m. Identity-free, so it
  does not depend on the sync it is finding.
- A (replay-sample × live-frame) distance table makes the exhaustive
  (rate ∈ [0.08, 1.5] log grid, offset) search array indexing; then a local
  refine (±6 % rate, ±4 frames at 0.25).
- Confidence = profile contrast (median cost over rates vs the optimum) ×
  absolute quality (cost vs the 3 m truncation).
- Speed ramp: a continuous two-rate time map with a breakpoint at 30–70 %
  of the replay; reported only when its rates differ by >20 % **and** it
  lowers the cost by >4 %.

## Pipeline integration

### New stage `replay_sync` (after `camera`, before `hmr_world`)

For each highlight group with ≥ 2 active shots that have tracks + a camera
track:

1. Estimate every non-reference member against the reference; when that is
   low-confidence (e.g. the reference doesn't show the moment — mancity g04
   s008), retry against members already placed and chain.
2. Write the result into `shots/sync_map.json` (`method="player_formation"`,
   `confidence`, new `playback_rate`) — **never over a `manual` alignment**,
   and never over an existing alignment with a low-confidence estimate.
3. **Retime** a confidently slow replay (no ramp, confidence ≥
   `retime_min_confidence`, |rate − 1| > `retime_tolerance`) to real time:
   - new clip frame k = native frame round(k / rate) (exact, decoded and
     re-encoded H.264 so the dashboard can play it); the native clip moves to
     `shots/native/<sid>.mp4` and is never deleted;
   - the shot's tracks and camera track are **remapped** frame-for-frame (no
     re-solve) and their native versions kept beside them;
   - the manifest records `speed_factor = 1 / rate`, `retimed = true`; the
     alignment becomes `playback_rate = 1.0`, `frame_offset = −offset`.
   From `hmr_world` on, every stage sees real-time footage — ball physics,
   pose smoothing, cross-shot fusion all stay valid without changes.
4. Near-real-time rates (within tolerance, e.g. 0.99 / 1.03) are stored as
   `playback_rate` and not retimed.
5. Ramps and low-confidence results are reported for the operator, not
   applied.
6. Diagnostics: `shots/replay_sync.json` (per member: estimate, decision,
   reason) and a `replay_sync` block in `quality_report.json`.

### Sync-map semantics

`Alignment.playback_rate` (default 1.0) generalises the mapping:
`ref_frame = playback_rate · shot_frame − frame_offset` (identical to today at
1.0). `GroupSync` gains `ref_frame_of(shot, f)` / `shot_frame_of(shot, ref)`;
consumers that only read `frame_offset` keep working because a retimed
replay is at 1.0 and near-real-time rates stay within a frame over a few
seconds. Ball Studio's solver takes the rate into its frame mapping.

### Dashboard (Prepare Shots → group sync timeline)

Per member: a speed badge ("0.34× slow motion · matched on players · 92 %",
"real time", "speed ramp — not applied"), a manual rate override (operator
wins), and "Retime to real time" / "Restore native clip" actions that call
the same retime code. Large UX change → impeccable review.

## Not in scope (noted)

- Piecewise retiming of ramps (detected and reported only).
- Using the slow replay's extra temporal resolution (retime keeps every
  1/rate-th frame; a high-frame-rate path would need per-shot fps everywhere).
- Pixel-only pre-camera detection (cues above are hints at best).

## Validation

- Unit tests on synthetic trajectories (rates 1.0 / 0.34 / 0.5, ramp,
  unrelated replay, too little data, feet projection).
- Real pairs with cameras: saka g04, origi, gberch (real time) — above.
- Slow motion: Liverpool g11 (s042/s043, GT 0.341 grade A) and Man City g05
  (s016/s017, GT 0.526 grade B), tracking + camera run in scratch dirs
  `output-speed-lv-g11`, `output-speed-mc-g05` — results below.

### Slow-motion results

Man City g05: the wide live shot s016 auto-calibrated; the tight slow replay
s017 did **not** (auto-anchors found no usable pitch landmarks — the camera
stage skips it). Tight close-ups are the common slow-motion framing, so the
geometric path cannot be the only path. Hence the second, camera-free path
below. (Liverpool g11 pending.)

## Two paths

1. **Automatic — players on the pitch** (`estimate_speed`), whenever both
   shots have tracks and a camera track. Runs in the `replay_sync` stage.
2. **Operator — mark matching moments** (`rate_from_moments`, committed with
   tests): the operator marks the same instant (a ball contact, a net impact)
   in the live clip and the replay, twice. Two pairs fix rate and offset
   exactly; three or more are least-squares fitted and their interval rates
   reveal a ramp (s012-style). This is exactly how the ground truth was made,
   needs no camera, and takes under a minute in the dashboard. Saved as a
   `manual` alignment (operator wins) with its `playback_rate`.

## Interfaces (build contract)

### Schema (`src/schemas/sync_map.py`)

- `Alignment.playback_rate: float = 1.0` (missing on load → 1.0).
- `GroupSync.ref_frame_of(shot_id, shot_frame) -> float` =
  `rate · shot_frame − frame_offset`; `shot_frame_of(shot_id, ref_frame) ->
  float` = `(ref_frame + frame_offset) / rate`.
- `Shot` gains `retimed: bool = False`, `native_frames: int = 0`
  (`speed_factor` keeps its meaning, `= 1 / rate` once retimed).

### Retime (`src/utils/replay_retime.py`)

- `retime_shot(output_dir, shot_id, rate) -> RetimeResult` — always from the
  native clip (`shots/native/<sid>.mp4`, created on first retime and never
  deleted), so repeated retimes don't compound. New frame `k` = native frame
  `round(k / rate)`; H.264 yuv420p at the manifest fps (browser-playable).
  Remaps `tracks/<sid>_tracks.json` and `camera/<sid>_camera_track.json` when
  present (natives kept under `tracks/native/`, `camera/native/`). Updates
  the manifest Shot. Returns the frame map + counts.
- `restore_native(output_dir, shot_id)` — puts the native clip, tracks and
  camera track back; `speed_factor = 1`, `retimed = False`.
- The caller updates the sync alignment (`playback_rate = 1.0`,
  `frame_offset = −offset`).

### Stage `replay_sync` (`src/stages/replay_sync.py`)

Config `replay_sync: {enabled: true, min_confidence: 0.5, max_cost_m: 2.2,
auto_retime: true, retime_tolerance: 0.08, retime_min_confidence: 0.6}`.
Writes `shots/replay_sync.json`:

```json
{"version": 1, "groups": [{"group_id": "g11", "reference_shot": "s042",
  "members": [{"shot_id": "s043", "against": "s042",
    "estimate": {"rate": 0.341, "offset": 127.6, "confidence": 0.81, "cost_m": 1.3,
                 "coverage": 0.97, "ramp": false, "rate_first": 0.341, "rate_second": 0.341},
    "decision": "applied_retimed", "reason": ""}]}]}
```

`decision` ∈ `applied | applied_retimed | kept_manual | low_confidence |
ramp_not_applied | no_camera | no_tracks`. Registered after `camera` in the
runner, fingerprint table (inputs: manifest, sync map, tracks, camera
tracks; upstream tracking + camera; hmr_world cascades from it), server
`STAGE_ORDER`, quality report.

### HTTP (dashboard)

- `GET /api/replay-sync` → `shots/replay_sync.json` (or `{"version":1,"groups":[]}`).
- `POST /api/sync/groups/{group_id}/moments` body
  `{"shot_id", "moments": [{"reference_frame", "shot_frame"}, ...], "retime": bool}`
  → `rate_from_moments`; saves a `manual` alignment with `playback_rate`;
  when `retime` and |rate − 1| > tolerance, retimes and re-bases the
  alignment. Returns the fit + the saved alignment.
- `POST /api/shots/{shot_id}/retime` `{"rate"}` and
  `POST /api/shots/{shot_id}/restore-native` — the same utilities, for the
  speed badge's actions.
