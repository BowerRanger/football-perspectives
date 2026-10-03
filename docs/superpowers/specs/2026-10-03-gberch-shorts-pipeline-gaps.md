# gberch YouTube Shorts — manual steps and the pipeline improvements that close them

Date: 2026-10-03 · Branch: `worktree-gberch-shorts`

## What was made

Three 9:16 Shorts of Gravenberch's goal (Liverpool v Chelsea, Anfield), all
rendered from the pipeline's 3-D reconstruction (camera → hmr_world →
refined_poses → ball, re-run with current defaults in `output-shorts/`):

| Short | Caption | Look | Angles |
|---|---|---|---|
| 1 Matchday | "WHAT GOAL IS THIS?" | bright cel-shaded, Anfield dressing | drone build-up → ball-chase strike → behind-goal 3× slow-mo → scorer orbit 4× slow-mo |
| 2 Keeper | "WOULD YOU SAVE THIS?" | floodlit night | goal-line establishing → Jorgensen eye-line (real time → 4× slow-mo → freeze at the strike) → behind-goal finish |
| 3 Comic | "WHAT GOAL IS THIS?" | posterised comic, heavy ink | drone → over Gravenberch's shoulder → chase 3× slow-mo → net orbit |

Render passes: `config/shorts/gberch_experiments.yaml`; edits: `config/shorts/gberch_*.yaml`
(EDLs for `scripts/compose_short.py`); kits: `config/clips/gberch.yaml`.
Outputs (gitignored scratch): `output-shorts/shorts/gberch_short{1_matchday,2_keeper,3_comic}.mp4`.
Presentation: https://claude.ai/artifact/MPX9F9gKE66naZicj25fhp

Reproduce (from the main checkout, after copying `output/` inputs to `output-shorts/`):

```bash
.venv311/bin/python recon.py run --input output-shorts/shots/gberch.mp4 --output output-shorts \
    --stages ball --from-stage ball --config config/clips/gberch.yaml
.venv311/bin/python scripts/splice_shot_arc.py output-shorts        # G16 hand fit
.venv311/bin/python scripts/render_experiments.py --output output-shorts --shot gberch \
    --experiments config/shorts/gberch_experiments.yaml --config config/clips/gberch.yaml --quality clean
for s in matchday keeper comic; do .venv311/bin/python scripts/compose_short.py \
    --edl config/shorts/gberch_$s.yaml --out output-shorts/shorts/gberch_$s.mp4; done
```

## Gap log — every step that was intuition or outside the pipeline

| # | Gap | What I did by hand | Status |
|---|---|---|---|
| G1 | Render stage ignored `players.json` `kit_role` (tracks label every team `unknown`), so every kit rendered grey | — | **Fixed** (render honours players.json, same as export) |
| G2 | No per-match kit palette; config ships placeholder colours | Sampled kits from footage (`scripts/sample_kit_colors.py`), then snapped the dark/desaturated samples to brand colours by eye; keeper long sleeves/gloves/white boots and referee yellow from crops | Design D1 |
| G3 | Kit geometry was height bands: T-posed arms + hands = shirt, no boots, socks to the ankle only, no hair | — | **Fixed** (`body_zones: anatomical`, skinning-weight garments) |
| G4 | Stale outputs: gberch ball track predated the `hybrid` default and the 29 Sep anchor edits; nothing flagged it. The ball detection cache also missed, costing a full ~25 min re-detect | Noticed via file mtimes; re-ran `ball` in a scratch copy | Design D4 |
| G5 | `gberch-2` active in the manifest; no shot filter on `recon.py run` | Hand-excluded it in the scratch manifest | Design D4 |
| G6 | Skin tone/hair: ViTPose head-patch sampling at broadcast resolution is unreliable (hair patch hits grass; two players' luminance inverted) | Set per-player `skin`/`hair` in players.json from public knowledge of the players | Loader **added**; auto-suggest = Design D1 |
| G7 | Experiments runner only read `default.yaml` | — | **Fixed** (`--config`) |
| G8 | No edit/caption step for social output; ffmpeg build has no `drawtext` | Wrote `scripts/compose_short.py` (EDL → trims, speed ramps, freezes, flash cuts, Pillow captions); chose every in/out point myself from the ball events | Design D2 |
| G9 | Rigs frame the all-player centroid across the whole shot | Added `focus` (ball/player) + orbit sweep window; picked focus, heights, FOVs, windows per pass by eye | Partly **fixed**; director = Design D2 |
| G10 | Slow motion was ffmpeg `setpts`/`minterpolate` (duplicated or smeared frames) | — | **Fixed** (`time_stretch`: Blender renders true in-between poses) |
| G11 | `vignette` drew a hard-edged bright disc (Blender 5 blur size is in px; ellipse size is width-relative) | — | **Fixed** |
| G12 | `pov` rig follows raw head facing — jittery, loses the ball | Picked the keeper from players.json role | **Fixed** (`eyes:<PID>` rig); auto-pick = D2 |
| G13 | Worktree lacked `data/models/smpl_neutral.npz` → silent capsule-body fallback | Symlinked the asset | Design D4 |
| G14 | Stadium is generic (grey/blue crowd) | Set Anfield seat/crowd colours | `crowd_colors` key **added**; venue library = D3 |
| G16 | **Ball flight at the finish was wrong.** The refreshed `hybrid` track put the ball "grounded" at 2.3 m and then at 3.8 m on the goal line (over the bar); the old `reference` track went wide of the far post (y=41) because `airborne_low` pins depth at z=1 m. Root cause: from frame 387 to 393 the detector locked onto a false object moving the *opposite* way (keeper glove / ad board), and nothing checks that a scored ball crosses the line inside the goal mouth | Zoomed frames 384–395 to find the true ball and added operator anchors 388/390/391 (the pipeline's own correction path). That was still ambiguous in depth, so I fitted a two-knot arc (body-pinned strike knot at frame 371 → your frame-394 anchor ray at the goal line, gravity + drag + bounded 8 m/s² curl, 21 px median residual) and spliced it in up to the side-net impact (`scripts/fit_shot_arc.py`, `scripts/splice_shot_arc.py`) | Design D6 |
| G17 | `--stages ball` reported `[SKIP] ball (cached)` after operator anchors changed; `ball.detection_cache` is off by default, so every anchor tweak costs a full ~50 min WASB pass | Forced with `--from-stage ball`; enabled the cache in the clip config | Design D4 |
| G18 | Style presets aren't kit-safe: `posterize: 5` crushed Chelsea blue to black and blotched skin; `duotone`/`saturation: 0` erase team identity entirely | Caught on contact sheets; rebuilt the comic look as a 2-band cel ramp + heavy ink + saturation | Design D1 (kit-safety lint) |
| G19 | Per-pass framing needed eyeballed iteration: 40 m top-down drone unreadable at 9:16, 70° keeper eye-line too wide, goal-line opener showed a defender, OTS at 1.6 m sat inside the striker, chase overran into the ad boards, eye-line clipped through the diving keeper's arm | 7 re-renders (~3–5 min each) and hand-picked cut points | Design D2 (framing check) |
| G15 | No audio | Silent AAC track (platform-safe); music to be added in-app | Design D5 |

## Improvement designs

### D1 — Appearance profile (kits, skin, hair) from evidence + a kit library

*Closes G2, G6 and the residual of G1.*

1. **Team clustering at tracking time**: replace the `FakeTeamClassifier`
   default with torso-colour clustering (Lab k-means on grass-masked
   torso crops, already prototyped in `kit_evidence.py`), so
   `team` is A/B/referee/keeper rather than `unknown`. Keepers come out as
   colour outliers inside each half's defensive third.
2. **Kit palette solver**: per team, sample shirt/shorts/socks bands
   (`sample_kit_colors.py`), then **white-balance against the pitch
   lines** (known #f5f5f0) and **exposure-normalise** so samples recover
   their true saturation. Snap the corrected colour to the nearest entry
   in a **kit library** (`config/kits/<club>/<season>.yaml`: home / away
   / third / GK kits with sleeves, socks, boots, trim), keyed by the match
   metadata the match-data autopopulate already gives us. If nothing
   matches within ΔE<12, keep the corrected sample.
3. **Skin/hair suggestions from close-ups**: highlights reels contain
   close-up and celebration shots (`prepare_shots` already classifies
   them). Face-detect there, match faces to track IDs by kit + number +
   time, and sample skin/hair at real resolution.
4. **Operator confirmation in the dashboard**: an Appearance panel with
   swatches per team and per player (suggested vs. confirmed); writes
   `players.json` `kit_role`/`skin`/`hair`, and operator input always
   wins.

Acceptance: on gberch, auto palette within ΔE<10 of `config/clips/gberch.yaml`
with no hand edits; team labels 100% correct for the 22 players.

### D2 — Event-aware shot director + a `shorts` stage

*Closes G8, G9, G12.*

- **Moments** from existing outputs: strike (last `player_touch` before a
  `goal_impact`), line-cross, net impact, keeper dive (peak lateral root
  velocity of the defending keeper), build-up start (first touch of the
  possession chain).
- **Template EDLs** expressed relative to moments, not frames, e.g.
  `{rig: chase, from: strike-40, to: impact+5}`, `{rig: eyes:@keeper, from:
  strike-10, to: line_cross, time_stretch: 4, freeze_at: strike+4}`.
  `@scorer` / `@keeper` resolve from events + players.json roles.
- **Framing checks** before render: project the focus subject into each
  candidate camera; reject passes where the ball leaves the 9:16 safe
  area or is occluded by a player for more than N frames. This is what I
  checked by eye on drafts.
- **`shorts` stage** (after `render`): renders each template's passes at
  `time_stretch`, then composes with `compose_short.py` logic (captions,
  chips, freeze, end card). Captions and the chosen template live in a
  `shorts/<shot>_shorts.json` sidecar editable from a dashboard panel
  with live preview.

Acceptance: `recon.py run --stages render,shorts` reproduces the three
gberch Shorts with zero frame numbers typed by hand.

### D3 — Venue dressing library

*Closes G14.* Extend `config/stadiums.yaml` (already per-club pitch data)
with a `dressing` block: seat colour, crowd palette, home/away-end split
(the away-end crowd in the away team's colours), roof style, board text.
The render stage looks it up from `AnchorSet.stadium` / match venue. Also
add a dedicated stand-tone palette key (the existing black-band issue
from low cameras).

### D4 — Freshness, scope and preflight

*Closes G4, G5, G13.*

- **Stage fingerprints** in the manifest: config-slice hash + input
  artefact hashes + code version per stage. `recon.py status` and the
  dashboard show **stale** when an upstream artefact (e.g. operator
  anchors) or a relevant default (`ball.trajectory`) changed after the
  stage last ran; `recon.py run --stale` re-runs only those.
- **`--shots gberch`** filter on `recon.py run` (without editing the
  manifest).
- **Render preflight**: fail loudly (or put a red banner in the quality
  report) when the SMPL body asset is missing, rather than silently
  rendering capsule mannequins. Fix the ball detection-cache fingerprint
  so a path-only change (worktree / scratch copy) doesn't force a full
  re-detect.

### D6 — Goal-aware, spin-aware flight solve for the decisive shot

*Closes G16, the accuracy gap that matters most for "what goal is this?".*

1. **Goal-mouth constraint**: when the event list contains a goal (a
   `goal_impact` anchor, or match-data goal at this time), the flight
   span ending in it must cross x=0 inside the mouth (|y−34| < 3.66,
   z < 2.44). An operator anchor near the line becomes a **line-cross
   knot** (ray ∩ goal-line plane), new goal_element `mouth`. That is the
   knot that made the fit well-posed here.
2. **Direction-consistency gate**: reject detections whose image-space
   velocity reverses against the fitted flight for ≥2 frames (the
   387–393 glove track) before they reach the hybrid blend.
3. **Bounded Magnus by default on shots**: `ball.hybrid.spin.enabled`
   is off; turn it on for spans that start at a `touch_type: shot`
   (the operator even tagged this one `instep_curl_right`), with the
   curl bound at about 10 m/s².
4. **Quality report check** that fails loudly: "goal event but trajectory
   misses the mouth / exceeds crossbar height at line-cross".
5. Add gberch's finish as a held-out case in `tests/test_ball_regression.py`
   (line-cross point within 0.3 m of the operator ray ∩ x=0).

### D5 — Sound

*Closes G15.* A crowd bed generated from the source audio with the
commentary band suppressed (re-uses the crowd-floor estimate in
`ball_cue_audio.py`), a "thump" on the strike frame and a roar swell on
net impact, all timed from D2's moments. Licensed music stays an in-app
choice.

### Next renderer fidelity steps (not gaps, but the "true to life" ceiling)

- Shirt numbers and names as decals (project onto the SMPL back/chest
  region by vertex-group, with no UVs needed). Players' numbers are
  public data in the match metadata.
- Hairstyle shapes (short/afro/long/bun) as head-attached meshes picked
  per player.
- Real bullet-time: let a camera track carry a `scene_frame` per frame,
  so time can freeze while the camera moves (the orbit currently fakes
  it with a 4× slow-mo sweep).

---

# Round 2 (2026-10-03/04): designs implemented, repeated on kroupi, origi, saka

## Status of the designs

| Design | Shipped as |
|---|---|
| D6 | `src/utils/ball_goal_constraint.py` (line-cross knot inference from operator airborne anchors near a goal_impact, `mouth` goal element, goal_check in diag + quality report, net containment, snap-into-mouth within margin), `ball_direction_gate.py`, shot-span bounded Magnus (only when the span has a goal event) |
| D4 | `src/pipeline/fingerprint.py` + runner: `[STALE]` re-runs on config/input/operator-file changes, `--stale`, `--shots`, `recon.py status`; ball detection cache on by default; render SMPL preflight (`render.allow_capsule_fallback`) |
| D1 | `appearance` stage (`src/stages/appearance.py`, `team_clustering`, `kit_palette` with pitch-line white balance, `kit_library` + `config/kits/*.yaml` with stripes/hoops/sleeve colours, `kit_resolution` precedence, `players_suggested.json`); D1.3 skin/hair still manual |
| D2 | `shorts` stage (`src/stages/shorts.py`, `shorts_moments`, `shorts_templates` + `config/shorts/templates/{matchday,keeper,comic}.yaml` in moment expressions, `shorts_framing`, `short_compositor`, pass cache, sidecar with operator block) |
| D3 | `config/stadiums.yaml` dressing library (anfield, vitality_stadium, emirates_stadium) via `stadium_dressing.resolve_dressing`; stand-tone floor fixes the black band; away end in the away kit |
| D5 | `src/utils/shorts_audio.py`: commentary-suppressed crowd bed (~15 dB), synthesised strike thump + net hit, real roar swell, loudnorm |

Plus: `render_pass_runner.resolve_style_payload` (kits + dressing + kit-safety lint),
`eyes:<PID>` hides the POV body, `--vertical-only`, `--python-exit-code 1` + output checks.

## Round-2 gap log (found running the new clips)

| # | Found | Fix |
|---|---|---|
| C1 | Skin/hair for 55 players set by hand (D1.3 deferred) | open — close-up face sampling + operator confirm |
| C3 | Duplicate saka ball runs raced on one output dir | open — per-output-dir run lock in recon.py |
| C4/C13 | kroupi line-cross at goal centre (no anchor near the line); with operator anchors 133/135/136 the ray landed 0.5 m outside the post | operator anchors (scratch) + `snap_into_mouth` |
| C5 | Wall-clock: cache fill 27 min / 4 h / 6.6 h (kroupi/origi/saka) under contention with the 5.5 h gate | open — job queue |
| C6 | kroupi: comic build-up cut at frame −21 | cuts clamp to the clip; out-of-clip candidates fall through (0df871e) |
| C6/C11 | Single-candidate slots dropped whole Shorts (kroupi net orbit through the keeper; origi scorer orbit lost the ball) | every slot has a fallback + test (54ad4dc) |
| C8/C15 | Ball re-runs still slow with the cache on | cause: zoom-crop/strike-window detections key on crop pixels whose origins follow the solver; open — quantise crop origins (needs gate) |
| C9 | origi: ter Stegen picked as scorer (auto touch at 447) | operator touches first; defending keeper/team excluded (9410821) |
| C10 | Second synced angles (origi02, s011) unused | open — cross-shot cutting |
| C12 | Toon shader darkened every kit (red → maroon, yellow → khaki) | lighting-only bands × colour (caee296) |
| C14 | Ball gate: gberch synth.p95, origi01 synth float_sink fail (goal knot trusts perturbed anchors in the synth mismatch scenario); s013 real p50 from shot curl on non-goal shots | s013 fixed (1ff9847); gberch/origi01 re-baseline = user decision |
| C16 | Concurrent region edits spliced `appearance:` inside export.virtual_cameras (14 keys swallowed) | moved (7fb8499) + structure test |
| C17 | Dashboard doesn't list appearance/shorts; `frontend/src/lib` (api.ts) was never committed (root `.gitignore` `lib/`) | open — restore lib, then panels with impeccable review |

Outputs (scratch, gitignored): `output-{shorts,kroupi-shorts,origi-shorts,saka-shorts}/shorts/<shot>_{matchday,keeper,comic}.mp4`.
Reproduce per clip: `recon.py run --output <dir> --stages ball,appearance,shorts --shots <shot> --config config/clips/<clip>.yaml`.
