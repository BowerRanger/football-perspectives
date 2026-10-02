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
