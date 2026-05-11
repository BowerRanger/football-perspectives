# Ball Anchor Editor

**Date:** 2026-05-11
**Status:** Draft — pending user review
**Scope:** new editor page + new `BallStage` Layer 5 (anchor injection) + small
extension to `fit_parabola_to_image_observations`

## Problem

The ball tracker still gets things wrong on real broadcast clips, even after
the four-layer improvements:

- WASB drops the ball for long stretches; Layer 4 (appearance bridge) caps
  at 8 frames, and longer gaps still emit `state="missing"`.
- The IMM can't reliably tell aerial from grounded from pixel kinematics
  alone (a low diagonal pass looks grounded; Layer 2 over-flags noisy
  ground projections).
- Monocular depth ambiguity means even well-tracked flight segments can
  fit to a wildly wrong world height.

These are all symptoms of one missing input: a small amount of human
ground truth. The camera stage already accepts user-provided pitch
landmarks; the ball stage should too.

## Goals

- Operator can scrub a clip in the dashboard, click on the ball at any
  frame, and mark a state tag (grounded, aerial, kick, catch, bounce,
  off-screen flight).
- Anchored frames override WASB on those frames.
- State tags drive flight-segment boundaries (kick/catch/bounce are
  hard knots; off-screen flight extends a flight without pixel data).
- A "Solve & Preview" button reruns just the IMM + fit + 3D-projection
  steps from the saved anchors, so iteration is sub-second rather than
  needing a full pipeline run.
- The editor matches the visual language of the existing camera anchor
  editor.

## Non-goals

- No new detector. WASB stays as the primary detection backend.
- No precise numeric height input. Height comes from coarse state
  buckets only (decided in brainstorm).
- No multi-ball support.
- No automatic anchor suggestion. The operator picks frames manually.

## Data model — `src/schemas/ball_anchor.py`

```python
from typing import Literal

BallAnchorState = Literal[
    "grounded",
    "airborne_low",       # 0–2 m
    "airborne_mid",       # 2–10 m
    "airborne_high",      # 10 m+
    "kick",               # ball leaves a foot, at ground level
    "catch",              # ball stops in a player's hands, ~1.5 m
    "bounce",             # ball hits the ground briefly
    "off_screen_flight",  # ball is airborne but not visible (no pixel)
]


@dataclass(frozen=True)
class BallAnchor:
    frame: int
    # None only when state == "off_screen_flight".
    image_xy: tuple[float, float] | None
    state: BallAnchorState


@dataclass(frozen=True)
class BallAnchorSet:
    clip_id: str
    image_size: tuple[int, int]
    anchors: tuple[BallAnchor, ...]
```

**State → assumed height map** (single source of truth in
`src/utils/ball_anchor_heights.py`):

| State              | Height (m) | Hard knot? |
|--------------------|-----------|------------|
| grounded           | 0.11      | yes        |
| airborne_low       | 1.0       | no (range center) |
| airborne_mid       | 6.0       | no (range center) |
| airborne_high      | 15.0      | no (range center) |
| kick               | 0.11      | yes        |
| catch              | 1.5       | yes        |
| bounce             | 0.11      | yes        |
| off_screen_flight  | n/a       | n/a        |

"Hard knot" means the parabola fit must pass exactly through this
(world_x, world_y, height) point; "no" means the height is used to
gate plausibility but the fit is not constrained to it.

## File layout

| Path | Status | Responsibility |
|---|---|---|
| `src/schemas/ball_anchor.py` | NEW | Dataclasses + JSON load/save |
| `src/utils/ball_anchor_heights.py` | NEW | Single source of truth for the state→height map |
| `src/utils/bundle_adjust.py` | MODIFY | Add optional `knot_frames: dict[int, np.ndarray] \| None` to `fit_parabola_to_image_observations` |
| `src/stages/ball.py` | MODIFY | Layer 5: inject anchors into the detection stream + drive flight-segment splits + pass knot frames |
| `src/web/server.py` | MODIFY | `GET/POST /ball-anchors/{shot_id}` + `POST /ball-anchors/{shot_id}/preview` |
| `src/web/static/ball_anchor_editor.html` | NEW | Editor page, modelled on `anchor_editor.html` |
| `src/web/static/index.html` | MODIFY | Link to the new editor from the ball panel |
| `tests/test_ball_anchor_schema.py` | NEW | Round-trip JSON load/save |
| `tests/test_ball_anchor_layer.py` | NEW | Unit tests for Layer 5 injection logic |
| `tests/test_bundle_adjust_knot_frames.py` | NEW | Multi-frame knot constraint tests |

## Server endpoints

Modelled on the existing `/anchors/{shot_id}` pair.

- `GET  /ball-anchors/{shot_id}` → returns the saved `BallAnchorSet`
  JSON, or 404 if the file does not exist.
- `POST /ball-anchors/{shot_id}` → validates the payload against the
  `BallAnchorSet` schema and writes
  `output/ball/<shot_id>_ball_anchors.json`.
- `POST /ball-anchors/{shot_id}/preview` → runs the ball stage's
  in-process fit pipeline (IMM + fit + 3D projection) without writing
  the ball_track to disk, returns the would-be `BallTrack` JSON so the
  editor can show a live preview overlay. Reuses the existing
  `BallStage._run_shot` machinery via a refactor that splits "compute"
  from "write".

## Layer 5 — anchor injection in `BallStage._run_shot`

Runs **first**, before all four existing layers. Pseudocode:

```
anchors = BallAnchorSet.load(...) if file exists else empty
anchor_by_frame = {a.frame: a for a in anchors.anchors}

# Detection loop
for frame_idx in range(n_frames):
    a = anchor_by_frame.get(frame_idx)
    if a is None:
        # Existing path: WASB → appearance bridge → IMM.
        ...
    elif a.state == "off_screen_flight":
        # No pixel. Skip WASB. Tell IMM to predict only.
        # Mark frame_idx in the "forced_flight" set for later use.
        steps.append(tracker.update(frame_idx, uv=None))
        forced_flight.add(frame_idx)
    else:
        # Hard pixel anchor; WASB ignored on this frame.
        uv = a.image_xy
        raw_confidences[frame_idx] = 1.0
        bridge.update_template(...)   # let the bridge keep a fresh template
        steps.append(tracker.update(frame_idx, uv))
        anchored_uv[frame_idx] = uv
        anchored_state[frame_idx] = a.state
```

After the detection loop, anchors influence flight-segment formation:

```
# Split flight runs at kick/catch/bounce.
events = {fi for fi, s in anchored_state.items()
          if s in {"kick", "catch", "bounce"}}
flight_runs = split_runs_at_events(flight_runs, events)

# Force a flight run wherever the operator marked airborne_* or
# off_screen_flight, regardless of IMM posterior.
for fi, s in anchored_state.items():
    if s.startswith("airborne_") or s == "off_screen_flight":
        extend_or_create_flight_run(flight_runs, fi)
for fi in forced_flight:
    extend_or_create_flight_run(flight_runs, fi)
```

For each flight run, when calling `fit_parabola_to_image_observations`:

```
knots = {}
for fi in range(run.start, run.end + 1):
    s = anchored_state.get(fi)
    if s in HARD_KNOT_STATES:        # grounded/kick/catch/bounce
        # World position via ankle_ray_to_pitch with plane_z set to the
        # state's height (see state→height table above).
        knots[fi] = ankle_ray_to_pitch(
            anchored_uv[fi],
            K=K_fi, R=R_fi, t=t_fi,
            plane_z=state_to_height(s),
        )

# kick at start → use as p0_fixed (already implemented in Task 6).
if run.start in knots:
    p0_fixed = knots.pop(run.start)
else:
    p0_fixed = None

# Pass the rest as knot_frames (new kwarg).
fit_parabola_to_image_observations(..., p0_fixed=p0_fixed, knot_frames=knots)
```

## `fit_parabola_to_image_observations` extension

Add an optional `knot_frames: dict[int, np.ndarray] | None` kwarg. Each
entry maps a frame index (relative to the segment start) to a world
position the parabola must pass through. Implementation: the LM
residual vector gains one 3-row block per knot:

```
residuals += [pts[knot_idx] - knot_world] * knot_weight
```

`knot_weight` is large (e.g. 1e3) but finite so the optimiser
gracefully handles tiny floating-point inconsistencies. Hard equality
would require constrained-LSQ which scipy.optimize doesn't ship for
LM; the heavy soft constraint is good enough for our scale.

## Editor UX — `ball_anchor_editor.html`

Reuses the camera anchor editor's CSS shell. Three columns:

**Left (220 px) — Tag palette**, a vertical list:

```
─ Ground states ─
▣ Grounded               (z = 0.11 m)

─ Aerial states ─
▣ Airborne — low         (0–2 m)
▣ Airborne — mid         (2–10 m)
▣ Airborne — high        (10+ m)
▣ Off-screen flight      (no pixel)

─ Events ─
▣ Kick (foot)
▣ Catch
▣ Bounce
```

Clicking a tag selects it (one selected at a time). The selected tag is
the one that will be applied to the next click on the video.

**Middle — Video stage** with the same controls as the kp2d viewer:
play/pause, prev frame, next frame, scrub. Above the controls: the video
with a transparent overlay canvas. Click anywhere on the video to place
an anchor at `(uv, current_frame, selected_tag)`. Right-click an existing
anchor dot to delete it. Drag a dot to nudge.

Off-screen flight is a special-case: when that tag is selected, clicking
the video has no effect; instead a "Mark current frame as off-screen
flight" button appears above the seek bar.

The overlay always shows:
- WASB raw detections as light gray dots (so you can compare).
- Anchors as colored circles (color per state group: grounded=green,
  aerial=orange, event=blue/pink).
- A camera-projected ground line at the current frame for orientation.

**Right (240 px) — Anchor list**, scrollable:

```
Frame 43   grounded
Frame 78   airborne_mid
Frame 84   kick               ●
Frame 91   off_screen_flight
Frame 95   catch
...
```

Clicking a row seeks the video to that frame and selects the anchor.
A small "●" indicator marks unsaved changes. Keyboard shortcuts (j/k
for previous/next anchor, delete to remove the selected one) mirror the
camera editor.

**Header**:

```
Football Perspectives — Ball Anchors      [Save]  [Solve & Preview]
Shot: origi01 ▾                                34 anchors  ●dirty
```

"Solve & Preview" calls `POST /ball-anchors/{shot_id}/preview` and
overlays the resulting per-frame world_xyz back-projected as crosshairs
on the video — operator sees the impact of their anchors immediately,
without writing a ball_track to disk.

## Workflow

1. Run `--from-stage camera` once so `camera_track.json` exists.
2. Open `/ball-anchor-editor?shot=origi01` from the ball panel link.
3. Scrub to obvious states (ball at feet, ball at apex, ball in goal,
   ball lost in crowd) and tag them.
4. ~5–15 anchors per clip is the typical target — not every frame.
5. Hit "Solve & Preview" and check the overlay; iterate.
6. Hit "Save" when satisfied.
7. Next full pipeline run reads the anchors automatically.

## Testing strategy

- **Schema round-trip**: load/save a synthetic `BallAnchorSet`, assert
  equality.
- **Layer 5 unit**: feed a fake detection stream + a small anchor set,
  assert that anchored frames use anchor uv (not WASB), that events
  split flight runs, that off-screen flights extend flight runs.
- **Knot frames**: synthetic flight where the operator knows the apex
  height; fit with vs without knot, assert the knotted fit lands within
  0.5 m of the truth at the apex frame while the unknotted fit drifts
  by >2 m.
- **Editor smoke**: load editor with a canned `BallAnchorSet`, assert
  every anchor renders and round-trips through POST.

## Risks

| Risk | Mitigation |
|---|---|
| Operator places anchors at wrong pixel (parallax / motion blur) | Hard knots use the anchor exactly. Provide a 3-frame look-ahead in the overlay so operator can verify before saving. |
| `airborne_*` height bucket midpoints are coarse | Layer 1 plausibility still gates the fit. If the bucket midpoint produces an implausible fit, the segment is dropped and the operator sees no preview — clear feedback that the tag was wrong. |
| Knot soft-constraint weight too low → knots drift | Start with `knot_weight = 1e3` (≈ 1000 px-equivalent error per metre); tune if integration test fails. |
| Off-screen flight anchors fed into IMM as uv=None confuse the bridge | The bridge only fires when WASB returns None AND we're not in an anchor frame. Off-screen anchor frames will skip the bridge entirely. |
| Editor / `BallStage` schema drift | Schema lives in `src/schemas/ball_anchor.py`; both editor (via JSON) and stage import the same module via the server. |

## Rollout

1. Schema + heights module.
2. `knot_frames` kwarg on `fit_parabola_to_image_observations` + tests.
3. Layer 5 anchor injection (no editor yet — exercised via fixture JSON).
4. Server endpoints + `/preview` plumbing.
5. Editor HTML.
6. Dashboard link from ball panel.

Each step ships independently with passing tests. The editor can be
tested manually before the dashboard link goes in.

## Open questions

None. Brainstorm settled:
- Off-screen flight tag included.
- Coarse height buckets only.
- Bounce/catch = hard knots.
- Anchor wins over WASB on annotated frames.
