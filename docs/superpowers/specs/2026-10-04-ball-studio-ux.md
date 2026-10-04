# Ball Studio — UX direction (impeccable audit before + critique)

Date: 2026-10-04 · Branch: `worktree-gberch-shorts` · Status: direction for the builder (no code yet)
Companion to `2026-10-04-ball-studio-design.md` (data model / solver / API). Mode: **Operate**.
Read first: `PRODUCT.md`, `DESIGN.md`, `frontend/README.md`, `.impeccable/surfaces/frontend-src-app-tsx.md`.

Method: `impeccable context`, then audit of the two nearest patterns (`/ball-anchor-editor?shot=origi01`,
`/viewer?shot=origi01`) against a live dashboard on `output-origi-shorts`, captured at 1440x900 (dark and
light) and 390x844 (dark); the impeccable detector run over `pages/ball-anchor-editor` and `pages/viewer`
(zero findings); then a critique of the spec's "UI" section. Captures are in `/tmp/a3/` (scratch, not
committed): `ball-d.png`, `ball-l.png`, `ball-m.png`, `viewer-d.png`, `viewer-m.png`. Light-mode viewer
and mobile light were not captured (theme flip changes chrome only; media wells are fixed).

## 1. Audit of the existing patterns

What already works and Ball Studio must inherit unchanged:

- Page header grammar: title + saved/dirty `DirtyBadge` ("60 anchors saved") + description, with
  `Save` and `Solve & preview` as header actions, one primary. Reuse it.
- Frame transport: one `FramePlayer` (scrub, frame input, Space / arrows / Shift+/-10 / Home / End), keys owned
  by the newest enabled player. The Studio has ONE master transport on the reference timeline.
- Safety: `useUnsavedGuard`, confirm-before-discard on shot switch, "Saving is disabled so existing anchors can't
  be overwritten" on load error (Studio does the same for a failed truth load).
- Anchor-tag colours are an established data vocabulary (grounded green, bounce pink, player touch cyan, goal
  impact amber, pitch fix lime). Studio events reuse them (section 5).
- Quality strip is click-to-seek with a labelled canvas; detector run is clean; light and dark both read.

Findings, ranked (P1 = would hurt Studio if copied):

| # | Sev | Finding | Consequence for Studio |
|---|---|---|---|
| A1 | P1 | The frame well is ~557 px wide at 1440 (256 sidebar + 250 tool rail + 320 events rail). Ball is ~8-12 px. No zoom, pan or loupe anywhere; click precision is whatever the video's native scale gives. The area under the quality strip (~280 px) is empty. | Two views plus a 3-D view cannot each get 557 px. Triangulation needs sub-pixel-ish picks, so zoom/loupe is mandatory, not polish. Collapse the sidebar to its icon rail on this page and give every pixel to the views. |
| A2 | P1 | Placement is pointer-only: `onClick` places, `onContextMenu` deletes; there is no pending/ghost state, no nudge, no undo (the dashboard has undo for tracks, not here). A mis-click is committed instantly. | Studio uses place-then-commit (pending pick), arrow-key nudge of a pending/selected point, and full undo/redo. |
| A3 | P2 | Right-click delete is undiscoverable and not keyboard reachable; the canvas has an `aria-label` but no focus model. | Selection model: click a mark to select, `Del` removes, inspector has a visible Delete button. |
| A4 | P2 | Viewer: ball is a ~6 px grey dot; sky is pure black above the pitch; legend card is open by default and covers the top-right; camera mode defaults to "Broadcast (solved)", which is a poor view for judging height. | Studio 3-D view needs a ball marker with ground drop-stem and height label, camera presets, and a legend that starts collapsed. |
| A5 | P2 | Quality strip explains itself with a prose legend ("Bars: detection confidence... Top ticks..."), no hover value, no tooltip, rows share one 28 px band. | Studio timeline uses separate labelled rows (segments / keys / events / residual / coverage) with hover readouts and a key row of chips. |
| A6 | P2 | `use-frame-player.ts` exists twice (`anchor-editor/`, `ball-anchor-editor/`) and the files differ; viewer imports `useScopedKeyboard` from `anchor-editor/`. | Do not add a third. Studio drives video from the master frame via one new hook that owns N `<video>` elements; place it in `pages/ball-studio/`. Flag the duplication to the generalist as follow-up. |
| A7 | P3 | The tag palette has 12 hues on a left rail with a long description card; the selected tag is the only mode state. | Studio's "what a click does" state is richer (key / observation / constraint). Show it as a compact mode bar over the views, not a tall rail. |
| A8 | P3 | Mobile: ball editor stacks cleanly with the honest "works best on a desktop screen" note; viewer fits. Good posture to extend (section 6). | Same honesty on mobile; Studio goes read-only there. |
| A9 | P3 | Canvas overlay text uses `system-ui`, not Geist. | Canvas labels: use the Geist stack at draw time, mono for numbers. |
| A10 | P3 | The "stage" well token is 0.18 lightness in the light theme (index.css) while DESIGN.md says near-black in both. | Out of scope; mention to the documenter. Studio uses `bg-stage` as-is. |

## 2. Critique of the proposed flow (spec "UI")

What is right: scrub, click A, epipolar line in B, click B, triangulate with per-view residual is the correct
core loop; the solved track being projected into every view (so the operator checks it against the real ball
in all angles) is the highest-value idea; "every edit re-solves" keeps the operator in a closed loop.

Problems and the resolution adopted below:

1. **Three panes + inspector + timeline in 1440x900 is cramped.** Views would be ~450 px. Resolution: sidebar
   rail by default, a defined grid (section 3), a *focus view* toggle that enlarges one view to ~980 px, and a
   per-view loupe/zoom. Layout presets are minimal: `Compare` (default) and `Focus A/B`.
2. **Two verbs on one click (`K` key vs `O` observation) and no pending state.** A stray click would mint a key.
   Resolution: a click creates a *pending pick* (ghost ring). It commits as: automatically a **key** when the
   second view's pick at the same instant completes a triangulation within tolerance (toggle "Auto-commit
   triangulations", on); `K` commits a single-view pick as a key (needs a constraint); `O` commits it as a soft
   observation; `Esc` drops it. Nothing reaches the truth document without a deliberate commit.
3. **Frame-step keys.** Spec says `,` `.`; the whole dashboard uses arrows, Shift+/-10, Home/End via
   `FramePlayer`. Resolution: arrows stay canonical (one vocabulary); `,` `.` are accepted aliases.
4. **Time sync is the hidden failure mode.** Offsets are integer frames; a ball at 25 m/s moves ~0.8 m per
   frame at 30 fps, so a one-frame sync error makes triangulation residuals look like a bad camera. The spec
   shows a residual but gives the operator no way to tell "bad pick" from "off-by-one sync". Resolution: when
   a triangulation residual exceeds the warn band, the residual popover offers a read-only *sync probe*:
   residual of the same two picks at offset -2..+2 (computed by the triangulate endpoint, see section 9). The
   Studio never edits `sync_map.json`; it links to the Prepare Shots sync timeline (operator data stays owned
   there).
5. **Constraint choice is buried.** "Pick a constraint" for single-view instants is the second most common
   path (the far view often lacks the ball). Resolution: constraint chips live on the mode bar next to the
   active view, with the live-previewed 3-D point as you hover (ray/constraint intersection shown as a ghost in
   all views and the 3-D scene before commit).
6. **No guidance on where to author next.** A dense per-frame track from sparse keys needs a "what next".
   Resolution: timeline rows show un-supported spans (segment without any observation), flagged physics
   sanity spans, and pipeline-vs-truth disagreement; `N` / `Shift+N` jump to the next/previous such span.
7. **Truth status.** Ground truth should be distinguishable from a work in progress. Resolution: `meta.status`
   `draft | reviewed` (section 9 request); header badge shows it; only `reviewed` groups are eligible for the
   future regression gate.
8. **Colour overload risk.** Keys, 3 view identities, 5 segment kinds, 8 event kinds, residual severity, plus
   pipeline ghost track. Resolution: one channel per meaning (section 5): shape for key source, hue for
   segment kind and for view identity (disjoint sets), semantic tokens for residual severity, and the
   pipeline track is always a neutral dashed ghost so truth is visually dominant.

## 3. Layout

Chrome: standard shell; on mount this page collapses the sidebar to the 48 px icon rail (restores on leave).
Header: sidebar trigger, title "Ball studio", `DirtyBadge`-style status (saved / unsaved / draft / reviewed),
description, actions right: group `Select`, outcome `Select`, `Save` (primary), `Solve` status chip.
No second primary anywhere.

At 1440x900 (content ~1392 px after the rail, 16 px gutter, header 64 px):

```
┌ header: Ball studio · [saved|unsaved] [draft|reviewed]      group ▾  outcome ▾  Solve ● 12 ms   Save ┐
│ mode bar: [Click: Triangulate | Ray constraint ▾ ground height plane depth player | Observation]   │
│           snap ▢ auto-commit ▣ rays ▣ pipeline ghost ▣ residuals ▣        loupe Z   layout ▣ ▢   │
├───────────────────────────┬───────────────────────────┬───────────────────────────────┤
│ View A  origi01  (ref)    │ View B  origi02  (-142)   │ 3-D scene                     │
│ video + overlay canvas    │ video + overlay canvas    │ pitch, goals, frusta, rays,   │
│ ~480 x 270 (16:9)         │ ~480 x 270                │ track, keys, players          │
│ [tag chips: A][frame 440] │ [B][shot frame 298]       │ ~360 x 300  presets ▾         │
├───────────────────────────┴───────────────────────────┼───────────────────────────────┤
│ master transport (FramePlayer, frame input, fps, loop) │ Inspector (selected item)      │
│ timeline rows (see section 4)                          │  ~360 x remaining              │
└───────────────────────────────────────────────────────┴───────────────────────────────┘
```

CSS grid: `grid-template-columns: minmax(0,1fr) minmax(0,1fr) 360px`; views row height
`clamp(260px, 36svh, 420px)` with `aspect-ratio` 16/9 media inside `bg-stage`; the 3-D well spans the same row;
the inspector spans rows 1-2 of the right column below the 3-D well so the left two columns hold the views
over the timeline. Each view and the 3-D well are a bordered stage well with a small `Card` overlay header
(canvas-overlay recipe from DESIGN.md: 80% background, blur, `shadow-sm`) holding view label, shot, local
frame, and view-local toggles. Views and 3-D are not `Panel`s (they are wells inside one editor surface, like
the ball editor's columns); the inspector and timeline are the only `Panel`s.

**Focus view** (`F`, or double-click a view header): the focused view takes the two left columns at ~980 px
wide (~550 tall), the other view collapses to a 200 px picture-in-picture strip above the 3-D well and stays
clickable (a click there still adds its pick; epipolar line stays visible on the focused view). `Tab` cycles
the active view; the active view has a `ring-info` outline (info = selection, per DESIGN.md).

**Zoom/pan/loupe (per view):** wheel zooms about the cursor (1x-8x), drag with middle mouse or `Space`+drag
pans (Space also plays; hold-to-pan uses `Alt+drag` instead to avoid the transport conflict), `Z` held shows a
4x loupe following the cursor with a crosshair and the pixel readout, `0` resets zoom. Picks are stored in
native video pixels (as the ball editor does), so zoom is purely visual.

**Large screens (>=1920):** same grid; views grow; no extra columns.

**Narrow laptop (1024-1279):** 3-D well moves below the views (2 columns of views, then 3-D + inspector
side by side, timeline last); focus view becomes the recommended mode.

## 4. Timeline (the second signature surface)

A `Panel` titled "Timeline", flush canvas, reference frames on x. Separate labelled rows (28 px each, labels in
a 96 px left gutter, sentence case):

1. **Segments** — filled bars by kind (colour + pattern, section 5); gaps with no segment hatched.
2. **Keys and events** — keys as shapes (section 5); events as small glyph markers (drawn, not unicode) with
   hover tooltips ("440 · touch · P023 r_foot").
3. **Footage** — one thin strip per view showing which reference frames have video (the offset makes ranges
   differ), plus ticks where that view has a pick or observation. Out-of-range frames are hatched
   ("no footage in origi02").
4. **Residual** — sparkline of worst per-view reprojection px per frame where observed; background bands at
   the 3 / 8 / 15 px thresholds; keys over threshold get a destructive tick.
5. **Flags** — physics sanity spans (z<0, speed, discontinuity, curl) as warning/destructive blocks with the
   flag name on hover; click seeks. `N` jumps to the next flag or unsupported span.
6. **Pipeline delta** (appears once the group has a dense truth and a pipeline track) — 3-D distance between
   pipeline track and truth as a muted bar chart; this is the "where does the model disagree" lead.

Interaction: click or drag seeks (as the quality strip); click a key/segment/event selects it (inspector);
double-click a segment boundary does nothing destructive; drag a key horizontally is NOT supported (frame is
identity; re-key instead). Playhead is a 2 px foreground line with the frame readout in mono. Hover shows a
vertical guide and a tooltip with frame, timecode, active segment, residual.

## 5. Visual encoding and colour tokens

Principle: chrome stays achromatic; hue is data. Truth is bold, the pipeline is ghosted. One channel per meaning.
All canvas colours read from the lists below (hex permitted: data colours); any chrome uses theme tokens.
Define them once in `pages/ball-studio/palette.ts` and add the same names to `DESIGN.md` "Data colours".

**View identity** (rays, frusta, epipolar lines, view badges; max 3 views):

| View | Hex | Use |
|---|---|---|
| A (reference) | `#f0abfc` | view A badge, A's ray in 3-D, A's frustum, the epipolar line drawn in B |
| B | `#5eead4` | same for B |
| C | `#93c5fd` | third angle if a group ever has one |

Always accompanied by the letter badge so it is not colour-only.

**Segment kind** (timeline bars, projected solved track, 3-D track). Colour + stroke pattern:

| Kind | Hex | Stroke |
|---|---|---|
| flight | `#38bdf8` | solid 3 px, gravity-arc ticks every 5 frames |
| roll | `#34d399` | solid 2 px (matches Grounded) |
| carried | `#a78bfa` | dotted 2 px (matches Catch) |
| linear | `#94a3b8` | dashed 2 px |
| static | `#64748b` | hollow ring, no line |

**Key source** — encoded by marker shape, drawn at the key's projection in every view (diamond family, 6 px
half-size at 1x, scaled by zoom up to a cap; the fill stays the segment colour of the following segment):

| Source | Marker |
|---|---|
| triangulated | filled diamond, white 1.5 px keyline |
| ray_ground / ray_height / ray_plane / ray_depth | hollow diamond with a tiny constraint glyph (ground line / h / plane / arrow) beside it |
| player | diamond with a short tether tick toward the joint |
| manual | square |

Selected key: `info` ring (token `--info`) 2 px plus its id label; a key in a view where it was NOT observed is
drawn at 60% alpha (it is a projection, not a pick). Observations (soft): small plus signs in the view colour.

**Residual severity** (chips, sparkline bands, key ticks): semantic tokens, not hex — `success` (<=3 px),
`warning` (3-8 px), `destructive` (>8 px; >15 px is a hard reject and the key refuses commit unless the operator
confirms "Commit with residual 17.2 px?"). Shown as `ToneBadge`s in the inspector; shown on canvas as the
reprojection vector (from the picked uv to the reprojected uv, true length, plus a magnified 4x ghost vector
when the true length is < 2 px so it stays visible).

**Events** reuse the ball-anchor tag colours so one vocabulary spans both editors: touch `#22d3ee`, bounce
`#f472b6`, post/crossbar/net `#f59e0b` (goal impact), line_cross `#a3e635`, keeper_save `#a78bfa`, out
`#94a3b8`. Each also has a glyph (circle / ring / bar / line / hand / x) for non-colour reading.

**Pipeline ghost track**: `#e2e8f0` at 55% alpha, 1.5 px dashed, no markers. Never coloured by kind; toggle
"Pipeline" on the mode bar. **Solved (truth) track**: kind-coloured, 3 px, with a short trail (+/-12 frames)
brighter than the rest of the path.

## 6. Epipolar lines, rays and residuals

- **Epipolar line** (view B, after a pending pick in A, and vice versa): the projection of A's ray, sampled
  densely along the ray from the camera centre to 60 m / below ground z=-1 m and projected through B's full
  model including lens distortion, so it is a polyline, not an assumed-straight line. Drawn in A's identity
  colour, 1.5 px, with a faint 6 px halo for visibility on turf. **Height ticks** every 0.11 m (ground),
  1, 2, 5, 10, 20 m with a small mono label ("2 m") — the operator reads the height they are choosing while
  sliding along the line. The segment of the line outside B's image is clipped; if the entire line is outside
  B's image, B shows an inline note "A's ray does not reach this view" and the click falls back to a
  single-view constraint.
- **Snap**: a B click within 8 px of the epipolar line is projected perpendicular onto it (snap indicator =
  filled dot on the line); hold `Alt` for a free click. Snap is a preference chip ("Snap to epipolar") because
  the ball's true centre can sit slightly off the line when calibration is imperfect, and that offset *is* the
  signal. After commit the residual vector shows exactly how far.
- **Rays in 3-D**: from each camera centre, thin lines in the view's colour through the pick, solid to the
  intersection/closest approach and fading beyond. When two rays are skew, the shortest connecting segment is
  drawn in `destructive` with its length in cm ("gap 38 cm"). The triangulated point is a ball marker with a
  vertical stem to the pitch and a height label; a ground ring marks the drop point. Pending picks show a
  ghost marker.
- **Single-view constraint preview**: before commit, hovering shows the ray hitting the constraint (ground
  plane, chosen height, goal line plane, depth handle, or player joint) as a ghost point in all views and in
  3-D. Depth mode adds a draggable handle on the ray in 3-D (the only 3-D editing gesture; it never moves a
  committed key without going through the same pending/commit step).
- **Residuals**: always per view, always in px, always with the same severity tokens. The inspector table lists
  every observation of the selected key with view, shot frame, uv, reprojected uv, residual px. The sync probe
  (section 2.4) opens from a residual chip.

## 7. States

| State | Behaviour |
|---|---|
| Loading groups / scene | `PanelSkeleton media` in the views grid shape; 3-D well shows the viewer's `LoadingOverlay` pattern with a progress bar (cameras, players, track). Videos load lazily; each view independently shows a spinner on its well. |
| No groups | `PanelEmpty`: "No synced groups yet. Run Prepare Shots (group + align) from the dashboard; the studio lists every group in `sync_map.json`." |
| Group with no truth (empty skeleton) | Views live, timeline empty with an in-place onboarding strip over the key row: "Scrub to a frame where the ball is visible, then click it in a view." No modal tours. The first key dismisses it forever (per-viewer `localStorage`, try/catch). |
| Single-angle group (kroupi) | Layout collapses to View A + 3-D; mode bar offers only single-view constraints; a quiet note: "One angle: depth comes from constraints, so residuals are not available." |
| Scene load error | `PanelError` + Retry; editing disabled with visible reason. |
| Truth load error | `PanelError` "Could not load ball truth for origi. Saving is disabled so the existing file can't be overwritten." + Retry (mirrors the ball editor). |
| Video missing for one view | That well shows the ball editor's `VideoOffIcon` state; the other view and the solver remain usable. |
| Solving | Header chip: spinner "Solving..."; the previously solved track stays drawn with 50% alpha; edits keep working (requests are debounced 150 ms and the stale one aborted). |
| Solved | Chip: `success` dot "Solved - 12 ms - 34 keys". |
| Solve error | Chip `destructive` "Solve failed"; the toast carries the server message and offending key id; the last good dense track stays drawn hatched (stale) and the offending key is outlined in `destructive` in the timeline. The truth is still saveable (operator data wins; it is just flagged "unsolved" in the saved meta by the server if it differs). |
| Flags present | Chip shows count with warning tone; list in the timeline Flags row and the inspector. |
| Unsaved | `DirtyBadge` "Unsaved changes", Save enabled, `useUnsavedGuard`, confirm on group switch ("Discard unsaved keys on origi?"), `beforeunload`. Undo history is per session and survives a failed save. |
| Save conflict | If the server reports the file changed since load (see section 9), a `useConfirm` offers "Reload theirs" / "Overwrite" (destructive); never silent. |
| Reviewed group | `reviewed` badge; editing is allowed but the first edit asks to flip back to `draft` ("This group is marked reviewed. Editing returns it to draft."). |

## 8. Keyboard map

Shared (from `FramePlayer`/`useFrameKeys`; the Studio mounts ONE master player, views are not players):
`Space` play, `Left/Right` +/-1, `Shift+Left/Right` +/-10, `Home/End`; aliases `,` / `.` for +/-1 and
`Shift+,` / `Shift+.` for +/-10.

Studio-specific (ignored while typing in inputs/menus/dialogs, same guards as `useFrameKeys`):

| Key | Action |
|---|---|
| click (in a view) | create/replace the pending pick for that view at this instant |
| `Enter` | commit the pending pick(s) as the natural thing (two views -> triangulated key; one view + constraint -> constrained key) |
| `K` | commit pending pick as a key (needs two views or a constraint) |
| `O` | commit pending pick as a soft observation |
| `Esc` | drop the pending pick, then clear selection, then exit focus view (one level per press) |
| `G` `H` `L` `D` `P` | single-view constraint: Ground, Height (opens the height input), goal-Line plane, Depth, Player joint |
| `E` | event menu at the playhead (touch, bounce, post, crossbar, net, line cross, out, keeper save); arrow keys + Enter to choose; touch prompts for player and bone |
| `[` `]` | previous / next key; `N` / `Shift+N` next / previous flag or unsupported span |
| `Tab` | cycle the active view; `F` toggle focus view; `Z` hold for loupe; `0` reset zoom |
| arrow keys with a pending pick focused (`Alt+arrows`) | nudge the pending pick 1 native px (`Alt+Shift` 5 px) |
| `Del` / `Backspace` | delete the selected key / observation / event (undoable; no confirm because undo exists) |
| `Ctrl/Cmd+Z`, `Ctrl/Cmd+Shift+Z` | undo / redo |
| `Ctrl/Cmd+S` (and bare `S`) | save |
| `?` | shortcuts sheet (`Kbd` legend in a `Sheet`) |

Accessibility notes: all non-canvas controls are real buttons/selects; the canvas has `aria-label` and a
live region ("Pending pick at 812, 604 in view A; epipolar line shown in view B") so state is spoken; every
canvas action has a keyboard path via nudge + `Enter`; the timeline is a `role="slider"` for seek plus a list
of keys/events reachable through `[` `]`. Target AA contrast for all chrome; canvas marks have a dark halo so
they hold on turf and crowd backgrounds in both themes.

## 9. Mobile (390 px) and tablet

Posture: **read-only review on phone**, stated on the page (consistent with the dashboard's monitoring
posture and the existing "works best on a desktop screen" note, but stricter because three synced views
cannot be annotated on a phone). At < 768 px:

- Header collapses like other pages (actions wrap); a permanent `info` note: "Authoring needs a desktop
  screen. This view is read-only: scrub, compare angles and check residuals."
- Stack: master transport (sticky under the header), segmented control `A | B | 3-D` showing one well at a
  time at full width (default A; the solved track, keys and ghost still drawn), timeline panel, then the
  inspector as a collapsible list of keys/events/flags (tap to seek).
- All commit controls (mode bar, K/O/E, Save, undo, delete) are absent, not disabled, except `Save` which is
  omitted because nothing can change. Pinch zooms the active well.
- Light/dark behave as elsewhere; wells stay near-black.

Tablet (768-1023): single-column stack of views as two columns A|B, 3-D below, inspector as a bottom `Sheet`;
authoring allowed but focus view recommended; pointer/touch picks use the loupe-on-press (long-press shows the
loupe, release commits the pending pick).

## 10. Component breakdown (builder guide)

All under `frontend/src/pages/ball-studio/` (route `/ball-studio?group=`), lazily loaded like other pages;
sidebar "Editors" gets a "Ball studio" entry (icon: lucide `Crosshair` or `Orbit`; not used by existing
items); the dashboard Ball panel may deep-link with the current group. Follow `frontend/README.md`: shadcn
only, `Panel*`, `useResource` + `getJson`, `useConfirm`/`usePrompt`, `useUnsavedGuard`, lucide, sonner,
`Kbd`.

State and logic (no JSX):

- `types.ts` — mirrors `ball_truth` schema and the scene/solve responses (coded against
  `2026-10-04-ball-studio-api.md`).
- `api.ts` — `getGroups`, `getScene`, `getTruth`, `putTruth`, `postSolve`, `postTriangulate`.
- `truth-doc.ts` + `use-truth-doc.ts` — pure reducer over the truth document (add/replace/remove key,
  segment kind edit, observation, event, outcome) with a bounded undo/redo stack; `dirty` is derived by
  comparing with the last saved JSON. Immutable updates only.
- `use-solver.ts` — debounced `postSolve` on every doc change, abort-on-supersede, returns
  `{status: idle|solving|solved|error, dense, flags, residuals, stale}`; keeps the last good dense.
- `camera-model.ts` — project/unproject per view and frame (K, R, t, distortion) from the scene payload, ray
  sampling for epipolar polylines, constraint intersections (client-side previews only; the server
  solver is authoritative). Unit-tested against the same fixtures as the backend where possible (vitest).
- `use-pick-session.ts` — the pending-pick state machine (idle -> picked A -> picked A+B -> committed), mode
  (triangulate / constraint / observation), snap and nudge, calling `postTriangulate` live.
- `use-view-videos.ts` — owns N `<video>` elements, maps reference frame -> shot frame via `frame_offset`,
  frame-exact seeking (decode via `requestVideoFrameCallback` when paused; fall back to
  `/api/video/{shot}/frame?frame_idx` JPEG stills if exactness fails), pause/play sync, out-of-range state per
  view. Risk to verify early: browser `currentTime` seeking can land off by a frame; clicks must be on the
  displayed frame.
- `use-studio-keys.ts` — the Studio shortcuts (guarded like `use-editor-shortcuts.ts`).
- `palette.ts` — section 5 colours and glyph/pattern definitions (single source for canvas, timeline, 3-D).

Presentational:

- `index.tsx` — page: header, layout state (compare/focus), guard, loading/error gating.
- `studio-header.tsx` — group `Select`, outcome `Select`, `SolveChip`, `Save`, status badges.
- `mode-bar.tsx` — click mode, constraint chips, snap/auto-commit/ghost/rays toggles (`ToggleGroup`, `Toggle`,
  `Switch`), loupe/layout toggles.
- `view-well.tsx` — `<video>` + overlay `<canvas>` + zoom/pan/loupe + overlay header card; props: view
  descriptor, frame, doc/solve derived marks, pick-session handlers.
- `view-overlay-draw.ts` — painters: solved track, ghost track, keys/observations, pending pick, epipolar
  polyline with height ticks, residual vectors, reprojected key marks.
- `scene-3d.tsx` + `scene-engine.ts` — plain three.js via the existing `ViewerEngine` pattern (reuse
  `pitch.ts`, `pitchToThree`; do not fork the SMPL loader — players only need root + key joints from the
  scene payload, drawn as capsules/points). Adds frusta, rays, skew-gap segment, ball + drop stem, track
  tube coloured by segment, camera presets (View A, View B, top, goal-end A, goal-end B, free), follow-ball.
- `timeline.tsx` + `timeline-draw.ts` — section 4 rows, hover tooltip, click/drag seek, selection.
- `inspector.tsx` — tabs (`Tabs`): Selection (key/segment/event editor: source, constraint, kind `Select` with
  the auto-proposed value marked, residual table, delete), Flags, Group (outcome, notes, status,
  keyboard sheet). Uses `Table`, `ToneBadge`, `Field`-style rows; confirm only on destructive group-wide
  actions.
- `event-menu.tsx` — `Command`-based popover for `E`.
- `onboarding-strip.tsx`, `empty-states.tsx` — section 7 states.
- `mobile-review.tsx` — the read-only stack for < 768 px (shares `view-well` in a no-input mode).
- Shared extraction candidates (do not copy-paste from the ball editor): `StudioToneBadge` over
  `ToneBadge`; if `quality-strip.tsx` canvas helpers are reusable for the timeline, move them to
  `components/`, not a cross-page import.

Tests (vitest, where the repo has them): `truth-doc` reducer + undo, `camera-model` round-trip
(project(unproject)=identity, epipolar sample lies on the other view's pick for a synthetic pair),
`use-pick-session` transitions, `palette` completeness (every key source / segment kind / event kind has a
definition), frame mapping `r = f_shot - frame_offset`.

## 11. Requests to other ICs (API/spec deltas surfaced by this review)

1. **A2 (backend):** `POST /triangulate` should accept an optional `offsets: [-2..2]` list and return the
   residual per offset (sync probe), or the frontend will call it five times; also return the skew gap (cm)
   and the per-view reprojected uv so the residual vectors can be drawn.
2. **A2:** truth `meta.status` (`draft|reviewed`) and `meta.updated_at` on GET; `PUT` should accept
   `expected_updated_at` and return 409 + the current file on mismatch (second tab / concurrent agent).
3. **A2:** scene payload should include per-shot `frame_range` (valid reference frames with footage) and image
   size `W,H` per shot, and goal-plane definitions so the frontend can offer the "goal line plane" constraint
   without hard-coding x=0/105.
4. **A1:** add a `ball-studio` entry to the sidebar editors list when it creates the route/stage plumbing, or
   leave the route registration to the builder (B) — decide before B starts to avoid a conflict in
   `App.tsx`/`app-sidebar.tsx`.
5. **DESIGN.md (documenter, after build):** add the data-colour sets of section 5 under "Data colours", the
   "truth bold / pipeline ghost" rule, and the canvas-overlay + stage-well usage for multi-view editors.

## 12. Finish checklist for the build (per CLAUDE.md)

- Build to this document, `DESIGN.md`, `frontend/README.md`.
- Run the impeccable detector on `pages/ball-studio/`.
- Capture 1440 dark + light and 390 mobile (and 1024) for: empty group, authored group with a triangulated
  key and epipolar line, solve error, focus view, mobile read-only.
- Run `impeccable-finish-reviewer` until disposition `ship`; update `DESIGN.md` through the documenter; append
  the before/after to this file (or a dated sibling).
