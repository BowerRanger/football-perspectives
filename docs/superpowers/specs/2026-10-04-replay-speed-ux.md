# Replay speed in the group sync timeline: impeccable BEFORE pass

Date: 2026-10-04 · Branch: `worktree-gberch-shorts` · Status: design, no code yet
Mode: **Operate** (single operator, precise annotation on footage).
Backend contract: `docs/superpowers/specs/2026-10-04-replay-speed.md` (HTTP: `GET /api/replay-sync`,
`POST /api/sync/groups/{gid}/moments`, `POST /api/shots/{sid}/retime`, `POST /api/shots/{sid}/restore-native`;
`Alignment.playback_rate`, `Shot.retimed`, `Shot.native_frames`).
Large UX change (new interaction pattern, destructive flow, extends an existing editor): this document is the
required audit; the AFTER pass (detector, 1440 dark+light and 390 captures, finish reviewer to `ship`, DESIGN.md
update) is the builder's, see the finish checklist.

Evidence: captures of the live dashboard on the highlights reel (g11 = "Highlight 11", s042 ref, s043 slow replay,
stored auto offset -129 at 0.81) in `/Users/joebower/.claude/jobs/88133eb1/tmp/ux-speed/` (`before-d-dark.png`,
`before-d-light.png`, `before-m-dark.png`, `before-g11-dark.png`). The server was stopped afterwards; nothing was saved.

## 1. Audit of the current sync timeline

Sources: `frontend/src/features/stages/prepare-shots/` `group-sync.tsx`, `sync-editor.tsx`, `sync-video-column.tsx`,
`sync-timeline.tsx`, `sync-offsets.tsx`.

What works and must be kept
- Two stage-well videos side by side, a Play both / Lock offset / nudge toolbar, a drag-or-arrow timeline, one
  slider row per shot. Edits flip the method to `manual`; unsaved state is guarded (`useUnsavedGuard`) and badged.
- Frame maths already goes through `lib/frame-time.ts` (`frameTime` / `frameAtTime`). Keep it; the new rate maths
  composes with it, never replaces it.
- Sign convention: `frame_offset = frame_in_active - frame_in_reference`; timeline block start = `-offset`.
  At rate 1 this equals the backend's `ref_frame = rate * shot_frame - frame_offset`. No change for existing data.

Findings (P1 blocks the feature, P2 must be fixed in the same build, P3 note)

| # | Sev | Finding | Fix in this build |
|---|---|---|---|
| A1 | P1 | The model assumes rate 1. The timeline block width is the clip's native frame count; s043 (slow, 0.34x) is drawn 187 reference-frames wide when it covers about 64. The picture is wrong exactly where the operator needs it right. | Block width = `frames * rate` in reference frames; show the native length in the tooltip. |
| A2 | P1 | "Play both" runs the member at 1x and re-seeks it every frame when drift > 0.08 s, so a slow replay is dragged forward by seeks. At 0.34x it visibly stutters, so the operator cannot judge sync by eye. | Set `video.playbackRate` to the rate and compute the target frame with the rate (section 6). |
| A3 | P1 | No speed information anywhere. `MethodBadge` shows only "Auto 0.81" / "Manual": the 0.81 is an offset confidence, with no hint that the clip is slow motion. | Speed badge per member (section 3). |
| A4 | P1 | No camera-free path: the only way to align without a good auto estimate is the Lock button, which fixes one offset and cannot express a rate. | Match moments (section 4). |
| A5 | P2 | `Lock offset to current frames` and the badges give no preview of the consequence. Offset-only edits on a slow replay silently produce a plausible-looking but wrong sync. | When the member's rate is not 1, Lock and the offset input are labelled "offset (at rate 0.34x)" and Lock is disabled with the reason as visible text, pointing to Match moments. |
| A6 | P2 | Mobile (390): the group tab list wraps over the videos ("Highlight 9..12" render on top of the Reference card; `before-m-dark.png`). TabsList is `h-auto` but fixed-height children overflow. | Fix: tab list becomes a horizontally scrolling single row (`overflow-x-auto`, `flex-nowrap`) below `md`. This is a pre-existing bug the build touches anyway. |
| A7 | P2 | Native `<video controls>` bars sit inside the stage well and are the only scrub affordance; frame stepping exists only through the timeline. No keyboard frame step at all, which makes marking an exact contact frame a fight. | Frame-step keys (section 4); keep native controls (they are the play/seek fallback) but the frame readout becomes the primary clock. |
| A8 | P2 | `Frame N . 0.00s` readout is the only per-video chrome; there is nowhere to hang a speed or mark state. | Video header chip row (section 2). |
| A9 | P2 | Toolbar has 7 controls in one wrapping row at 1440 and no grouping; adding more would exceed the dense-toolbar budget. | Split into transport row and alignment row (section 2). |
| A10 | P3 | Group tab labels read "Highlight 11 (2)" while the data ids are `g11`; the operator and the sidecars speak `g11`. | Add the mono id to the tooltip only; no relabel. |
| A11 | P3 | Timeline block colours are info (reference), success (active), `white/20` (others). Low confidence is a warning ring. Fine, and reused for the speed states (section 5). | none |
| A12 | P3 | Light-theme captures keep the stage well dark, correct per the Fixed Surround Rule. | none |

Pattern check against DESIGN.md: shadcn components throughout, one Panel (no nesting; the inner bordered columns
are `rounded-lg border` wells inside the Panel body, existing), tokens not hex, lucide icons, `useConfirm`, sentence
case, mono only for ids and numbers. One deviation to retire: `bg-white/20` and `border-white/20` literals on the
timeline blocks sit on the stage well and are acceptable there (media surround), keep.

## 2. Direction

THESIS. Extend, do not invent. The sync editor already is "two videos, one offset, a timeline". Speed adds one
number per member (the rate) and one new way to set both numbers (paired moments). Everything stays inside the Group
sync Panel; no new page, no modal editor. Ball Studio's "place, then commit" and frame-exact transport are the
interaction reference.

### Layout at 1440 (inside the existing Group sync Panel, top to bottom)

1. **Group tabs** (unchanged).
2. **Two video wells** (unchanged grid, `md:grid-cols-2`). Each well's header row becomes: label + shot select
   (existing) on the left; right side a chip row. Reference well: `ToneBadge info "Reference"` and mono
   `frame N`. Member well: the **speed badge** (below), mono `frame N`, and when marking, a mark chip (below).
   Under each video the existing frame readout stays, extended: `Frame 112 . 4.50s . reference frame 112.0` (member
   only: maps the member frame to the reference clock with the rate, so the operator sees the equivalent instant).
3. **Transport row** (new split of the toolbar): `Play both` (default variant, the one primary on screen when not
   marking), step buttons `-1 f` `+1 f` (IconButtons, apply to the focused well), a segmented `Speed` control
   (ToggleGroup, size sm): `Fit` (default, plays the member at its detected/set rate), `1x` (raw, for comparing
   against the old behaviour). Hidden when the member rate is 1.
4. **Alignment row**: `Match moments` (outline) toggles the moments tray open; `Lock offset to current frames`
   (outline, existing; disabled with visible reason when the rate is not 1); offset input + nudges (existing);
   `Rate` read-only mono value with an edit pencil (operator override: opens a one-field `usePrompt`, 0.05 to 4,
   becomes a `manual` alignment); right side: `Unsaved changes` badge + `Save group`. When the moments tray holds
   unsaved pairs, `Save group` is replaced by the tray's own `Save alignment` (one primary per header rule).
5. **Moments tray** (collapsible region directly under the alignment row, same well styling `rounded-lg border`,
   not a Panel). Contents in section 4.
6. **Timeline** (existing component, extended): block width = frames * rate; a slow member block gets a diagonal
   hatch overlay and its rate label (`s043 . 0.34x . global 129-193`); ramp members draw two segments with a notch at
   the breakpoint; mark pairs draw as thin vertical connector ticks (reference position to member position, in the
   pair's sequence number) so a mis-paired mark is visible. Playhead unchanged.
7. **Per-member rows** (existing `SyncOffsetRows`), extended: each row = id, `SpeedBadge`, method badge, slider +
   offset input, and a kebab-free action pair at the end: `Retime` / `Restore` (outline, size xs). On mobile they
   wrap under the id.

At 390 the whole region is **read-only** (see section 8).

### Speed badge (component `SpeedBadge`, used in the video chip row, the timeline tooltip and each member row)

Always `ToneBadge` pill + lucide icon + text. Text = `<rate>x <kind> . <source> . <conf> %`, mono only for the
numbers. The tooltip carries the sentence form and the decision reason from `replay_sync.json`.

| State | Condition | Badge text | Tone | Icon |
|---|---|---|---|---|
| detecting | `replay_sync` stage running or no entry yet while tracks/camera exist | `Detecting speed...` | muted | `LoaderCircle` (spin) |
| real time | rate within `retime_tolerance` of 1 (0.92 to 1.08) | `real time` (+ `1.03x` if the stored rate is not exactly 1) | success | `Gauge` |
| slow | rate < 0.92, not ramp, applied | `0.34x slow motion . matched on players . 81 %` | info | `Turtle` |
| ramp | decision `ramp_not_applied` | `speed ramp: 0.27 to 0.41x, not applied` | warning | `TrendingUp` |
| no camera | decision `no_camera` / `no_tracks` | `no camera: mark moments` (clickable: opens the tray) | warning | `VideoOff` |
| low confidence | `low_confidence` or confidence < 0.5 | `0.34x? . matched on players . 38 %, check` | warning (ring) | `CircleHelp` |
| manual | alignment `method="manual"` with `playback_rate` | `0.34x slow motion . marked by you . 2 pairs` | info (solid) | `Hand` |
| retimed | `Shot.retimed` | `retimed to real time (was 0.34x)` + `native kept` | success | `History` |
| sped up | rate > 1.08 | `1.24x faster than live . check` | warning | `FastForward` |

Source words: `matched on players` (`player_formation`), `marked by you` (manual), `from stride` never shown (not
shipped). Confidence prints as an integer percent. Real-time with the reel-wide `speed_factor` is NEVER shown (it
is documented unusable). Every badge is text-first: colour never carries the state alone (icons differ, words differ).

Retimed, manual and kept-manual states always show `manual` or `retimed` before any auto info: operator data wins
visually as well as functionally. An auto estimate that was refused because a manual alignment exists is a muted
secondary line in the tooltip ("auto found 0.34x, kept your marks").

## 3. States, in the member row and video chip

- **detecting**: skeleton-width muted badge; actions disabled with the text "waiting for the replay_sync stage".
  Poll `GET /api/replay-sync` on stage completion via `usePipeline()` refresh, not on a timer.
- **real time**: badge only. No retime button (nothing to do). Match moments still available (an operator may
  disagree).
- **slow, applied, not retimed**: `Retime to real time` (outline xs) available. Playback runs at the rate.
- **ramp**: no retime button (backend does not retime ramps); tooltip explains piecewise retiming is out of
  scope; `Match moments` is the tool, and with >= 3 pairs the tray shows the ramp.
- **no camera**: the badge itself is the call to action; the row shows `Match moments` as an outline xs button.
- **low confidence**: playback still runs at the estimated rate but the badge is warning; `Retime` is available
  only after the operator confirms or marks (disabled, reason: "confidence below 60 %: confirm by marking moments
  or set the rate").
- **manual**: shows pair count; `Retime` available; editing the offset again keeps `manual`.
- **retimed**: `Restore native clip` (outline xs). Timeline block is drawn at rate 1 length; tooltip lists the
  native frame count (`Shot.native_frames`). The video element src stays `/api/video/{sid}` (server now serves
  the retimed clip); bump a `?v=` cache-bust on retime/restore success so the `<video>` reloads.

Tone/colour (DESIGN.md): only the four semantic hues. Info = a rate the pipeline or operator set (selection/hint
family), success = real time or retimed (complete), warning = needs a look, muted = pending. No new hue, no new
data channel. Timeline blocks keep their current fills; slow blocks add a hatch (second channel: pattern) and a
text rate label; ramp adds the notch glyph. Residual severity reuses the Ball Studio vocabulary (<= 3 reference
frames success, <= 8 warning, above destructive) as `ToneBadge`, text `residual 1.4 f`.

## 4. Match moments, step by step

Model: the tray holds pairs `{reference_frame, shot_frame}` in clip frame numbers (the same numbers the backend
takes). Local fit mirrors `rate_from_moments` (least squares of `reference_frame = rate * shot_frame - offset`);
shown live, final numbers come from the server's response on save and replace the local ones.

1. Operator opens the tray (`Match moments` button, or the `no camera` badge, or key `M`). The active member is the
   one in the right-hand well; the tray header names it: `Match moments for s043 against s042`.
2. They scrub both wells to the same real-world instant (a ball contact, a net impact; copy hint in the tray when
   empty). Step with frame keys. The frame readouts are mono and frame-exact (`frameAtTime`).
3. `1` marks the reference's current frame, `2` marks the member's. Each mark becomes a **pending mark**: a ring
   chip in that well's header (`mark 4812`, ring in `info`, with a lucide `Crosshair` icon) and a tick on its
   scrub bar overlay. Marks can be redone before commit by pressing the key again.
4. When both pending marks exist, the tray's mode line reads: `Pair 1 ready: ref 48 = s043 112. Enter adds it.`
   `Add pair` button shows `Enter`. `Esc` discards pending marks. Nothing enters the fit until commit (Ball
   Studio's place-then-commit rule).
5. Committed pairs list as rows: `#`, ref frame (mono), member frame (mono), interval rate to the previous pair
   (`0.33x`), a small delete icon button (`Backspace` on the focused row). The list sorts by reference frame; an
   out-of-order pair (member time running backwards) is flagged destructive text "order inverted".
6. **Live result strip** above the list, updating as each pair is added:
   - 0 pairs: hint only. 1 pair: `1 pair: offset only. Add a second moment to get the rate.` (offset is shown,
     rate stays the current rate).
   - 2 pairs: `rate 0.341x . offset -127.6 . residual 0.0 f` (two points always fit exactly; the strip says
     `2 pairs fit exactly, add a third to check`).
   - >= 3 pairs: `rate 0.338x . offset -126.9 . residual 1.4 f` and, if the interval rates differ by > 20 %,
     a warning line `Speed ramp: 0.27x then 0.41x` with the same text as the badge; pairs that sit furthest from the
     line are marked with a warning dot.
   - Fit is nonsense (rate <= 0 or > 4): destructive text, Save disabled with the reason.
7. **Preview on the timeline and in playback**: the member block re-draws with the fitted rate/offset at once (as
   unsaved, dashed outline), and `Play both` runs with the fitted values, so the operator can watch the contact
   happen together before saving. Pairs draw as connector ticks.
8. `Save alignment` (the one primary in the tray) POSTs `moments` with `retime: false`. Success toast:
   `Saved s043: 0.34x, offset -127.6 (marked by you)`; the alignment turns `manual`; badge becomes the manual state;
   `onSaved()` refreshes the model. A second line in the toast action: `Retime to real time` (runs the confirm
   below), because the very next thing the operator wants is usually that. Failure: sonner error toast with the
   server message, tray keeps its pairs.
9. Leaving with unsaved pairs trips `useUnsavedGuard` (what: "marked moments").
10. `Clear all` (ghost) uses `useConfirm` only when >= 3 pairs.

### Keyboard map (active when focus is not in an input or select; shown as `Kbd` hints on the buttons)

| Key | Action |
|---|---|
| `Space` | Play / pause both |
| `Left` / `Right` | Step the focused well 1 frame (`Shift` = 10). Focus = last clicked well; shown by an info ring on that well |
| `,` / `.` | Same as Left / Right (for keyboards where arrows scroll) |
| `Tab` (inside the video area) / `F` | Switch focused well |
| `1` | Mark reference frame (pending) |
| `2` | Mark member frame (pending) |
| `Enter` | Add the pending pair |
| `Esc` | Discard pending marks (second press closes the tray) |
| `Backspace` / `Delete` | Remove the focused pair row |
| `Alt+Left` / `Alt+Right` | Nudge the member's pending mark 1 frame without moving the reference (precise fine pick) |
| `M` | Toggle the tray |
| `Mod+S` | Save the tray (or Save group when the tray is empty) |

`Space` and arrows are swallowed only when the editor region has focus (`tabIndex=0` region with `role="group"`,
labelled), never page-wide. Existing timeline arrow behaviour on focused blocks stays untouched (it handles its
own keydown and stops propagation). All of it is also operable by the visible buttons (no key-only features).

## 5. Playback at the detected rate

- Rate source for the active member: pending tray fit if any, else the saved/auto `playback_rate`, else 1.
  `Speed: Fit | 1x` toggle switches between that and raw.
- Member element: `video.playbackRate = rate` while playing (clamp 0.0625 to 16, the browser range; a 0.08 rate
  is allowed, the toggle warns below the browser floor). The reference plays at 1.
- Target mapping (one function, `memberFrameForRef(refFrame, rate, offset)` in `sync-editor.tsx` or a small
  `sync-rate.ts`): `member_frame = (ref_frame + offset) / rate`, then `frameTime(round(member_frame), fps)` for
  seeks. Seeks stay frame-exact and only fire on paused scrubs or when drift exceeds a threshold; while playing,
  the drift threshold is measured in reference seconds (0.08 s) after rate conversion, so a correctly-rated player
  is not re-seeked. Re-apply `playbackRate` after every `loadedmetadata` / src change (the element resets it).
- Cursor and timeline stay in reference frames (as today); the member frame readout gains the reference
  equivalent (section 2).
- Retimed members have rate 1 and need none of this; playing a retimed member at `Fit` is just 1x.

## 6. Retime and restore

Both go through `useConfirm` (retime is reversible because the native clip is kept, so both are normal-weight confirms with a mono list of what changes; neither needs typed confirmation). Buttons: outline xs.
While the request runs: the row's buttons disable, the badge shows `Retiming...` (spinner), a sonner loading
toast resolves to success/error. These re-encode a clip, so allow tens of seconds; do not block other rows.
Neither runs while a pipeline job is in flight (the same `usePipeline()` busy gate other destructive actions use;
reason shown as text). After success: `onSaved()` plus video cache-bust.

**Retime to real time** (shown when slow/manual, not ramp, not already retimed)

- Title: `Retime s043 to real time?`
- Body: `s043 plays at 0.34x. This re-encodes the clip so it plays at the live shot's real-time speed (repeated slow-motion frames are dropped), then remaps its tracks and camera frame for frame without re-solving. The original clip, tracks and camera are kept and you can restore them.`
- Mono list: `shots/s043.mp4  replaced (187 -> 64 frames)`, `shots/native/s043.mp4  kept`, `tracks/s043_tracks.json  remapped`, `camera/s043_camera_track.json  remapped`, `shots/sync_map.json  rate 1.0, offset -127`.
- Warning line (shown when stages downstream have output for this shot): `hmr_world and later outputs for s043 were made from the slow clip. Re-run from hmr_world after retiming.`
- Buttons: `Cancel` / `Retime clip` (primary). Counts come from the member's `native_frames` and `frames * rate`.
- Success toast: `Retimed s043 to real time`, action `Undo` (calls restore-native). Undo is available for the
  toast's lifetime; restore stays in the row afterwards.

**Restore native clip** (shown when `retimed`)

- Title: `Restore the native clip for s043?`
- Body: `This puts back the original slow-motion clip, its tracks and camera, and returns the alignment to rate 0.34x. Anything you edited on the retimed tracks since the retime is lost.`
- Mono list mirrors the retime list reversed.
- Buttons: `Cancel` / `Restore clip` (primary).
- If the operator edited tracks since the retime (server can tell via mtime; if the contract does not expose it,
  the copy above covers it).
- Success toast: `Restored the native clip for s043`.

Edge copy: retime from a low-confidence estimate is blocked (reason text, section 3); `Retime` from a manual
alignment is allowed (the operator's own marks are the strongest source). The moments Save toast offers the retime
action; the `retime: true` body flag is used only when the operator ticks the retime step in that follow-up, never
silently.

## 7. Component breakdown (builder)

New files in `frontend/src/features/stages/prepare-shots/` (each < 250 lines):

| File | Contents |
|---|---|
| `replay-speed.ts` | Types `SpeedInfo`, `SpeedState`; `deriveSpeedState(alignment, shot, replaySyncMember)`; `fitMoments(pairs)` (LSQ, residual, interval rates, ramp flag, mirrors backend); `memberFrameForRef`, `refFrameForMember`. Pure and unit-tested. |
| `use-replay-sync.ts` | `useReplaySync()` -> `GET /api/replay-sync` (version 1, groups[].members[]), refresh on stage completion; returns a lookup `by shot id`. |
| `speed-badge.tsx` | `SpeedBadge` (table in section 2); tooltip with reason; clickable variant for `no camera`. |
| `moments-tray.tsx` | Tray UI: mode line, pending-mark state, live result strip, pair list, Add / Discard / Save / Clear. Props: `reference`, `member`, `fps`s, `currentRate`, `onPreview(fit | null)`, `onSaved`. Owns pair state and posts to `/api/sync/groups/{gid}/moments`. |
| `use-moment-keys.ts` | The keyboard map above as a hook scoped to the editor region ref. |
| `retime-actions.tsx` | `RetimeButton` / `RestoreButton` using `useConfirm`, `usePipeline()` busy gate, sonner promise toasts, endpoints `/api/shots/{id}/retime` and `/restore-native`; returns video cache-bust key. |

Edits to existing files:
- `types.ts`: `SyncAlignment.playback_rate?: number`; `Shot.retimed?: boolean`, `native_frames?: number`; new `ReplaySyncMember`
  type matching the `replay_sync.json` example (`estimate`, `decision`, `reason`).
- `sync-editor.tsx` (283 lines now): carry `rates` state beside `offsets`; save must send `playback_rate` back in
  `/api/sync` alignments or it would reset a manual rate to 1 (check `POST /api/sync` accepts it; if not, ask the
  backend IC); play loop per section 5; split transport / alignment rows; embed the tray; preview fit overrides.
  If it passes ~350 lines, extract the play loop into `use-sync-playback.ts`.
- `sync-video-column.tsx`: header chip row, member frame equivalent, mark chip slot, `playbackRate` effect, focus ring,
  cache-bust query on `src`.
- `sync-timeline.tsx`: block width by rate, hatch for slow, ramp notch, connector ticks, tooltip with rate and native
  length; `AlignMethod` gains `rate`.
- `sync-offsets.tsx`: `SpeedBadge` + `Retime` / `Restore` in each row; `MethodBadge` stays for the confidence of offset.
- `group-sync.tsx`: description copy (section 8 of copy list) and the mobile tab-list fix (A6).
- `ShotModel` / `buildShotModel`: pass `replay_sync` member data into `GroupView` (or read the hook in the editor).

Reuse, do not add: `ToneBadge`, `Button` (outline/xs), `ToggleGroup`, `Kbd`, `IconButton`, `Slider`, `Input`,
`useConfirm`, `usePrompt`, `useUnsavedGuard`, `usePipeline`, sonner. No new colours, no new primitives, no nested Panel.

Microcopy list: group-sync description becomes: `Align shots within a highlight. Scrub both videos to the same instant and lock the offset, drag clips on the timeline, or mark matching moments to set speed and offset. Edits become manual and survive re-alignment.`

## 8. Mobile (390) and light theme

Posture is monitoring (PRODUCT.md): **read-only at 390**. Shown: the group tabs (now a scrolling single row),
videos stacked, the `SpeedBadge` per member, method badge and offset values as text, the timeline (horizontally
scrollable, no drag), and `Play both` (playback at the rate works). Hidden below `md`: `Match moments`, marking
keys, tray, Lock, sliders-as-inputs become display-only text, `Retime` / `Restore`, `Save group`. One line says it:
`Marking moments and retiming are desktop-only.` (muted, visible, replaces the toolbar). Light theme: all chrome on
tokens (success/warning/info/destructive resolve to the darker light values); the stage wells and the timeline stay
dark.

## 9. Critique of the proposed flow

- Strong: reuses the existing pattern; place-then-commit prevents accidental pairs; live result strip gives the
  feedback the ground-truth labelling needed (the operator can see a bad pair by its residual immediately).
- Risk: the operator marks a contact on the wrong frame by one. Mitigation: frame-exact stepping, Alt+arrow fine
  nudge, residual in frames, connector ticks, and playback preview before save.
- Risk: two keys `1`/`2` are hidden. Mitigation: every key is also a visible button with a `Kbd` hint; the mode
  line says what is pending and what Enter does.
- Risk: retime confuses with "sync". Mitigation: Retime lives in the member row, separated from the tray; the
  confirm lists exactly what moves.
- Open question for the backend IC: does `POST /api/sync` round-trip `playback_rate`? If not, saving offsets
  after a manual rate would drop it. Also whether `POST .../moments` returns `residual` and per-interval rates
  (the UI computes them locally anyway, but must reconcile with the server's).
- Open question: a progress signal for retime (re-encode of a long clip). Without it, the UI shows an indeterminate
  loading toast.

## 10. Finish checklist (builder; do not skip)

1. Implement per section 7; run `cd frontend && npx tsc -b --noEmit`, `npm run build`, commit `src/web/static/app/` with
   the source; run `.venv311/bin/python -m pytest tests/test_web_*.py -q` (endpoint markers `/api/replay-sync`,
   `/moments`, `/retime`, `/restore-native` must appear in both `frontend/src` and the bundle).
2. Unit tests for `fitMoments`, `deriveSpeedState`, `memberFrameForRef` (rates 1, 0.34, a ramp, inverted order).
3. Impeccable detector on every changed file; fix findings.
4. Captures, Playwright against `recon.py serve` on a scratch/copied output (never save into the reel dir):
   1440 dark and light for each state in section 2 (detecting, real time, slow, ramp, no camera, low confidence, manual,
   retimed), the tray with 1, 2 and 4 pairs, both confirms; 390 dark (read-only state, tab list fix verified).
5. Behavioural checks: playback at 0.34x stays in step (no stutter-seeks); marks are frame-exact (`frame-time.ts`); unsaved
   guard fires with pending pairs; keys do nothing inside inputs; operator manual alignment is not overwritten by
   a re-run (`kept_manual` shows as such).
6. Finish reviewer (`impeccable-finish-reviewer`) until disposition is `ship`.
7. Documenter: add to DESIGN.md "Group sync speed": speed badge states table, moments tray (place-then-commit reuse),
   slow-block hatch, confirm copy conventions. Update `frontend/README.md` if a convention is added.
8. Record AFTER captures and the reviewer verdict by appending an "After" section to this file.
