> **Status note (2026-09-29):** this re-audit was taken before the final fix batch (commit 16c18bf). Since then F-1 (HMR run-for-selection confirm), F-2 (failed anchor load disables Save), F-3 (named slider thumbs), F-4 (light-theme AA tokens, muted rows) and F-10/F-11 are fixed; F-5 (job cancel) and the P2/P3 backlog remain open.

# Rebuilt Dashboard Re-audit + UX Critique (React 19 + Vite + shadcn/ui)

**Target:** the SPA in `frontend/src` (built into `src/web/static/app/`), served by `src/web/server.py` at `http://localhost:8766` (output `output`, shots `gberch` / `gberch-2`).
**Worktree:** `web-shadcn-revamp` @ `1ec6f04`. All `file:line` refs are relative to `frontend/src/` unless prefixed `server.py`.
**Rubric:** the same as the baseline (`impeccable/reference/audit.md` 5 dims 0-4; Nielsen guide in `critique.md`).
**Baseline:** `docs/superpowers/specs/2026-09-28-dashboard-ux-audit.md`. Audit **5/20**, heuristics **16/40**, 69 findings (P0 2 / P1 18 / P2 36 / P3 13).

**Evidence used**
- 22 "after" screenshots in `shots/final/`, plus new ones in `shots/reaudit/`: `ball-editor-noshot.png`, `mobile-sidebar-open.png`, `mobile-tracking.png`, `mobile-export.png`.
- **axe-core 4.x scan**, run through Playwright on 12 routes × 3 configs (dark 1440, light 1440, dark 390). Raw output: `tmp/axe_dark.json`, `axe_light.json`, `axe_mobile.json`. Runner: `tmp/axe_run.js`. The scan also records touch-target sizes and horizontal scroll.
- **Keyboard tab-walk and title/heading probe** (`tmp/kbd_probe.js`).
- **Impeccable detector:** 0 findings, down from 29.
- **oxlint:** 63 warnings (`tmp/lint.txt`, from the build job).
- A code read of every finding below. Nothing destructive was clicked.

Measured totals, compared with the baseline:

| Metric | Legacy | Rebuilt |
|---|---|---|
| Hex literals in UI source (excl. `components/ui`) | 787 (91 unique) | 94, all data/canvas palettes (`lib/format.ts`, `tags.ts`, pitch/overlay draw files) |
| Inline `style={{}}` | 524 | 21 (data geometry / player colours) |
| CSS custom properties | 0 | full shadcn token set + `success/warning/info/stage`, dark + light |
| `@media` / responsive | 0 | Sidebar→Sheet under `md`, mobile editor layouts, 0 routes with horizontal page scroll at 390px |
| `aria-*` / roles | 0 | pervasive (labels on icon buttons, `aria-live` log status, `role=toolbar`, labelled nav groups) |
| `prefers-reduced-motion` | 0 | handled (`index.css:180-186`) |
| `title`-only hints | 78 | ~6 (`multi-shot-status.tsx:43`, `shot-tile.tsx:151,156`, `dropped-tray.tsx:83`, `sync-offsets.tsx:60`, `panel.tsx:111`) |
| Native `alert/confirm/prompt` | 6 | 0 (`hooks/use-dialogs.tsx`) |
| Frame-transport implementations | 5 + 2 | 6 (still not shared, see F-7) |
| Lenient `getJsonOrNull` call sites (error → empty) | ~all fetches | 73 of 91 fetches |
| axe serious/critical violations | not run | every route has ≥1 (slider name). Light theme adds 1-20 contrast nodes/route. Viewer has 22 critical (invalid ARIA) |

---

## 1. Audit Health Score

| # | Dimension | Before | After | Key finding |
|---|---|---|---|---|
| 1 | Accessibility | 1 | **3** | Real landmarks (`nav`, `main`, `h1`, card `h2`), every control is keyboard-reachable with a visible focus ring, icon buttons are labelled, and the log uses `aria-live`. Remaining AA failures: every frame `Slider` thumb is unnamed (axe serious on 11/12 routes, `components/ui/slider.tsx:45-50`); light-theme tone badges are 3.6-4.2:1; suppressed rows use `opacity-60` (2.0-3.9:1); the viewer player list is `button[role=listitem][aria-pressed]` ×22 (axe critical). |
| 2 | Performance | 2 | **3** | Route/stage code-splitting, with three.js lazy (737 KB chunk, main 385 KB). WebGL disposal (`ball-3d-scene.ts:147-158`, `viewer/engine.ts:356-365`), render-on-dirty in the viewer, rAF-batched logs, parallel fetches. Gaps: the log dock re-renders up to 5,000 non-virtualised spans running 3 regexes each per flush (`log-dock.tsx:51-69`); N+1 per-shot/per-player fetches remain (`multi-shot-status.tsx:104`, `refined-poses/index.tsx:35-40`, `ball-anchor-editor/api.ts:112-120`). |
| 3 | Responsive Design | 0 | **3** | The sidebar becomes an off-canvas Sheet at 390px (verified: 1 `role=dialog`). Header actions wrap, editors have stacked/tabbed mobile bodies, and no route scrolls horizontally. Gaps: small touch targets at 390px (Tracking 45/125 interactive < 24px; Export 45/61, including 16px POV/OTS checkboxes). Panel header actions crush the title column and clip ("Open full scre…") on Export (`mobile-export.png`, `panel.tsx:35-41`). |
| 4 | Theming | 1 | **3** | A full token system with dark default and working light theme. Hex only for data. Gaps: light-theme `--success/--warning/--info` are too light for 12px text on their own `/15` tints (`index.css:79-84` with `status.tsx:73-78`). A few chrome literals remain: `text-white/40` in `sync-timeline.tsx:168`, canvas chrome hex in `ball/topdown-canvas.tsx`, inline team-colour border in `tracking/player-row.tsx:94-101`. |
| 5 | Implementation Integrity | 1 | **2** | Coherent component system; detector clean; one ball editor that round-trips the full `BallAnchorSet`. Several verified issues are not isolated: (a) error-as-empty survives in 73 `getJsonOrNull` calls, including both editors' *saved-anchor loads*, so a failed GET followed by Save can overwrite ground truth (F-2); (b) 6 hand-rolled transports with diverging features (F-7); (c) in-panel run paths bypass the new destructive-run policy (F-1); (d) player-label drift ("Referee" vs "Ref") (F-13); (e) README conventions broken in places (raw `<button>` in `viewer/overlays.tsx:70`); 63 lint warnings. |
| **Total** | | **5/20** | **14/20** | **Good** (band 14-17). Up from Critical. The remaining weakness is integrity: errors are still masked as empty states, and the destructive-action policy is only partly applied. |

### Implementation Integrity Verdict: **PASS, with reservations**

The rebuild expresses one product-specific system:
- One `Panel` container, the `StatusBadge`/`ToneBadge` state vocabulary, tokens instead of hex, lucide icons, and `useConfirm`/`usePrompt` instead of native dialogs.
- One `PipelineProvider` job store with SSE reattach.
- One `BallAnchorEditor` used both in the Ball stage and at `/ball-anchor-editor`.
- The anchor editor is a real in-shell component (no iframe).
- The detector's 0 findings are confirmed by code read: no side-tab borders, no glyph icons, no uppercase tracked titles.

The reservations are behavioural, not visual:
1. The legacy "error looks like empty" pattern (baseline S-13) was ported, not fixed, in most read paths (`lib/api.ts:49-58` used 73×). In the two editors this pattern is a data-safety hole.
2. The new "confirm + dry-run + clean-after-admission" policy covers only the header Re-run. `/api/run-shot` still wipes per-shot output with no confirmation from HMR World's "Run for selection".
3. Six separate transports (the shared `<FramePlayer>` the baseline asked for was not built).

---

## 2. Nielsen Heuristics

| # | Heuristic | Before | After | Justification |
|---|---|---|---|---|
| 1 | Visibility of system status | 2 | **3** | **Good:** tri-state status (Complete / Partial / Not run / Running / Failed) in the header and sidebar, a log dock with elapsed time and a "Connection lost — reconnecting…" badge, reattach on reload, skeletons, toasts, and editor dirty/saved chips ("59 anchors saved"). **Gaps:** no progress or ETA for 35-60 min runs; "Partial" doesn't say which shots are missing; Refined Poses shows **Complete** next to "Needs HMR World first" with the controls disabled (`desktop-index-refined_poses.png`). |
| 2 | Match with real world | 2 | **3** | Humanised stage names; "Rendered in 36m 33s" is labelled; no `undefined`. Identifiers still leak into copy: "Run hmr_world first" (`refined-poses/index.tsx:104`, `hmr-world/index.tsx:93`, `camera/index.tsx:88`, `export/index.tsx:106`); toasts like "hmr_world running for gberch__P001" (`hmr-world/index.tsx:78`); a raw `unknown` team chip on every tracked player. |
| 3 | User control and freedom | 1 | **2** | **Good:** unsaved-edit guards (route block + `beforeunload` + shot-switch confirm), Cancel on every dialog, restorable dropped-shot tray, undo for dismissing an auto event. **Missing:** running jobs can't be cancelled (no endpoint), and there's no undo for track merges, deletes, "Ignore unknown" or anchor deletes (confirmation only). No ⌘Z. |
| 4 | Consistency and standards | 1 | **3** | One primary colour, one button set, one dialog system, and consistent Continue / Re-run clean wording. **Drift:** six transports with different feature sets (a frame-number input only in the pitch-anchor editor, Home/End only in some); the Export picker says "Referee"/"MacAllister" where HMR, Refined and the Viewer say "Ref"/"P001"; "Rerun camera tracking" vs "Re-run clean". |
| 5 | Error prevention | 1 | **3** | **Good:** the header Re-run lists exactly what will be deleted (server dry-run `/api/output/{stage}/artifacts`), requires the stage name typed for `prepare_shots`/`tracking`, and clears only after the run is admitted (`server.py` `clean_first`). Also: the global run lockout, visible blocked reasons, and scoped confirms for bulk track operations. **Remaining holes:** "Run for selection" wipes a shot's HMR cache unconfirmed (F-1), and a failed anchor load followed by Save can clobber ground truth (F-2). |
| 6 | Recognition rather than recall | 2 | **3** | A searchable landmark palette with coordinates; inline tag help; `Kbd` shortcut hints under every player; `?shot=` carried into the editors; labelled selects everywhere; hover-only hints down from 78 to ~6. The shot choice still isn't shared across stage panels: HMR, Render and Tracking each default to `shots[0]`. |
| 7 | Flexibility and efficiency | 2 | **3** | ←/→, Shift±10, Space, Home/End and ⌘S in the editors; number-key tag selection in the ball editor; deep links `/?stage=` and `?shot=`; resizable editor panes; collapsible icon sidebar; bulk track tools. There's no command palette and no undo stack. |
| 8 | Aesthetic and minimalist design | 2 | **3** | Calm, consistent hierarchy; sentence-case titles; the segments table is collapsed and its empty columns hidden. Noise remains: an all-zero "Multi-view" column on Refined Poses, and on Ball the summary sits above the editor. At phone width the panel header squeezes the description to one word per line. |
| 9 | Error recognition and recovery | 1 | **2** | Where errors are surfaced they are good: `PanelError` with the server `detail` (Render has Retry), toasts carrying `detail`, log "First error" jump, reconnect handling. But the baseline root cause remains in HMR, Refined Poses, Export, the viewer scene load and both editors' data hooks: a 500 still renders "No refined tracks yet — run hmr_world first" (`refined-poses/index.tsx:31-32,100-106`). |
| 10 | Help and documentation | 2 | **3** | A description under every stage header and panel; keyboard-reachable HoverCard metric help with legends (`camera/camera-metrics.tsx:47-62`); inline tag help; empty states that name the next step; `Kbd` hints. There's no link to the specs or docs. |
| **Total** | | **16/40** | **28/40** | **Good** (28-35). Solid foundation; the weak areas are control/undo and error masking. |

### Persona check (vs baseline red flags)

- **Alex (power user):**
  - Fixed: stages are links with URLs, the shot survives into the editors, and every player has keyboard stepping.
  - Still true: "Run for selection" on All players silently discards the per-player cache and commits most of an hour with no estimate and no cancel.
- **Sam (a11y):**
  - Fixed: sidebar, palette, tags, anchor list, tile menus and viewer players are all reachable; log status is announced; status carries text.
  - Still true: frame sliders announce only as "slider" with no name; in the light theme the tone badges and warning numbers fail contrast.
- **Joe mid-run:**
  - Fixed: reloading reattaches the log and the lockout; camera reruns from the editor show in the log dock; header Re-run on Tracking now demands a typed "tracking".
  - Still true: a job started from another tab after this tab loaded is invisible here (reattach runs once, `hooks/use-pipeline.tsx:270-286`).

---

## 3. Resolution of the baseline P0/P1 findings

There are 20 baseline P0/P1 issues (2 P0 + 18 P1; PS-1 and RP-1 count as one, as in the baseline).

| ID | Sev | Status | Evidence |
|---|---|---|---|
| S-1 Re-run wipes output unconfirmed, before admission | P0 | **Resolved** | `components/stage-actions.tsx:65-100`: dry-run list from `/api/output/{stage}/artifacts`, destructive `AlertDialog`, typed confirm for `prepare_shots`/`tracking`. `server.py` `RunRequest.clean_first` clears only after the 409/429 checks. Continue is the filled primary (screenshots). The client no longer calls `DELETE /api/output`. *Residual: the in-panel `/api/run-shot` path, see F-1.* |
| B-1 Inline ball Save erases chains/dismissals/end_frame/landmark | P0 | **Resolved** | One editor. `ball-anchor-editor/types.ts:113-128` normalises every field (`spin`, `confidence`, `end_frame`, `landmark`). The payload sends `shot_chains` + `dismissed_auto` (`use-ball-anchor-editor.ts:148-156`). *Residual: no server merge/PATCH and no load→save identity test; new load-failure hole F-2.* |
| S-2 Run state lost on reload; no cancel | P1 | **Partial** | `GET /api/jobs?status=running` + SSE replay reattach (`use-pipeline.tsx:270-286`). **No cancel endpoint or button**; only `jobs[0]` is reattached; no polling for jobs started elsewhere. |
| S-3 SSE has no error handling | P1 | **Resolved** | `use-pipeline.tsx:194-218`: close, exponential backoff up to 15 s, a status probe, 404 → "server restarted, job gone", and a "reconnecting" badge in the dock. |
| S-4 Binary status misreports partial output | P1 | **Resolved** | `server.py` `partial` flag + `_stage_has_output`. `components/status.tsx:5-32` has a tri-state badge and a distinct half-filled sidebar dot; HMR, Ball, Export and Render show "Partial". *Residual: no per-shot count or tooltip; the Refined "Complete + Needs HMR" contradiction (F-10).* |
| S-5 Header vs in-panel run buttons disagree; status wiped | P1 | **Resolved** | One `StageActions` in the page header. In-panel triggers (HMR, Render, ingest, anchor rerun) all read `isRunning` and route through `attachToJob`, so the log dock persists. `BlockedNote` gives a visible reason. *Residual: in-panel runs skip dependency gating and confirmation (F-1).* |
| S-6 Stage nav not keyboard-reachable, no routes | P1 | **Resolved** | `app-sidebar.tsx:60-80` uses `<Link to="/?stage=…">`. The tab-walk reaches all 8 stages and 3 editors with a visible focus ring. *Residual: no `aria-current` (P3, F-18).* |
| S-7 No landmarks, headings, ARIA | P1 | **Resolved** | `<nav aria-label="Dashboard">`, shadcn `<main>`, `h1` in `page-header.tsx:32`, panel titles as `role=heading aria-level=2` (`panel.tsx:36`), `section aria-label="Run log"` + `aria-live` (`log-dock.tsx:128-153`). *Residual: heading-order skips and region nits (F-21).* |
| S-8 Contrast failures | P1 | **Partial** | Dark theme: muted-foreground at L 0.72 on card L 0.185 (~6.9:1); primary passes. **Light theme fails**: axe finds 1-20 contrast nodes on 10/12 routes (tone badges 3.57-4.2:1; `text-warning` on white 4.39:1); `opacity-60` suppressed rows fail in both themes (F-4). |
| S-9 Desktop-only layout | P1 | **Resolved** | `SidebarProvider` + Sheet under `md` (`mobile-sidebar-open.png`), wrapping header, mobile bodies for both editors and the viewer, no horizontal page scroll at 390px on any of 12 routes. *Residual: touch targets and panel-header squeeze (F-8, F-9).* |
| PS-1 / RP-1 "undefined frames/flagged" | P1 | **Resolved** | `fmt*` never prints undefined (`lib/format.ts:38-58`). Refined summary tiles show foot-lock, PhysPT, cleanup and jitter (`desktop-index-refined_poses.png`). `multi-shot-status.tsx:80-92` reads `total_frames` / `foot_lock` / `cleanup`. |
| T-1 Disabled toolbar buttons look enabled | P1 | **Resolved** | shadcn variants with real disabled styling (`tracking/toolbar.tsx`; greyed "Merge selected" and "Delete selected" in `desktop-index-tracking.png`), counts in labels. |
| T-2 Bulk track changes: no confirm, no undo | P1 | **Partial** | A scoped confirm for Merge by name, Ignore unknown, Delete ignored and deletes (`track-editor.tsx:133-211`). **No undo and no server snapshot.** The Merge-by-name dialog counts only *this* shot's names although it rewrites every shot (`:175`). |
| C-1 Anchor editor as fixed iframe nesting the dashboard | P1 | **Resolved** | `<AnchorEditor embedded />` in-shell (`camera/index.tsx:36-56`), a ResizablePanelGroup body, no nested header, "Open full screen" is a router link. |
| C-2 Iframe camera rerun invisible to run state | P1 | **Resolved** | `anchor-editor/use-anchor-editor.ts:165-195`: confirm → save → `/api/run-shot` → `pipeline.attachToJob` (log dock + lockout). |
| B-2 Two diverged ball editors | P1 | **Resolved** | `features/stages/ball/index.tsx:143` renders the same `BallAnchorEditor` as the route; the legacy HTML was deleted. |
| E-1 Viewer iframe 780px below the table; Close nests the dashboard | P1 | **Resolved** | The viewer is a React component directly under a 2-row status table, with "Open full screen" and no Close (`desktop-index-export.png`). The engine disposes on unmount and renders only when dirty. |
| RN-1 "Render" renders every shot next to a shot select | P1 | **Resolved** | Separate "Render gberch" (`/api/run-shot`) and "Render all shots" (`render/index.tsx:93-99`). *Residual: no estimate or confirm on Render all.* |
| A-1 Unsaved anchors silently discarded | P1 | **Resolved** | `hooks/use-unsaved-guard.tsx` (useBlocker + `beforeunload`), shot-switch confirm (`use-anchor-editor.ts:114-129`), ⌘S (`use-editor-keys.ts:55`); the same in the ball editor (`workspace.tsx:26`). *No localStorage draft.* |
| BA-1 No `?shot=` → black canvas | P1 | **Resolved** | `/ball-anchor-editor` resolves to the first shot and exposes a ShotSelect with anchor counts (`shots/reaudit/ball-editor-noshot.png`). |

**Resolved 17 / Partial 3 / Open 0** of 20.

Other baseline items worth noting:

| Status | Items | Notes |
|---|---|---|
| **Resolved** | S-10, S-11 (partly), S-12, S-14, S-15, S-16, S-17, S-18, PS-2, PS-4, PS-5, PS-6 (text badges), PS-7, PS-8, T-3, T-4, T-5, C-3, H-2, H-3, RP-2, RP-3, B-3, E-2, RN-2, A-2, A-3, A-5, A-6, BA-2, BA-3, BA-4, BA-5, V-1, V-2, V-3 (render-on-dirty), V-4 | |
| **Partial** | S-13, T-6, B-4, PS-3, A-4 | S-13: error masking remains (F-6). T-6: six transports (F-7). B-4: the summary still sits above the editor. PS-3: keyboard "Move to group" exists; drag still has no keyboard sensor. A-4: confirm, but no undo. |
| **Open** | H-1 (escalated, F-1), E-3 (F-13), RN-3, V-5 | |

---

## 4. Remaining findings by severity

**Counts: P0 0 · P1 5 · P2 10 · P3 12 (27 total, down from 69).**

### P1

**F-1 [P1] HMR "Run for selection" deletes the shot's HMR output without confirmation, and bypasses Continue's cache**
- **Location:**
  - `features/stages/hmr-world/index.tsx:71-86, 136`: POSTs `/api/run-shot` straight away; the default scope is "All players".
  - `server.py` `post_run_shot` (~2593-2637): unlinks `hmr_world/{shot}_*` after admission.
- **Category:** Error prevention / Implementation Integrity.
- **Impact:** One click discards every cached per-player `.npz` for the shot and commits a 35-60 min GVHMR recompute. There's no dry-run list, no estimate and no cancel. The header "Continue" would have resumed from that cache. This is exactly what the new policy (S-1) exists to prevent, but only the header path follows it.
- **Fix:**
  - Reuse `useRerunFlow`'s confirm, fed by a shot-filtered artifacts dry run (`/api/output/hmr_world/artifacts?shot=`).
  - Add a server `clean` flag to `/api/run-shot` that defaults to `false`, so a scoped run resumes by default.
  - Put an estimate from past job durations in the dialog.

**F-2 [P1] Both editors treat a failed load as "no saved anchors", so the next Save overwrites operator ground truth**
- **Location:**
  - `pages/ball-anchor-editor/api.ts:53`: `getJsonOrNull('/ball-anchors/{shot}')`. The server returns 500 on a corrupt or partially written file (`server.py` `get_ball_anchors_for_shot`); a network error also yields `null`.
  - `pages/anchor-editor/use-anchor-data.ts:126-135`: the same pattern, plus `clip_id` mismatch → empty.
- **Category:** Implementation Integrity / Error prevention.
- **Impact:** After a transient failure the editor looks like a fresh shot. Place one anchor, press Save (or ⌘S), and every previously saved chain, dismissal and pitch-anchor frame is replaced. This is the same data-loss class as baseline B-1.
- **Fix:**
  - Use `getJson` for the saved set. On error, render `PanelError` + Retry and disable Save.
  - Server side, accept an `If-Match`/`base_count` and refuse a write that shrinks the file unless `force` is set.
  - Add a load-failure → Save-disabled test.

**F-3 [P1] Every frame slider is unnamed for assistive tech**
- **Location:** `components/ui/slider.tsx:45-50`. Consumer `aria-label`s (e.g. `viewer/transport.tsx:152`, `hmr-world/transport-bar.tsx:38`) land on the Root `<span>`, not on the `role="slider"` Thumb. There's no `aria-valuetext`.
- **Category:** Accessibility (WCAG 4.1.2).
- **Impact:** axe `aria-input-field-name` (serious) on 11 of 12 routes. Screen readers announce "slider, 0" for the primary scrubber of every player, the sync offsets and the touch-confidence control.
- **Fix:** Add `thumbLabel` / `valueText` props to `Slider` and forward them to `SliderPrimitive.Thumb` (`aria-label`, `aria-valuetext="Frame 120 of 428"`).

**F-4 [P1] Light theme (and dimmed rows in both themes) fail AA contrast**
- **Location:**
  - `index.css:79-84`: light `--success` L 0.52, `--warning` L 0.58, `--info` L 0.55.
  - `components/status.tsx:73-78` (`text-X` on `bg-X/15`).
  - `pages/ball-anchor-editor/events-list.tsx:121` (`opacity-60` on suppressed rows).
  - `features/stages/prepare-shots/sync-timeline.tsx:168` (`text-white/40`, 10px).
- **Category:** Accessibility (WCAG 1.4.3) / Theming.
- **Impact:** The light theme has 20 contrast nodes on Prepare Shots, 17 on Ball, 10 on HMR World, 9 on Refined Poses and 7 on Camera. Measured: badges 3.57-4.2:1; warning confidence numbers 4.39:1 on white; suppressed event text 2.05:1 (light) and 3.43-3.92:1 (dark).
- **Fix:**
  - Darken the light tone tokens to about L 0.45 for text use (or add `--X-text` tokens).
  - Replace `opacity-60` with `text-muted-foreground` on text children and keep opacity for icons only.
  - Raise the tick labels to `text-stage-foreground/70` at 11px.

**F-5 [P1] Running jobs can't be cancelled (carried over from S-2)**
- **Location:** `server.py` has no `/api/jobs/{id}/cancel`; `components/log-dock.tsx` has only minimise and dismiss.
- **Category:** User control.
- **Impact:** A mistaken Run all / HMR / Render ties up the single-job lockout for up to an hour; the only exit is killing the server.
- **Fix:** Add a cooperative cancel flag checked between stages and players in `_run_job`, a `POST /api/jobs/{id}/cancel`, and a destructive "Stop" button in the dock header while `status === "running"`.

### P2

**F-6 [P2] Error-as-empty survives in 73 fetches (S-13 partial)**
- **Location:** `lib/api.ts:49-58`, used by `refined-poses/index.tsx:31-32,100-106`, `hmr-world/*`, `export/index.tsx:25-38`, `viewer/load-scene.ts` (9×), `multi-shot-status.tsx:104`.
- **Impact:** A 500 renders "No refined tracks yet — run hmr_world first", sending the operator to re-run a stage that succeeded.
- **Fix:** Make `getJsonOrNull` return `null` only on 404 and throw otherwise. Panels map throws to `PanelError` + Retry.

**F-7 [P2] Six hand-rolled frame transports**
- **Location:** `anchor-editor/transport-bar.tsx`, `ball-anchor-editor/transport.tsx`, `viewer/transport.tsx`, `hmr-world/transport-bar.tsx`, `camera/overlaid-pitch-map.tsx:103-156`, `tracking/track-video.tsx:112-190`.
- **Impact:** Features diverge: the frame-number input exists only in the anchor editor, and Home/End only in some. Keyboard scoping is also implemented four ways (window vs element listeners).
- **Fix:** Extract one `<FramePlayer>` (buttons + Slider with valuetext + frame Input + `Kbd` legend + a scoped key hook). This also fixes F-3 in one place.

**F-8 [P2] Touch targets below 24px on phones**
- **Location:** `tracking/player-row.tsx:71-75` (16px Checkbox), `features/stages/export/camera-picker.tsx` (16px POV/OTS checkboxes), `size="icon-xs"` split/delete buttons.
- **Impact:** axe at 390px: Tracking 45/125, Export 45/61, Render 58/68, HMR 24/38.
- **Fix:** Add hit-area padding (`after:absolute after:-inset-2`) or `pointer-coarse:size-9` on checkbox and icon-xs.

**F-9 [P2] Panel header actions crush the title at phone width**
- **Location:** `components/panel.tsx:35-41` (`CardAction` grid column); `shots/reaudit/mobile-export.png`.
- **Impact:** On the Export 3D viewer the description wraps one word per line and "Open full screen" is clipped.
- **Fix:** Make the header a flex-wrap, or put actions on their own row below the title under a container query (`@container (max-width: 32rem)`).

**F-10 [P2] Gating treats a *partial* dependency as missing**
- **Location:** `hooks/use-pipeline.tsx:297-303` (`missingDeps` checks `complete` only); `desktop-index-refined_poses.png`.
- **Impact:** Refined Poses reads "Complete" + "Needs HMR World first" with Re-run and Continue disabled, because gberch-2 has no HMR even though gberch does. The operator can't refresh a stage whose inputs exist.
- **Fix:** Treat `partial` deps as "allowed with warning": enable the buttons and confirm "HMR World is missing for gberch-2 — continue with 1/2 shots?".

**F-11 [P2] Viewer player list uses invalid ARIA**
- **Location:** `pages/viewer/overlays.tsx:63-76` (`div[role=list] > button[role=listitem][aria-pressed]`, a raw `<button>` against the README).
- **Impact:** axe `aria-allowed-attr` (critical) and `aria-allowed-role` ×22. Selection state is lost to AT.
- **Fix:** `<ul>` > `<li>` > `<Button variant="ghost" aria-pressed>`.

**F-12 [P2] The log dock isn't virtualised**
- **Location:** `components/log-dock.tsx:51-69` (3 regex tests per line per render), `:81` (a full scan for `hasError`).
- **Impact:** Chatty stages re-render up to 5,000 spans per rAF flush. `MAX_LOG_LINES` also silently drops the head of long logs, including early tracebacks.
- **Fix:** Classify lines once on ingest (store `{text, level}`), track `firstErrorIndex` incrementally, and virtualise with `@tanstack/react-virtual`. Note the truncation in the header.

**F-13 [P2] Player-label drift across panels (E-3 open)**
- **Location:** Export picker "Referee", "MacAllister P001" (`mobile-export.png`) vs HMR, Refined and Viewer "Ref P010", "P001" (`desktop-index-hmr_world.png`, `desktop-viewer.png`).
- **Fix:** One `usePlayerLabel(playerId)` backed by a single names source (tracks, or the export `display_name`).

**F-14 [P2] The job store sees only one job, and only once**
- **Location:** `hooks/use-pipeline.tsx:270-286` (reattach runs once and takes `jobs[0]`).
- **Impact:** A job started from another tab or the CLI after load, or a second concurrent per-shot job, isn't shown. That tab's lockout is then wrong.
- **Fix:** Poll `/api/jobs?status=running` (e.g. every 5 s while idle) and model `jobs[]`, not a single `runningLabel`.

**F-15 [P2] No undo for destructive edits (T-2, A-4 partial)**
- **Location:** `tracking/track-editor.tsx:133-211` ("This cannot be undone"); `anchor-editor/use-anchor-editor.ts:198-218`.
- **Impact:** Merges, deletes and anchor-frame deletes rely on a confirm alone. The Merge-by-name dialog under-reports its scope: it counts this shot, but rewrites every shot.
- **Fix:** Take a server snapshot of `tracks/*.json` before bulk operations and show an Undo toast. Give the editors an immutable-history undo stack (⌘Z), which the doc model already makes cheap. Use a server dry-run count across all shots.

### P3

| ID | Issue | Location | Fix |
|---|---|---|---|
| F-16 | Empty-state and toast copy leaks identifiers ("Run hmr_world first", "gberch__P001") | `refined-poses/index.tsx:104`, `hmr-world/index.tsx:78,93`, `camera/index.tsx:88`, `export/index.tsx:106` | Use `humanizeStageName` in the copy |
| F-17 | Refined Poses "Multi-view" column is all 0 on single-shot data | `refined-poses/index.tsx:55-56,69` | Hide all-zero columns, as `segments-table.tsx` already does |
| F-18 | Active sidebar item has `data-active` but no `aria-current="page"` | `components/ui/sidebar.tsx:511`, `app-sidebar.tsx:65-67` | Pass `aria-current={isActive ? "page" : undefined}` |
| F-19 | `document.title` is set only on the dashboard; editors and the viewer are all "Football Perspectives" | `pages/dashboard.tsx:72` | Set the title in each page (e.g. "Ball anchors · gberch") |
| F-20 | No skip link; 13 tab stops before content | `App.tsx:19-37` | A "Skip to content" link targeting `<main id>` |
| F-21 | Heading order skips (`h3` "Events…" with no `h2` on the ball editor; `h4` "Player cameras"); sidebar brand and "Theme" outside landmarks; unlabeled icon `<th>` | `ball-anchor-editor/events-list.tsx`, `export/camera-picker.tsx`, `app-sidebar.tsx:43-45,115`, `refined-poses/index.tsx:59` | Normalise the levels; wrap the header in a `div role="banner"`-like region or move it into `nav`; add `<span className="sr-only">Diagnostics</span>` |
| F-22 | Prepare Shots tabs emit `aria-controls` pointing at unmounted content (axe critical, cosmetic) | prepare-shots tabs | `forceMount` + `hidden`, or drop the Tabs for a ToggleGroup |
| F-23 | The Broadcast render card has a blank poster (RN-3 open) | `render/camera-grid` (`desktop-index-render.png`) | `#t=0.5` on the src, or a poster endpoint |
| F-24 | The documented viewer confidence timeline doesn't exist (V-5 open) | `pages/viewer/*` vs CLAUDE.md | Add a camera/HMR confidence lane to the transport, or fix the docs |
| F-25 | Team chip shows raw "unknown" on every player, with an inline hex border | `tracking/player-row.tsx:94-101` | Hide it when unknown, or read "No team"; use a token/data class |
| F-26 | Camera "Open full screen" drops the current shot | `camera/index.tsx:43` | Link to `/anchor_editor?shot=<current>` |
| F-27 | Hover-only hints remain (stale-camera reason, group/method badges) | `multi-shot-status.tsx:43,64`, `shot-tile.tsx:151,156`, `sync-offsets.tsx:60`, `panel.tsx:111` | A Tooltip on a focusable trigger, or inline text for "Stale — re-run camera" |

---

## 5. Patterns and systemic issues

1. **The error-masking fetch was ported rather than retired.** `getJsonOrNull` is used for 73/91 reads. It turns failures into "not run yet" copy in panels (F-6) and into empty documents in editors (F-2).
2. **The destructive-run policy is header-only.** The dry-run → confirm → clean-after-admission flow is excellent, but `/api/run-shot` callers (HMR selection, the anchor-editor rerun, Render shot) each decide for themselves. Only the anchor editor confirms (F-1).
3. **Shared primitives stopped short of the player.** Panel, Status, Dialogs and the Pipeline store are shared; the frame transport isn't (F-7). That is also why the slider naming defect repeats seven times (F-3).
4. **The light theme wasn't contrast-validated.** The dark default passes; the light tone tokens were chosen for hue, not for 12px text (F-4).

## 6. Positive findings (keep)

- **The destructive re-run flow** (`stage-actions.tsx:65-100` + `server.py` `clean_first`, `/api/output/{stage}/artifacts`) is the best part of the rebuild. It lists the exact paths, requires a typed confirm for operator-edited stages, and deletes only after the run is admitted. Tests are in `tests/test_web_api_run_clean_first.py`.
- **A job store with SSE reattach and a reconnect state machine** (`use-pipeline.tsx:144-221`). Log lines are batched per animation frame, and the log dock has Follow, Copy, Download and First error.
- **One ball editor** with an immutable document model and derived dirty state (`use-anchor-doc.ts`). It round-trips every `BallAnchorSet` field.
- **Unsaved-edit protection** that covers route changes, stage changes, shot switches and reload (`use-unsaved-guard.tsx`).
- **An honest tri-state status**: the Partial dot and badge now match what panels show.
- **Keyboard-first editors**: a searchable, grouped landmark palette with coordinates; `Kbd` legends; ⌘S; Shift±10; Home/End; tag number keys.
- **Accessible metric help** (HoverCard on a button with a legend) keeps the legacy `CAM_METRIC_HELP` content.
- **Performance hygiene**: lazy stage chunks, three.js split out, WebGL/controls/ResizeObserver disposal, and render-on-dirty in the viewer.
- **The design system**: tokens only for chrome, `bg-stage` media wells in both themes, sentence-case titles, lucide icons, `tabular-nums` numerics, and a detector score of 0.

## 7. Recommended actions

1. **[P1] `/impeccable harden`**: route every `/api/run-shot` caller through the dry-run/confirm/resume policy (F-1); switch editor loads to a strict fetch with Save disabled on error, plus server shrink-guard (F-2); add job cancel (F-5).
2. **[P1] `/impeccable audit` → fix**: forward the Slider thumb `aria-label`/`aria-valuetext` (F-3); fix the viewer list ARIA (F-11).
3. **[P1] `/impeccable colorize`**: recalibrate the light tone tokens and drop `opacity-60` on text (F-4).
4. **[P2] `/impeccable harden`**: `getJsonOrNull` → 404-only; `PanelError` + Retry everywhere (F-6); a multi-job store with polling (F-14); undo toasts and ⌘Z (F-15).
5. **[P2] `/impeccable distill`**: extract `<FramePlayer>` (F-7) and `usePlayerLabel` (F-13).
6. **[P2] `/impeccable adapt`**: coarse-pointer hit areas (F-8); wrap panel header actions (F-9).
7. **[P2] `/impeccable clarify`**: allow partial dependencies with a warning (F-10); humanise the leaked identifiers (F-16).
8. **[P2] `/impeccable optimize`**: a virtualised, pre-classified log (F-12).
9. **`/impeccable polish`**: the P3 table.
