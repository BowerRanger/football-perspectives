# Legacy Dashboard Audit + UX Critique (before shadcn/React rewrite)

**Target:** Football Perspectives pipeline dashboard (legacy vanilla JS), served by `src/web/server.py` (FastAPI) at `http://localhost:8765`.
**Files audited:** `src/web/static/index.html` (5341 lines), `src/web/static/js/prepare_shots_panel.js` (1581), `src/web/static/js/render_panel.js` (328), `src/web/static/anchor_editor.html` (1454), `src/web/static/ball_anchor_editor.html` (893), `src/web/static/viewer.html` (1196), plus `src/web/server.py` where the UI's behaviour depends on it.
**Evidence:** 16 screenshots (1440x900 desktop, 390 mobile @2x), impeccable detector output (29 findings), live API responses, and a code read of every finding below. All line numbers refer to the worktree `web-shadcn-revamp` at `700e3f9`.
**Surface type:** Operate mode. A single desktop operator (the developer) runs stages, watches logs, and edits operator ground truth (pitch anchors, ball anchors, track names, shot groups, sync offsets).

Measured totals across the six UI files:

| Metric | Count |
|---|---|
| Hex colour literals | **787** (91 unique colours) |
| Inline style assignments (`style="..."`, `.style.x =`, `cssText`) | **524** |
| CSS custom properties (`var(--`) | **0** |
| `@media` queries | **0** |
| `:focus` / `:focus-visible` rules | **0** |
| `aria-*` / `role=` / `tabindex` attributes | **0** |
| `prefers-reduced-motion` handling | **0** (pulse animations run unconditionally) |
| Tooltips implemented as `title` only | **78** |
| Native `alert` / `confirm` / `prompt` calls | **6** |
| Copy-pasted frame-transport implementations | **5** in `index.html` + 2 standalone editors |
| Dead code (defined, never called) | ~600 lines (`renderMatchInfoForm`, `buildCameraPitchMap`, `buildFramePreview`) |

---

## 1. Audit Health Score

| # | Dimension | Score | Key finding |
|---|-----------|-------|-------------|
| 1 | Accessibility | **1** | 0 ARIA attributes in any file. Sidebar navigation uses clickable `<div>`s (`index.html:437-440`). No landmarks or headings in the dashboard. 21 contrast failures, e.g. panel titles at 3.5:1. |
| 2 | Performance | **2** | The ball 3D preview leaks a WebGL context and a `requestAnimationFrame` loop on every visit (`index.html:4819`, `4916-4921`). The log uses `textContent +=` for every line, which is O(n²) (`639-641`). Status fetches run one after another (`1296`). Pages are otherwise light. |
| 3 | Responsive Design | **0** | There are no media queries anywhere. The 240px sidebar never collapses, so at 390px the content column is about 150px wide and gets clipped. The anchor editor's video column collapses to zero width. |
| 4 | Theming | **1** | There are no tokens: 787 hex literals (91 unique) and 524 inline styles. A few CSS classes (`.btn`, `.panel`, `.badge`) are the only shared layer, and there is no light theme. |
| 5 | Implementation Integrity | **1** | There are two ball-anchor editors, and they have already drifted apart in a way that loses data. Other problems: 5 copies of the transport code, about 600 dead lines, and UI fields that no longer match the backend schema (`undefined` rendered on screen). |
| **Total** | | **5/20** | **Critical (fundamental issues)**. The rating is driven by two paths that lose operator data and by the complete absence of a11y, responsive and theming infrastructure. The tool does work on a desktop for its single operator. |

### Implementation Integrity Verdict: **FAIL**

The dashboard reads as one product and has a consistent dark "slate + indigo" look. The implementation behind it is not a system:
- **The ball editor exists twice.** An inline copy of about 700 lines (`index.html:3970-4676`) says it "mirrors" `ball_anchor_editor.html`. It lacks shot chains, dismissals, span end frames and pitch fixes, and its Save overwrites those fields with nothing (P0, finding B-1).
- **The same code is hand-written over and over.** The same play/step/seek transport appears at `index.html:1450`, `2552`, `2732`, `2986` and `3501`, each with the same ▶ glyph for both "Play" and "Next frame".
- **The UI has drifted from the backend schema.** The refined-poses UI reads `total_fused_frames`, `single_view_frames` and `high_disagreement_frames`, which the backend no longer emits, so the page shows literal `undefined` (`index.html:1331-1333`, `5053-5055`).
- **The design system is ad hoc.** Every button colour is passed per call (`makeToolbarBtn(text, "#4f46e5")`, `index.html:2020-2025`), so "primary" appears as indigo, sky, teal and slate.
- Detector findings were all confirmed. Notes:
  - The `#fff on #6366f1` (4.47:1) findings are real but borderline.
  - The side-tab border (`prepare_shots_panel.js:667`) is a true positive of low impact.
  - "Monotonous spacing" in the ball editor is a judgement call, not a defect.
  - The detector missed the two P0s, the missing focus and keyboard support, and the lack of any responsive layout. Those came from reading the code and screenshots.

---

## 2. Nielsen Heuristics

| # | Heuristic | Score | Justification |
|---|-----------|-------|---------------|
| 1 | Visibility of system status | **2** | Good: SSE log streaming, pulsing sidebar dots, Running/Error badges. Bad: status is binary and often wrong (HMR World, Ball, Export and Render all say "Not run" next to visible output); there are no loading skeletons; the in-panel "Job running" line gets wiped by a re-render; runs started from the iframe editor are invisible. |
| 2 | Match between system and real world | **2** | Domain vocabulary suits the developer-operator. But raw identifiers leak ("Run refined_poses", `hmr_world`), and so does `undefined`. The unlabeled "2193.1s" chip reads like a clip duration. |
| 3 | User control and freedom | **1** | Nothing can be undone: track merge, Ignore Unknown, anchor delete. Running jobs can't be cancelled (the server has no endpoint for it). Re-run wipes data with one click. The iframe "← Dashboard" and "✕ Close" controls load the dashboard inside itself. |
| 4 | Consistency and standards | **1** | Four colours are used for "primary". "Rerun" and "Re-run" both appear. The two ball editors have different features. The same player is "P001" in one panel and "MacAllister" in another, "Ref" in one and "Referee" in another. Three run paths are gated differently. Only one editor has keyboard shortcuts. |
| 5 | Error prevention | **1** | Two P0 data-loss paths. The most destructive button has no confirmation while less destructive ones (Re-run split, track delete) do. Disabled buttons look enabled. Dependency gating and the run lockout are real strengths. |
| 6 | Recognition rather than recall | **2** | Labels are mostly visible. However, 78 hints live only in `title` attributes, the ~60-row landmark palette has no search, the shot choice resets on every visit, and glyph icons are ambiguous (▶ vs ▶). |
| 7 | Flexibility and efficiency of use | **2** | Good: anchor-editor keys (←/→/Space/Esc), bulk track operations, Run All / Continue, the last stage is remembered. Missing: keys in the other four players, a command palette, deep links, and a remembered shot per stage. |
| 8 | Aesthetic and minimalist design | **2** | The palette is calm and cohesive. Against that: uppercase tracked titles on every panel, dead table columns (spin, residual all "—" or 0.00), long tables above the actual editors, and duplicated headers inside iframes. |
| 9 | Error recognition, diagnosis, recovery | **1** | `fetchJsonOrNull` turns every HTTP or network error into an empty state that looks normal ("No tracks yet…"). "Failed to start run" hides the server's `detail`. `alert()` shows raw errors. A few in-panel dispatch errors do show `detail`. |
| 10 | Help and documentation | **2** | Good empty-state copy that names the next step, and good camera-metric help text. The help is hover-only (`title`), with no link to the docs or specs. |
| **Total** | | **16/40** | **Poor**: the core experience works for an expert but is fragile. |

### Persona red flags

- **Alex (impatient power user, primary):**
  - Can't Tab or arrow through stages; the sidebar is `<div onclick>`.
  - There are no deep links (`/stages/ball?shot=gberch`), and the shot picker resets on every stage change.
  - Frame-step keys exist only in the anchor editor.
  - Launching "Run for selection" on All players kicks off a 35–60 min GVHMR job with no estimate and no way to cancel it.
- **Sam (accessibility-dependent):**
  - The sidebar, landmark palette, tag list, anchor list, viewer player list and tile menus can't be reached by keyboard.
  - Nothing is announced by a screen reader: the log status and toasts have no `aria-live`.
  - Status is conveyed by colour alone: red/green dots, confidence colours, ✓/✗ glyphs.
  - Text contrast is 3.1–3.5:1 on every panel title and table header.
- **Project persona, "Joe mid-run" (operator-developer during a 45-minute hmr_world job):**
  - Reloading the tab loses the log and the run lockout.
  - The anchor editor's "Rerun camera tracking" runs outside the dashboard's lockout.
  - Clicking header "Re-run Stage" on Tracking to "just refresh" deletes every player name and merge made so far.

---

## 3. Executive Summary

- **Audit Health Score: 5/20 (Critical)**. **Nielsen: 16/40 (Poor)**.
- **Issues found: 69.** P0: **2**, P1: **18**, P2: **36**, P3: **13**.

**Top 5 issues**
1. **[P0] Re-run Stage deletes the stage's output with no confirmation, before the run is even accepted.** `DELETE /api/output/{stage}` removes `shots/` (manifest, groups, manual sync offsets) or `tracks/` (operator names, merges, splits). Then `/api/run` may still reject with 409/429 (`index.html:676-682`, `server.py:295-307, 842-867`).
2. **[P0] Saving from the Ball panel's inline editor erases operator ball data.** The inline editor posts only `anchors`, so the server resets `shot_chains` and `dismissed_auto` to `[]`. The loader also drops `end_frame`, `landmark` and `confidence` (`index.html:4576-4578, 4612-4621`, `server.py:2080-2085`).
3. **[P1] Run state is fragile and split three ways.** It lives only in page memory, so a reload loses it. SSE has no `onerror`. The header, in-panel and iframe run buttons follow different gating.
4. **[P1] Status is wrong.** Stage completion is binary, so partial output shows as "Not run", and schema drift renders `undefined frames, undefined flagged`.
5. **[P1] There is no a11y, responsive or theming foundation.** 0 ARIA, 0 media queries, 0 tokens, clickable-`div` navigation, 21 contrast failures.

**Recommended next steps:** fix the two P0s first. They are behavioural contracts the rewrite must honour: confirm-and-scope destructive runs, and round-trip the full `BallAnchorSet`. Then build the shell around one job store (`/api/jobs` + SSE reattach). Then establish shadcn tokens and components (Sidebar, AlertDialog, Select, Tooltip, Skeleton, Toggle, DropdownMenu) and a single `<FramePlayer>`.

---

## 4. Detailed Findings by Page

Format: **[P?] Name**, then Location, Category, Impact, and **Rewrite:** the concrete recommendation for the shadcn/React build.

### 4.1 Shell (sidebar, header, log panel), `index.html`

**S-1 [P0] Re-run Stage deletes output without confirmation, before the job is admitted**
- **Location:** `index.html:676-682` (`rerun-btn` handler); `server.py:842-867` (`DELETE /api/output/{stage}`); `server.py:295-307` (`_STAGE_ARTIFACTS`)
- **Category:** Error prevention / Implementation Integrity
- **Impact:**
  - One click on the most visually prominent header button wipes the stage directory:
    - `prepare_shots` → `shots/`, including `shots_manifest.json`, groups and `sync_map.json` (manual offsets)
    - `tracking` → `tracks/`, including every operator-entered name, merge and split
    - `refined_poses`, `hmr_world` (a 35-60 min recompute), `export`
  - This contradicts CLAUDE.md's "operator input always wins".
  - The DELETE response is never checked, and it runs before `/api/run`, which can return 409 (hmr_world in flight) or 429 (job cap). The output can be gone with nothing running.
  - The in-panel "⟳ Re-run split" is *less* destructive, yet it does confirm (`prepare_shots_panel.js:350`).
- **Rewrite:**
  - Make **Continue** (non-destructive) the default action.
  - Demote "Re-run clean" into a SplitButton/DropdownMenu item that opens an **AlertDialog**. The dialog lists exactly what will be deleted (add a server dry-run, e.g. `DELETE ...?dry_run=1`) and calls out operator-edited artefacts in destructive styling.
  - Require a typed confirmation for `prepare_shots` and `tracking`.
  - Move the delete server-side into `/api/run` (`clean: true`), so it happens only after the job is admitted.

**S-2 [P1] Run state lives only in page memory, so a reload loses the lockout and the log**
- **Location:** `index.html:326` (`runningStage`), `557-580`; `server.py` has only `/api/jobs/{id}/status|logs` (817, 830) and no job list
- **Category:** Visibility of system status / Error prevention
- **Impact:**
  - Reloading or leaving and returning during a long hmr_world or render job clears the Running badge, the pulsing dot and the log.
  - Every run button re-enables, so the operator can double-dispatch on top of a live job. The server guards only hmr_world against this (409).
- **Rewrite:**
  - Add `GET /api/jobs?status=running`.
  - Keep a global job store (TanStack Query polling or zustand) that reattaches to the SSE stream on load. The server already replays `log_lines` (`server.py:511-515`), so reattaching gives the full log.
  - Add a Cancel action backed by a new endpoint.

**S-3 [P1] The SSE stream has no error handling**
- **Location:** `index.html:631-665` (`_streamJobLogs`: listeners only for `log` and `done`)
- **Category:** Error recovery
- **Impact:** If the server restarts or the connection drops, `done` never arrives. The stage stays "Running" and every run trigger stays locked until a manual reload.
- **Rewrite:** Handle `onerror` by polling `/api/jobs/{id}/status`, and show an inline "Connection lost — reconnecting…" `Alert` in the log dock. Use exponential-backoff reconnect.

**S-4 [P1] Binary stage status misreports partial output**
- **Location:** `server.py:596-605` (`complete: bool`); `index.html:498-510` (badge); screenshots `desktop-index-hmr_world/ball/export/render.png`
- **Category:** Visibility of system status
- **Impact:** Each of these contradicts what the panel below shows:
  - HMR World says "NOT RUN" above a table of 22 processed players.
  - Ball says "NOT RUN" above a ball-track summary.
  - Export says "NOT RUN" while its own table shows `gberch` exported ✓.
  - Render says "NOT RUN" above 10 rendered cameras.
  - The cause is that completion requires *all* shots. The operator learns to ignore the badge.
- **Rewrite:** Use a tri-state `StatusBadge` (complete / partial / none) with a per-shot count ("1/2 shots"). Add a matching sidebar indicator, e.g. a half-filled dot, and a tooltip listing the missing shots.

**S-5 [P1] Header and in-panel run buttons disagree, and the in-panel status is wiped**
- **Location:** `index.html:511-535` (header gating); in-panel buttons at `3136-3200` (HMR "Run for selection"), `4979-5020` ("Run refined_poses"), `render_panel.js:34-107` ("Render"); `attachToJob` → `selectStage` → `output-panel.innerHTML = ""` (`index.html:627, 533-534`)
- **Category:** Consistency / Visibility
- **Impact:**
  - On Refined Poses, header Continue and Re-run are disabled ("Requires: hmr_world") while the in-panel "Run refined_poses" is enabled for the same stage (screenshot).
  - In-panel buttons ignore the any-stage-running lockout.
  - After dispatch, the "Job X running — log panel above" line is erased because the panel re-renders.
- **Rewrite:**
  - Build one `<StageRunControls>` per stage, driven by the job store: Continue / Re-run clean / scoped run (shot, player).
  - Put it in the page header, not duplicated in the body.
  - A disabled control shows its reason as visible helper text, not only in a tooltip.

**S-6 [P1] Stage navigation can't be reached by keyboard and has no routes**
- **Location:** `index.html:437-440` (`div.stage-item` with `onclick`), CSS `41-54`
- **Category:** Accessibility (WCAG 2.1.1, 4.1.2)
- **Impact:** The stages can't be reached with Tab. There's no `aria-current`, no URL per stage, and back/forward don't work.
- **Rewrite:** Use the shadcn `Sidebar` + `SidebarMenuButton` (`asChild` link) with a route per stage (`/stages/:stage`) and `aria-current="page"`. Keep the index number and status dot as decoration.

**S-7 [P1] No landmarks, no heading hierarchy, zero ARIA**
- **Location:** `index.html:274-310` (all `div`), `286` (stage title is a `<span>`), `makePanel` `723-731` (panel titles are `div`s); `aria-*`/`role` count is 0 in all six files
- **Category:** Accessibility (WCAG 1.3.1, 2.4.6)
- **Impact:** Screen-reader users get one flat document with no navigation by region or heading.
- **Rewrite:**
  - `<nav>` for the sidebar and `<main>` for content, with the stage title as `<h1>`.
  - Card titles as `<h2>`.
  - The log dock as a `<section aria-label="Run log">` with an `aria-live="polite"` status.

**S-8 [P1] Contrast failures on titles, headers, badges and primary buttons**
- **Location:**
  - `#64748b` on `#1a1d27` = 3.5:1: `.panel-title` `155-163`, `th` `188-197`, `#sidebar-header` `32-38`, `.stage-index` `72`
  - `.badge-pending` `#64748b` on `#1e293b` = 3.1:1 (`124`)
  - `.btn-primary` `#fff` on `#6366f1` = 4.47:1 (`176`)
  - The detector reports 21 low-contrast findings across files
- **Category:** Accessibility (WCAG 1.4.3)
- **Impact:** All structural labels are hard to read on a dim screen or in daylight.
- **Rewrite:** Set the `--muted-foreground` token to at least 4.5:1 on `--card`, and give the primary colour at least 4.5:1 against its foreground. Validate both themes.

**S-9 [P1] The layout is desktop-only; nothing adapts**
- **Location:** `index.html:10-27` (`body{display:flex;overflow:hidden}`, `#sidebar{min-width:240px}`); 0 `@media` rules in any file; screenshots `mobile-index-*.png`
- **Category:** Responsive
- **Impact:**
  - At 390px the sidebar takes 60% of the width.
  - The title wraps, the badge clips to "COMPL", and the header buttons and output select are off-screen.
  - Cards clip text mid-word ("Re-ingest the sa").
  - The operator mostly uses a desktop, but a half-width split window (~720px) already squeezes the fixed 300px player list and the 220/240px editor rails.
- **Rewrite:**
  - Wrap the shell in `SidebarProvider`, collapsing to an off-canvas `Sheet` below `md` and an icon rail at `md`.
  - Collapse header actions into a `DropdownMenu` below `lg`.
  - Use container queries for cards.

**S-10 [P2] The output-directory switcher uses `prompt`/`alert` and an unstyled, unlabeled select**
- **Location:** `index.html:291` (native select, `title` only; renders white and unstyled in screenshots), `397-399` (`window.prompt`), `413`, `418` (`window.alert`)
- **Category:** Consistency / Error recovery / Accessibility
- **Impact:** Native dialogs block the page and are unstyled. Server errors are shown raw. The select has no label, and a destructive-scope choice (which reconstruction you are writing to) looks like debug UI.
- **Rewrite:**
  - A labelled `Select` ("Workspace") with a "New output…" item that opens a `Dialog` with a validated `Input` (pattern hint `output-<name>`) and inline `FormMessage` errors.
  - A `sonner` toast on success.
  - Keep it disabled while a job runs (the legacy UI already does this, `index.html:536-545`).

**S-11 [P2] The log dock is quadratic and bare**
- **Location:** `index.html:294-303` (dock, `✕` glyph close), `639-641` (`logOutput.textContent += line + "\n"`)
- **Category:** Performance / Efficiency
- **Impact:** Long GVHMR or render logs re-serialise the whole string on each line. There's no follow/pause, no copy, no filter, and no error-line highlighting. On error the user is told to "scroll log above for details".
- **Rewrite:** A `LogViewer`: a line array plus a virtualised list (`@tanstack/react-virtual`), a "Follow" `Toggle`, Copy/Download buttons, and ERROR/Traceback lines highlighted with a "Jump to first error" button. Make it a resizable bottom dock (`ResizablePanel`) rather than a sticky 45vh overlay.

**S-12 [P2] Switching stages quickly mixes panels from two stages**
- **Location:** `index.html:700-735` (`loadOutput` clears, then awaits); e.g. `renderTracking` `1340-1375` appends after `await`
- **Category:** Implementation Integrity
- **Impact:** Clicking stage A then B quickly lets A's late render append into B's view.
- **Rewrite:** Route-based pages plus TanStack Query with `AbortSignal`, so each stage renders only its own data.

**S-13 [P2] Errors are shown as empty states**
- **Location:** `index.html:811-817` (`fetchJsonOrNull` returns `null` on any non-OK response or exception); used by every panel
- **Category:** Error recovery (H9)
- **Impact:** A 500 from `/tracking/shots` renders "No tracks yet — run tracking…", which sends the operator off to re-run a stage that already succeeded.
- **Rewrite:** Query hooks with explicit `isError` handling: a destructive `Alert` with the status/detail and a Retry button, kept separate from the `EmptyState` component.

**S-14 [P2] No loading states**
- **Location:** every `render*` function (e.g. Export does its per-shot fetches before painting, `5142-5167`; Multi-shot status fetches serially, `1296-1297`)
- **Category:** Visibility
- **Impact:** The panel is blank while it loads, which looks like "no data".
- **Rewrite:** `Skeleton` placeholders shaped like the eventual cards and tables. Fetch in parallel.

**S-15 [P2] Explanations for disabled buttons live only in `title`**
- **Location:** `index.html:289`, `523-535` (`rerunBtn.title = blockedReason`)
- **Category:** Accessibility / Recognition
- **Impact:** Disabled buttons don't receive focus, and many browsers suppress their tooltips, so "Requires: hmr_world" is effectively invisible.
- **Rewrite:** Visible helper text under the header ("Blocked: HMR World not complete"), plus a `Tooltip` wrapped around a focusable `span` for the long form.

**S-16 [P2] Native, inconsistently styled `<select>`s everywhere**
- **Location:** `index.html:291` (header, unstyled white); per-panel inline-styled selects at `1351`, `3122`, `3133`, `3761`, `5298`; `render_panel.js:48`; `viewer.html:62-75`; `ball_anchor_editor.html:93`
- **Category:** Theming / Consistency
- **Impact:** Some selects are white and native, some are dark and inline-styled. The shot selects carry no label.
- **Rewrite:** shadcn `Select` with a `Label`. Build a shared `<ShotSelect>` bound to a `?shot=` search param.

**S-17 [P3] Uppercase letter-spaced titles everywhere**
- **Location:** `.panel-title` `index.html:155-163`, `th` `188-197`, sidebar header `32-38`, plus inline copies (`1464`, `prepare_shots_panel.js`)
- **Category:** Aesthetic
- **Impact:** Constant small caps flatten the hierarchy and slow scanning. Combined with 3.5:1 contrast they are the least readable text on the page.
- **Rewrite:** Sentence-case `CardTitle` (text-sm, font-medium, foreground). Keep uppercase at most for one tiny eyebrow style.

**S-18 [P3] Unicode glyphs used as icons**
- **Location:** `✕` `index.html:300`; `▶ ◀ ▶` `1450-1452` (Play and Next are the *same* glyph, and Play has no title); `✂ ×` `1699, 1731`; `✓ ✗ ⚠` `1283-1291`; `↗` `2853`; `⟳ ⋮` `prepare_shots_panel.js:342, 563`; `⇥ ⇤ ↩ ✕` `ball_anchor_editor.html:313-369`
- **Category:** Consistency / Accessibility
- **Impact:** Glyphs render differently across fonts, carry no accessible names, and Play is indistinguishable from step-forward.
- **Rewrite:** `lucide-react` icons (Play/Pause, ChevronLeft/Right, Scissors, Trash2, ExternalLink, RefreshCw, MoreVertical, Check, X, AlertTriangle), each icon-only button with an `aria-label` and a `Tooltip`.

### 4.2 Prepare Shots, `js/prepare_shots_panel.js` + `index.html:1275-1338`

**PS-1 [P1] Multi-shot status prints "undefined frames, undefined flagged"**
- **Location:** `index.html:1328-1334`
- **Category:** Implementation Integrity (schema drift)
- **Impact:** The same root cause as RP-1: `quality_report.refined_poses` now has `total_frames` / `cleanup` / `foot_lock` / `physpt_takeover`, not `total_fused_frames` / `high_disagreement_frames`. The first screen the operator sees shows garbage.
- **Rewrite:** Type every API response with zod and render only fields that are defined. Show `total_frames` and the foot-lock/cleanup counters. *(PS-1 and RP-1 are counted as one issue in the totals.)*

**PS-2 [P2] The tile overflow menu (⋮) works only with a mouse**
- **Location:** `prepare_shots_panel.js:581-605` (`div` rows, close on `mousedown` only, no Esc, no arrow keys, no focus management)
- **Category:** Accessibility
- **Rewrite:** shadcn `DropdownMenu`.

**PS-3 [P2] Regrouping depends on drag-and-drop**
- **Location:** `prepare_shots_panel.js:491-500` (dragstart), `650` (new-group zone), `680` (card drop); a partial alternative is the ◀/▶ "Move to adjacent group" buttons at `782-800`
- **Category:** Accessibility / Efficiency
- **Impact:** Moving a shot to a non-adjacent group, or into a new group, needs a precise mouse drag.
- **Rewrite:** `@dnd-kit` with its keyboard sensor, plus a "Move to…" submenu in the tile `DropdownMenu` listing every group and "New group".

**PS-4 [P2] The lightbox isn't a dialog**
- **Location:** `prepare_shots_panel.js:228-260`
- **Category:** Accessibility
- **Impact:** No `role="dialog"`, no focus trap, focus isn't returned to the tile, and the Esc handler is attached to `window` globally.
- **Rewrite:** `Dialog` with a `DialogTitle` (shot id + group) and the video player.

**PS-5 [P2] The Re-run split confirmation uses native `confirm()`**
- **Location:** `prepare_shots_panel.js:350`
- **Category:** Consistency
- **Impact:** This correct safeguard looks like a browser error. It's also the *only* confirmation in the shell's run family (see S-1).
- **Rewrite:** A destructive `AlertDialog` listing what is replaced (shots, groups, discards, sync offsets) and what survives (match info).

**PS-6 [P2] Multi-shot status: glyph-only cells, fetched one shot at a time**
- **Location:** `index.html:1280-1297` (`✓/✗/⚠` with `title` only; `await` inside a `for` loop per shot)
- **Category:** Accessibility / Performance
- **Impact:** Screen readers hear "check mark". The "stale anchors" warning (a genuinely valuable signal) is hover-only. N shots means N sequential round-trips.
- **Rewrite:** `Badge` cells with text ("Stale", "Missing", "22 players"), a single batched `/api/output/shot-status` call, and a `Table` with a sticky header.

**PS-7 [P3] Side-tab accent border on group cards**
- **Location:** `prepare_shots_panel.js:667` (`border-left:4px solid ${accent}`); detector `side-tab`
- **Category:** Aesthetic
- **Rewrite:** Show group identity as a small colour dot plus a label in the card header, or a 2px top rule shared with the tile ribbon.

**PS-8 [P3] About 300 lines of dead Match Info form**
- **Location:** `index.html:974-1274` (`renderMatchInfoForm`, never called; see the note at `prepare_shots_panel.js:41-42`)
- **Category:** Implementation Integrity
- **Rewrite:** Don't port it. If match info is wanted, rebuild it as a `Sheet` form off the header (`react-hook-form` + zod).

### 4.3 Tracking, `index.html:1340-2070`

**T-1 [P1] Disabled toolbar buttons look enabled**
- **Location:** `makeToolbarBtn` `index.html:2020-2025` (inline `background`, no disabled style); `mergeBtn.disabled = true` `1406`, `deleteBtn` `1413`, `interpBtn` `1416`; screenshot shows vivid indigo "Merge Selected" and red "Delete Selected" while disabled
- **Category:** Error prevention / Consistency
- **Impact:** The operator clicks and nothing happens. The affordance is false.
- **Rewrite:** shadcn `Button` variants (`default`, `secondary`, `destructive`) with built-in disabled styling. Show the selection count in the label, as the legacy UI does ("Merge Selected (3)").

**T-2 [P1] Bulk track changes have no confirmation and no undo**
- **Location:** "Ignore Unknown" `index.html:1409-1410` (renames every unnamed track in the shot to `ignore`); "Merge by Name" `1407-1408` (rewrites `player_id` across **every shot**); merges have no undo
- **Category:** User control / Error prevention
- **Impact:** One click rewrites operator-curated identity data across shots, and the only way back is manual re-editing.
- **Rewrite:** An `AlertDialog` that states the scope ("Merge 4 names across 2 shots; 7 tracks affected"), backed by a server dry-run. Then a `sonner` toast with **Undo** (server-side snapshot of `tracks/*.json` before the mutation).

**T-3 [P2] Track deletes use native `confirm()`**
- **Location:** `index.html:1739`, `1842`
- **Category:** Consistency
- **Rewrite:** A destructive `AlertDialog`, or better, delete immediately and offer an Undo toast.

**T-4 [P2] Player-row controls are unlabeled and use colour alone**
- **Location:**
  - `index.html:1580-1593` (row checkbox, no label)
  - `1596-1598` (red/green "named" dot, colour only)
  - `1633-1637` (name input, placeholder "Name…" only)
  - `1697-1731` (`✂` and `×` buttons, `title` only)
- **Category:** Accessibility (WCAG 1.3.1, 1.4.1, 4.1.2)
- **Rewrite:**
  - `Checkbox` with an sr-only label ("Select P009")
  - `Input` with `aria-label="Name for P009"`
  - the status dot paired with sr-only text ("unnamed")
  - icon buttons with `aria-label` and a `Tooltip`
  - render the list as a `Table`, or as a `ScrollArea` of rows

**T-5 [P2] No keyboard frame stepping outside the anchor editor**
- **Location:** `index.html` has no `keydown` handler (tracking, camera map, kp2d, HMR trajectory, inline ball editor); `ball_anchor_editor.html` has none either; the anchor editor has one (`anchor_editor.html:1401-1410`)
- **Category:** Flexibility / Consistency
- **Rewrite:** A shared `<FramePlayer>` with ←/→ (±1), Shift+←/→ (±10), Space (play/pause), Home/End, shortcuts shown in tooltips, and input fields ignored (as the anchor editor already does).

**T-6 [P2] The transport is copy-pasted five times, with identical Play and Next glyphs**
- **Location:** `index.html:1450-1452`, `2552-2554`, `2732-2734`, `2986-2988`, `3501-3503`
- **Category:** Implementation Integrity
- **Rewrite:** One `<FramePlayer>` component (see T-5): a `Slider` with `aria-valuetext="Frame 120 of 428"` and a frame `Input`.

**T-7 [P3] The shot choice is forgotten**
- **Location:** `index.html:1352-1374` (always loads `data.shots[0]`)
- **Category:** Efficiency
- **Rewrite:** A `?shot=` search param shared across stage routes.

**T-8 [P3] Fixed-width player list**
- **Location:** `index.html:1463` (`width:300px;min-width:300px;max-height:560px`)
- **Category:** Responsive
- **Rewrite:** A `ResizablePanelGroup` (video | players). The list stacks under the video below `lg`.

### 4.4 Camera Tracking, `index.html:2146-2878` + embedded `anchor_editor.html`

**C-1 [P1] The anchor editor is embedded as a fixed 780px iframe that nests the dashboard inside itself**
- **Location:** `index.html:2843-2865` (`height:780px`); `anchor_editor.html:232` (`<a href="/">← Dashboard</a>`, no `target="_top"`, no embed detection)
- **Category:** Implementation Integrity / Responsive / Control
- **Impact:**
  - The editor has its own title bar, shot select and "← Dashboard" link inside the dashboard (camera screenshot).
  - Clicking that link loads the entire dashboard *inside* the iframe.
  - The 780px height cuts the palette and anchor list off below the fold of the page scroll.
  - Keyboard shortcuts only work once the iframe has focus.
- **Rewrite:** Build the anchor editor as a React route/component (`/stages/camera/anchors?shot=`) inside the shell, as a full-height `ResizablePanelGroup` (palette | video | anchors). Remove the duplicate header. Replace "Open in new tab ↗" with a "Focus mode" toggle that hides the sidebar.

**C-2 [P1] "Rerun camera tracking" in the iframe is invisible to the dashboard's run state**
- **Location:** `anchor_editor.html:1281-1337` (polls `/api/jobs/{id}/status`, no log view); the dashboard's `runningStage` is never set
- **Category:** Error prevention / Visibility
- **Impact:**
  - While it runs, the header "Re-run Stage" (which DELETEs `camera/*_camera_track.json`) stays enabled.
  - The sidebar dot isn't pulsing.
  - On error the message says "see logs", but the logs aren't shown anywhere.
- **Rewrite:** All dispatches go through the shared job store. A camera run started from the editor shows in the log dock and applies the global lockout.

**C-3 [P2] Metric help is hover-only**
- **Location:** `index.html:2240-2265` (`CAM_METRIC_HELP` delivered via `title` on dotted-underlined labels)
- **Category:** Help / Accessibility
- **Impact:** The content is excellent ("A high value means a glitchy or shaky track…"), but keyboard users can't reach it.
- **Rewrite:** `HoverCard` or `Tooltip` on a focusable trigger. Show thresholds as a small legend (green < 0.5°, amber < 1.5°).

### 4.5 HMR World, `index.html:3107-3290`

*(Its "NOT RUN" badge over 22 players is covered by S-4.)*

**H-1 [P2] "Run for selection" starts a 35–60 min job with no estimate, confirmation or cancel**
- **Location:** `index.html:3136-3200`; the server has no cancel endpoint
- **Category:** Error prevention / User control
- **Impact:** With "All players" selected (the default), one click commits most of an hour of CPU/MPS time and can't be stopped.
- **Rewrite:** When the scope is All players, show a confirmation `AlertDialog` with an estimate from `render_timings`/history. Add a Cancel button in the log dock (new `POST /api/jobs/{id}/cancel`).

**H-2 [P2] Confidence is shown by colour alone, with inconsistent thresholds**
- **Location:** HMR table (colour classes via `cls`); refined table `index.html:5087-5088` (`>0.7` green, `>0.4` amber). Screenshots show 0.690 amber next to 0.707 green with no legend.
- **Category:** Accessibility (WCAG 1.4.1) / Consistency
- **Rewrite:** One `<ConfidenceCell>`: the value, a mini bar, and a text tier on hover. Define the thresholds once and add a legend in the card header.

**H-3 [P3] The shot/player select labels aren't `<label>`s**
- **Location:** `index.html:3118-3133` (`span` "Shot:", "Player:")
- **Category:** Accessibility
- **Rewrite:** `Label htmlFor` + `Select`. The player picker becomes a `Combobox` (22+ players).

### 4.6 Refined Poses, `index.html:4972-5130`

**RP-1 [P1] The summary shows "Fused frames: undefined (undefined single-view, undefined flagged)"**
- **Location:** `index.html:5049-5056`. `/refined_poses/summary` now returns `total_frames`, `cleanup.*`, `jitter.*`, `foot_lock.*`, `physpt_takeover.*`, none of which are surfaced.
- **Category:** Implementation Integrity / Visibility
- **Impact:** The one pipeline-wide summary is broken. The quality levers the project actually tunes (foot lock `spans_locked`/`mean_pin_err_m_after`, PhysPT takeover, cleanup clamps) are hidden.
- **Rewrite:** A zod-typed summary shown as stat tiles: players, total frames, foot-lock spans locked/skipped plus pin error before→after, PhysPT spans taken over, cleanup clamps. Fields that are missing render as "—", never `undefined`.

**RP-2 [P3] The button is labelled with a code identifier**
- **Location:** `index.html:4982` ("Run refined_poses")
- **Category:** Match real world
- **Rewrite:** "Run Refined Poses", or fold it into `StageRunControls` (S-5).

**RP-3 [P3] Operator-entered names are interpolated into `innerHTML`**
- **Location:** `index.html:5085-5090` (`player_name` in the `{html: ...}` cell)
- **Category:** Implementation Integrity (self-XSS)
- **Rewrite:** JSX escapes by default. Never use `dangerouslySetInnerHTML` for data.

### 4.7 Ball, `index.html:3751-4970`

**B-1 [P0] Saving from the inline ball editor erases operator ball data**
- **Location:**
  - `index.html:4576-4578`: payload `{clip_id, image_size, anchors}` only
  - `4612-4621`: the loader keeps only `frame, image_xy, state, player_id, bone, goal_element, touch_type, spin`, dropping `end_frame`, `landmark` and `confidence`
  - `server.py:2080-2085`: `shot_chains = []` and `dismissed_auto = []` by default; `2311-2351`: the file is overwritten
- **Category:** Implementation Integrity / Error prevention
- **Impact:** Pressing Save in the dashboard's Ball panel silently deletes:
  - every shot chain
  - every dismissed auto-suggestion (they reappear on the next run)
  - every span event's end frame
  - every pitch-fix landmark reference authored in the standalone ball editor
- This is operator ground truth, and CLAUDE.md is explicit that it must never be overwritten.
- **Rewrite:**
  - One `BallAnchorEditor` component used both in the Ball page and at the standalone route.
  - A zod schema mirroring `BallAnchorPayload` that round-trips *every* field.
  - A server-side `PATCH` or merge so partial clients can't clobber fields.
  - A regression test: load → save with no edits → the file is byte-identical.

**B-2 [P1] Two diverged ball-editor implementations**
- **Location:** `index.html:3970-4676` (~700 lines, "Mirrors src/web/static/ball_anchor_editor.html"); feature counts inline vs standalone: `shot_chains` 0 vs 3, `dismissed_auto` 0 vs 3, pitch fix 0 vs 7, `end_frame` 0 vs 5
- **Category:** Implementation Integrity
- **Impact:** It's unclear which editor is authoritative, B-1 is the proof that they drift, and every fix must be made twice.
- **Rewrite:** A single component. The Ball page embeds it in-shell, with no iframe and no copy.

**B-3 [P2] The WebGL preview leaks on every visit**
- **Location:** `index.html:4819` (`new THREE.WebGLRenderer`), `4916-4921` (unbounded `requestAnimationFrame(animate)`), `4948-4955` (`ResizeObserver` never disconnected); panels are torn down with `innerHTML = ""`
- **Category:** Performance
- **Impact:**
  - Each Ball visit or shot switch adds a render loop that runs forever, plus another GL context.
  - After about 16 contexts the browser drops the oldest, which can be the viewer's.
  - Fans spin while the operator sits on another stage.
- **Rewrite:** A `useEffect` cleanup that calls `cancelAnimationFrame`, `renderer.dispose()`, `controls.dispose()` and `ro.disconnect()`. Render on demand (on frame or orbit change), or use `@react-three/fiber` with `frameloop="demand"`.

**B-4 [P2] Tables of noise come before the editor**
- **Location:**
  - `index.html:3776` (the shot picker is hidden when there's only one shot)
  - `3830-3865`: the flight-segments table
    - its "#" column is the segment id, which equals the start frame
    - Fit residual is 0.00 and spin is "—" on every row under `trajectory: hybrid`
    - on gberch it runs to 23 rows, placed above the editor and previews
- **Category:** Aesthetic / Minimalism
- **Rewrite:** Summary stat tiles, then the editor (the primary task), then the segments in a `Collapsible` `Table` that hides all-empty columns. Always show the shot picker.

### 4.8 Export, `index.html:5133-5335`

**E-1 [P1] The viewer iframe is fixed at 780px, sits below a 22-row table, and "✕ Close" nests the dashboard**
- **Location:** `index.html:5291-5335` (`height:780px`); `viewer.html:44` (`onclick="window.location.href='/'"`)
- **Category:** Implementation Integrity / Control / Responsive
- **Impact:** The operator scrolls past the whole camera picker to reach the viewer. Clicking the viewer's Close loads the dashboard inside the Export panel.
- **Rewrite:** The viewer becomes a React component (react-three-fiber) placed *first* in Export, filling the available height. The camera picker goes in a side `Sheet` or a `Tabs` pane. There's no Close button when embedded.

**E-2 [P2] The auto-save confirmation sits below the fold**
- **Location:** `index.html:5213-5262` (status `<p>` appended after the full player table)
- **Category:** Visibility
- **Impact:** Toggling POV for the first player "saves" with feedback 20 rows below the click.
- **Rewrite:** A `sonner` toast ("Saved 6 cameras for gberch — re-run Export"), plus a sticky card footer showing the persisted count.

**E-3 [P2] Player names differ between panels**
- **Location:** Export uses `display_name` (`index.html:5244`: "MacAllister", "Referee"); HMR/Refined/Viewer show "P001", "Ref" for the same people (screenshots)
- **Category:** Consistency
- **Rewrite:** One `usePlayerLabel(playerId)` resolver (name + id chip) used everywhere.

**E-4 [P3] One metadata and one player fetch per shot**
- **Location:** `index.html:5159-5167`
- **Category:** Performance
- **Rewrite:** A batched `/api/export/status` endpoint, or parallel queries with caching.

### 4.9 Render, `js/render_panel.js`

**RN-1 [P1] The "Render" button renders every shot, although it sits next to the Shot select**
- **Location:** `render_panel.js:34-54` (button, then select), `85-90` (`body: {stages: "render"}`, no shot)
- **Category:** Match real world / Error prevention
- **Impact:** The layout says "render this shot", but it starts a multi-minute headless-Blender job for all active shots.
- **Rewrite:** A `SplitButton`: "Render gberch" (a `/api/run-shot` scope) and "Render all shots". Show an estimate from `render_timings.json`.

**RN-2 [P2] Each card shows an unlabeled "2193.1s" chip**
- **Location:** `render_panel.js:170-173` (the shot's total `render_seconds`, repeated on every card)
- **Category:** Match real world
- **Impact:** It reads as the clip length (36 minutes?). It's actually the total render wall-clock for the shot.
- **Rewrite:** Show "Rendered in 36m 33s" once in the card header. Per card, show the camera clip duration and size.

**RN-3 [P3] Some cards show an empty thumbnail**
- **Location:** `render_panel.js:150-155` (`preload="metadata"`, no `poster`; the broadcast card is blank in the screenshot)
- **Category:** Aesthetic
- **Rewrite:** A poster-frame endpoint, or `#t=0.5` on the src. Use `AspectRatio` wrappers and hover-scrub.

### 4.10 Anchor Editor, `anchor_editor.html`

**A-1 [P1] Unsaved anchors are silently discarded**
- **Location:** `anchor_editor.html:390` (clip `change` → `loadClip`); `232` ("← Dashboard"); there's no dirty flag or `beforeunload`
- **Category:** Error prevention
- **Impact:** Minutes of landmark clicking vanish on a shot switch or back-navigation. This is operator ground truth.
- **Rewrite:** A dirty-state store with an unsaved-changes `AlertDialog` (Save / Discard / Cancel) on shot change and route leave, a draft autosaved to `localStorage` per shot, and ⌘S to save.

**A-2 [P2] "Done — Open viewer" re-enters the rerun handler**
- **Location:** `anchor_editor.html:1282` (click listener, never removed), `1327-1329` (`onclick` reassigned to navigate)
- **Category:** Implementation Integrity
- **Impact:** Clicking "Done — Open viewer" first runs the original listener. That re-POSTs the anchors, and a second camera run may be dispatched before navigation aborts it. Inside the iframe, it also navigates the iframe itself.
- **Rewrite:** A state-machine button (idle → saving → running → done) driven by the job store. "Open viewer" is a separate `Link`.

**A-3 [P2] The ~60-row landmark palette can't be searched or reached by keyboard, and coordinates are cut off**
- **Location:** `anchor_editor.html:551-564` (`div` rows with click handlers); the screenshot shows truncated coords ("43.4,")
- **Category:** Recognition / Accessibility
- **Rewrite:** A `Command` (cmdk) list: type-to-filter, grouped (centre / left box / right box / far touchline / mowing), arrow-key selection, coordinates in a monospace secondary line, and already-placed landmarks marked for the current frame.

**A-4 [P2] Deleting an anchor has no undo**
- **Location:** `anchor_editor.html:1141-1147`, `1168-1175`
- **Category:** User control
- **Rewrite:** An undo stack (⌘Z), plus a toast with Undo on delete.

**A-5 [P2] The layout breaks at narrow widths**
- **Location:** `anchor_editor.html:45-49` (`grid-template-columns: 220px 1fr 240px`); in `mobile-anchor-editor.png` the video column collapses to zero
- **Category:** Responsive
- **Rewrite:** A `ResizablePanelGroup` on desktop. Below `md`, the video goes full-width and the palette and anchors move into `Tabs` or `Sheet`s.

**A-6 [P2] A third "primary" colour, with failing contrast**
- **Location:** `anchor_editor.html:231` (`#fff` on `#0ea5e9` = 2.8:1, sky blue next to the indigo Save)
- **Category:** Theming / Accessibility (WCAG 1.4.3)
- **Rewrite:** "Save" is the primary action. "Save & rerun camera" is a `secondary` or split-button action, using tokens only.

### 4.11 Ball Anchor Editor, `ball_anchor_editor.html`

**BA-1 [P1] Without `?shot=` there's no empty state, just a black canvas**
- **Location:** `ball_anchor_editor.html:829-833` (the status text "Pass ?shot=<id> in the URL", 11px grey, top-right); `836-839` (comment "no /api/shots list endpoint", yet `/api/output/shots` exists and `index.html:3753` uses it); screenshots `desktop-ball-editor.png`, `mobile-ball-editor.png`
- **Category:** Help / Visibility
- **Impact:** The editor shows an empty shot select, blank Tags and Anchors panels, a black video area and unstyled native transport buttons. It looks broken.
- **Rewrite:** An `EmptyState` card in the video area: "Choose a shot to annotate", a `Select` populated from `/api/output/shots`, each option showing anchor counts. Hide the side rails until a shot is loaded.

**BA-2 [P2] The shot select can't switch shots**
- **Location:** `ball_anchor_editor.html:836-839` (populated with the single current shot only; no `change` handler)
- **Category:** Consistency
- **Rewrite:** A real `ShotSelect` bound to the route param, with the unsaved-changes guard (A-1).

**BA-3 [P2] 10px functional text**
- **Location:** `ball_anchor_editor.html:59`, `104`, `114`, `117`, `127`, `130`, `134`, `285`, `312`, `719`; the detector found 6 instances (Touch authoring, Goal impact, Pitch fix labels and help)
- **Category:** Accessibility
- **Rewrite:** A 12px minimum for functional text (`text-xs`) and 11px only for decorative eyebrows. Help copy becomes `FormDescription`.

**BA-4 [P2] Native transport buttons with duplicate glyphs and no keyboard**
- **Location:** `ball_anchor_editor.html:143-146` (`&#9654; &#9664; &#9654;`, unstyled white in the screenshot); no `keydown` handler in the file
- **Category:** Consistency / Efficiency
- **Rewrite:** The shared `<FramePlayer>` (T-5/T-6).

**BA-5 [P3] Tag descriptions exist only in `title`**
- **Location:** `ball_anchor_editor.html:257-270` (`row.title = tag.description`); the tag rows are `div`s
- **Category:** Help / Accessibility
- **Rewrite:** A `ToggleGroup` (single) of tags, with the selected tag's description shown inline under the list.

### 4.12 Viewer, `viewer.html`

**V-1 [P2] Toggle state is encoded in label text**
- **Location:** `viewer.html:58-60` ("Ball: ON", "Skeleton: ON", "Mesh: OFF"; the state is shown by the label and the background colour), `1146-1156`
- **Category:** Accessibility / Consistency
- **Rewrite:** `Toggle` buttons (`aria-pressed`) with fixed labels and icons (Circle, Bone, Box).

**V-2 [P2] The resize handler assumes a 60px control bar**
- **Location:** `viewer.html:1185-1189` (`window.innerHeight - 60`); the controls use `flex-wrap` (`11`)
- **Category:** Responsive
- **Impact:** When controls wrap to two or three rows (narrow windows), the canvas is sized wrong. In the mobile screenshot the match header collides with the player list.
- **Rewrite:** Size the canvas from the container with a `ResizeObserver` (r3f does this). Put overlay chrome in a flex column, with the player list moving into a `Sheet` below `md`.

**V-3 [P2] The render loop never idles**
- **Location:** `viewer.html:1163-1182` (renders every rAF while paused, including inside the Export iframe)
- **Category:** Performance
- **Rewrite:** Render on demand. Pause when `document.hidden` or when the canvas is off-screen (`IntersectionObserver`).

**V-4 [P2] Player list rows are clickable divs, and the selects have no labels**
- **Location:** `viewer.html:882-896` (`.player-row` click); `62-75` (speed, camera and shot selects have no label, and the shot select only has `title`)
- **Category:** Accessibility
- **Rewrite:** A list of `Button`s (`aria-pressed` for the focused player), and labelled `Select`s in a compact toolbar.

**V-5 [P3] The documented confidence timeline doesn't exist**
- **Location:** CLAUDE.md ("confidence timeline highlighting frames where camera or HMR are uncertain") vs `viewer.html` (no timeline strip; `confidence` is only parsed at `878`)
- **Category:** Implementation Integrity (doc/UI drift)
- **Rewrite:** Reuse the anchor editor's confidence strip in `<FramePlayer>` (camera confidence + HMR confidence lanes, click to seek), or correct the docs.

---

## 5. Patterns and Systemic Issues

1. **Destructive actions aren't treated as a category.** Confirmations exist for split and track delete, but not for Re-run Stage (wipes operator data), Merge by Name, Ignore Unknown, anchor delete, or the partial-payload ball Save. Nothing has undo. The rewrite needs one policy: reversible actions get an Undo toast, irreversible actions get an AlertDialog that states their scope, and operator-edited artefacts get a typed confirmation.
2. **There is no job model in the client.** Run state lives in three places (the shell's `runningStage`, the in-panel buttons' local `disabled`, and the iframe's polling loop), none of which survives a reload. A single job store (server list + SSE reattach + cancel) would fix S-2, S-3, S-5, C-2 and H-1 together.
3. **The UI and API have no shared schema.** The `undefined` summaries (PS-1/RP-1), the ball payload loss (B-1) and the stale comment about a missing shots endpoint (BA-1) all come from hand-read JSON. Add zod schemas generated from, or mirrored against, the Pydantic models (`BallAnchorPayload`, summary sidecars).
4. **Code is copied instead of shared.** There are 2 ball editors, 5+2 frame transports, 8 shot selects and 3 viewers (the viewer iframe, the ball 3D preview, the HMR trajectory canvas). The rewrite needs `<FramePlayer>`, `<ShotSelect>`, `<BallAnchorEditor>`, `<SceneViewer>`, `<StageRunControls>`, `<ConfidenceCell>` and `<StatusBadge>` as first-class components.
5. **There's no design system.** 787 hex literals (91 unique colours), 524 inline style assignments, 0 tokens, and button colour chosen per call site (indigo `#6366f1`/`#4f46e5`, sky `#0ea5e9`, teal `#0f766e`, slate `#334155`/`#374151`, red `#7f1d1d`). Every contrast failure comes from the same `#64748b` muted colour reused on four different surfaces.
6. **Accessibility is absent, not partial.** Zero ARIA, clickable `div`s for every list (stages, landmarks, tags, anchors, viewer players, tile menus), colour-only status, and 78 `title`-only hints.
7. **Iframes stand in for composition.** The embedded editors bring their own headers, navigation (which nests the dashboard), fixed heights and a separate keyboard focus.
8. **Errors are masked as emptiness.** `fetchJsonOrNull` is used everywhere, and a failure looks like "not run yet".

---

## 6. Positive Findings (preserve in the rewrite)

- **Run lockout while any stage runs** (`index.html:505-545`): every run trigger and the output-dir switcher are disabled mid-run, with the reason given. The server adds a 409 guard for concurrent hmr_world (`server.py:754-800`). Keep both, and extend the client lockout to *all* run paths.
- **Dependency gating with human reasons** (`STAGE_DEPS`, `index.html:333-347`; "Requires: hmr_world"). The hard-versus-soft dependency split is deliberate and documented; keep it.
- **Continue versus Re-run** (`index.html:684-693`): resuming without wiping the per-player cache is the right default for 35–60 min stages. Make it the *primary* action.
- **SSE log streaming with server-side replay** (`server.py:511-526`): the full `log_lines` are replayed on connect, so reattach only needs a job-list endpoint.
- **The last viewed stage is restored from localStorage** (`index.html:455-469, 486`), which survives trips into the editors and viewer. Keep it, and extend it to the selected shot via the URL.
- **Operator-wins flows:**
  - discarded shots go to a restorable "dropped tray", with no destructive delete (`prepare_shots_panel.js:782`)
  - "Re-align" keeps manual offsets (`723`)
  - dismissed auto-anchors persist, with an undo (`ball_anchor_editor.html:345, 369`)
  - stale-anchor ⚠ detection in multi-shot status (`index.html:1282-1285`)
  - the split re-run confirms with an explicit scope (`prepare_shots_panel.js:350`)
- **Autosave with honest persisted-state feedback**: camera selections (`index.html:5201-5262`; the comment "status line reflects exactly what's on disk"), and track renames that propagate to the video overlay without a round-trip (`1646-1683`).
- **Anchor editor craft:**
  - ←/→/Space/Esc shortcuts that ignore focused inputs (`1401-1410`)
  - click-to-seek confidence strip with anchor ticks (`1190-1234`)
  - sub-pixel snap-to-lines, projected-pitch and detected-line overlays
  - a focused name field highlights that player's bbox (`index.html:1641-1650`)
- **Empty-state copy that names the next step**, e.g. "No refined tracks yet — run hmr_world first, then click Run refined_poses above." (`5039`).
- **Camera metric explanations** (`CAM_METRIC_HELP`, `index.html:2240-2245`): excellent content that just needs an accessible delivery.
- **A stable per-player colour palette** shared across panels (`index.html:2866-2870`), `tabular-nums` on numeric columns, and range-request video serving for smooth scrubbing.
- **Toasts and an Esc-closable lightbox** in Prepare Shots: the right interaction ideas, needing proper components.

---

## 7. Improvement Backlog for the Rewrite

1. [Shell] Make Continue the primary action. Put "Re-run clean" behind an AlertDialog listing the paths to be deleted, with a typed confirmation for `prepare_shots`/`tracking`.
2. [Shell/API] Move the output wipe into `/api/run` (`clean: true`) so it happens only after the job is admitted. Never DELETE from the client first.
3. [Shell/API] Add `GET /api/jobs?status=running` plus a global job store that reattaches the SSE log on load.
4. [Shell/API] Add `POST /api/jobs/{id}/cancel` and a Cancel button in the log dock.
5. [Shell] Handle SSE `onerror` with a "reconnecting" Alert and fall back to status polling.
6. [Shell] Use a tri-state StatusBadge (complete / partial n/m / none) in the header and sidebar.
7. [Shell] One `StageRunControls` per stage, used by every run path; blocked reasons as visible helper text.
8. [Shell] shadcn Sidebar with a route per stage, `aria-current`, and a Sheet under `md`.
9. [Shell] Landmarks (`nav`/`main`), stage `h1`, card `h2`, `aria-live` for run status and toasts.
10. [Shell] A token system (`--muted-foreground` ≥ 4.5:1, primary ≥ 4.5:1), dark by default, light theme supported, no hex literals in components.
11. [Shell] A virtualised LogViewer with Follow, Copy/Download, error highlighting and a resizable bottom dock.
12. [Shell] A workspace Select plus a "New output" Dialog with validation, replacing `prompt`/`alert`.
13. [Shell] TanStack Query with abort; separate `isError` (Alert + Retry) from empty (EmptyState); Skeletons while loading.
14. [Shell] lucide icons with `aria-label` + Tooltip on every icon-only button; no Unicode glyph icons.
15. [Shell] Sentence-case CardTitles instead of uppercase tracked titles.
16. [Prepare Shots] Render `total_frames` and foot-lock counters from a zod-typed quality report; never render `undefined`.
17. [Prepare Shots] DropdownMenu tile menu with a "Move to…" submenu (all groups + New group); dnd-kit keyboard sensor.
18. [Prepare Shots] Lightbox as a Dialog; Re-run split confirmation as an AlertDialog.
19. [Prepare Shots] Multi-shot status as a Table of text Badges ("Stale", "Missing") from one batched request.
20. [Tracking] Button variants with real disabled styling (Merge/Delete/Interp).
21. [Tracking] AlertDialog stating the scope plus an Undo toast (server snapshot) for Merge by Name, Ignore Unknown, merges and deletes.
22. [Tracking] Labelled row controls (Checkbox sr-only label, name Input `aria-label`, status text alongside the dot).
23. [All players] A shared `<FramePlayer>`: Slider with `aria-valuetext`, frame Input, ←/→/Shift/Space/Home/End keys, optional confidence lanes.
24. [All pages] A `?shot=` search param shared across routes, plus a shared `<ShotSelect>` with Label.
25. [Camera] The anchor editor as an in-shell route using ResizablePanelGroup; no iframe, duplicate header or nested-dashboard link.
26. [Camera/Anchor Editor] "Save & rerun camera" dispatched through the shared job store (shows the log, applies the lockout).
27. [Anchor Editor] Dirty tracking with an unsaved-changes AlertDialog, a localStorage draft, and ⌘S / ⌘Z.
28. [Anchor Editor] A Command (cmdk) landmark palette: search, grouping, keyboard selection, full coordinates.
29. [Anchor Editor] Replace the "Done — Open viewer" rebinding with a state-machine button plus a separate Link.
30. [HMR World] Confirm All-players runs with an estimated runtime; a Combobox player picker.
31. [HMR/Refined] A single `<ConfidenceCell>` (value + bar + tier) with shared thresholds and a legend.
32. [Refined Poses] Summary stat tiles (frames, foot-lock spans and pin error, PhysPT takeovers, cleanup clamps) from the typed summary.
33. [Ball] One `<BallAnchorEditor>` that round-trips the full `BallAnchorSet` (chains, dismissed, end_frame, landmark, confidence); server PATCH/merge; a load→save identity test.
34. [Ball] Dispose the 3D preview on unmount (cancel rAF, `renderer.dispose`, observer disconnect) and render on demand.
35. [Ball] Order the page as summary tiles → editor → collapsible segments table with empty columns hidden; always show the shot picker.
36. [Ball Anchor Editor] An EmptyState with a shot picker from `/api/output/shots` when `?shot=` is missing; a working shot switcher.
37. [Ball Anchor Editor] A 12px minimum for functional text; tags as a ToggleGroup with an inline description.
38. [Export] A react-three-fiber viewer shown first at full height; camera picker in Tabs/Sheet; auto-save feedback via toast; one `usePlayerLabel` resolver.
39. [Render] A "Render this shot / Render all" SplitButton with an estimate; render time shown once and labelled; poster frames on cards.
40. [Viewer] Toggle buttons with `aria-pressed`, container-driven canvas sizing, render-on-demand, a player list of Buttons, and the documented confidence timeline.

---

## 8. Recommended Actions (impeccable command mapping)

1. **[P0] `/impeccable harden`**: the destructive Re-run flow (S-1) and the ball-anchor round-trip (B-1, B-2).
2. **[P1] `/impeccable harden`**: the job store, SSE reattach/error handling and unified run controls (S-2, S-3, S-5, C-2, H-1, RN-1).
3. **[P1] `/impeccable clarify`**: tri-state status, the `undefined` summaries, and render-scope and render-time labels (S-4, PS-1/RP-1, RN-1, RN-2).
4. **[P1] `/impeccable adapt`**: responsive shell and editors (S-9, A-5, V-2, C-1, E-1).
5. **[P1] `/impeccable colorize`**: tokens, contrast and the single primary colour (S-8, A-6, T-1, pattern 5).
6. **[P1] `/impeccable audit`**: re-run after the rewrite to verify a11y (S-6, S-7, T-4, PS-2, PS-4, V-1, V-4).
7. **[P2] `/impeccable layout`**: Ball and Export page order; the ResizablePanel editors (B-4, E-1, T-8).
8. **[P2] `/impeccable optimize`**: WebGL disposal, render-on-demand, the virtualised log, batched fetches (B-3, V-3, S-11, PS-6, E-4).
9. **[P2] `/impeccable onboard`**: empty and error states (BA-1, S-13, S-14).
10. **[P3] `/impeccable typeset`**: sentence-case titles, the 12px floor (S-17, BA-3).
11. **`/impeccable polish`**: final pass.
