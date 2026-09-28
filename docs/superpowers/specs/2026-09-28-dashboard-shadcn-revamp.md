# Dashboard revamp: React + shadcn/ui (2026-09-28)

## Decision

User request: a complete revision of the web dashboard UX, converted to shadcn throughout, with
impeccable run against it to find and implement improvements. User decisions (2026-09-28):

- **Stack:** full React + Vite + TypeScript + shadcn/ui rewrite (chosen over a shadcn-styled CSS
  layer on the vanilla JS).
- **Theme:** dark by default, with a light / system toggle.
- The built SPA is committed to `src/web/static/app/`, so `recon.py serve` still needs no Node.

## Evidence that drove the changes

`2026-09-28-dashboard-ux-audit.md` is the impeccable audit and heuristic critique of the legacy
UI:

- Audit health scored 5/20 (Critical); Nielsen heuristics scored 16/40.
- 69 findings: 2 P0, 18 P1, 36 P2, 13 P3.
- Code metrics: 787 hex literals, 524 inline styles, 0 ARIA attributes, 0 media queries.

## Architecture

- `frontend/` holds the source (see `frontend/README.md` for conventions).
- Every page route (`/`, `/anchor_editor`, `/ball-anchor-editor`, `/viewer`) returns the same SPA
  shell with `Cache-Control: no-store`. Client-side routing then picks the page, inside one shared
  shadcn `Sidebar` shell.
- The stage is chosen by `?stage=` (falling back to localStorage, then the first incomplete
  stage). The shot is chosen by `?shot=`, which carries across the editor links.
- One lazily-loaded chunk per stage panel and editor, so three.js and the canvas editors only load
  when opened.
- The editors are real components rather than iframes:
  - the Camera panel embeds `<AnchorEditor embedded />`;
  - the Export panel embeds `<Viewer embedded shot />`;
  - the Ball panel and the ball-anchor page share one editor implementation.
- `PipelineProvider` is the one job store:
  - stage list with complete / partial status;
  - live run state and the SSE log;
  - reattach on load;
  - reconnect with backoff;
  - one lockout shared by every run trigger.

## Additive API changes (backward compatible)

| Change | Why |
|---|---|
| `POST /api/run` `clean_first: bool` | Re-run clears the stage's generated output only after the run is accepted. The legacy flow DELETEd first, so a 409/429 lost output (audit S-1, P0). |
| `GET /api/output/{stage}/artifacts` | Dry run for the re-run confirm dialog: lists exactly what will be cleared. |
| `GET /api/jobs[?status=running]` | Reattach to an in-flight run after a reload (S-2). |
| `/api/stages` → `partial` | Tri-state status: stages with some output no longer read "Not run" (S-4). |
| `_clear_stage_outputs` / `_stage_output_paths` | Shared by DELETE, `clean_first` and the dry run. |

## Visual system

- shadcn/ui radix-nova on the neutral base, Geist + Geist Mono.
- Monochrome primary. Colour is spent only on state (success / warning / destructive / info) and
  on data (the fixed player palette, pitch drawing).
- Media sits in a near-black `stage` well in both themes.
- The direction contract lives in `.impeccable/surfaces/frontend-src-app-tsx.md`.

## Not done / follow-ups

- Job cancel needs a cancellable runner; pipeline jobs run in daemon threads.
- Server-side undo snapshots for track merges and deletes; today those are confirm-gated only.
- Typed schemas (zod) mirrored from the Pydantic payload models (audit pattern 3).
