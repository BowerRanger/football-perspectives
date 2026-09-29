# Dashboard frontend (React + Vite + shadcn/ui)

The operator dashboard served by `python recon.py serve`. Source lives here;
the production build is written to `../src/web/static/app/` and **committed**,
so running the dashboard never needs Node. FastAPI serves the SPA shell for
`/`, `/anchor_editor`, `/ball-anchor-editor` and `/viewer`; every JSON / SSE /
video endpoint in `src/web/server.py` is unchanged.

```bash
cd frontend
npm install              # .npmrc forces dev deps even when npm omit=dev is global
npm run dev              # Vite on :5173, proxies the API to FP_API (default http://localhost:8765)
npm run build            # typecheck + build into ../src/web/static/app (commit the result)
npx tsc -b --noEmit      # typecheck only
npm test                 # vitest unit tests (src/**/*.test.ts)
```

Run a backend for dev with `python recon.py serve --output ./output --port 8765`.

## Layout

| Path | What |
|---|---|
| `src/main.tsx`, `src/App.tsx` | Providers (theme, tooltip, router, dialogs, pipeline) and routes inside the sidebar shell |
| `src/components/ui/` | shadcn/ui components (radix-nova). Add more with `npx shadcn@latest add <name>`; don't hand-edit unless fixing a bug |
| `src/components/` | App-level building blocks: `panel.tsx` (Panel / PanelEmpty / PanelError / PanelSkeleton / StatList), `status.tsx` (StatusDot / StatusBadge / ToneBadge), `page-header.tsx`, `app-sidebar.tsx`, `log-dock.tsx`, `stage-actions.tsx` |
| `src/hooks/use-pipeline.tsx` | Stage list, live run state, `startRun` / `rerunStage` / `attachToJob`, SSE log stream, `outputVersion` |
| `src/hooks/use-dialogs.tsx` | `useConfirm()` / `usePrompt()` — promise-based shadcn dialogs replacing `window.confirm/prompt` |
| `src/lib/api.ts` | `getJson`, `getJsonOrNull`, `postJson`, `putJson`, `deleteJson`, `postForm`, `ApiError`, `errorMessage`, `qs` |
| `src/lib/format.ts` | `fmt*` (never prints "undefined"), player palette (`playerColour`, `playerLabel`), `withAlpha`, `confidenceTone`, `TONE_TEXT`, `cssVar` |
| `src/lib/stages.ts` | Stage names, labels, descriptions, hard dependencies |
| `src/features/stages/<stage>/` | One lazily-loaded panel per pipeline stage (default export, no props) |
| `src/pages/` | Routed pages: `dashboard.tsx`, `anchor-editor/`, `ball-anchor-editor/`, `viewer/` |

## Conventions

- **shadcn throughout.** Buttons, inputs, selects, tabs, tables, dialogs, tooltips, badges, toggles, sliders
  come from `@/components/ui`. No raw `<button>`/`<select>` styling, no inline `style={{}}` for chrome
  (inline style is fine for data-driven geometry: canvas sizes, player colours, positions).
- **Tokens, not hex.** Chrome colours are Tailwind token classes (`bg-card`, `text-muted-foreground`,
  `text-success`, `bg-stage`…). Hex is allowed only for data colours (player palette, pitch drawing,
  canvas overlays) — read theme tokens for canvas chrome with `cssVar("--foreground")`.
- **State vocabulary.** success = complete/ok, warning = running/marginal, destructive = failed or
  destructive action, info = selection/hint. `StatusBadge` for stage state, `ToneBadge` for data state.
- **Panels.** `Panel` (a Card) is the only container; sentence-case titles; never nest Panels.
  Loading → `PanelSkeleton`/`Skeleton`; empty → `PanelEmpty` that says what to do next; failure → `PanelError`.
- **Errors are never empty states.** Load a panel's main payload with `useResource` (`@/hooks/use-resource`) and
  `getJson` / `getJsonOr404` (null only on 404 = "not produced yet"); render `PanelError` + Retry on failure.
  `getJsonOrNull` swallows every failure and is reserved for optional lookups (a test caps its use).
- **One frame transport.** Every video / canvas / 3D player uses `<FramePlayer>` (`@/components/frame-player`):
  play/step/scrub with a spoken value, optional frame input, and the shared Space / ←→ / Shift±10 / Home End
  shortcuts (`useFrameKeys`; only the newest enabled player listens — pass `keyboard={false}` to secondaries).
- **Runs** go through `usePipeline()`; the log dock is virtualised and has Cancel (`POST /api/jobs/{id}/cancel`).
  Destructive track edits return an `undo_id`; offer it via a toast action (`POST /api/tracks/undo`).
- **Media wells.** Video, canvases and three.js sit in `bg-stage` (near-black in both themes).
- **Destructive actions** go through `useConfirm({ destructive: true })`. Errors surface via
  `toast.error(title, { description })` (sonner) — never `alert()`.
- **Icons** are lucide-react. No unicode glyphs (✕ ⟳ ↗ ▶) as icons.
- **Numbers** use `tabular-nums` (tables get it automatically) and Geist Mono (`font-mono`) for ids,
  frames and coordinates only.
- **Keyboard.** Every clickable thing is a button or link; editors keep their shortcuts and show them
  with `Kbd`.
- Stage panels re-mount when a job finishes (`outputVersion`), so a plain fetch-on-mount is enough.
