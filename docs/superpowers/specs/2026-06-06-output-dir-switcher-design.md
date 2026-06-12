# Output-directory switcher (web dashboard)

**Date:** 2026-06-06
**Status:** Approved, implementing

## Problem

The pipeline produces multiple sibling output directories (`output/`,
`output-origi/`, `output-kroupi/`, …) for different experiments. Today the
served directory is fixed at launch via `recon.py serve --output <dir>`. To
look at a different reconstruction you must stop the server and relaunch.

We want a dropdown in the dashboard top bar to switch the active output
directory at runtime — both for **display** (stage status, viewer, panels)
and for **writing** (running pipeline stages targets the active dir).

## Scope (decided)

- **Placement:** dashboard only (`index.html` `#stage-header`, with the
  top-right button cluster). The viewer is display-only and unchanged.
- **Contents:** every `output*` sibling directory found on disk, **plus** a
  "New output…" entry that creates an empty `output-<name>/` and switches to it.
- The active directory is **server-wide**: one server process holds one active
  dir, so every page/endpoint reads from it. Switching from the dashboard
  affects the viewer/anchor-editor too (after their own reload).

## Architecture

### Runtime switch mechanism — shared closure cell

`create_app(output_dir, …)` defines ~89 endpoint closures that all reference
the same enclosing `output_dir` variable (one cell). A switch endpoint
reassigns that cell with `nonlocal output_dir`, so every existing closure
immediately sees the new value — **no changes to the other 89 references.**
`app.state.output_dir` is kept in sync for anything that reads state.

Rejected alternatives:
- Rewrite all 89 references to a `current_output_dir()` helper reading
  `app.state` — large mechanical edit of a 118 KB file, no functional gain.
- Restart uvicorn with a new dir — heavy, drops in-flight jobs.

In-flight pipeline jobs are unaffected: `_run_job` captures its `output_dir`
at submit time, so switching mid-run never redirects a running job.

Switching is **in-memory only**; a server restart reverts to the CLI
`--output`. This is intentional, not a gap.

### Discovery

Glob `output*` **directories** that are direct children of the resolved
served path's parent (e.g. parent = repo root). Always include the current
served dir even if its name doesn't match `output*`. Return sorted basenames.

### Endpoints (`src/web/server.py`)

- `GET /api/output-dirs` → `{ "current": "output", "dirs": ["output",
  "output-kroupi", "output-origi"], "parent": "/abs/parent" }`.
- `PUT /api/output-dirs/active` body `{ "name": "output-kroupi" }` → validate
  `name` is in the discovered whitelist (resolved, direct child of parent,
  name matches `output*`); reassign closure cell + `app.state`; return the new
  state. Reject path-traversal / non-whitelisted names with 400.
- `POST /api/output-dirs` body `{ "name": "foo" }` → sanitize to
  `[A-Za-z0-9_-]+`; final dir name is `name` if it already starts with
  `output`, else `output-<name>`; `mkdir` under parent; switch to it; return
  new state. 400 on empty/invalid name.

### Frontend (`src/web/static/index.html`)

- A `<select id="output-dir-select">` appended to `#stage-header` after the
  Re-run button, populated from `GET /api/output-dirs` on load, current dir
  selected. A trailing `<option value="__new__">New output…</option>`.
- On change:
  - `__new__` → `prompt()` for a name → `POST /api/output-dirs` → `location.reload()`.
    Cancel restores the previous selection.
  - otherwise → `PUT /api/output-dirs/active` → `location.reload()` so all
    stage statuses/panels reflect the new dir.
- The selector is `disabled` while a pipeline run is in progress (jobs are
  unaffected, but the guard avoids confusion).

## Testing (`tests/test_web_api_output_dirs.py`, `TestClient`)

- Discovery lists `output*` siblings + the current dir, sorted.
- `PUT …/active` switches: a subsequent `GET /api/stages` reflects the new dir
  (prove the closure-cell reassignment reaches existing endpoints).
- `POST` creates `output-<name>/` on disk and switches to it.
- Invalid names / path traversal (`../`, absolute, non-`output*`) → 400 and no
  switch / no directory created.

## Out of scope (YAGNI)

- Persisting the active dir across restarts.
- Deleting/renaming output dirs from the UI.
- Per-tab independent active dirs.
