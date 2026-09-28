---
version: 1
slug: "frontend-src-app-tsx"
primary_target: "frontend/src/App.tsx"
related_targets: ["frontend/src/pages/dashboard.tsx"]
---

## Scope and mode

Whole operator dashboard (React SPA in `frontend/`): pipeline stage panels, pitch-anchor editor, ball-anchor editor, track editor, 3D viewer. Mode: **Operate**. Single technical operator, long desktop sessions, precise annotation on broadcast frames; occasional phone glance at a long run.

## Direction contract

THESIS: The frame is the hero; the chrome is shadcn played straight. One quiet, consistent component vocabulary (shadcn/ui radix-nova, neutral base) so every stage, editor and viewer reads as one tool, and colour is spent only on data and state. Refuses the legacy look of per-panel ad-hoc indigo/sky/slate buttons, uppercase tracked panel titles and inline-styled chrome.

OWN-WORLD: shadcn neutral tokens, dark default with light toggle. Geist for UI, Geist Mono for frame numbers, coordinates and ids (data only). Monochrome primary buttons. State vocabulary: success (complete), warning (running), destructive (failed / destructive action), info (selection/hint). Player identity colours are fixed data colours shared across panels. Video, canvas and 3D sit in a near-black `stage` well regardless of theme. Cards with sentence-case titles are the only container; never nested.

STORY: The operator sees at a glance which stage is complete, running or failed and which shot is weak, drills into a stage panel, fixes data in an editor (anchors, ball events, tracks, sync), and re-runs with an explicit confirm before anything destructive. Nothing they entered can be silently overwritten.

FIRST VIEWPORT: Left collapsible shadcn Sidebar: product name, output-directory switcher, Pipeline list (8 stages, status dot + index), Editors list, theme toggle, live-run entry. Sticky page header: sidebar trigger, stage title + status badge + one-line description, actions right (Run all ghost, Continue outline + Re-run primary in a button group, gated with tooltips). Content: stage panels as Cards; run log docks to the bottom of the content area. Primary action sits top-right of the header.

FORM: Brief-pinned (user chose "Full React + Vite + shadcn/ui rewrite", 2026-09-28): shadcn/ui radix-nova as the committed world; no concept roll — a user-pinned direction beats the roll. Seed key: none (pinned).

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Memorable moment

The run log dock plus live sidebar dots: starting a stage visibly pulses its dot, the log follows the tail and flips to the traceback on failure, and every panel refreshes itself when the job finishes.

## Unresolved

- Mobile is a monitoring posture (sidebar becomes a sheet, editors show a "best on desktop" note while staying usable), not a full annotation experience.
