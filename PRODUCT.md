# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React 19 + Vite + TypeScript + Tailwind CSS v4 + shadcn/ui, living in `frontend/` and built into `src/web/static/app/` (committed, so `python recon.py serve` needs no Node toolchain). FastAPI (`src/web/server.py`) serves the SPA shell and the JSON/SSE/video API. User decision 2026-09-28: full React + shadcn/ui rewrite, dark theme by default with a light toggle.

## Users

A single technical operator (the project's developer) running the Football Perspectives reconstruction pipeline on their own Mac. They sit at a desktop display for long sessions and switch between running pipeline stages, reading logs, and doing precise manual annotation on broadcast video frames. The pages are also opened on a laptop, and occasionally glanced at on a phone to check a long run. (Inferred from the repository; the operator confirmed the stack and theme.)

## Product Purpose

A local dashboard for a CLI pipeline that turns a single broadcast football camera clip (or a whole highlights reel) into 3D player animation and ball trajectories. It runs and re-runs the eight stages (prepare_shots → tracking → camera → hmr_world → refined_poses → ball → export → render), streams their logs, surfaces per-stage quality diagnostics, and hosts the manual editors whose operator input the solvers treat as ground truth. Success means the operator can see what state each stage and shot is in, spot where quality is weak, fix it with an editor, and re-run, without leaving the dashboard.

## Positioning

Operator input is authoritative: manual pitch anchors, ball anchors, track merges/renames and sync offsets always override automatic passes. The dashboard exists to make that human-in-the-loop correction fast and precise on real broadcast frames.

## Operating Context

- Launched with `python recon.py serve --output ./output/ [--port N]`; one output directory active at a time (switchable).
- Heavy stages run for tens of minutes locally; logs stream over SSE and only one run may be in flight.
- Editors: pitch-landmark anchor editor (`/anchor_editor`), ball anchor editor (`/ball-anchor-editor?shot=`), track editor (tracking stage), prepare-shots groups board + sync timeline, 3D viewer (`/viewer`).
- Evaluation clips: gberch (primary), origi, japan, kroupi, saka output dirs.

## Capabilities and Constraints

- All API routes in `src/web/server.py` are the contract; the frontend rewrite must not change them.
- Video frames, canvases and 3D (three.js) are the working material; chrome must never compete with them.
- Destructive actions exist (re-run wipes a stage's output, track deletes, dropping shots).
- Player identity colours are data and must stay stable across panels.

## Product Principles

1. Operator data wins: never let an automatic pass or a stray click overwrite manual work without an explicit confirm.
2. State is always visible: which stage runs, which shot is weak, what is unsaved.
3. The frame is the hero: controls stay dense, quiet and out of the image.
4. One vocabulary: the same action looks and behaves the same on every page.

## Accessibility & Inclusion

Target WCAG 2.2 AA for chrome (contrast, keyboard, focus, labels). Canvas annotation is pointer-first by nature; keyboard shortcuts should cover frame stepping and common editor actions.
