---
name: Football Perspectives Dashboard
description: Operator console for a broadcast-to-3D football reconstruction pipeline; shadcn played straight, colour spent only on data and state.
colors:
  background: "oklch(0.145 0 0)"
  foreground: "oklch(0.985 0 0)"
  card: "oklch(0.185 0 0)"
  popover: "oklch(0.205 0 0)"
  primary: "oklch(0.922 0 0)"
  primary-foreground: "oklch(0.205 0 0)"
  muted: "oklch(0.269 0 0)"
  muted-foreground: "oklch(0.72 0 0)"
  border: "oklch(1 0 0 / 10%)"
  input: "oklch(1 0 0 / 15%)"
  ring: "oklch(0.556 0 0)"
  sidebar: "oklch(0.17 0 0)"
  sidebar-accent: "oklch(0.25 0 0)"
  stage: "oklch(0.11 0 0)"
  stage-foreground: "oklch(0.96 0 0)"
  success: "oklch(0.76 0.15 152)"
  warning: "oklch(0.82 0.14 78)"
  destructive: "oklch(0.704 0.191 22.216)"
  info: "oklch(0.74 0.12 240)"
  background-light: "oklch(1 0 0)"
  foreground-light: "oklch(0.145 0 0)"
  primary-light: "oklch(0.205 0 0)"
  muted-foreground-light: "oklch(0.5 0 0)"
  border-light: "oklch(0.922 0 0)"
  sidebar-light: "oklch(0.985 0 0)"
  stage-light: "oklch(0.18 0 0)"
  success-light: "oklch(0.45 0.13 150)"
  warning-light: "oklch(0.49 0.12 65)"
  destructive-light: "oklch(0.577 0.245 27.325)"
  info-light: "oklch(0.47 0.15 250)"
typography:
  headline:
    fontFamily: "'Geist Variable', ui-sans-serif, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 600
    lineHeight: 1.5
    letterSpacing: "-0.025em"
  title:
    fontFamily: "'Geist Variable', ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 600
    lineHeight: 1.375
  body:
    fontFamily: "'Geist Variable', ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.43
  label:
    fontFamily: "'Geist Variable', ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 500
    lineHeight: 1.33
  data:
    fontFamily: "'Geist Mono Variable', ui-monospace, 'SF Mono', monospace"
    fontSize: "0.75rem"
    fontWeight: 400
    lineHeight: 1.625
    fontFeature: "'tnum'"
rounded:
  sm: "6px"
  md: "8px"
  lg: "10px"
  xl: "14px"
  pill: "26px"
  full: "9999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.primary-foreground}"
    rounded: "{rounded.lg}"
    padding: "0 10px"
    height: "32px"
  button-primary-sm:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.primary-foreground}"
    rounded: "{rounded.md}"
    padding: "0 10px"
    height: "28px"
  button-outline:
    backgroundColor: "{colors.input}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.lg}"
    padding: "0 10px"
    height: "32px"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.foreground}"
    rounded: "{rounded.lg}"
    padding: "0 10px"
    height: "32px"
  button-destructive:
    backgroundColor: "{colors.destructive}"
    textColor: "{colors.destructive}"
    rounded: "{rounded.lg}"
    padding: "0 10px"
    height: "32px"
  panel:
    backgroundColor: "{colors.card}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.xl}"
    padding: "16px"
  panel-header:
    typography: "{typography.title}"
    padding: "12px 16px"
  input:
    backgroundColor: "{colors.input}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.lg}"
    height: "32px"
  status-badge:
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    padding: "2px 8px"
    height: "20px"
  stage-well:
    backgroundColor: "{colors.stage}"
    textColor: "{colors.stage-foreground}"
    rounded: "{rounded.lg}"
  sidebar:
    backgroundColor: "{colors.sidebar}"
    textColor: "{colors.foreground}"
    width: "256px"
  sidebar-item-active:
    backgroundColor: "{colors.sidebar-accent}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.md}"
---

# Design System: Football Perspectives Dashboard

## Overview

**Creative North Star: "The Frame Is the Hero"**

The dashboard is a quiet instrument around loud material. Broadcast frames, pitch canvases and the three.js scene are the work; everything else is shadcn/ui (radix-nova style, neutral base) used as shipped, so every stage panel, editor and viewer reads as one tool. The chrome is achromatic: black, white and a few greys. Colour appears only where it means something: a stage's state, a metric's quality, a player's identity, a pitch drawing.

Density is desktop-operator density: 14px body, 12px secondary text, 32px controls (28px in headers and toolbars), 8 to 16px gaps. Dark is the default theme with a light toggle; both themes share the same structure and the same near-black media well, so footage never changes its surround when the theme flips.

**Key Characteristics:**
- Neutral, flat chrome; hue is reserved for state and data.
- One container: the Panel (a Card with a sentence-case title), never nested.
- Media always sits in the near-black stage well, in both themes.
- Geist for UI, Geist Mono only for ids, frame numbers, coordinates and logs.
- State is always visible: sidebar status dots, header status badge, docked run log.

## Colors

A monochrome neutral scale carries all chrome; four semantic hues and a fixed player palette carry all meaning. Tokens are OKLCH custom properties in `frontend/src/index.css`; the unsuffixed values above are the dark (default) theme, `-light` keys are the light theme's counterparts.

### Primary
- **Chalk White** (primary, dark theme) / **Ink** (primary-light): the monochrome primary. Fills the one primary action per header (Continue, Run for selection) and the primary button in dialogs. It has no hue on purpose.

### Neutral
- **Floodlit Night** (background): page canvas in dark mode; white in light mode.
- **Raised Charcoal** (card): Panel surface, one step above the background.
- **Touchline Grey** (muted-foreground): descriptions, captions, labels in stat lists, key hints.
- **Hairline** (border, 10% white in dark): panel header rules, table rows, the page header's bottom edge.
- **Tunnel Black** (stage): the media well behind video, canvases, three.js and the run log body. Near-black in both themes.

### Semantic (state vocabulary)
- **Pitch Green** (success): complete / ok. A half-filled success dot means partial output.
- **Amber Flag** (warning): running / marginal. The running dot pulses.
- **Red Card** (destructive): failed, and destructive actions.
- **Replay Blue** (info): selection and hints only; also the text-selection tint.

Light-theme semantic values are darker (lightness 0.45 to 0.58) so text on white clears WCAG AA; do not reuse the dark-theme values on light surfaces.

### Data colours
The player palette (16 fixed hex values in `frontend/src/lib/format.ts`) is data, not theme: a player keeps the same colour in tables, overlays, trajectories and the 3D viewer, in both themes. Chart tokens (`--chart-1..5`) mirror the semantic hues. Hex is allowed only for data colours and canvas drawing.

### Named Rules
**The Colour-Is-Meaning Rule.** Chrome is achromatic. If a hue appears, it is a state (success / warning / destructive / info) or a data identity (player, pitch). A coloured button, heading or decorative accent is off-system.

**The Fixed Surround Rule.** Footage and canvases sit in `stage` in both themes; theme switching never changes what surrounds the image.

## Typography

**Display Font:** none (the system has no display tier)
**Body Font:** Geist Variable (with ui-sans-serif, system-ui)
**Label/Mono Font:** Geist Mono Variable (with ui-monospace, SF Mono)

**Character:** One neutral grotesque at small sizes and two weights (400/500 for text, 600 for titles). Mono is a data voice, never a style.

### Hierarchy
- **Headline** (600, 16px, tracking-tight): page title in the sticky header, one per page.
- **Title** (600, 14px): Panel titles, sentence case.
- **Body** (400, 14px): panel content, stat lists, tables, controls.
- **Label** (400-500, 12px): descriptions under titles, captions, badges, hints, button text in small buttons (0.8rem).
- **Data** (Geist Mono, 12px, tabular numerals): ids, frame numbers, coordinates, file paths, the run log.

### Named Rules
**The Sentence-Case Rule.** Titles and labels are sentence case at normal tracking. No uppercase tracked titles.

**The Mono-Is-Data Rule.** Geist Mono is used for values the operator reads as identifiers or numbers (P004, frame 212, 49.0, -31.6), never for headings, buttons or key hints (`kbd` is set back to the sans).

## Layout

A collapsible left sidebar (256px; icon rail 48px; below 768px it becomes an off-canvas sheet of 288px) holds the product name, output-directory switcher, the eight pipeline stages with status dot and index, the editors list, the last-run-log entry and the theme toggle. The content column opens with a sticky page header (sidebar trigger, title, status badge, one-line description; actions right, wrapping below the title on narrow screens), then a vertical stack of Panels with 16px gaps and 16px page padding (24px from 768px up).

Inside panels, rhythm is 8px between related controls, 12px between groups, 16px between blocks. Multi-column content uses responsive grids that start single-column and add columns at sm / md / xl. Tables and media go full-bleed inside a Panel (flush body); everything else gets 16px padding.

The run log docks at the bottom of the content column (sticky, max 45% of the viewport), not in a separate page. Mobile is a monitoring posture: the same pages stack, editors stay usable but are best on desktop.

## Elevation & Depth

Flat by default. Panels are separated from the page by a one-step tonal lift (card over background) and a 1px ring at 10% foreground, with no shadow. Borders do the rest: a hairline under the page header and under each Panel header.

### Shadow Vocabulary
- **Floating dock** (`box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.2), 0 4px 6px -4px rgb(0 0 0 / 0.2)`): the docked run log, which floats over scrolling content.
- **Canvas overlay** (`box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)` with an 80% background and backdrop blur): control cards that sit on top of the 3D viewer and editor canvases.

The sticky page header uses a 95% (80% where supported) background with backdrop blur so content scrolling beneath it stays legible.

### Named Rules
**The Only-Floaters-Cast Rule.** Shadows belong only to things that float over other content (log dock, overlays on media, popovers and dialogs from shadcn). Panels in the flow never cast a shadow.

## Shapes

Softly rounded, derived from one base radius of 10px. Panels and the log dock use 14px, buttons and inputs 10px (8px for small header buttons), list rows and small wells 6-8px, badges are pills, status dots and swatches are full circles. Dashed 1px borders mark drop targets only (file drop zone, "drag a shot here" slot). Media thumbnails clip to 10px corners inside the stage well.

## Components

### Buttons
Monochrome and compact; lucide icons lead the label at 16px (14px in small buttons).
- **Shape:** gently rounded (10px; 8px at small size), 32px tall (28px small, 24px xs).
- **Primary:** Chalk White fill with Ink text in dark, the inverse in light. At most one per header or dialog.
- **Outline:** input-tinted fill with a hairline border; the secondary action (Re-run clean, Open full screen).
- **Ghost:** transparent until hover (muted fill); tertiary actions such as Run all and log-dock tools.
- **Destructive:** Red Card text on a 10-20% Red Card tint, never a solid red block.
- **Hover / Focus:** primary fades to 80%, others fill with muted. Focus is a 3px ring at 50% of `ring`. Pressed buttons nudge down 1px. Disabled at 50% opacity, with the reason shown as visible text or a tooltip on a focusable wrapper.
- **Button group:** Re-run clean (outline) + Continue (primary) are joined; Continue is the non-destructive default.

### Status dot and badges
- **StatusDot** (8px circle): complete = solid success; partial = half-filled success with a success ring; running = pulsing warning; failed = destructive; not run = ring only, readable without colour.
- **StatusBadge** (pill, 20px, 12px text): the state hue at 15% fill, 25% border, full-strength text, with a dot inside. Labels: Complete, Partial, Running, Failed, Not run.
- **ToneBadge:** the same recipe for data quality (coverage %, px error), plus a muted variant for "not detected".

### Panels (Cards / Containers)
- **Corner Style:** 14px.
- **Background:** card token, 1px ring at 10% foreground.
- **Shadow Strategy:** none (see Elevation).
- **Header:** 12px x 16px, hairline bottom border, 14px semibold sentence-case title, optional 12px muted description, actions right.
- **Internal Padding:** 16px, or flush for tables, video and canvases.
- **States:** loading uses a panel-shaped skeleton; empty uses an icon, a title and a description of what to do next; failure uses a destructive alert.

### Inputs / Fields
- **Style:** 32px, 10px radius, input-tinted fill in dark, hairline border in light. Selects and native selects match.
- **Focus:** the shared 3px ring at 50% `ring`.
- **Error / Disabled:** destructive border and 20% destructive ring; 50% opacity when disabled.

### Navigation
- **Sidebar:** sidebar surface one step off the background. Group labels (Pipeline, Editors) in 12px medium at 70% foreground. Items are 14px, 32px rows with 8px radius; the active item fills with sidebar-accent. Pipeline items carry a StatusDot left and a mono index right. A live run replaces "Last run log" with a spinning warning loader and "<stage> running…".
- **Mobile:** below 768px the sidebar is a sheet opened from the header trigger.

### Run log dock (signature)
A floating Panel at the bottom of the content column: header row with terminal icon, "<Stage> log" title, StatusBadge, mono elapsed time, and ghost tools right (First error, Follow, copy, download, minimise, close). The body is the stage well in Geist Mono 12px at 85% stage-foreground; warnings render in warning, traceback lines in destructive on a destructive tint. On failure the dock border turns 50% destructive and a sonner error toast points to it.

### Stage well
Near-black `stage` surface for video, canvases, three.js and thumbnails. Canvas chrome (axes, labels) reads theme tokens at draw time; data marks use the player palette.

### Dialogs
Destructive actions go through a promise-based confirm dialog; re-running a stage first lists the generated outputs it will clear in a mono list, and stages holding operator edits require typing the stage name.

## Do's and Don'ts

### Do:
- **Do** use shadcn components from `components/ui` for every control and keep chrome on theme tokens (bg-card, text-muted-foreground, text-success).
- **Do** spend hue only on state (success, warning, destructive, info) and data (player palette, pitch drawing).
- **Do** put every video, canvas and 3D view in the `stage` well.
- **Do** wrap every content block in a Panel with a sentence-case title; give empty states a next step.
- **Do** keep one primary button per header or dialog; make the non-destructive path the primary.
- **Do** use Geist Mono with tabular numerals for ids, frames, coordinates and logs, and nowhere else.
- **Do** show why a control is disabled as visible text or a reachable tooltip.

### Don't:
- **Don't** colour buttons, titles or decorative accents; the primary stays monochrome.
- **Don't** nest Panels or add shadows to in-flow containers.
- **Don't** use uppercase tracked titles or labels.
- **Don't** use unicode glyphs as icons; icons are lucide.
- **Don't** use `alert()` or `window.confirm`; errors are toasts, destructive actions are confirm dialogs.
- **Don't** theme the player palette; a player's colour is identical in both themes and every panel.
