// Single source for Ball Studio's data colours, glyphs and severity bands
// (canvas views, timeline and 3-D all read from here). Chrome uses theme
// tokens; hex is allowed here because these are DATA colours.
import type { EventKind, KeySource, SegmentKind } from "./types"

/** View identity: rays, frusta, epipolar lines, view badges (max 3). Always paired with a letter badge. */
export const VIEW_COLOURS = ["#f0abfc", "#5eead4", "#93c5fd"] as const
export const VIEW_LETTERS = ["A", "B", "C"] as const

export const viewColour = (i: number): string => VIEW_COLOURS[Math.min(Math.max(i, 0), VIEW_COLOURS.length - 1)]
export const viewLetter = (i: number): string => VIEW_LETTERS[Math.min(Math.max(i, 0), VIEW_LETTERS.length - 1)]

export type StrokePattern = "solid" | "dotted" | "dashed" | "ring"

export interface SegmentStyle {
  colour: string
  width: number
  pattern: StrokePattern
  label: string
}

export const SEGMENT_KINDS: readonly SegmentKind[] = ["flight", "roll", "carried", "linear", "static"]

export const SEGMENT_STYLE: Record<SegmentKind, SegmentStyle> = {
  flight: { colour: "#38bdf8", width: 3, pattern: "solid", label: "Flight" },
  roll: { colour: "#34d399", width: 2, pattern: "solid", label: "Roll" },
  carried: { colour: "#a78bfa", width: 2, pattern: "dotted", label: "Carried" },
  linear: { colour: "#94a3b8", width: 2, pattern: "dashed", label: "Linear" },
  static: { colour: "#64748b", width: 2, pattern: "ring", label: "Static" },
}

export type KeyMarker = "diamond-filled" | "diamond-hollow" | "diamond-tether" | "square"

export const KEY_SOURCES: readonly KeySource[] = [
  "triangulated",
  "ray_ground",
  "ray_height",
  "ray_plane",
  "ray_depth",
  "player",
  "manual",
]

export const KEY_SOURCE_STYLE: Record<KeySource, { marker: KeyMarker; glyph: string; label: string }> = {
  triangulated: { marker: "diamond-filled", glyph: "", label: "Triangulated" },
  ray_ground: { marker: "diamond-hollow", glyph: "_", label: "Ray on ground" },
  ray_height: { marker: "diamond-hollow", glyph: "h", label: "Ray at height" },
  ray_plane: { marker: "diamond-hollow", glyph: "|", label: "Ray on plane" },
  ray_depth: { marker: "diamond-hollow", glyph: ">", label: "Ray at depth" },
  player: { marker: "diamond-tether", glyph: "", label: "Player joint" },
  manual: { marker: "square", glyph: "", label: "Manual 3-D" },
}

export const EVENT_KINDS: readonly EventKind[] = [
  "touch",
  "bounce",
  "post",
  "crossbar",
  "net",
  "line_cross",
  "keeper_save",
  "out",
]

export type EventGlyph = "circle" | "ring" | "bar" | "line" | "hand" | "x"

/** Reuses the ball-anchor tag colours so one vocabulary spans both editors. */
export const EVENT_STYLE: Record<EventKind, { colour: string; glyph: EventGlyph; label: string }> = {
  touch: { colour: "#22d3ee", glyph: "circle", label: "Touch" },
  bounce: { colour: "#f472b6", glyph: "ring", label: "Bounce" },
  post: { colour: "#f59e0b", glyph: "bar", label: "Post" },
  crossbar: { colour: "#f59e0b", glyph: "bar", label: "Crossbar" },
  net: { colour: "#f59e0b", glyph: "bar", label: "Net" },
  line_cross: { colour: "#a3e635", glyph: "line", label: "Goal line cross" },
  keeper_save: { colour: "#a78bfa", glyph: "hand", label: "Keeper save" },
  out: { colour: "#94a3b8", glyph: "x", label: "Out of play" },
}

/** Pipeline output is always a neutral ghost so the authored truth stays dominant. */
export const PIPELINE_GHOST = { colour: "#e2e8f0", alpha: 0.55, width: 1.5 } as const

/** Reprojection residual bands (px). >15 px is a hard reject at the server. */
export const RESIDUAL_WARN_PX = 3
export const RESIDUAL_BAD_PX = 8
export const RESIDUAL_REJECT_PX = 15

export type Severity = "success" | "warning" | "destructive"

export function residualSeverity(px: number): Severity {
  if (px <= RESIDUAL_WARN_PX) return "success"
  if (px <= RESIDUAL_BAD_PX) return "warning"
  return "destructive"
}

/** Canvas-side severity colours (the same hues as the success/warning/destructive tokens). */
export const SEVERITY_CANVAS: Record<Severity, string> = {
  success: "#4ade80",
  warning: "#fbbf24",
  destructive: "#f87171",
}

export const WORST_RESIDUAL = (r: Record<string, number> | undefined): number | null => {
  if (!r) return null
  const v = Object.values(r)
  return v.length ? Math.max(...v) : null
}
