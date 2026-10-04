// Timeline painter: labelled rows on the reference timeline. Pure drawing +
// a hit-test helper, so the React shell stays thin.
import { cssVar } from "@/lib/format"
import {
  EVENT_STYLE,
  PIPELINE_GHOST,
  RESIDUAL_BAD_PX,
  RESIDUAL_REJECT_PX,
  RESIDUAL_WARN_PX,
  SEGMENT_STYLE,
  SEVERITY_CANVAS,
  residualSeverity,
  viewColour,
  viewLetter,
} from "./palette"
import type { SegmentKind, SolveResult, TruthDoc } from "./types"

export const GUTTER = 96
export const ROW_H = 26
const PAD = 4

export interface TimelineModel {
  doc: TruthDoc
  solved: SolveResult | null
  stale: boolean
  range: readonly [number, number]
  frame: number
  selectedKey: string | null
  selectedEvent: number | null
  hoverFrame: number | null
  views: { shotId: string; offset: number; nFrames: number; repeats?: readonly number[] }[]
  /** Pipeline track on the reference timeline (for the delta row). */
  pipeline: { frames: readonly number[]; xyz: readonly (readonly number[] | null)[] } | null
  keyKinds: ReadonlyMap<string, SegmentKind>
}

export interface Row {
  id: string
  label: string
  y: number
  h: number
}

export function layoutRows(m: Pick<TimelineModel, "views" | "pipeline" | "solved">): Row[] {
  const rows: Row[] = []
  let y = PAD
  const add = (id: string, label: string, h = ROW_H) => {
    rows.push({ id, label, y, h })
    y += h + 2
  }
  add("segments", "Segments")
  add("keys", "Keys, events")
  add("footage", "Footage", Math.max(ROW_H, m.views.length * 9 + 4))
  add("residual", "Residual")
  add("flags", "Flags")
  if (m.pipeline && m.solved) add("delta", "vs pipeline")
  return rows
}

export const timelineHeight = (m: Pick<TimelineModel, "views" | "pipeline" | "solved">): number => {
  const rows = layoutRows(m)
  const last = rows[rows.length - 1]
  return last.y + last.h + PAD
}

export function frameToX(frame: number, range: readonly [number, number], width: number): number {
  const span = Math.max(1, range[1] - range[0])
  return GUTTER + ((frame - range[0]) / span) * (width - GUTTER - 8)
}

export function xToFrame(x: number, range: readonly [number, number], width: number): number {
  const span = Math.max(1, range[1] - range[0])
  const f = range[0] + ((x - GUTTER) / (width - GUTTER - 8)) * span
  return Math.min(range[1], Math.max(range[0], Math.round(f)))
}

export type Hit = { type: "key"; id: string } | { type: "event"; index: number } | null

/** Which key/event marker (if any) sits under the pointer in the keys row. */
export function hitTest(m: TimelineModel, width: number, x: number, y: number): Hit {
  const row = layoutRows(m).find((r) => r.id === "keys")
  if (!row || y < row.y || y > row.y + row.h) return null
  let best: { hit: Hit; d: number } | null = null
  for (const k of m.doc.keys) {
    const d = Math.abs(frameToX(k.frame, m.range, width) - x)
    if (d <= 7 && (!best || d < best.d)) best = { hit: { type: "key", id: k.id }, d }
  }
  m.doc.events.forEach((e, i) => {
    const d = Math.abs(frameToX(e.frame, m.range, width) - x)
    if (d <= 6 && (!best || d < best.d - 1)) best = { hit: { type: "event", index: i }, d }
  })
  return (best as { hit: Hit } | null)?.hit ?? null
}

function hatch(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, colour: string) {
  ctx.save()
  ctx.beginPath()
  ctx.rect(x, y, w, h)
  ctx.clip()
  ctx.strokeStyle = colour
  ctx.lineWidth = 1
  ctx.beginPath()
  for (let i = -h; i < w; i += 6) {
    ctx.moveTo(x + i, y + h)
    ctx.lineTo(x + i + h, y)
  }
  ctx.stroke()
  ctx.restore()
}

function eventGlyph(ctx: CanvasRenderingContext2D, glyph: string, x: number, y: number, colour: string) {
  ctx.strokeStyle = colour
  ctx.fillStyle = colour
  ctx.lineWidth = 2
  ctx.beginPath()
  switch (glyph) {
    case "circle":
      ctx.arc(x, y, 4, 0, Math.PI * 2)
      ctx.fill()
      break
    case "ring":
      ctx.arc(x, y, 4, 0, Math.PI * 2)
      ctx.stroke()
      break
    case "bar":
      ctx.moveTo(x, y - 5)
      ctx.lineTo(x, y + 5)
      ctx.moveTo(x - 3, y - 5)
      ctx.lineTo(x + 3, y - 5)
      ctx.stroke()
      break
    case "line":
      ctx.moveTo(x - 5, y)
      ctx.lineTo(x + 5, y)
      ctx.moveTo(x, y - 5)
      ctx.lineTo(x, y + 5)
      ctx.stroke()
      break
    case "hand":
      ctx.moveTo(x - 4, y + 4)
      ctx.lineTo(x - 4, y - 3)
      ctx.moveTo(x, y + 4)
      ctx.lineTo(x, y - 5)
      ctx.moveTo(x + 4, y + 4)
      ctx.lineTo(x + 4, y - 3)
      ctx.stroke()
      break
    default:
      ctx.moveTo(x - 4, y - 4)
      ctx.lineTo(x + 4, y + 4)
      ctx.moveTo(x + 4, y - 4)
      ctx.lineTo(x - 4, y + 4)
      ctx.stroke()
  }
}

export function drawTimeline(canvas: HTMLCanvasElement, width: number, height: number, dpr: number, m: TimelineModel): void {
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, width, height)
  const fg = cssVar("--stage-foreground", "#f5f5f5")
  const muted = "rgba(245,245,245,0.62)"
  const border = "rgba(245,245,245,0.22)"
  const rows = layoutRows(m)
  const X = (f: number) => frameToX(f, m.range, width)
  const rowOf = (id: string) => rows.find((r) => r.id === id)

  ctx.font = '12px "Geist Variable", ui-sans-serif, system-ui, sans-serif'
  ctx.textBaseline = "middle"
  for (const r of rows) {
    ctx.fillStyle = muted
    ctx.textAlign = "left"
    ctx.fillText(r.label, 4, r.y + Math.min(r.h, ROW_H) / 2)
    ctx.fillStyle = border
    ctx.globalAlpha = 0.35
    ctx.fillRect(GUTTER, r.y, width - GUTTER - 8, r.h)
    ctx.globalAlpha = 1
  }

  // Segments row
  const seg = rowOf("segments")!
  const keys = m.doc.keys
  if (keys.length >= 2) {
    hatch(ctx, X(keys[0].frame), seg.y, X(keys[keys.length - 1].frame) - X(keys[0].frame), seg.h, border)
  }
  ctx.globalAlpha = m.stale ? 0.5 : 1
  for (const s of m.solved?.segments ?? []) {
    const st = SEGMENT_STYLE[s.kind]
    const x0 = X(s.frame_range[0])
    const x1 = X(s.frame_range[1])
    const y = seg.y + 4
    const h = seg.h - 8
    ctx.fillStyle = st.colour
    if (st.pattern === "ring") {
      ctx.strokeStyle = st.colour
      ctx.lineWidth = 2
      ctx.strokeRect(x0, y, Math.max(2, x1 - x0), h)
    } else {
      ctx.fillRect(x0, y, Math.max(2, x1 - x0), h)
      if (st.pattern !== "solid") {
        ctx.fillStyle = cssVar("--stage", "#111")
        const step = st.pattern === "dotted" ? 4 : 8
        for (let x = x0 + 2; x < x1; x += step * 1.6) ctx.fillRect(x, y, step * 0.5, h)
      }
    }
    if (s.auto) {
      ctx.fillStyle = "rgba(0,0,0,0.55)"
      ctx.font = '10px "Geist Variable", ui-sans-serif'
      ctx.textAlign = "left"
      if (x1 - x0 > 34) ctx.fillText("auto", x0 + 4, seg.y + seg.h / 2)
    }
  }
  ctx.globalAlpha = 1

  // Keys and events
  const kr = rowOf("keys")!
  const cy = kr.y + kr.h / 2
  for (const e of m.doc.events.map((ev, i) => ({ ev, i }))) {
    const st = EVENT_STYLE[e.ev.kind]
    eventGlyph(ctx, st.glyph, X(e.ev.frame), kr.y + kr.h - 8, st.colour)
    if (m.selectedEvent === e.i) {
      ctx.strokeStyle = "#60a5fa"
      ctx.lineWidth = 2
      ctx.strokeRect(X(e.ev.frame) - 7, kr.y + kr.h - 15, 14, 14)
    }
  }
  for (const k of keys) {
    const x = X(k.frame)
    const colour = SEGMENT_STYLE[m.keyKinds.get(k.id) ?? "flight"].colour
    const bad = (m.solved?.keys.find((s) => s.id === k.id)?.status ?? "ok") === "error"
    ctx.beginPath()
    ctx.moveTo(x, cy - 12 + 4)
    ctx.lineTo(x + 5, cy - 4)
    ctx.lineTo(x, cy + 2)
    ctx.lineTo(x - 5, cy - 4)
    ctx.closePath()
    ctx.fillStyle = colour
    ctx.fill()
    ctx.lineWidth = bad ? 2 : 1
    ctx.strokeStyle = bad ? SEVERITY_CANVAS.destructive : fg
    ctx.stroke()
    if (m.selectedKey === k.id) {
      ctx.strokeStyle = "#60a5fa"
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(x, cy - 4, 9, 0, Math.PI * 2)
      ctx.stroke()
    }
  }

  // Footage: one thin strip per view, ticks where it has a pick
  const fr = rowOf("footage")!
  m.views.forEach((v, i) => {
    const y = fr.y + 3 + i * 9
    const a = Math.max(m.range[0], 0 - v.offset)
    const b = Math.min(m.range[1], v.nFrames - 1 - v.offset)
    hatch(ctx, GUTTER, y, width - GUTTER - 8, 6, border)
    ctx.fillStyle = viewColour(i)
    ctx.globalAlpha = 0.45
    if (b > a) ctx.fillRect(X(a), y, X(b) - X(a), 6)
    ctx.globalAlpha = 1
    ctx.fillStyle = viewColour(i)
    ctx.font = '600 9px "Geist Mono Variable", ui-monospace'
    ctx.textAlign = "right"
    ctx.fillText(viewLetter(i), GUTTER - 4, y + 3)
    // Repeated (pulldown) frames: dark ticks in the strip.
    if (v.repeats?.length) {
      ctx.fillStyle = "rgba(0,0,0,0.55)"
      for (const sf of v.repeats) {
        const r = sf - v.offset
        if (r >= m.range[0] && r <= m.range[1]) ctx.fillRect(X(r), y, Math.max(1, X(r + 1) - X(r)), 6)
      }
      ctx.fillStyle = viewColour(i)
    }
    const mark = (shotFrame: number) => {
      ctx.fillRect(X(shotFrame - v.offset) - 1, y - 2, 2, 10)
    }
    for (const k of m.doc.keys) for (const o of k.observations) if (o.shot_id === v.shotId) mark(o.shot_frame)
    for (const o of m.doc.observations) if (o.shot_id === v.shotId) mark(o.shot_frame)
  })

  // Residual sparkline with severity bands
  const rr = rowOf("residual")!
  const maxPx = 20
  const Y = (px: number) => rr.y + rr.h - 2 - (Math.min(px, maxPx) / maxPx) * (rr.h - 4)
  for (const [px, c] of [
    [RESIDUAL_WARN_PX, SEVERITY_CANVAS.success],
    [RESIDUAL_BAD_PX, SEVERITY_CANVAS.warning],
    [RESIDUAL_REJECT_PX, SEVERITY_CANVAS.destructive],
  ] as const) {
    ctx.strokeStyle = c
    ctx.globalAlpha = 0.35
    ctx.setLineDash([2, 3])
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(GUTTER, Y(px))
    ctx.lineTo(width - 8, Y(px))
    ctx.stroke()
  }
  ctx.setLineDash([])
  ctx.globalAlpha = 1
  const worst = new Map<number, number>()
  for (const o of m.solved?.observations ?? []) {
    if (o.residual_px === null) continue
    const view = m.views.find((v) => v.shotId === o.shot_id)
    if (!view) continue
    const ref = o.shot_frame - view.offset
    worst.set(ref, Math.max(worst.get(ref) ?? 0, o.residual_px))
  }
  const pts = [...worst.entries()].sort((a, b) => a[0] - b[0])
  if (pts.length) {
    ctx.strokeStyle = muted
    ctx.lineWidth = 1
    ctx.beginPath()
    pts.forEach(([f, px], i) => (i ? ctx.lineTo(X(f), Y(px)) : ctx.moveTo(X(f), Y(px))))
    ctx.stroke()
    for (const [f, px] of pts) {
      ctx.fillStyle = SEVERITY_CANVAS[residualSeverity(px)]
      ctx.beginPath()
      ctx.arc(X(f), Y(px), 2.5, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  // Flags
  const fl = rowOf("flags")!
  for (const f of m.solved?.flags ?? []) {
    const segRange = f.ref?.segment !== undefined ? m.solved?.segments[f.ref.segment]?.frame_range : undefined
    const a = f.frame ?? segRange?.[0]
    if (a === undefined) continue
    const b = f.frame !== undefined ? f.frame : (segRange?.[1] ?? a)
    ctx.fillStyle = f.level === "error" ? SEVERITY_CANVAS.destructive : SEVERITY_CANVAS.warning
    ctx.fillRect(X(a) - 1.5, fl.y + 4, Math.max(3, X(b) - X(a) + 3), fl.h - 8)
  }
  for (const s of m.solved?.segments ?? []) {
    if (s.n_soft_obs === 0 && s.kind === "flight") {
      ctx.strokeStyle = SEVERITY_CANVAS.warning
      ctx.setLineDash([3, 3])
      ctx.lineWidth = 1
      ctx.strokeRect(X(s.frame_range[0]), fl.y + 4, Math.max(3, X(s.frame_range[1]) - X(s.frame_range[0])), fl.h - 8)
      ctx.setLineDash([])
    }
  }

  // Pipeline delta
  const dr = rowOf("delta")
  if (dr && m.pipeline && m.solved) {
    const dense = new Map<number, readonly number[]>()
    m.solved.dense.frames.forEach((f, i) => dense.set(f, m.solved!.dense.xyz[i]))
    ctx.fillStyle = PIPELINE_GHOST.colour
    ctx.globalAlpha = 0.6
    m.pipeline.frames.forEach((f, i) => {
      const p = m.pipeline!.xyz[i]
      const q = dense.get(f)
      if (!p || !q) return
      const d = Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2])
      const h = Math.min(dr.h - 4, (Math.min(d, 2) / 2) * (dr.h - 4))
      ctx.fillRect(X(f), dr.y + dr.h - 2 - h, Math.max(1, X(f + 1) - X(f)), h)
    })
    ctx.globalAlpha = 1
  }

  // Hover guide and playhead
  if (m.hoverFrame !== null) {
    ctx.strokeStyle = muted
    ctx.globalAlpha = 0.6
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(X(m.hoverFrame) + 0.5, 0)
    ctx.lineTo(X(m.hoverFrame) + 0.5, height)
    ctx.stroke()
    ctx.globalAlpha = 1
  }
  ctx.strokeStyle = fg
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(X(m.frame), 0)
  ctx.lineTo(X(m.frame), height)
  ctx.stroke()
  ctx.fillStyle = fg
  ctx.font = '600 11px "Geist Mono Variable", ui-monospace'
  ctx.textAlign = X(m.frame) > width - 50 ? "right" : "left"
  ctx.fillText(String(m.frame), X(m.frame) + (X(m.frame) > width - 50 ? -5 : 5), 8)
}
