// Canvas overlay painters for the frame well. Colours are data colours
// (anchor tag colours, state colours) so they stay legible on any frame.

import { tagColour } from "./tags"
import type { Layers, PredictedPoint } from "./use-ball-anchor-editor"
import type { AutoAnchor, BallAnchor } from "./types"

export interface OverlayScene {
  width: number
  height: number
  frame: number
  anchors: BallAnchor[]
  autoAnchors: AutoAnchor[]
  layers: Layers
  predictedByFrame: Map<number, PredictedPoint>
  previewByFrame: Map<number, [number, number]>
}

const PRED_COLOURS: Record<string, string> = { flight: "#fb923c", grounded: "#34d399" }

function ring(ctx: CanvasRenderingContext2D, uv: [number, number], r: number) {
  ctx.beginPath()
  ctx.arc(uv[0], uv[1], r, 0, Math.PI * 2)
  ctx.stroke()
}

function dot(ctx: CanvasRenderingContext2D, uv: [number, number], r: number) {
  ctx.beginPath()
  ctx.arc(uv[0], uv[1], r, 0, Math.PI * 2)
  ctx.fill()
}

function drawPredicted(ctx: CanvasRenderingContext2D, p: PredictedPoint, k: number) {
  const colour = PRED_COLOURS[p.state] ?? "#94a3b8"
  const [u, v] = p.uv
  ctx.strokeStyle = colour
  ctx.fillStyle = colour
  ctx.lineWidth = 2 * k
  ring(ctx, p.uv, 10 * k)
  dot(ctx, p.uv, 3 * k)
  ctx.beginPath()
  for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
    ctx.moveTo(u + dx * 4 * k, v + dy * 4 * k)
    ctx.lineTo(u + dx * 16 * k, v + dy * 16 * k)
  }
  ctx.stroke()
  const label = `${p.state} z=${p.z.toFixed(2)}m`
  ctx.font = `bold ${Math.round(13 * k)}px system-ui, sans-serif`
  const tw = ctx.measureText(label).width + 10 * k
  ctx.fillStyle = "rgba(0,0,0,0.75)"
  ctx.fillRect(u - tw / 2, v - 34 * k, tw, 17 * k)
  ctx.fillStyle = colour
  ctx.fillText(label, u - tw / 2 + 5 * k, v - 21 * k)
}

function drawAnchors(ctx: CanvasRenderingContext2D, s: OverlayScene, k: number) {
  for (const a of s.autoAnchors) {
    if (a.frame !== s.frame || !a.image_xy) continue
    ctx.save()
    ctx.strokeStyle = tagColour(a.state)
    ctx.globalAlpha = 0.8
    ctx.lineWidth = 2 * k
    ctx.setLineDash([5 * k, 4 * k])
    ring(ctx, a.image_xy, 12 * k)
    ctx.restore()
  }
  for (const a of s.anchors) {
    if (a.frame !== s.frame || !a.image_xy) continue
    const c = tagColour(a.state, "#ffffff")
    ctx.strokeStyle = c
    ctx.fillStyle = c
    ctx.lineWidth = 2 * k
    ring(ctx, a.image_xy, 12 * k)
    dot(ctx, a.image_xy, 3 * k)
  }
}

export function drawOverlay(canvas: HTMLCanvasElement, s: OverlayScene): void {
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  const k = Math.max(1, s.width / 1280)
  const predicted = s.layers.predicted ? s.predictedByFrame.get(s.frame) : undefined
  if (predicted) drawPredicted(ctx, predicted, k)
  if (s.layers.anchors) drawAnchors(ctx, s, k)
  const preview = s.layers.preview ? s.previewByFrame.get(s.frame) : undefined
  if (preview) {
    ctx.strokeStyle = "rgba(255,255,255,0.7)"
    ctx.lineWidth = 1.5 * k
    ring(ctx, preview, 14 * k)
  }
}

/** Ball-quality strip painter. Returns nothing; draws into `canvas`. */
export function drawQualityStrip(
  canvas: HTMLCanvasElement,
  quality: import("./types").BallQuality | null,
  anchors: BallAnchor[],
  frame: number,
  emptyLabel: string,
  textColour: string,
  bgColour: string,
): void {
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  const w = canvas.width
  const h = canvas.height
  ctx.fillStyle = bgColour
  ctx.fillRect(0, 0, w, h)
  const n = quality?.n_frames ?? 0
  if (!quality || !n) {
    ctx.fillStyle = textColour
    ctx.font = "12px system-ui, sans-serif"
    ctx.textAlign = "center"
    ctx.fillText(emptyLabel, w / 2, h / 2 + 4)
    return
  }
  const barW = Math.max(1, w / n)
  for (const f of quality.frames ?? []) {
    const c01 = f.gap_fill ? 0 : Math.max(0, Math.min(1, f.confidence ?? 0))
    ctx.fillStyle = `rgb(${Math.round(255 * (1 - c01))},${Math.round(200 * c01 + 30)},60)`
    ctx.fillRect(f.frame * barW, 0, Math.ceil(barW), h)
  }
  for (const it of quality.annotate_next ?? []) {
    ctx.fillStyle = it.reason === "underconstrained_flight" ? "rgba(239,68,68,0.7)" : "rgba(251,146,60,0.65)"
    ctx.fillRect(it.start * barW, h - 7, (it.end - it.start + 1) * barW, 7)
  }
  ctx.fillStyle = "#38bdf8"
  for (const e of quality.events ?? []) ctx.fillRect(e.frame * barW, 0, Math.max(1, barW), 4)
  ctx.fillStyle = "#a855f7"
  for (const a of anchors) ctx.fillRect(a.frame * barW, 0, Math.max(1, barW), 4)
  const cx = (frame / Math.max(1, n - 1)) * w
  ctx.fillStyle = "rgba(255,255,255,0.95)"
  ctx.fillRect(Math.max(0, cx - 1), 0, 2, h)
}
