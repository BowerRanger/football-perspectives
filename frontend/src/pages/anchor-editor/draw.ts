// Canvas overlay drawing. Backing store is in image pixels, so every size is
// multiplied by `scale` (image px per displayed px) to stay crisp at any zoom.
// Colours here are data colours (overlay legibility over video), not chrome.
import { cameraForFrame, PITCH_POLYLINES, projectPoint } from "./projection"
import type {
  AnchorFrame,
  CameraTrack,
  DetectedLinesByFrame,
  Landmark,
  Vec2,
  ViewOptions,
} from "./types"

const FONT = "-apple-system, Segoe UI, sans-serif"
const AMBER = "#facc15"
const CYAN = "#22d3ee"

export interface OverlayInput {
  width: number
  height: number
  scale: number
  frame: number
  view: ViewOptions
  anchor: AnchorFrame | undefined
  track: CameraTrack | null
  detected: DetectedLinesByFrame
  landmarks: readonly Landmark[]
  pendingLineStart: Vec2 | null
}

function drawProjectedPitch(ctx: CanvasRenderingContext2D, o: OverlayInput) {
  const cam = cameraForFrame(o.track, o.frame)
  if (!cam) return
  ctx.lineWidth = 1.5 * o.scale
  ctx.strokeStyle = "rgba(255, 200, 0, 0.55)"
  for (const poly of PITCH_POLYLINES) {
    ctx.beginPath()
    let drawing = false
    for (const p of poly) {
      const proj = projectPoint(p, cam.K, cam.R, cam.t, cam.distortion)
      if (!proj) {
        drawing = false
        continue
      }
      if (drawing) ctx.lineTo(proj[0], proj[1])
      else {
        ctx.moveTo(proj[0], proj[1])
        drawing = true
      }
    }
    ctx.stroke()
  }
}

function drawDetectedLines(ctx: CanvasRenderingContext2D, o: OverlayInput) {
  const lines = o.detected[String(o.frame)]?.lines
  if (!lines || lines.length === 0) return
  ctx.save()
  ctx.lineWidth = 2 * o.scale
  ctx.strokeStyle = "rgba(34, 211, 238, 0.9)"
  for (const ln of lines) {
    const [[x1, y1], [x2, y2]] = ln.image_segment
    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.stroke()
  }
  ctx.restore()
}

function drawLandmarkLabels(ctx: CanvasRenderingContext2D, o: OverlayInput) {
  const cam = cameraForFrame(o.track, o.frame)
  if (!cam || o.landmarks.length === 0) return
  const dotR = 3 * o.scale
  ctx.font = `${11 * o.scale}px ${FONT}`
  ctx.lineWidth = 3 * o.scale
  for (const lm of o.landmarks) {
    const proj = projectPoint(lm.world_xyz, cam.K, cam.R, cam.t, cam.distortion)
    if (!proj) continue
    const [u, v] = proj
    if (u < 0 || u > o.width || v < 0 || v > o.height) continue
    ctx.beginPath()
    ctx.arc(u, v, dotR, 0, 2 * Math.PI)
    ctx.fillStyle = "rgba(255, 200, 0, 0.85)"
    ctx.fill()
    ctx.strokeStyle = "rgba(0, 0, 0, 0.7)"
    ctx.strokeText(lm.name, u + dotR + 3 * o.scale, v - 3 * o.scale)
    ctx.fillStyle = "rgba(255, 220, 120, 0.95)"
    ctx.fillText(lm.name, u + dotR + 3 * o.scale, v - 3 * o.scale)
  }
}

function labelled(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, fill: string, scale: number) {
  ctx.strokeStyle = "rgba(0,0,0,0.7)"
  ctx.lineWidth = 3 * scale
  ctx.strokeText(text, x, y)
  ctx.fillStyle = fill
  ctx.fillText(text, x, y)
}

function drawAnchorLines(ctx: CanvasRenderingContext2D, anchor: AnchorFrame, s: number) {
  const dotR = 4 * s
  for (const ln of anchor.lines) {
    const [[x1, y1], [x2, y2]] = ln.image_segment
    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.lineWidth = 3 * s
    ctx.strokeStyle = AMBER
    ctx.stroke()
    for (const [x, y] of [[x1, y1], [x2, y2]]) {
      ctx.beginPath()
      ctx.arc(x, y, dotR * 0.8, 0, 2 * Math.PI)
      ctx.fillStyle = AMBER
      ctx.fill()
    }
    ctx.font = `${11 * s}px ${FONT}`
    labelled(ctx, ln.name, (x1 + x2) / 2 + 6 * s, (y1 + y2) / 2 - 4 * s, "#fef9c3", s)
  }
}

function drawPendingStart(ctx: CanvasRenderingContext2D, start: Vec2, s: number) {
  const [px, py] = start
  ctx.beginPath()
  ctx.arc(px, py, 4 * s, 0, 2 * Math.PI)
  ctx.fillStyle = AMBER
  ctx.fill()
  ctx.beginPath()
  ctx.arc(px, py, 8 * s, 0, 2 * Math.PI)
  ctx.lineWidth = 2 * s
  ctx.strokeStyle = "#fef08a"
  ctx.stroke()
}

function drawAnchorPoints(ctx: CanvasRenderingContext2D, anchor: AnchorFrame, s: number) {
  const dotR = 4 * s
  const ringR = 8 * s
  for (const lm of anchor.points) {
    const [x, y] = lm.image_xy
    ctx.beginPath()
    ctx.arc(x, y, dotR, 0, 2 * Math.PI)
    ctx.fillStyle = CYAN
    ctx.fill()
    ctx.beginPath()
    ctx.arc(x, y, ringR, 0, 2 * Math.PI)
    ctx.lineWidth = 2 * s
    ctx.strokeStyle = "#fff"
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(x - ringR * 1.5, y)
    ctx.lineTo(x + ringR * 1.5, y)
    ctx.moveTo(x, y - ringR * 1.5)
    ctx.lineTo(x, y + ringR * 1.5)
    ctx.lineWidth = s
    ctx.strokeStyle = "rgba(255, 255, 255, 0.6)"
    ctx.stroke()
    ctx.font = `${12 * s}px ${FONT}`
    labelled(ctx, lm.name, x + ringR + 4 * s, y - 4 * s, "#fff", s)
  }
}

export function drawOverlay(canvas: HTMLCanvasElement, o: OverlayInput) {
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  if (o.view.pitch) drawProjectedPitch(ctx, o)
  if (o.view.detected) drawDetectedLines(ctx, o)
  if (o.view.labels) drawLandmarkLabels(ctx, o)
  if (!o.view.anchors) return
  if (o.anchor) drawAnchorLines(ctx, o.anchor, o.scale)
  if (o.pendingLineStart) drawPendingStart(ctx, o.pendingLineStart, o.scale)
  if (o.anchor) drawAnchorPoints(ctx, o.anchor, o.scale)
}

/** Colour ramp for the coverage strip (red -> green by confidence). */
export function confidenceColour(confidence: number | undefined): string {
  const c01 = Math.max(0, Math.min(1, confidence ?? 0))
  return `rgb(${Math.round(255 * (1 - c01))},${Math.round(200 * c01 + 30)},60)`
}
