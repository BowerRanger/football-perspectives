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
/** Minimum CSS px for landmark labels; multiplied by `scale` to stay legible at any zoom. */
const LABEL_PX = 12

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

interface Rect {
  x: number
  y: number
  w: number
  h: number
}

function overlaps(a: Rect, b: Rect): boolean {
  return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h
}

/** Greedy placement: first of right/left/above/below that stays in frame and clears every placed label. */
function pickLabelRect(
  u: number,
  v: number,
  tw: number,
  th: number,
  gap: number,
  placed: readonly Rect[],
  o: OverlayInput,
): Rect | null {
  const candidates: Rect[] = [
    { x: u + gap, y: v - th / 2, w: tw, h: th },
    { x: u - gap - tw, y: v - th / 2, w: tw, h: th },
    { x: u - tw / 2, y: v - gap - th, w: tw, h: th },
    { x: u - tw / 2, y: v + gap, w: tw, h: th },
  ]
  return (
    candidates.find(
      (r) => r.x >= 0 && r.y >= 0 && r.x + r.w <= o.width && r.y + r.h <= o.height && !placed.some((p) => overlaps(r, p)),
    ) ?? null
  )
}

function drawLandmarkLabels(ctx: CanvasRenderingContext2D, o: OverlayInput, placed: Rect[]) {
  const cam = cameraForFrame(o.track, o.frame)
  if (!cam || o.landmarks.length === 0) return
  const dotR = 3 * o.scale
  for (const lm of o.landmarks) {
    const proj = projectPoint(lm.world_xyz, cam.K, cam.R, cam.t, cam.distortion)
    if (!proj) continue
    const [u, v] = proj
    if (u < 0 || u > o.width || v < 0 || v > o.height) continue
    ctx.beginPath()
    ctx.arc(u, v, dotR, 0, 2 * Math.PI)
    ctx.fillStyle = "rgba(255, 200, 0, 0.85)"
    ctx.fill()
    placeLabel(ctx, lm.name, u, v, dotR + 3 * o.scale, "rgba(255, 220, 120, 0.95)", o, placed)
  }
}

/** Draw `text` beside (u, v) on the first free side; skip it when every side collides. */
function placeLabel(
  ctx: CanvasRenderingContext2D,
  text: string,
  u: number,
  v: number,
  gap: number,
  fill: string,
  o: OverlayInput,
  placed: Rect[],
) {
  const th = LABEL_PX * o.scale
  const pad = 2 * o.scale
  ctx.save()
  ctx.font = `${th}px ${FONT}`
  ctx.textBaseline = "middle"
  ctx.lineWidth = 3 * o.scale
  const tw = ctx.measureText(text).width
  const rect = pickLabelRect(u, v, tw, th, gap, placed, o)
  if (rect) {
    placed.push({ x: rect.x - pad, y: rect.y - pad, w: rect.w + 2 * pad, h: rect.h + 2 * pad })
    const ty = rect.y + rect.h / 2
    ctx.strokeStyle = "rgba(0, 0, 0, 0.7)"
    ctx.strokeText(text, rect.x, ty)
    ctx.fillStyle = fill
    ctx.fillText(text, rect.x, ty)
  }
  ctx.restore()
}

function drawAnchorLines(ctx: CanvasRenderingContext2D, anchor: AnchorFrame, o: OverlayInput, placed: Rect[]) {
  const s = o.scale
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
    if (showNames(o)) placeLabel(ctx, ln.name, (x1 + x2) / 2, (y1 + y2) / 2, 6 * s, "#fef9c3", o, placed)
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

function drawAnchorPoints(ctx: CanvasRenderingContext2D, anchor: AnchorFrame, o: OverlayInput, placed: Rect[]) {
  const s = o.scale
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
    if (showNames(o)) placeLabel(ctx, lm.name, x, y, ringR + 4 * s, "#fff", o, placed)
  }
}

/** Below this rendered width (CSS px) point/line names pile up, so they show only when "landmark labels" is on. */
const NARROW_CSS_PX = 640

function showNames(o: OverlayInput): boolean {
  return o.width / o.scale >= NARROW_CSS_PX || o.view.labels
}

export function drawOverlay(canvas: HTMLCanvasElement, o: OverlayInput) {
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  if (o.view.pitch) drawProjectedPitch(ctx, o)
  if (o.view.detected) drawDetectedLines(ctx, o)
  // Shared label layout: marker rings are obstacles; anchor names claim space before catalogue labels.
  const placed: Rect[] = []
  if (o.view.anchors && o.anchor) {
    const r = 12 * o.scale
    for (const p of o.anchor.points) placed.push({ x: p.image_xy[0] - r, y: p.image_xy[1] - r, w: 2 * r, h: 2 * r })
  }
  if (o.view.anchors) {
    if (o.anchor) drawAnchorLines(ctx, o.anchor, o, placed)
    if (o.pendingLineStart) drawPendingStart(ctx, o.pendingLineStart, o.scale)
    if (o.anchor) drawAnchorPoints(ctx, o.anchor, o, placed)
  }
  if (o.view.labels) drawLandmarkLabels(ctx, o, placed)
}

/** Colour ramp for the coverage strip (red -> green by confidence). */
export function confidenceColour(confidence: number | undefined): string {
  const c01 = Math.max(0, Math.min(1, confidence ?? 0))
  return `rgb(${Math.round(255 * (1 - c01))},${Math.round(200 * c01 + 30)},60)`
}
