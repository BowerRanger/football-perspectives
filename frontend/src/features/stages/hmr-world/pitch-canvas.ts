// Top-down pitch drawing for the trajectory panel. Pitch metres in, canvas
// pixels out; y is flipped (y_world = 68 is the far touchline, canvas top).
// All colours here are data/illustration colours, not theme chrome.

import type { CameraSample } from "./types"

export const PITCH_L = 105
export const PITCH_W = 68
const MARGIN_X = 8
const MARGIN_FAR = 5
const MARGIN_NEAR = 30 // room below the near touchline so the camera marker lands on canvas
const SCALE = 6
const PADDING = 20

export const CANVAS_W = (PITCH_L + 2 * MARGIN_X) * SCALE + PADDING * 2
export const CANVAS_H = (PITCH_W + MARGIN_FAR + MARGIN_NEAR) * SCALE + PADDING * 2

const LINE = "#e2e8f0"

export function pitch2px(x: number, y: number): [number, number] {
  return [PADDING + (x + MARGIN_X) * SCALE, PADDING + (PITCH_W + MARGIN_FAR - y) * SCALE]
}

function dot(ctx: CanvasRenderingContext2D, x: number, y: number, r: number) {
  ctx.beginPath()
  ctx.arc(x, y, r, 0, Math.PI * 2)
  ctx.fill()
}

function rectPath(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number) {
  const [a, b] = pitch2px(x0, y0)
  const [c, d] = pitch2px(x1, y1)
  ctx.beginPath()
  ctx.rect(Math.min(a, c), Math.min(b, d), Math.abs(c - a), Math.abs(d - b))
  ctx.stroke()
}

function drawEnd(ctx: CanvasRenderingContext2D, left: boolean) {
  const PEN_DEPTH = 16.5
  const PEN_HALF = 20.16
  const SIX_DEPTH = 5.5
  const SIX_HALF = 9.16
  const SPOT = 11
  const ARC_R = 9.15
  const GOAL_HALF = 3.66
  const yMid = PITCH_W / 2
  const halfAngle = Math.acos((PEN_DEPTH - SPOT) / ARC_R)
  const xGoal = left ? 0 : PITCH_L
  const sgn = left ? 1 : -1

  rectPath(ctx, xGoal, yMid - PEN_HALF, xGoal + sgn * PEN_DEPTH, yMid + PEN_HALF)
  rectPath(ctx, xGoal, yMid - SIX_HALF, xGoal + sgn * SIX_DEPTH, yMid + SIX_HALF)
  const xSpot = xGoal + sgn * SPOT
  const [sx, sy] = pitch2px(xSpot, yMid)
  ctx.fillStyle = LINE
  dot(ctx, sx, sy, 1.6)

  // Penalty "D": the part of the 9.15 m circle outside the box.
  const startAng = left ? -halfAngle : Math.PI - halfAngle
  const endAng = left ? halfAngle : Math.PI + halfAngle
  const SEGMENTS = 32
  ctx.beginPath()
  for (let i = 0; i <= SEGMENTS; i++) {
    const a = startAng + (endAng - startAng) * (i / SEGMENTS)
    const [px, py] = pitch2px(xSpot + ARC_R * Math.cos(a), yMid + ARC_R * Math.sin(a))
    if (i === 0) ctx.moveTo(px, py)
    else ctx.lineTo(px, py)
  }
  ctx.stroke()

  // Goal mouth as a thicker stub so the two ends read differently.
  ctx.lineWidth = 3
  const [ax, ay] = pitch2px(xGoal, yMid - GOAL_HALF)
  const [bx, by] = pitch2px(xGoal, yMid + GOAL_HALF)
  ctx.beginPath()
  ctx.moveTo(ax, ay)
  ctx.lineTo(bx, by)
  ctx.stroke()
  ctx.lineWidth = 1.5
}

export function drawPitch(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = "#0a1f10"
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H)
  ctx.fillStyle = "#0b3b1a"
  const [tlx, tly] = pitch2px(0, PITCH_W)
  ctx.fillRect(tlx, tly, PITCH_L * SCALE, PITCH_W * SCALE)
  ctx.strokeStyle = LINE
  ctx.lineWidth = 1.5

  const corners: [number, number][] = [[0, 0], [PITCH_L, 0], [PITCH_L, PITCH_W], [0, PITCH_W]]
  ctx.beginPath()
  corners.forEach(([cx, cy], i) => {
    const [x, y] = pitch2px(cx, cy)
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.closePath()
  ctx.stroke()

  const m1 = pitch2px(PITCH_L / 2, 0)
  const m2 = pitch2px(PITCH_L / 2, PITCH_W)
  ctx.beginPath()
  ctx.moveTo(m1[0], m1[1])
  ctx.lineTo(m2[0], m2[1])
  ctx.stroke()
  const c = pitch2px(PITCH_L / 2, PITCH_W / 2)
  ctx.beginPath()
  ctx.arc(c[0], c[1], 9.15 * SCALE, 0, Math.PI * 2)
  ctx.stroke()
  ctx.fillStyle = LINE
  dot(ctx, c[0], c[1], 1.6)

  drawEnd(ctx, true)
  drawEnd(ctx, false)
}

/** Camera position + heading; flags an out-of-bounds solve (mirror-solution tell). */
export function drawCameraMarker(ctx: CanvasRenderingContext2D, cam: CameraSample) {
  const [cx, cy] = pitch2px(cam.pos[0], cam.pos[1])
  let [fx, fy] = cam.fwd
  const mag = Math.hypot(fx, fy) || 1
  fx /= mag
  fy /= mag
  const dx = fx
  const dy = -fy // pitch2px flips y
  const size = 10
  ctx.fillStyle = "#fde68a"
  ctx.strokeStyle = "#854d0e"
  ctx.lineWidth = 1.5
  ctx.beginPath()
  ctx.moveTo(cx + dx * size, cy + dy * size)
  ctx.lineTo(cx - dx * 5 - dy * 6, cy - dy * 5 + dx * 6)
  ctx.lineTo(cx - dx * 5 + dy * 6, cy - dy * 5 - dx * 6)
  ctx.closePath()
  ctx.fill()
  ctx.stroke()

  const off = cam.pos[0] < -3 || cam.pos[0] > 108 || cam.pos[1] < -3 || cam.pos[1] > 71
  const label = `cam (${cam.pos[0].toFixed(1)}, ${cam.pos[1].toFixed(1)}, ${cam.z.toFixed(1)})`
  ctx.font = "bold 10px sans-serif"
  const tw = ctx.measureText(label).width + 6
  const lx = Math.max(2, Math.min(CANVAS_W - tw - 2, cx + 10))
  const ly = Math.max(14, Math.min(CANVAS_H - 4, cy - 12))
  ctx.fillStyle = off ? "rgba(127,29,29,0.85)" : "rgba(0,0,0,0.65)"
  ctx.fillRect(lx, ly - 11, tw, 14)
  ctx.fillStyle = off ? "#fecaca" : "#fde68a"
  ctx.fillText(label, lx + 3, ly)
}
