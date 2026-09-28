import type { IndexedShot, PitchCameraPose } from "./types"
import { nearestInShot } from "./pitch-shots"

// Canvas drawing for the top-down pitch map. Pitch colours are DATA (a
// stylised pitch), so hex is intentional here.

export const PITCH_L = 105
export const PITCH_W = 68
const MARGIN_X = 8
const MARGIN_FAR = 5
const MARGIN_NEAR = 30
const SCALE = 6
const PADDING = 20
export const MAP_W = (PITCH_L + 2 * MARGIN_X) * SCALE + PADDING * 2
export const MAP_H = (PITCH_W + MARGIN_FAR + MARGIN_NEAR) * SCALE + PADDING * 2

const LINE = "#e2e8f0"
const FONT = "ui-sans-serif, system-ui, sans-serif"

export function pitchToPx(x: number, y: number): [number, number] {
  return [PADDING + (x + MARGIN_X) * SCALE, PADDING + (PITCH_W + MARGIN_FAR - y) * SCALE]
}

function spot(ctx: CanvasRenderingContext2D, x: number, y: number) {
  ctx.fillStyle = LINE
  ctx.beginPath()
  ctx.arc(x, y, 1.6, 0, Math.PI * 2)
  ctx.fill()
}

function rectPath(ctx: CanvasRenderingContext2D, xa: number, ya: number, xb: number, yb: number) {
  const [pax, pay] = pitchToPx(xa, ya)
  const [pbx, pby] = pitchToPx(xb, yb)
  ctx.beginPath()
  ctx.rect(Math.min(pax, pbx), Math.min(pay, pby), Math.abs(pbx - pax), Math.abs(pby - pay))
  ctx.stroke()
}

function drawGoalEnd(ctx: CanvasRenderingContext2D, left: boolean) {
  const PEN_DEPTH = 16.5
  const PEN_HALF = 20.16
  const SIX_DEPTH = 5.5
  const SIX_HALF = 9.16
  const SPOT_FROM_GOAL = 11.0
  const ARC_R = 9.15
  const GOAL_HALF = 3.66
  const yMid = PITCH_W / 2
  const xGoal = left ? 0 : PITCH_L
  const sgn = left ? 1 : -1
  const halfAngle = Math.acos((PEN_DEPTH - SPOT_FROM_GOAL) / ARC_R)
  rectPath(ctx, xGoal, yMid - PEN_HALF, xGoal + sgn * PEN_DEPTH, yMid + PEN_HALF)
  rectPath(ctx, xGoal, yMid - SIX_HALF, xGoal + sgn * SIX_DEPTH, yMid + SIX_HALF)
  const xSpot = xGoal + sgn * SPOT_FROM_GOAL
  const [sx, sy] = pitchToPx(xSpot, yMid)
  spot(ctx, sx, sy)
  const startAng = left ? -halfAngle : Math.PI - halfAngle
  const endAng = left ? halfAngle : Math.PI + halfAngle
  ctx.beginPath()
  const N_SEG = 32
  for (let i = 0; i <= N_SEG; i++) {
    const a = startAng + (endAng - startAng) * (i / N_SEG)
    const [px, py] = pitchToPx(xSpot + ARC_R * Math.cos(a), yMid + ARC_R * Math.sin(a))
    if (i === 0) ctx.moveTo(px, py)
    else ctx.lineTo(px, py)
  }
  ctx.stroke()
  // Goal mouth in highlight yellow so left/right goals are easy to pick out.
  ctx.strokeStyle = "#fde68a"
  ctx.lineWidth = 3
  const [gAx, gAy] = pitchToPx(xGoal, yMid - GOAL_HALF)
  const [gBx, gBy] = pitchToPx(xGoal, yMid + GOAL_HALF)
  ctx.beginPath()
  ctx.moveTo(gAx, gAy)
  ctx.lineTo(gBx, gBy)
  ctx.stroke()
  ctx.strokeStyle = LINE
  ctx.lineWidth = 1.5
}

export function drawPitch(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = "#0a1f10"
  ctx.fillRect(0, 0, MAP_W, MAP_H)
  ctx.fillStyle = "#0b3b1a"
  const [x0, y0] = pitchToPx(0, PITCH_W)
  ctx.fillRect(x0, y0, PITCH_L * SCALE, PITCH_W * SCALE)
  ctx.strokeStyle = LINE
  ctx.lineWidth = 1.5
  const corners: [number, number][] = [[0, 0], [PITCH_L, 0], [PITCH_L, PITCH_W], [0, PITCH_W]]
  ctx.beginPath()
  corners.forEach(([x, y], i) => {
    const [px, py] = pitchToPx(x, y)
    if (i === 0) ctx.moveTo(px, py)
    else ctx.lineTo(px, py)
  })
  ctx.closePath()
  ctx.stroke()
  const m1 = pitchToPx(PITCH_L / 2, 0)
  const m2 = pitchToPx(PITCH_L / 2, PITCH_W)
  ctx.beginPath()
  ctx.moveTo(m1[0], m1[1])
  ctx.lineTo(m2[0], m2[1])
  ctx.stroke()
  const c = pitchToPx(PITCH_L / 2, PITCH_W / 2)
  ctx.beginPath()
  ctx.arc(c[0], c[1], 9.15 * SCALE, 0, Math.PI * 2)
  ctx.stroke()
  spot(ctx, c[0], c[1])
  drawGoalEnd(ctx, true)
  drawGoalEnd(ctx, false)
}

function drawMarker(ctx: CanvasRenderingContext2D, id: string, colour: string, pose: PitchCameraPose, live: boolean) {
  const { pos, z, fwd, isAnchor } = pose
  const [cx, cy] = pitchToPx(pos[0], pos[1])
  const [dx, dy] = [fwd[0], -fwd[1]] // canvas y is flipped vs world y
  const size = 10
  const sight = pitchToPx(pos[0] + fwd[0] * 30, pos[1] + fwd[1] * 30)
  ctx.strokeStyle = colour + (live ? "b3" : "55")
  ctx.lineWidth = live ? 1.4 : 1
  ctx.beginPath()
  ctx.moveTo(cx, cy)
  ctx.lineTo(sight[0], sight[1])
  ctx.stroke()

  // Outlined gold at anchor frames so anchored vs interpolated is visible.
  ctx.fillStyle = colour
  ctx.strokeStyle = isAnchor ? "#facc15" : "#0a0e1a"
  ctx.lineWidth = isAnchor ? 2 : 1.6
  ctx.beginPath()
  ctx.moveTo(cx + dx * size, cy + dy * size)
  ctx.lineTo(cx - dx * 5 - dy * 6, cy - dy * 5 + dx * 6)
  ctx.lineTo(cx - dx * 5 + dy * 6, cy - dy * 5 - dx * 6)
  ctx.closePath()
  ctx.fill()
  ctx.stroke()

  const off = pos[0] < -3 || pos[0] > 108 || pos[1] < -3 || pos[1] > 71
  const label = `${id} (${pos[0].toFixed(1)}, ${pos[1].toFixed(1)}, ${z.toFixed(1)})`
  ctx.font = `bold 10px ${FONT}`
  const tw = ctx.measureText(label).width + 6
  const lx = Math.max(2, Math.min(MAP_W - tw - 2, cx + 12))
  const ly = Math.max(14, Math.min(MAP_H - 4, cy - 10))
  ctx.fillStyle = off ? "rgba(127,29,29,0.92)" : "rgba(0,0,0,0.72)"
  ctx.fillRect(lx, ly - 11, tw, 14)
  ctx.fillStyle = off ? "#fecaca" : colour
  ctx.fillText(label, lx + 3, ly)
}

export function renderPitchFrame(ctx: CanvasRenderingContext2D, shots: readonly IndexedShot[], fi: number) {
  drawPitch(ctx)
  for (const shot of shots) {
    drawMarker(ctx, shot.id, shot.colour, nearestInShot(shot, fi), fi >= shot.minFrame && fi <= shot.maxFrame)
  }
}
