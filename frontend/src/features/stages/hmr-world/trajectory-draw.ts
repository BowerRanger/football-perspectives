import { playerColour, playerLabel, withAlpha } from "@/lib/format"
import { drawCameraMarker, drawPitch, PITCH_L, PITCH_W, pitch2px } from "./pitch-canvas"
import type { CameraSample, PlayerRef, PosePreview, TrajectoryPlayer } from "./types"

export const TRAIL_LEN = 30 // frames of fading trail behind each dot
export const CONF_THRESH = 0.2 // hide samples below this confidence

export interface TrajectoryInput extends PlayerRef {
  colour?: string
  data: PosePreview
}

/** Per-player frame -> {pos, conf} maps for O(1) lookup while scrubbing. */
export function buildTrajectoryPlayers(inputs: readonly TrajectoryInput[]): TrajectoryPlayer[] {
  return inputs.map((entry) => {
    const frames = entry.data.frames ?? []
    const rootT = entry.data.root_t ?? []
    const conf = entry.data.confidence ?? []
    const byFrame = new Map<number, { pos: number[]; conf: number }>()
    frames.forEach((f, i) => byFrame.set(f, { pos: rootT[i], conf: conf[i] ?? 1 }))
    return {
      pid: entry.player_id,
      label: playerLabel(entry),
      colour: entry.colour ?? playerColour(0),
      byFrame,
      firstFrame: frames.length ? frames[0] : 0,
      lastFrame: frames.length ? frames[frames.length - 1] : 0,
    }
  })
}

export function frameRange(players: readonly TrajectoryPlayer[]): { min: number; max: number } {
  let min = Infinity
  let max = -Infinity
  for (const p of players) {
    if (p.byFrame.size === 0) continue
    min = Math.min(min, p.firstFrame)
    max = Math.max(max, p.lastFrame)
  }
  return Number.isFinite(min) ? { min, max } : { min: 0, max: 0 }
}

function drawTrail(ctx: CanvasRenderingContext2D, p: TrajectoryPlayer, fi: number) {
  const trail: { f: number; pos: number[] }[] = []
  for (let f = Math.max(p.firstFrame, fi - TRAIL_LEN); f <= fi; f++) {
    const e = p.byFrame.get(f)
    if (!e || !e.pos || e.pos.length < 2 || e.conf < CONF_THRESH) continue
    trail.push({ f, pos: e.pos })
  }
  if (trail.length === 0) return
  ctx.lineWidth = 1.6
  ctx.lineCap = "round"
  for (let i = 1; i < trail.length; i++) {
    const aN = (trail[i - 1].f - (fi - TRAIL_LEN)) / TRAIL_LEN
    const aB = (trail[i].f - (fi - TRAIL_LEN)) / TRAIL_LEN
    const alpha = Math.max(0, Math.min(1, (aN + aB) / 2))
    ctx.strokeStyle = withAlpha(p.colour, 0.15 + 0.6 * alpha)
    const [x0, y0] = pitch2px(trail[i - 1].pos[0], trail[i - 1].pos[1])
    const [x1, y1] = pitch2px(trail[i].pos[0], trail[i].pos[1])
    ctx.beginPath()
    ctx.moveTo(x0, y0)
    ctx.lineTo(x1, y1)
    ctx.stroke()
  }
  const head = trail[trail.length - 1]
  const [hx, hy] = pitch2px(head.pos[0], head.pos[1])
  ctx.fillStyle = p.colour
  ctx.beginPath()
  ctx.arc(hx, hy, 4, 0, Math.PI * 2)
  ctx.fill()
  // Off-pitch dots get a warning ring so calibration errors are obvious.
  if (head.pos[0] < 0 || head.pos[0] > PITCH_L || head.pos[1] < 0 || head.pos[1] > PITCH_W) {
    ctx.strokeStyle = "#ef4444"
    ctx.lineWidth = 1.2
    ctx.beginPath()
    ctx.arc(hx, hy, 7, 0, Math.PI * 2)
    ctx.stroke()
  }
  ctx.font = "bold 10px sans-serif"
  const tw = ctx.measureText(p.label).width + 6
  ctx.fillStyle = "rgba(0,0,0,0.6)"
  ctx.fillRect(hx + 6, hy - 8, tw, 14)
  ctx.fillStyle = p.colour
  ctx.fillText(p.label, hx + 9, hy + 2)
}

export function renderTrajectoryFrame(
  ctx: CanvasRenderingContext2D,
  players: readonly TrajectoryPlayer[],
  hidden: ReadonlySet<string>,
  fi: number,
  camera: CameraSample | undefined,
) {
  drawPitch(ctx)
  if (camera) drawCameraMarker(ctx, camera)
  for (const p of players) {
    if (!hidden.has(p.pid)) drawTrail(ctx, p, fi)
  }
}
