import * as React from "react"

import type { BallPreviewTrack } from "@/pages/ball-anchor-editor/api"

type Frames = NonNullable<BallPreviewTrack["frames"]>

const PITCH_L = 105
const PITCH_W = 68
const MARGIN = 8
const SCALE = 5.5
const PAD = 20
const W = (PITCH_L + 2 * MARGIN) * SCALE + PAD * 2
const H = (PITCH_W + 2 * MARGIN) * SCALE + PAD * 2

const px = (x: number, y: number): [number, number] => [PAD + (x + MARGIN) * SCALE, PAD + (PITCH_W + MARGIN - y) * SCALE]

function rectStroke(ctx: CanvasRenderingContext2D, xa: number, ya: number, xb: number, yb: number) {
  const [ax, ay] = px(xa, ya)
  const [bx, by] = px(xb, yb)
  ctx.strokeRect(Math.min(ax, bx), Math.min(ay, by), Math.abs(bx - ax), Math.abs(by - ay))
}

function drawPitch(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = "#0a1f10"
  ctx.fillRect(0, 0, W, H)
  const [x0, y0] = px(0, PITCH_W)
  ctx.fillStyle = "#0b3b1a"
  ctx.fillRect(x0, y0, PITCH_L * SCALE, PITCH_W * SCALE)
  ctx.strokeStyle = "#e2e8f0"
  ctx.lineWidth = 1.4
  rectStroke(ctx, 0, 0, PITCH_L, PITCH_W)
  const [m1x, m1y] = px(PITCH_L / 2, 0)
  const [m2x, m2y] = px(PITCH_L / 2, PITCH_W)
  ctx.beginPath()
  ctx.moveTo(m1x, m1y)
  ctx.lineTo(m2x, m2y)
  ctx.stroke()
  const [cx, cy] = px(PITCH_L / 2, PITCH_W / 2)
  ctx.beginPath()
  ctx.arc(cx, cy, 9.15 * SCALE, 0, Math.PI * 2)
  ctx.stroke()
  const mid = PITCH_W / 2
  rectStroke(ctx, 0, mid - 20.16, 16.5, mid + 20.16)
  rectStroke(ctx, PITCH_L, mid - 20.16, PITCH_L - 16.5, mid + 20.16)
  rectStroke(ctx, 0, mid - 9.16, 5.5, mid + 9.16)
  rectStroke(ctx, PITCH_L, mid - 9.16, PITCH_L - 5.5, mid + 9.16)
}

function strokeRun(ctx: CanvasRenderingContext2D, frames: Frames, state: string, colour: string, lw: number) {
  ctx.strokeStyle = colour
  ctx.lineWidth = lw
  ctx.beginPath()
  let drawing = false
  for (const f of frames) {
    if (!f.world_xyz || f.state !== state) {
      drawing = false
      continue
    }
    const [x, y] = px(f.world_xyz[0], f.world_xyz[1])
    if (drawing) ctx.lineTo(x, y)
    else ctx.moveTo(x, y)
    drawing = true
  }
  ctx.stroke()
}

function drawLegend(ctx: CanvasRenderingContext2D) {
  ctx.font = "12px system-ui, sans-serif"
  ctx.fillStyle = "#4ade80"
  ctx.fillRect(PAD + 4, 6, 4, 10)
  ctx.fillStyle = "#e2e8f0"
  ctx.fillText("grounded", PAD + 12, 15)
  ctx.fillStyle = "#fb923c"
  ctx.fillRect(PAD + 84, 6, 4, 10)
  ctx.fillStyle = "#e2e8f0"
  ctx.fillText("flight", PAD + 92, 15)
}

/** Top-down pitch view of the solved ball trajectory; the dot follows the current frame. */
export function TopdownCanvas({ frames, frame }: { frames: Frames; frame: number }) {
  const ref = React.useRef<HTMLCanvasElement | null>(null)
  const byFrame = React.useMemo(() => new Map(frames.map((f) => [f.frame, f])), [frames])

  React.useEffect(() => {
    const ctx = ref.current?.getContext("2d")
    if (!ctx) return
    drawPitch(ctx)
    strokeRun(ctx, frames, "grounded", "#4ade80", 1.6)
    strokeRun(ctx, frames, "flight", "#fb923c", 2.2)
    const cur = byFrame.get(frame)
    if (cur?.world_xyz) {
      const [x, y] = px(cur.world_xyz[0], cur.world_xyz[1])
      ctx.fillStyle = cur.state === "flight" ? "#fb923c" : "#fafafa"
      ctx.strokeStyle = "#0a0e1a"
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(x, y, 5, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
    }
    drawLegend(ctx)
  }, [frames, byFrame, frame])

  return (
    <div className="overflow-hidden rounded-lg bg-stage">
      <canvas
        ref={ref}
        width={W}
        height={H}
        className="block h-auto w-full"
        role="img"
        aria-label="Top-down view of the ball trajectory on the pitch"
      />
    </div>
  )
}
