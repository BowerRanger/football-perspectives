import * as React from "react"

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { confidenceColour } from "./draw"
import type { CameraTrack } from "./types"

interface CoverageStripProps {
  track: CameraTrack | null
  frame: number
  totalFrames: number
  onSeek: (frame: number) => void
}

const ANCHOR_TICK = "#a855f7"

function paintStrip(canvas: HTMLCanvasElement, track: CameraTrack, total: number, w: number, h: number) {
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  ctx.clearRect(0, 0, w, h)
  const barW = Math.max(1, w / Math.max(1, total))
  track.frames.forEach((f, i) => {
    const x = ((f.frame ?? i) / Math.max(1, total)) * w
    ctx.fillStyle = confidenceColour(f.confidence)
    ctx.fillRect(x, 0, Math.ceil(barW), h)
    if (f.is_anchor) {
      ctx.fillStyle = ANCHOR_TICK
      ctx.fillRect(x, 0, Math.ceil(barW), 4)
    }
  })
}

/** Camera confidence / anchor coverage under the video; click to seek. */
export function CoverageStrip({ track, frame, totalFrames, onSeek }: CoverageStripProps) {
  const wrapRef = React.useRef<HTMLButtonElement>(null)
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const [width, setWidth] = React.useState(0)
  const height = 28

  React.useLayoutEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const measure = () => setWidth(el.clientWidth)
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  React.useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !track || width <= 0) return
    paintStrip(canvas, track, totalFrames, width, height)
  }, [track, totalFrames, width])

  const seekFromEvent = (ev: React.MouseEvent<HTMLButtonElement>) => {
    if (!track) return
    const rect = ev.currentTarget.getBoundingClientRect()
    const t = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width))
    onSeek(Math.round(t * Math.max(0, totalFrames - 1)))
  }

  const cursor = totalFrames > 1 ? (frame / (totalFrames - 1)) * 100 : 0

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          ref={wrapRef}
          type="button"
          disabled={!track}
          onClick={seekFromEvent}
          aria-label="Camera confidence timeline. Click to seek. Purple ticks mark anchor frames."
          className="relative block shrink-0 cursor-pointer overflow-hidden border-t bg-stage focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:cursor-default"
          style={{ height }}
        >
          <canvas ref={canvasRef} width={width} height={height} className="block size-full" />
          {track ? (
            <span
              aria-hidden
              className="absolute inset-y-0 w-0.5 bg-stage-foreground"
              style={{ left: `calc(${cursor}% - 1px)` }}
            />
          ) : (
            <span className="absolute inset-0 flex items-center justify-center text-xs text-stage-foreground/60">
              No camera track yet. Run the camera stage to see confidence here.
            </span>
          )}
        </button>
      </TooltipTrigger>
      <TooltipContent>Camera confidence per frame (green high, red low). Purple ticks are anchor frames. Click to seek.</TooltipContent>
    </Tooltip>
  )
}
