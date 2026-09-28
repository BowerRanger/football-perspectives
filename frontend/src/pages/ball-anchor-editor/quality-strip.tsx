import * as React from "react"

import { Button } from "@/components/ui/button"
import { cssVar } from "@/lib/format"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { drawQualityStrip } from "./overlay-draw"
import type { EditorController } from "./use-ball-anchor-editor"

const HEIGHT = 30

/** Per-frame detection confidence strip; click to seek. Backed by /ball-quality/{shot}. */
export function QualityStrip({ ctrl }: { ctrl: EditorController }) {
  const wrapRef = React.useRef<HTMLDivElement | null>(null)
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
  const [width, setWidth] = React.useState(0)
  const { quality, player, docApi } = ctrl
  const n = quality?.n_frames ?? 0

  React.useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setWidth(el.clientWidth))
    ro.observe(el)
    setWidth(el.clientWidth)
    return () => ro.disconnect()
  }, [])

  React.useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !width) return
    canvas.width = width
    canvas.height = HEIGHT
    drawQualityStrip(
      canvas,
      quality,
      docApi.doc.anchors,
      player.frame,
      "No ball quality yet — run the Ball stage",
      cssVar("--muted-foreground", "#a1a1aa"),
      cssVar("--stage", "#0a0a0a"),
    )
  }, [width, quality, docApi.doc.anchors, player.frame])

  const seekFromPointer = (clientX: number) => {
    if (!n || !wrapRef.current) return
    const rect = wrapRef.current.getBoundingClientRect()
    const frac = (clientX - rect.left) / Math.max(1, rect.width)
    player.seekTo(Math.round(Math.max(0, Math.min(1, frac)) * (n - 1)))
  }

  const weak = quality?.annotate_next ?? []
  const items = weak.slice(0, 3)
  const nextWeak = () => {
    const sorted = [...weak].sort((a, b) => a.start - b.start)
    const target = sorted.find((w) => w.start > player.frame) ?? sorted[0]
    if (target) player.seekTo(target.start)
  }

  return (
    <div className="flex flex-col gap-1.5">
      <div
        ref={wrapRef}
        role="slider"
        tabIndex={0}
        aria-label="Ball quality timeline — click to seek"
        aria-valuemin={0}
        aria-valuemax={Math.max(0, n - 1)}
        aria-valuenow={player.frame}
        title="Ball quality — click to seek"
        className="cursor-pointer overflow-hidden rounded-md border bg-stage outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
        style={{ height: HEIGHT }}
        onClick={(e) => seekFromPointer(e.clientX)}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") player.step(-1)
          if (e.key === "ArrowRight") player.step(1)
        }}
      >
        <canvas ref={canvasRef} className="block h-full w-full" />
      </div>
      <p className="text-xs text-muted-foreground">
        Bars: detection confidence, red to green. Top ticks: auto events (blue), manual anchors (purple). Bottom bands:
        underconstrained flight (red), detection gap (orange).
      </p>
      {items.length ? (
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <Button size="xs" variant="secondary" onClick={nextWeak}>
            Annotate next weak span
          </Button>
          <span>Ranked:</span>
          {items.map((it) => (
            <Tooltip key={`${it.reason}-${it.start}`}>
              <TooltipTrigger asChild>
                <Button size="xs" variant="outline" onClick={() => player.seekTo(it.start)}>
                  {it.reason === "underconstrained_flight" ? "flight" : "gap"} {it.start}–{it.end}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {it.reason === "underconstrained_flight"
                  ? "Flight span with < 2 hard knots — add a bracketing kick/bounce/grounded anchor inside it"
                  : "Long detection gap — confirm the ball state through it"}
              </TooltipContent>
            </Tooltip>
          ))}
        </div>
      ) : null}
    </div>
  )
}
