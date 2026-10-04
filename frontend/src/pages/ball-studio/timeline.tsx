import * as React from "react"

import { Panel } from "@/components/panel"
import { shotVideoRange } from "./camera-model"
import { EVENT_STYLE } from "./palette"
import { GUTTER, drawTimeline, frameToX, hitTest, timelineHeight, xToFrame, type TimelineModel } from "./timeline-draw"
import type { Studio } from "./use-studio"

export function Timeline({ studio, onboarding }: { studio: Studio; onboarding?: React.ReactNode }) {
  const wrapRef = React.useRef<HTMLDivElement | null>(null)
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
  const [width, setWidth] = React.useState(0)
  const [hover, setHover] = React.useState<{ frame: number; x: number } | null>(null)
  const dragging = React.useRef(false)
  const { docApi, solver, scene, frame, selection } = studio
  const solved = solver.result

  const model: TimelineModel = React.useMemo(
    () => ({
      doc: docApi.doc,
      solved,
      stale: solver.stale,
      range: studio.range,
      frame,
      selectedKey: selection?.type === "key" ? selection.id : null,
      selectedEvent: selection?.type === "event" ? selection.index : null,
      hoverFrame: hover?.frame ?? null,
      views: studio.shots.map((s) => ({ shotId: s.shot_id, offset: s.frame_offset, nFrames: shotVideoRange(s)[1] + 1, repeats: s.repeat_frames })),
      pipeline: scene.pipeline_tracks[0] ? { frames: scene.pipeline_tracks[0].frames, xyz: scene.pipeline_tracks[0].xyz } : null,
      keyKinds: studio.keyKinds,
    }),
    [docApi.doc, solved, solver.stale, studio.range, frame, selection, hover, studio.shots, scene.pipeline_tracks, studio.keyKinds],
  )
  const height = timelineHeight(model)

  React.useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setWidth(el.clientWidth))
    ro.observe(el)
    setWidth(el.clientWidth)
    return () => ro.disconnect()
  }, [])

  React.useEffect(() => {
    const c = canvasRef.current
    if (!c || !width) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    c.width = Math.round(width * dpr)
    c.height = Math.round(height * dpr)
    drawTimeline(c, width, height, dpr, model)
  }, [model, width, height])

  const local = (e: React.PointerEvent): [number, number] => {
    const r = wrapRef.current!.getBoundingClientRect()
    return [e.clientX - r.left, e.clientY - r.top]
  }

  const onDown = (e: React.PointerEvent) => {
    const [x, y] = local(e)
    const hit = hitTest(model, width, x, y)
    if (hit) {
      if (hit.type === "key") {
        const k = docApi.doc.keys.find((kk) => kk.id === hit.id)
        if (k) studio.setFrame(k.frame)
        studio.setSelection({ type: "key", id: hit.id })
      } else {
        studio.setFrame(docApi.doc.events[hit.index].frame)
        studio.setSelection({ type: "event", index: hit.index })
      }
      return
    }
    dragging.current = true
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    studio.setFrame(xToFrame(x, studio.range, width))
  }

  const onMove = (e: React.PointerEvent) => {
    const [x] = local(e)
    if (x < GUTTER) {
      setHover(null)
      return
    }
    const f = xToFrame(x, studio.range, width)
    setHover({ frame: f, x })
    if (dragging.current) studio.setFrame(f)
  }

  const tip = hover ? describeFrame(studio, hover.frame) : null

  return (
    <Panel title="Timeline" flush contentClassName="relative">
      <div
        ref={wrapRef}
        role="slider"
        tabIndex={0}
        aria-label="Reference timeline: click or drag to seek"
        aria-valuemin={studio.range[0]}
        aria-valuemax={studio.range[1]}
        aria-valuenow={frame}
        className="relative w-full cursor-pointer bg-stage outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
        style={{ height }}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={() => {
          dragging.current = false
        }}
        onPointerLeave={() => setHover(null)}
      >
        <canvas ref={canvasRef} className="block size-full" style={{ width: "100%", height }} />
        {onboarding ? (
          <div className="pointer-events-none absolute top-[34px] right-3 left-[104px] flex h-[26px] items-center">{onboarding}</div>
        ) : null}
        {hover && tip ? (
          <div
            className="pointer-events-none absolute z-10 rounded-md border bg-popover px-2 py-1 text-xs text-popover-foreground shadow-md"
            style={{ left: Math.min(Math.max(GUTTER, hover.x + 10), Math.max(GUTTER, width - 220)), bottom: height + 4 }}
          >
            {tip}
          </div>
        ) : null}
      </div>
      <span className="sr-only" aria-live="polite">
        Frame {frame}
      </span>
    </Panel>
  )
}

function describeFrame(studio: Studio, f: number): React.ReactNode {
  const { docApi, solver, scene } = studio
  const key = docApi.doc.keys.find((k) => k.frame === f)
  const ev = docApi.doc.events.find((e) => e.frame === f)
  const seg = solver.result?.segments.find((s) => f >= s.frame_range[0] && f <= s.frame_range[1])
  const worst = solver.result?.observations
    .filter((o) => o.residual_px !== null && o.shot_frame - (studio.shots.find((s) => s.shot_id === o.shot_id)?.frame_offset ?? 0) === f)
    .reduce((m, o) => Math.max(m, o.residual_px ?? 0), 0)
  return (
    <span className="flex flex-wrap items-center gap-x-2">
      <span className="font-mono tabular-nums">{f}</span>
      <span className="font-mono text-muted-foreground tabular-nums">{(f / scene.fps).toFixed(2)}s</span>
      {key ? <span>key {key.id}</span> : null}
      {ev ? <span>{EVENT_STYLE[ev.kind].label.toLowerCase()}</span> : null}
      {seg ? <span>{seg.kind}</span> : null}
      {worst ? <span className="font-mono tabular-nums">{worst.toFixed(1)} px</span> : null}
    </span>
  )
}

export { frameToX }
