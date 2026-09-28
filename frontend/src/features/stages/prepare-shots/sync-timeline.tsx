import * as React from "react"

import { Kbd } from "@/components/ui/kbd"
import { cn } from "@/lib/utils"

export interface AlignMethod {
  method: string
  confidence: number
}

interface TimelineProps {
  shotIds: string[]
  framesByShot: Record<string, number>
  fpsByShot: Record<string, number>
  referenceShot: string
  activeShot: string
  offsets: Record<string, number>
  methods: Record<string, AlignMethod>
  cursorFrame: number
  onCommitOffset: (shotId: string, offset: number) => void
  onPick: (shotId: string) => void
  onScrub: (globalFrame: number) => void
}

type Drag =
  | { kind: "block"; shotId: string; startX: number; startOffset: number; live: number }
  | { kind: "scrub" }

const ROW_HEIGHT = 36
const RULER_HEIGHT = 20
const TARGET_WIDTH = 900

function methodNote(m: AlignMethod | undefined): string {
  if (!m) return ""
  return m.method === "manual" ? " · manual" : ` · auto ${m.confidence.toFixed(2)}`
}

/**
 * Video-editor style timeline. Each shot is a block at its real-time
 * position (start = -offset, reference pinned at 0). Drag or use arrow keys
 * to change an offset; click a block to make it the active clip; drag empty
 * timeline (or the red handle) to scrub the reference video.
 */
export function SyncTimeline(props: TimelineProps) {
  const { shotIds, framesByShot, fpsByShot, referenceShot, activeShot, offsets, methods, cursorFrame } = props
  const contentRef = React.useRef<HTMLDivElement>(null)
  const [drag, setDrag] = React.useState<Drag | null>(null)
  const dragRef = React.useRef<Drag | null>(null)
  const propsRef = React.useRef(props)
  propsRef.current = props

  const startOf = (id: string, off: Record<string, number>) => (id === referenceShot ? 0 : -(off[id] ?? 0))

  // Layout is computed from committed offsets only, so it stays fixed while
  // a block is being dragged.
  const geo = React.useMemo(() => {
    let minStart = Infinity
    let maxEnd = -Infinity
    for (const id of shotIds) {
      const s = id === referenceShot ? 0 : -(offsets[id] ?? 0)
      minStart = Math.min(minStart, s)
      maxEnd = Math.max(maxEnd, s + (framesByShot[id] ?? 0))
    }
    if (!Number.isFinite(minStart)) {
      minStart = 0
      maxEnd = 0
    }
    const pad = Math.max(60, Math.round((maxEnd - minStart) * 0.05))
    const spanStart = minStart - pad
    const spanFrames = Math.max(1, maxEnd + pad - spanStart)
    const pxPerFrame = Math.max(0.4, Math.min(8, TARGET_WIDTH / spanFrames))
    return { spanStart, spanFrames, pxPerFrame, width: Math.round(spanFrames * pxPerFrame) }
  }, [shotIds, framesByShot, offsets, referenceShot])

  const scrubTo = React.useCallback(
    (clientX: number) => {
      const rect = contentRef.current?.getBoundingClientRect()
      if (!rect) return
      propsRef.current.onScrub(geo.spanStart + (clientX - rect.left) / geo.pxPerFrame)
    },
    [geo],
  )

  const active = drag !== null
  React.useEffect(() => {
    if (!active) return
    const onMove = (ev: PointerEvent) => {
      const d = dragRef.current
      if (!d) return
      if (d.kind === "scrub") {
        scrubTo(ev.clientX)
        return
      }
      const dxFrames = Math.round((ev.clientX - d.startX) / geo.pxPerFrame)
      const next: Drag = { ...d, live: d.startOffset - dxFrames }
      dragRef.current = next
      setDrag(next)
    }
    const onUp = () => {
      const d = dragRef.current
      dragRef.current = null
      setDrag(null)
      if (d?.kind === "block" && d.live !== d.startOffset) propsRef.current.onCommitOffset(d.shotId, d.live)
    }
    window.addEventListener("pointermove", onMove)
    window.addEventListener("pointerup", onUp)
    window.addEventListener("pointercancel", onUp)
    return () => {
      window.removeEventListener("pointermove", onMove)
      window.removeEventListener("pointerup", onUp)
      window.removeEventListener("pointercancel", onUp)
    }
  }, [active, geo, scrubTo])

  const beginDrag = (d: Drag) => {
    dragRef.current = d
    setDrag(d)
  }

  const ordered = [
    referenceShot,
    ...shotIds.filter((id) => id !== referenceShot).sort((a, b) => startOf(a, offsets) - startOf(b, offsets)),
  ]
  const tickStride = Math.max(1, Math.round(120 / geo.pxPerFrame / 10) * 10)
  const ticks: number[] = []
  for (let f = Math.ceil(geo.spanStart / tickStride) * tickStride; f <= geo.spanStart + geo.spanFrames; f += tickStride) {
    ticks.push(f)
  }
  const px = (frame: number) => Math.round((frame - geo.spanStart) * geo.pxPerFrame)

  const onBlockKey = (ev: React.KeyboardEvent, id: string, off: number) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault()
      props.onPick(id)
    } else if ((ev.key === "ArrowLeft" || ev.key === "ArrowRight") && id !== referenceShot) {
      ev.preventDefault()
      const step = (ev.shiftKey ? 10 : 1) * (ev.key === "ArrowRight" ? -1 : 1)
      props.onCommitOffset(id, off + step)
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs text-muted-foreground">
        Each block sits at its real-time position (reference anchored at frame 0). Drag a clip, or focus it and press{" "}
        <Kbd>←</Kbd> <Kbd>→</Kbd> (<Kbd>Shift</Kbd> for 10 frames) to slide it. Click a clip to make it active. Drag
        empty timeline to scrub the reference.
      </p>
      <div className="overflow-x-auto rounded-md bg-stage">
        <div
          ref={contentRef}
          className="relative cursor-ew-resize select-none"
          style={{ width: geo.width, height: RULER_HEIGHT + ROW_HEIGHT * ordered.length + 8 }}
          onPointerDown={(ev) => {
            if (ev.button !== 0 || ev.target !== ev.currentTarget) return
            ev.preventDefault()
            beginDrag({ kind: "scrub" })
            scrubTo(ev.clientX)
          }}
        >
          <div
            className="pointer-events-none absolute inset-x-0 top-0 border-b border-white/10"
            style={{ height: RULER_HEIGHT }}
          >
            {ticks.map((f) => (
              <span
                key={f}
                className="absolute top-0 bottom-0 border-l border-white/10 pl-1 text-[10px] text-white/40 tabular-nums"
                style={{ left: px(f) }}
              >
                {f}
              </span>
            ))}
          </div>
          {ordered.map((id, i) => {
            const isRef = id === referenceShot
            const committed = isRef ? 0 : (offsets[id] ?? 0)
            const off = drag?.kind === "block" && drag.shotId === id ? drag.live : committed
            const start = isRef ? 0 : -off
            const len = framesByShot[id] ?? 1
            const m = isRef ? undefined : methods[id]
            const low = !!m && m.method !== "manual" && m.confidence < 0.5
            const label = isRef
              ? `${id} (ref, frames 0–${len})`
              : `${id} · global ${start}–${start + len}${methodNote(m)}`
            return (
              <div
                key={id}
                role="button"
                tabIndex={0}
                aria-label={label}
                aria-pressed={id === activeShot}
                title={`${id}\nframes: ${len}\nfps: ${(fpsByShot[id] ?? 25).toFixed(2)}\noffset: ${off}\nglobal range: ${start}–${start + len}`}
                className={cn(
                  "absolute overflow-hidden rounded border px-1.5 py-1 text-xs font-medium text-ellipsis whitespace-nowrap text-white outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  isRef ? "cursor-pointer border-info bg-info/80" : "cursor-grab border-white/20 active:cursor-grabbing",
                  !isRef && (id === activeShot ? "border-success bg-success/80" : "bg-white/20"),
                  low && "ring-2 ring-warning",
                )}
                style={{
                  left: px(start),
                  top: RULER_HEIGHT + 4 + i * ROW_HEIGHT,
                  width: Math.max(2, Math.round(len * geo.pxPerFrame)),
                  height: ROW_HEIGHT - 6,
                }}
                onClick={(ev) => {
                  ev.stopPropagation()
                  if (!isRef) props.onPick(id)
                }}
                onKeyDown={(ev) => onBlockKey(ev, id, committed)}
                onPointerDown={(ev) => {
                  ev.stopPropagation()
                  if (ev.button !== 0 || isRef) return
                  beginDrag({ kind: "block", shotId: id, startX: ev.clientX, startOffset: committed, live: committed })
                }}
              >
                {label}
              </div>
            )
          })}
          <div
            className="pointer-events-none absolute bottom-0 w-0.5 bg-destructive"
            style={{ left: px(cursorFrame), top: RULER_HEIGHT }}
          />
          <div
            role="slider"
            tabIndex={-1}
            aria-label="Reference playhead"
            aria-valuenow={Math.round(cursorFrame)}
            title="Drag to scrub the reference video"
            className="absolute z-10 size-3 cursor-ew-resize rounded-full border-2 border-white bg-destructive"
            style={{ left: px(cursorFrame) - 5, top: RULER_HEIGHT - 6 }}
            onPointerDown={(ev) => {
              if (ev.button !== 0) return
              ev.preventDefault()
              ev.stopPropagation()
              beginDrag({ kind: "scrub" })
              scrubTo(ev.clientX)
            }}
          />
        </div>
      </div>
    </div>
  )
}
