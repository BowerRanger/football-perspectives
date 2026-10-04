import * as React from "react"

import { Kbd } from "@/components/ui/kbd"
import { cn } from "@/lib/utils"

import { isRealTime, type MomentPair } from "./replay-speed"

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
  /** Reference frames per shot frame, per shot (1 when absent). */
  rates?: Record<string, number>
  methods: Record<string, AlignMethod>
  /** Shots whose speed ramps (drawn with a notch; the rate is not applied). */
  rampShots?: string[]
  /** Shots whose rate is approximate (label shows ≈). */
  approxShots?: string[]
  /** Match-moments pairs for the active shot, drawn as connectors. */
  pairs?: MomentPair[]
  /** The active shot's offset and rate are an unsaved preview (dashed outline). */
  previewActive?: boolean
  /** Shot driven by unsaved pairs: cannot be dragged or slid. */
  lockedShot?: string | null
  /** Phone width: no dragging, no keyboard slide. */
  readOnly?: boolean
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

/** Diagonal hatch marks a block that is slowed down (second channel besides the rate label). */
const SLOW_HATCH =
  "repeating-linear-gradient(135deg, transparent 0 6px, rgb(255 255 255 / 0.2) 6px 8px)"

/**
 * Video-editor style timeline. Each shot is a block at its real-time
 * position (start = -offset, reference pinned at 0). Drag or use arrow keys
 * to change an offset; click a block to make it the active clip; drag empty
 * timeline (or the red handle) to scrub the reference video.
 */
export function SyncTimeline(props: TimelineProps) {
  const { shotIds, framesByShot, fpsByShot, referenceShot, activeShot, offsets, methods, cursorFrame, readOnly } = props
  const rates = props.rates ?? {}
  const rateOf = (id: string) => (id === referenceShot ? 1 : (rates[id] ?? 1))
  /** Length of a block on the reference clock. */
  const spanOf = (id: string) => (framesByShot[id] ?? 0) * rateOf(id)
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
      maxEnd = Math.max(maxEnd, s + spanOf(id))
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shotIds, framesByShot, offsets, referenceShot, props.rates])

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
    } else if ((ev.key === "ArrowLeft" || ev.key === "ArrowRight") && id !== referenceShot && !readOnly && id !== props.lockedShot) {
      ev.preventDefault()
      const step = (ev.shiftKey ? 10 : 1) * (ev.key === "ArrowRight" ? -1 : 1)
      props.onCommitOffset(id, off + step)
    }
  }

  const refRow = ordered.indexOf(referenceShot)
  const actRow = ordered.indexOf(activeShot)
  const pairs = props.pairs ?? []
  const actStart = drag?.kind === "block" && drag.shotId === activeShot ? -drag.live : -(offsets[activeShot] ?? 0)
  const rowTop = (i: number) => RULER_HEIGHT + 4 + i * ROW_HEIGHT
  const pairConnectors =
    pairs.length > 0 && refRow >= 0 && actRow >= 0 && refRow !== actRow ? (
      <svg
        aria-hidden
        className="pointer-events-none absolute inset-0 z-[5]"
        width={geo.width}
        height={RULER_HEIGHT + ROW_HEIGHT * ordered.length + 8}
      >
        {pairs.map((p, k) => {
          const x1 = px(p.reference_frame)
          const x2 = px(actStart + rateOf(activeShot) * p.shot_frame)
          const yA = rowTop(refRow) + (ROW_HEIGHT - 6) / 2
          const yB = rowTop(actRow) + (ROW_HEIGHT - 6) / 2
          return (
            <g key={`${p.reference_frame}:${p.shot_frame}`}>
              <line x1={x1} y1={yA} x2={x2} y2={yB} stroke="black" strokeOpacity={0.55} strokeWidth={4} />
              <line x1={x1} y1={yA} x2={x2} y2={yB} stroke="white" strokeWidth={1.5} />
              <circle cx={x1} cy={yA} r={3.5} fill="white" stroke="black" strokeOpacity={0.55} />
              <circle cx={x2} cy={yB} r={3.5} fill="white" stroke="black" strokeOpacity={0.55} />
              <text
                x={x1 + 6}
                y={yA - 6}
                stroke="black"
                strokeOpacity={0.7}
                strokeWidth={3}
                paintOrder="stroke"
                className="fill-white text-[10px] font-semibold"
              >
                {k + 1}
              </text>
            </g>
          )
        })}
      </svg>
    ) : null

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs text-muted-foreground">
        Each block sits at its real-time position (reference anchored at frame 0); a slowed replay is drawn as long as
        the live time it covers, hatched, with its rate.{" "}
        {readOnly ? (
          "Click a clip to make it active."
        ) : (
          <>
            Drag a clip, or focus it and press <Kbd>←</Kbd> <Kbd>→</Kbd> (<Kbd>Shift</Kbd> for 10 frames) to slide it.
            Click a clip to make it active. Drag empty timeline to scrub the reference.
          </>
        )}
      </p>
      <div className="overflow-x-auto rounded-md bg-stage">
        <div
          ref={contentRef}
          className="relative cursor-ew-resize select-none"
          style={{ width: geo.width, height: RULER_HEIGHT + ROW_HEIGHT * ordered.length + 8 }}
          onPointerDown={(ev) => {
            if (ev.button !== 0 || ev.target !== ev.currentTarget || readOnly) return
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
                className="absolute top-0 bottom-0 border-l border-white/10 pl-1 text-[11px] text-stage-foreground/70 tabular-nums"
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
            const native = framesByShot[id] ?? 1
            const rate = rateOf(id)
            const len = native * rate
            const slow = !isRef && !isRealTime(rate) && rate < 1
            const ramp = !isRef && (props.rampShots ?? []).includes(id)
            const preview = !isRef && id === activeShot && !!props.previewActive
            const m = isRef ? undefined : methods[id]
            const low = !!m && m.method !== "manual" && m.confidence < 0.5
            const span = `${Math.round(start)}–${Math.round(start + len)}`
            const approx = !isRef && (props.approxShots ?? []).includes(id)
            const rateNote = isRef || isRealTime(rate) ? "" : ` · ${approx ? "≈" : ""}${rate.toFixed(2)}×`
            const label = isRef
              ? `${id} (ref, frames 0–${native})`
              : `${id}${rateNote} · global ${span}${methodNote(m)}${preview ? " · unsaved" : ""}`
            // Lead with id and rate; drop the range and method when the block is narrow,
            // and draw the label beside a block too narrow for even that.
            const widthPx = Math.max(2, Math.round(len * geo.pxPerFrame))
            const short = isRef ? `${id} (ref)` : `${id}${rateNote}${preview ? " · unsaved" : ""}`
            const fits = (text: string) => widthPx >= text.length * 6.6 + 14
            const inside = fits(label) ? label : fits(short) ? short : ""
            const rowTopPx = RULER_HEIGHT + 4 + i * ROW_HEIGHT
            return (
              <React.Fragment key={id}>
              <div
                role="button"
                tabIndex={0}
                aria-label={label}
                aria-pressed={id === activeShot}
                data-timeline-block={id}
                title={`${id}\nclip frames: ${native}\nrate: ${approx ? "≈" : ""}${rate.toFixed(3)}× (covers ${len.toFixed(0)} live frames)\nfps: ${(fpsByShot[id] ?? 25).toFixed(2)}\noffset: ${off}\nglobal range: ${span}`}
                className={cn(
                  "absolute overflow-hidden rounded border px-1.5 py-1 text-xs font-medium text-ellipsis whitespace-nowrap text-white outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  isRef ? "cursor-pointer border-info bg-info/80" : "cursor-grab border-white/20 active:cursor-grabbing",
                  !isRef && (id === activeShot ? "border-success bg-success/80" : "bg-white/20"),
                  low && "ring-2 ring-warning",
                  preview && "border-dashed border-white/80",
                  readOnly && "cursor-pointer active:cursor-pointer",
                )}
                style={{
                  left: px(start),
                  top: rowTopPx,
                  width: widthPx,
                  height: ROW_HEIGHT - 6,
                  backgroundImage: slow ? SLOW_HATCH : undefined,
                }}
                onClick={(ev) => {
                  ev.stopPropagation()
                  if (!isRef) props.onPick(id)
                }}
                onKeyDown={(ev) => onBlockKey(ev, id, committed)}
                onPointerDown={(ev) => {
                  ev.stopPropagation()
                  if (ev.button !== 0 || isRef || readOnly || id === props.lockedShot) return
                  beginDrag({ kind: "block", shotId: id, startX: ev.clientX, startOffset: committed, live: committed })
                }}
              >
                {inside}
                {ramp ? (
                  <span
                    aria-hidden
                    className="absolute inset-y-0 left-1/2 w-0.5 -skew-x-12 bg-warning"
                    title="Speed ramp: the rate changes here (not applied)"
                  />
                ) : null}
              </div>
              {inside === "" ? (
                <span
                  aria-hidden
                  className="pointer-events-none absolute text-xs font-medium whitespace-nowrap text-stage-foreground"
                  style={{ left: px(start) + widthPx + 6, top: rowTopPx + 4 }}
                >
                  {short}
                </span>
              ) : null}
              </React.Fragment>
            )
          })}
          {pairConnectors}
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
