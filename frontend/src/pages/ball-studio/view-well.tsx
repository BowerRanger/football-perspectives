import * as React from "react"
import { Maximize2Icon, Minimize2Icon, VideoOffIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { cn } from "@/lib/utils"
import type { SnapResult } from "./pick-session"
import { shotVideoRange } from "./camera-model"
import { viewColour, viewLetter } from "./palette"
import type { GroupShot, Vec2 } from "./types"
import type { ViewStatus } from "./use-view-videos"
import { drawView, type ViewDrawInput } from "./view-overlay-draw"

export type ViewDrawProps = Omit<ViewDrawInput, "width" | "height" | "dpr" | "zoom" | "tx" | "ty" | "imageSize" | "hover">

interface Transform {
  zoom: number
  tx: number
  ty: number
}

const IDENTITY: Transform = { zoom: 1, tx: 0, ty: 0 }
const MAX_ZOOM = 8
const SNAP_SCREEN_PX = 8
const LOUPE_SIZE = 168
const LOUPE_FACTOR = 4

interface ViewWellProps {
  index: number
  shot: GroupShot
  shotFrame: number
  status: ViewStatus
  videoRef: (el: HTMLVideoElement | null) => void
  draw: ViewDrawProps
  active: boolean
  /** Pointer picking enabled (false on the mobile read-only review). */
  interactive: boolean
  compact?: boolean
  focused?: boolean
  loupe: boolean
  resetSignal: number
  /** Resolve a candidate pixel (snap to the epipolar line when close). */
  resolve: (uv: Vec2, radiusNative: number, alt: boolean) => SnapResult
  onPick: (uv: Vec2) => void
  onHover: (uv: Vec2 | null) => void
  onActivate: () => void
  onToggleFocus?: () => void
  onVideoError: () => void
  /** Short inline note under the header, e.g. "A's ray does not reach this view". */
  note?: string | null
  /** The displayed frame repeats the previous image (pulldown). */
  isRepeat?: boolean
  /** Cap the well height (focus view); the picture letterboxes in its aspect ratio. */
  maxHeight?: string
  className?: string
}

/** One synced video + overlay canvas in a `bg-stage` well, with wheel zoom, Alt/middle pan and a Z loupe. */
export function ViewWell(props: ViewWellProps) {
  const { index, shot, shotFrame, status, interactive, active, draw, loupe } = props
  const wrapRef = React.useRef<HTMLDivElement | null>(null)
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
  const videoElRef = React.useRef<HTMLVideoElement | null>(null)
  const loupeRef = React.useRef<HTMLCanvasElement | null>(null)
  const [size, setSize] = React.useState({ w: 0, h: 0 })
  const [tf, setTf] = React.useState<Transform>(IDENTITY)
  const [hover, setHover] = React.useState<{ uv: Vec2; snapped: boolean } | null>(null)
  const [cursor, setCursor] = React.useState<[number, number] | null>(null)
  const panning = React.useRef<{ x: number; y: number; tx: number; ty: number; moved: boolean } | null>(null)
  const colour = viewColour(index)
  const imageSize = shot.image_size
  const registerVideo = props.videoRef
  // Stable callback ref: an inline one would detach/re-attach (and re-add listeners) on every render.
  const setVideoEl = React.useCallback(
    (el: HTMLVideoElement | null) => {
      videoElRef.current = el
      registerVideo(el)
    },
    [registerVideo],
  )

  React.useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }))
    ro.observe(el)
    setSize({ w: el.clientWidth, h: el.clientHeight })
    return () => ro.disconnect()
  }, [])

  React.useEffect(() => setTf(IDENTITY), [props.resetSignal])

  const clampTf = React.useCallback(
    (t: Transform): Transform => {
      const zoom = Math.min(MAX_ZOOM, Math.max(1, t.zoom))
      return {
        zoom,
        tx: Math.min(0, Math.max(size.w - size.w * zoom, t.tx)),
        ty: Math.min(0, Math.max(size.h - size.h * zoom, t.ty)),
      }
    },
    [size.w, size.h],
  )

  const k = size.w > 0 ? (size.w / imageSize[0]) * tf.zoom : 1
  const toNative = React.useCallback(
    (clientX: number, clientY: number): Vec2 => {
      const rect = wrapRef.current!.getBoundingClientRect()
      return [(clientX - rect.left - tf.tx) / k, (clientY - rect.top - tf.ty) / k]
    },
    [tf.tx, tf.ty, k],
  )

  // Paint the overlay every render (cheap: a few hundred points).
  React.useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !size.w) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const pw = Math.round(size.w * dpr)
    const ph = Math.round(size.h * dpr)
    if (canvas.width !== pw) canvas.width = pw
    if (canvas.height !== ph) canvas.height = ph
    drawView(canvas, { ...draw, width: size.w, height: size.h, dpr, imageSize, zoom: tf.zoom, tx: tf.tx, ty: tf.ty, hover })
  })

  // Loupe: 4x crop of the displayed frame around the cursor.
  React.useEffect(() => {
    const lc = loupeRef.current
    const v = videoElRef.current
    if (!lc || !loupe || !hover || !v) return
    const ctx = lc.getContext("2d")
    if (!ctx) return
    const src = LOUPE_SIZE / LOUPE_FACTOR
    const vw = v.videoWidth || imageSize[0]
    const vh = v.videoHeight || imageSize[1]
    const sx = (hover.uv[0] / imageSize[0]) * vw - src / 2
    const sy = (hover.uv[1] / imageSize[1]) * vh - src / 2
    ctx.imageSmoothingEnabled = false
    ctx.clearRect(0, 0, LOUPE_SIZE, LOUPE_SIZE)
    try {
      ctx.drawImage(v, sx, sy, src, src, 0, 0, LOUPE_SIZE, LOUPE_SIZE)
    } catch {
      /* frame not decoded yet */
    }
    ctx.strokeStyle = colour
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(LOUPE_SIZE / 2, 0)
    ctx.lineTo(LOUPE_SIZE / 2, LOUPE_SIZE)
    ctx.moveTo(0, LOUPE_SIZE / 2)
    ctx.lineTo(LOUPE_SIZE, LOUPE_SIZE / 2)
    ctx.stroke()
  })

  const radiusNative = SNAP_SCREEN_PX / Math.max(k, 1e-6)
  const clientToLocal = (e: { clientX: number; clientY: number }): [number, number] => {
    const rect = wrapRef.current!.getBoundingClientRect()
    return [e.clientX - rect.left, e.clientY - rect.top]
  }

  // Wheel zoom about the cursor. Native listener so preventDefault works (React wheel handlers are passive).
  const tfRef = React.useRef(tf)
  React.useEffect(() => {
    tfRef.current = tf
  })
  React.useEffect(() => {
    const el = canvasRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      setTf((t) => {
        const nz = Math.min(MAX_ZOOM, Math.max(1, t.zoom * Math.exp(-e.deltaY * 0.0015)))
        return clampTf({ zoom: nz, tx: mx - ((mx - t.tx) * nz) / t.zoom, ty: my - ((my - t.ty) * nz) / t.zoom })
      })
    }
    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  }, [clampTf])

  const onPointerDown = (e: React.PointerEvent) => {
    props.onActivate()
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      panning.current = { x: e.clientX, y: e.clientY, tx: tf.tx, ty: tf.ty, moved: false }
      ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
      if (e.button === 1) e.preventDefault()
    } else if (e.button === 0) {
      panning.current = { x: e.clientX, y: e.clientY, tx: tf.tx, ty: tf.ty, moved: false }
      ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    }
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const p = panning.current
    if (p && (e.buttons & 4 || e.altKey) && Math.hypot(e.clientX - p.x, e.clientY - p.y) > 4) {
      p.moved = true
      setTf((t) => clampTf({ ...t, tx: p.tx + e.clientX - p.x, ty: p.ty + e.clientY - p.y }))
      return
    }
    if (!interactive) return
    const uv = toNative(e.clientX, e.clientY)
    const r = props.resolve(uv, radiusNative, e.altKey)
    setHover({ uv: r.uv, snapped: r.snapped })
    setCursor(clientToLocal(e))
    props.onHover(r.uv)
  }

  const onPointerUp = (e: React.PointerEvent) => {
    const p = panning.current
    panning.current = null
    if (!p || p.moved || e.button !== 0 || !interactive) return
    if (status !== "ready" && status !== "error") return
    const uv = toNative(e.clientX, e.clientY)
    if (uv[0] < 0 || uv[1] < 0 || uv[0] > imageSize[0] || uv[1] > imageSize[1]) return
    props.onPick(props.resolve(uv, radiusNative, e.altKey).uv)
  }

  const letter = viewLetter(index)
  const blocked = status === "out_of_range"
  const frameUrl = `${shot.frame_url}?frame_idx=${shotFrame}`

  return (
    <div
      className={cn(
        "relative min-w-0 overflow-hidden rounded-lg bg-stage ring-1 ring-border",
        active && interactive && "ring-2 ring-info",
        props.className,
      )}
    >
      <div ref={wrapRef} className="relative w-full overflow-hidden" style={{
          aspectRatio: `${imageSize[0]} / ${imageSize[1]}`,
          ...(props.maxHeight ? { width: `min(100%, calc(${props.maxHeight} * ${imageSize[0] / imageSize[1]}))`, marginInline: "auto" } : {}),
        }}>
        <div
          className="absolute inset-0 origin-top-left"
          style={{ transform: `translate(${tf.tx}px, ${tf.ty}px) scale(${tf.zoom})` }}
        >
          {status === "error" ? (
            <img src={frameUrl} alt={`Shot ${shot.shot_id} frame ${shotFrame}`} className="size-full object-fill" />
          ) : (
            <video
              ref={setVideoEl}
              src={shot.video_url}
              preload="auto"
              muted
              playsInline
              aria-label={`Shot ${shot.shot_id} video`}
              className={cn("size-full object-fill", tf.zoom >= 3 && "[image-rendering:pixelated]", blocked && "invisible")}
              onError={props.onVideoError}
            />
          )}
        </div>
        <canvas
          ref={canvasRef}
          className={cn("absolute inset-0 size-full touch-none", interactive ? "cursor-crosshair" : "cursor-default")}
          style={{ cursor: status === "seeking" ? "progress" : undefined }}
          role="img"
          aria-label={`${letter} view of ${shot.shot_id}, frame ${shotFrame}. ${interactive ? "Click the ball to place a pending pick." : "Read-only."}`}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={() => {
            setHover(null)
            setCursor(null)
            props.onHover(null)
          }}
          onDoubleClick={props.onToggleFocus}
          onContextMenu={(e) => e.preventDefault()}
        />

        {status === "loading" ? (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-stage-foreground">
            <Spinner className="size-5" />
          </div>
        ) : null}
        {blocked ? (
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-1 bg-[repeating-linear-gradient(135deg,transparent_0_8px,rgb(255_255_255/0.04)_8px_16px)] p-4 text-center text-stage-foreground">
            <VideoOffIcon className="size-5 opacity-70" aria-hidden />
            <p className="text-sm font-medium">No footage in {shot.shot_id}</p>
            <p className="text-xs opacity-70">
              Shot frame <span className="font-mono">{shotFrame}</span> is outside 0-{shotVideoRange(shot)[1]}. Scrub elsewhere.
            </p>
          </div>
        ) : null}

        {/* Overlay header: canvas-overlay recipe (80% background, blur). */}
        <div className="pointer-events-none absolute top-2 left-2 flex items-center gap-2 rounded-md border bg-background/80 px-2 py-1 text-xs shadow-sm backdrop-blur">
          <span
            className="inline-flex size-4 items-center justify-center rounded-sm font-mono text-[10px] font-semibold text-black"
            style={{ backgroundColor: colour }}
            aria-hidden
          >
            {letter}
          </span>
          <span className="font-medium">{shot.shot_id}</span>
          <span className="font-mono text-muted-foreground tabular-nums">
            {shot.frame_offset === 0 ? "ref" : shot.frame_offset > 0 ? `+${shot.frame_offset}` : shot.frame_offset}
          </span>
          <span className="text-muted-foreground">
            frame <span className="font-mono tabular-nums">{shotFrame}</span>
          </span>
          {tf.zoom > 1.01 ? <span className="font-mono text-muted-foreground tabular-nums">{tf.zoom.toFixed(1)}x</span> : null}
          {props.isRepeat ? (
            <span className="rounded-sm bg-warning/20 px-1 text-warning" title="This frame repeats the previous image (25 to 30 fps pulldown). Pick on a fresh frame.">
              repeat
            </span>
          ) : null}
          {shot.excluded ? <span className="text-warning">excluded shot</span> : null}
        </div>
        {props.onToggleFocus ? (
          <Button
            variant="outline"
            size="icon-xs"
            className="absolute top-2 right-2 bg-background/80 backdrop-blur"
            aria-label={props.focused ? "Leave focus view" : "Focus this view"}
            onClick={props.onToggleFocus}
          >
            {props.focused ? <Minimize2Icon /> : <Maximize2Icon />}
          </Button>
        ) : null}
        {props.note ? (
          <p className="pointer-events-none absolute bottom-2 left-2 rounded-md border bg-background/80 px-2 py-1 text-xs backdrop-blur">
            {props.note}
          </p>
        ) : null}

        {loupe && hover && cursor ? (
          <canvas
            ref={loupeRef}
            width={LOUPE_SIZE}
            height={LOUPE_SIZE}
            aria-hidden
            className="pointer-events-none absolute rounded-md border-2 bg-stage shadow-md"
            style={{
              width: LOUPE_SIZE,
              height: LOUPE_SIZE,
              borderColor: colour,
              left: Math.min(Math.max(0, cursor[0] + 24), Math.max(0, size.w - LOUPE_SIZE)),
              top: Math.min(Math.max(0, cursor[1] + 24), Math.max(0, size.h - LOUPE_SIZE)),
            }}
          />
        ) : null}
      </div>
    </div>
  )
}
