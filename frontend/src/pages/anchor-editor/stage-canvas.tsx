import * as React from "react"
import { VideoOffIcon } from "lucide-react"

import { Kbd } from "@/components/ui/kbd"
import { Skeleton } from "@/components/ui/skeleton"
import { cn } from "@/lib/utils"
import { drawOverlay, type OverlayInput } from "./draw"
import type { Vec2 } from "./types"

export interface VideoMeta {
  width: number
  height: number
  duration: number
}

export interface StageCanvasProps {
  shot: string
  videoRef: React.RefObject<HTMLVideoElement | null>
  imageSize: Vec2
  overlay: Omit<OverlayInput, "width" | "height" | "scale">
  videoHandlers: { onPlay: () => void; onPause: () => void; onEnded: () => void }
  onMeta: (meta: VideoMeta) => void
  /** A landmark or line is armed: clicks place points. */
  armed: boolean
  hint: string | null
  onPlace: (xy: Vec2) => void
  onDeleteAnchor: () => void
}

interface Box {
  w: number
  h: number
}

/** Largest box of the given aspect that fits inside `outer`. */
function fit(outer: Box, aspect: number): Box {
  if (outer.w <= 0 || outer.h <= 0) return { w: 0, h: 0 }
  return outer.w / outer.h > aspect ? { w: outer.h * aspect, h: outer.h } : { w: outer.w, h: outer.w / aspect }
}

function useElementBox(ref: React.RefObject<HTMLElement | null>): Box {
  const [box, setBox] = React.useState<Box>({ w: 0, h: 0 })
  React.useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const measure = () => setBox({ w: el.clientWidth, h: el.clientHeight })
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])
  return box
}

export function StageCanvas(props: StageCanvasProps) {
  const { shot, videoRef, imageSize, overlay, videoHandlers, onMeta, armed, hint, onPlace, onDeleteAnchor } = props
  const wellRef = React.useRef<HTMLDivElement>(null)
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const outer = useElementBox(wellRef)
  const [iw, ih] = imageSize
  const aspect = iw > 0 && ih > 0 ? iw / ih : 16 / 9
  const box = fit(outer, aspect)
  const scale = iw > 0 && box.w > 0 ? iw / box.w : 1
  const [ready, setReady] = React.useState(false)
  const [failed, setFailed] = React.useState(false)

  React.useEffect(() => {
    setReady(false)
    setFailed(false)
  }, [shot])

  React.useEffect(() => {
    const canvas = canvasRef.current
    if (canvas) drawOverlay(canvas, { ...overlay, width: iw, height: ih, scale })
  }, [overlay, iw, ih, scale])

  const toImageXY = (ev: React.MouseEvent<HTMLCanvasElement>): Vec2 => {
    const rect = ev.currentTarget.getBoundingClientRect()
    if (rect.width === 0 || rect.height === 0) return [0, 0]
    return [((ev.clientX - rect.left) * iw) / rect.width, ((ev.clientY - rect.top) * ih) / rect.height]
  }

  return (
    <div ref={wellRef} className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-stage">
      <div className="relative" style={{ width: box.w, height: box.h }}>
        <video
          key={shot}
          ref={videoRef}
          src={`/api/video/${encodeURIComponent(shot)}`}
          preload="metadata"
          playsInline
          muted
          aria-label={`Video for shot ${shot}`}
          className="block size-full"
          onLoadedMetadata={(e) => {
            setReady(true)
            const v = e.currentTarget
            onMeta({ width: v.videoWidth, height: v.videoHeight, duration: v.duration })
          }}
          onError={() => setFailed(true)}
          {...videoHandlers}
        />
        <canvas
          ref={canvasRef}
          width={iw || undefined}
          height={ih || undefined}
          aria-label="Frame overlay. Click to place the selected landmark; right-click to delete this frame's anchor."
          className={cn("absolute inset-0 size-full", armed ? "cursor-crosshair" : "cursor-default")}
          onClick={(ev) => onPlace(toImageXY(ev))}
          onContextMenu={(ev) => {
            ev.preventDefault()
            onDeleteAnchor()
          }}
        />
      </div>
      {!ready && !failed ? <Skeleton className="absolute inset-4 rounded-lg" /> : null}
      {failed ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-stage p-4 text-center text-stage-foreground">
          <VideoOffIcon className="size-6 opacity-70" aria-hidden />
          <p className="text-sm font-medium">Could not load the video for {shot}</p>
          <p className="text-xs opacity-70">Run prepare_shots to (re)create shots/{shot}.mp4, then reload.</p>
        </div>
      ) : null}
      {hint ? (
        <div
          role="status"
          className="pointer-events-none absolute top-3 left-1/2 flex max-w-[90%] -translate-x-1/2 items-center gap-2 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground shadow-lg"
        >
          <span className="truncate">{hint}</span>
          <Kbd className="bg-primary-foreground/20 text-primary-foreground">Esc</Kbd>
        </div>
      ) : null}
    </div>
  )
}
