import * as React from "react"
import { ChevronLeftIcon, ChevronRightIcon, PauseIcon, PlayIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Kbd, KbdGroup } from "@/components/ui/kbd"
import { Slider } from "@/components/ui/slider"
import { drawTrackOverlay, hitTestBoxes } from "./overlay"
import type { FrameBox } from "./types"
import { videoUrl } from "./api"

export interface TrackVideoHandle {
  currentFrame: () => number
  seekToFrame: (frame: number) => void
}

interface TrackVideoProps {
  ref?: React.Ref<TrackVideoHandle>
  shotId: string
  fps: number
  boxesByFrame: ReadonlyMap<number, FrameBox[]>
  nameByTrack: ReadonlyMap<string, string>
  highlightIds: ReadonlySet<string>
  onPickTrack: (trackId: string) => void
}

const SKIP_TAGS = new Set(["INPUT", "TEXTAREA", "SELECT"])

function isTypingTarget(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false
  return SKIP_TAGS.has(t.tagName) || t.isContentEditable
}

function isActivatable(t: EventTarget | null): boolean {
  return t instanceof HTMLElement && (t.tagName === "BUTTON" || t.tagName === "A" || t.getAttribute("role") === "slider")
}

/** Video + bbox overlay canvas + transport. Owns playback and keyboard shortcuts. */
export function TrackVideo({ ref, shotId, fps, boxesByFrame, nameByTrack, highlightIds, onPickTrack }: TrackVideoProps) {
  const videoRef = React.useRef<HTMLVideoElement>(null)
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const [frame, setFrame] = React.useState(0)
  const [maxFrame, setMaxFrame] = React.useState(0)
  const [playing, setPlaying] = React.useState(false)

  const latest = React.useRef({ boxesByFrame, nameByTrack, highlightIds, fps })
  latest.current = { boxesByFrame, nameByTrack, highlightIds, fps }

  const draw = React.useCallback(() => {
    const v = videoRef.current
    const c = canvasRef.current
    if (!v || !c) return
    const { boxesByFrame: idx, nameByTrack: names, highlightIds: hl, fps: f } = latest.current
    const fi = Math.round(v.currentTime * f)
    setFrame(fi)
    drawTrackOverlay(c, idx.get(fi) ?? [], hl, names)
  }, [])

  React.useEffect(draw, [draw, boxesByFrame, nameByTrack, highlightIds])

  React.useEffect(() => {
    if (!playing) return
    let raf = requestAnimationFrame(function loop() {
      draw()
      raf = requestAnimationFrame(loop)
    })
    return () => cancelAnimationFrame(raf)
  }, [playing, draw])

  const seekToFrame = React.useCallback(
    (fi: number) => {
      const v = videoRef.current
      if (!v) return
      v.pause()
      const dur = Number.isFinite(v.duration) ? v.duration : Infinity
      v.currentTime = Math.min(dur, Math.max(0, fi / fps))
    },
    [fps],
  )
  const step = React.useCallback(
    (delta: number) => {
      const v = videoRef.current
      if (v) seekToFrame(Math.round(v.currentTime * fps) + delta)
    },
    [fps, seekToFrame],
  )
  const togglePlay = React.useCallback(() => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) void v.play()
    else v.pause()
  }, [])

  React.useImperativeHandle(
    ref,
    () => ({
      currentFrame: () => Math.round((videoRef.current?.currentTime ?? 0) * fps),
      seekToFrame,
    }),
    [fps, seekToFrame],
  )

  React.useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.metaKey || e.ctrlKey || e.altKey || isTypingTarget(e.target)) return
      const big = e.shiftKey ? 10 : 1
      if (e.key === "ArrowLeft" && !isActivatable(e.target)) step(-big)
      else if (e.key === "ArrowRight" && !isActivatable(e.target)) step(big)
      else if (e.key === " " && !isActivatable(e.target)) togglePlay()
      else return
      e.preventDefault()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [step, togglePlay])

  function onCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const c = e.currentTarget
    if (!c.width) return
    const rect = c.getBoundingClientRect()
    const x = (e.clientX - rect.left) * (c.width / rect.width)
    const y = (e.clientY - rect.top) * (c.height / rect.height)
    const fi = Math.round((videoRef.current?.currentTime ?? 0) * fps)
    const hit = hitTestBoxes(boxesByFrame.get(fi) ?? [], x, y)
    if (hit) onPickTrack(hit.track_id)
    else togglePlay()
  }

  function onLoadedMetadata() {
    const v = videoRef.current
    const c = canvasRef.current
    if (!v || !c) return
    setMaxFrame(Math.round(v.duration * fps))
    c.width = v.videoWidth
    c.height = v.videoHeight
    draw()
  }

  return (
    <div className="flex min-w-0 flex-col gap-2">
      <div className="relative overflow-hidden rounded-lg bg-stage">
        <video
          ref={videoRef}
          src={videoUrl(shotId)}
          preload="metadata"
          aria-label={`Shot ${shotId} video`}
          className="block h-auto w-full"
          onLoadedMetadata={onLoadedMetadata}
          onPlay={() => setPlaying(true)}
          onPause={() => {
            setPlaying(false)
            draw()
          }}
          onSeeked={draw}
        />
        <canvas
          ref={canvasRef}
          aria-label="Track overlay: click a box to edit its player"
          className="absolute inset-0 size-full cursor-pointer"
          onClick={onCanvasClick}
        />
      </div>
      <div className="flex items-center gap-2">
        <Button size="icon-sm" onClick={togglePlay} aria-label={playing ? "Pause" : "Play"}>
          {playing ? <PauseIcon /> : <PlayIcon />}
        </Button>
        <Button size="icon-sm" variant="outline" onClick={() => step(-1)} aria-label="Previous frame">
          <ChevronLeftIcon />
        </Button>
        <Button size="icon-sm" variant="outline" onClick={() => step(1)} aria-label="Next frame">
          <ChevronRightIcon />
        </Button>
        <Slider
          className="flex-1"
          min={0}
          max={Math.max(1, maxFrame)}
          step={1}
          value={[Math.min(frame, Math.max(1, maxFrame))]}
          onValueChange={(v) => seekToFrame(v[0] ?? 0)}
        />
        <span className="min-w-20 text-right text-xs text-muted-foreground tabular-nums">Frame <span className="font-mono">{frame}</span></span>
      </div>
      <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
            <KbdGroup className="font-sans">
              <Kbd>Space</Kbd> play
            </KbdGroup>
            <KbdGroup className="font-sans">
              <Kbd>←</Kbd>
              <Kbd>→</Kbd> step
            </KbdGroup>
            <KbdGroup className="font-sans">
              <Kbd>Shift</Kbd>+<Kbd>←</Kbd>
              <Kbd>→</Kbd> ±10
            </KbdGroup>
            <span>Click a box to edit that player · shortcuts pause while typing</span>
      </p>
    </div>
  )
}
