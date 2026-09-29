import * as React from "react"

import { FramePlayer } from "@/components/frame-player"
import { PanelEmpty } from "@/components/panel"
import { MAP_H, MAP_W, renderPitchFrame } from "./pitch-draw"
import type { IndexedShot } from "./types"

interface PitchMapProps {
  shots: readonly IndexedShot[]
}

/** Frame cursor + rAF playback shared by every shot on the map. */
function usePlayback(minFrame: number, maxFrame: number, fps: number) {
  const [frame, setFrame] = React.useState(minFrame)
  const [playing, setPlaying] = React.useState(false)
  const frameRef = React.useRef(frame)
  frameRef.current = frame

  React.useEffect(() => {
    if (!playing) return
    let last = 0
    let acc = 0
    let raf = requestAnimationFrame(function tick(ts) {
      if (last) acc += (ts - last) / 1000
      last = ts
      const advance = Math.floor(acc * fps)
      if (advance > 0) {
        acc -= advance / fps
        const next = Math.min(maxFrame, frameRef.current + advance)
        setFrame(next)
        if (next >= maxFrame) {
          setPlaying(false)
          return
        }
      }
      raf = requestAnimationFrame(tick)
    })
    return () => cancelAnimationFrame(raf)
  }, [playing, fps, maxFrame])

  const seek = React.useCallback(
    (f: number) => {
      setPlaying(false)
      setFrame(Math.min(maxFrame, Math.max(minFrame, f)))
    },
    [minFrame, maxFrame],
  )
  const toggle = React.useCallback(() => {
    if (playing) return setPlaying(false)
    if (frameRef.current >= maxFrame) setFrame(minFrame)
    setPlaying(true)
  }, [playing, minFrame, maxFrame])

  return { frame, playing, seek, toggle }
}

/**
 * One canvas, one pitch, every shot's camera marker at its per-frame centre,
 * in the shot's colour. Anchor frames get a gold outline; shots that don't
 * cover the current frame hold at their nearest end.
 */
export function OverlaidPitchMap({ shots }: PitchMapProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const minFrame = shots.length ? Math.min(...shots.map((s) => s.minFrame)) : 0
  const maxFrame = shots.length ? Math.max(...shots.map((s) => s.maxFrame)) : 0
  const fps = shots[0]?.fps || 30
  const { frame, playing, seek, toggle } = usePlayback(minFrame, maxFrame, fps)

  const wrapRef = React.useRef<HTMLDivElement>(null)
  const [cssW, setCssW] = React.useState(MAP_W)
  const dpr = typeof window === "undefined" ? 1 : window.devicePixelRatio || 1

  React.useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setCssW(Math.max(1, Math.round(el.clientWidth))))
    ro.observe(el)
    setCssW(Math.max(1, Math.round(el.clientWidth)))
    return () => ro.disconnect()
  }, [shots.length])

  // Backing store matches displayed size x DPR so labels are crisp at any width.
  const pxW = Math.round(cssW * dpr)
  const pxH = Math.round((cssW * MAP_H * dpr) / MAP_W)

  React.useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d")
    if (ctx) renderPitchFrame(ctx, shots, frame, cssW / MAP_W, dpr)
  }, [shots, frame, cssW, dpr, pxW, pxH])

  if (shots.length === 0) {
    return (
      <PanelEmpty
        title="No camera tracks yet"
        description="Run camera tracking (after placing anchors below) to see camera positions on the pitch."
      />
    )
  }

  return (
    <div className="flex flex-col gap-2" aria-label="Camera pitch map">
      <div ref={wrapRef} className="overflow-hidden rounded-lg bg-stage">
        <canvas
          ref={canvasRef}
          width={pxW}
          height={pxH}
          role="img"
          aria-label="Top-down pitch with each shot's camera position and view direction"
          className="block h-auto w-full"
        />
      </div>
      <FramePlayer
        frame={frame}
        min={minFrame}
        max={maxFrame}
        playing={playing}
        onTogglePlay={toggle}
        onSeek={seek}
        fps={fps}
        keyHints
      />
      <p className="text-xs text-muted-foreground">Gold outline = anchored frame</p>
    </div>
  )
}
