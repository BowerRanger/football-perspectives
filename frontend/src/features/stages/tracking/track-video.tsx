import * as React from "react"

import { FramePlayer } from "@/components/frame-player"
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

/** Video + bbox overlay canvas + transport. Playback and shortcuts come from FramePlayer. */
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
      <FramePlayer
        frame={frame}
        max={Math.max(1, maxFrame)}
        playing={playing}
        onTogglePlay={togglePlay}
        onSeek={seekToFrame}
        fps={fps}
        keyHints
      />
      <p className="text-xs text-muted-foreground">Click a box to edit that player · shortcuts pause while typing</p>
    </div>
  )
}
