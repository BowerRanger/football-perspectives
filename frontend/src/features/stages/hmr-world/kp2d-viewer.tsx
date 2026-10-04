import { frameAtTime, frameTime } from "@/lib/frame-time"
import * as React from "react"

import { Panel } from "@/components/panel"
import { FramePlayer } from "@/components/frame-player"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { playerLabel } from "@/lib/format"
import { drawSkeleton } from "./skeleton"
import { useCameraTrack } from "./use-camera-track"
import type { Coloured, Kp2dPreview, PlayerRef } from "./types"

export type Kp2dPlayer = Coloured<PlayerRef> & { data: Kp2dPreview }

const ALL = "__all__"

interface FrameEntry {
  player: Kp2dPlayer
  keypoints: number[][]
}

function indexByFrame(players: readonly Kp2dPlayer[]): Map<number, FrameEntry[]> {
  const map = new Map<number, FrameEntry[]>()
  for (const p of players) {
    for (const f of p.data.frames) {
      const arr = map.get(f.frame) ?? []
      arr.push({ player: p, keypoints: f.keypoints })
      map.set(f.frame, arr)
    }
  }
  return map
}

/** Shot video with a COCO-17 skeleton overlay; "All players" or one player. */
export function Kp2dViewer({ shotId, players }: { shotId: string; players: readonly Kp2dPlayer[] }) {
  const { fps } = useCameraTrack()
  const videoRef = React.useRef<HTMLVideoElement>(null)
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const [selected, setSelected] = React.useState(ALL)
  const [frame, setFrame] = React.useState(0)
  const [maxFrame, setMaxFrame] = React.useState(1000)
  const [playing, setPlaying] = React.useState(false)
  const byFrame = React.useMemo(() => indexByFrame(players), [players])

  const currentFrame = React.useCallback(() => frameAtTime(videoRef.current?.currentTime ?? 0, fps), [fps])

  // rAF loop while playing keeps the overlay locked to the video clock.
  React.useEffect(() => {
    if (!playing) return
    let raf = 0
    const loop = () => {
      setFrame(currentFrame())
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [playing, currentFrame])

  React.useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext("2d")
    if (!canvas || !ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    for (const e of byFrame.get(frame) ?? []) {
      if (selected !== ALL && e.player.player_id !== selected) continue
      drawSkeleton(ctx, e.keypoints, e.player.colour, playerLabel(e.player))
    }
  }, [byFrame, frame, selected])

  const seekTo = (f: number) => {
    const v = videoRef.current
    if (!v) return
    v.pause()
    v.currentTime = Math.max(0, Math.min(v.duration || 0, frameTime(f, fps)))
    setFrame(f)
  }

  const onSelect = (value: string) => {
    setSelected(value)
    if (value === ALL) return
    // Jump to the middle of the player's frame range so the overlay isn't blank.
    const p = players.find((x) => x.player_id === value)
    if (p && p.data.frames.length) seekTo(p.data.frames[Math.floor(p.data.frames.length / 2)].frame)
  }

  const onMeta = () => {
    const v = videoRef.current
    const c = canvasRef.current
    if (!v || !c) return
    c.width = v.videoWidth
    c.height = v.videoHeight
    setMaxFrame(Math.round(v.duration * fps))
    setFrame(currentFrame())
  }

  return (
    <Panel
      title="Keypoint viewer"
      description="GVHMR's internal ViTPose keypoints over the shot clip."
      actions={
        <Select value={selected} onValueChange={onSelect}>
          <SelectTrigger size="sm" className="w-52" aria-label="Player">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL}>All players ({players.length})</SelectItem>
            {players.map((p) => (
              <SelectItem key={p.player_id} value={p.player_id}>
                {playerLabel(p)} ({p.data.frames.length} frames)
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      }
    >
      <div className="flex flex-col gap-3">
        <div className="relative overflow-hidden rounded-md bg-stage">
          <video
            ref={videoRef}
            src={`/api/video/${encodeURIComponent(shotId)}`}
            preload="metadata"
            muted
            playsInline
            className="block h-auto w-full"
            onLoadedMetadata={onMeta}
            onPlay={() => setPlaying(true)}
            onPause={() => {
              setPlaying(false)
              setFrame(currentFrame())
            }}
            onSeeked={() => setFrame(currentFrame())}
          />
          <canvas ref={canvasRef} aria-label="Keypoint overlay" className="pointer-events-none absolute inset-0 size-full" />
        </div>
        <FramePlayer
          frame={frame}
          max={maxFrame}
          fps={fps}
          playing={playing}
          onTogglePlay={() => {
            const v = videoRef.current
            if (!v) return
            if (v.paused) void v.play()
            else v.pause()
          }}
          onSeek={seekTo}
          label="Keypoint frame"
          keyHints
        />
      </div>
    </Panel>
  )
}
