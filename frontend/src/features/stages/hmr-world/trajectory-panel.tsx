import * as React from "react"

import { Panel } from "@/components/panel"
import { Badge } from "@/components/ui/badge"
import { getJsonOrNull } from "@/lib/api"
import { cn } from "@/lib/utils"
import { TransportBar } from "./transport-bar"
import { CANVAS_H, CANVAS_W } from "./pitch-canvas"
import {
  buildTrajectoryPlayers,
  frameRange,
  renderTrajectoryFrame,
  type TrajectoryInput,
} from "./trajectory-draw"
import { useCameraTrack } from "./use-camera-track"
import { useTrajectoryPlayback } from "./use-trajectory-playback"

interface TrajectoryPanelProps {
  players: readonly TrajectoryInput[]
  /** Shot whose clip plays beside the pitch; falls back to the first tracked shot. */
  shotId: string | null
  title?: string
}

function useClipSrc(shotId: string | null): string | null {
  const [fallback, setFallback] = React.useState<string | null>(null)
  React.useEffect(() => {
    if (shotId) return
    let alive = true
    void getJsonOrNull<{ shots?: string[] }>("/tracking/shots").then((d) => {
      if (alive && d?.shots?.[0]) setFallback(d.shots[0])
    })
    return () => {
      alive = false
    }
  }, [shotId])
  const id = shotId ?? fallback
  return id ? `/api/video/${encodeURIComponent(id)}` : null
}

function LegendBadge({ label, colour, hidden, onToggle }: { label: string; colour: string; hidden: boolean; onToggle: () => void }) {
  return (
    <button type="button" onClick={onToggle} aria-pressed={!hidden} className="rounded-full focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none">
      <Badge variant="outline" className={cn("cursor-pointer gap-1.5 hover:bg-muted", hidden && "opacity-40")}>
        <span aria-hidden className="size-2 rounded-full" style={{ backgroundColor: colour }} />
        {label}
      </Badge>
    </button>
  )
}

/** Top-down per-player trajectories on a pitch, synced to the shot clip. */
export function TrajectoryPanel({ players: inputs, shotId, title = "Top-down trajectories" }: TrajectoryPanelProps) {
  const players = React.useMemo(() => buildTrajectoryPlayers(inputs), [inputs])
  const { min, max } = React.useMemo(() => frameRange(players), [players])
  const { fps, cameraByFrame } = useCameraTrack()
  const clipSrc = useClipSrc(shotId)
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const videoRef = React.useRef<HTMLVideoElement>(null)
  const [videoReady, setVideoReady] = React.useState(false)
  const [hidden, setHidden] = React.useState<ReadonlySet<string>>(new Set())
  const { frame, playing, seek, toggle, step } = useTrajectoryPlayback({ min, max, fps, videoRef, videoReady })

  React.useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d")
    if (!ctx) return
    const cam = cameraByFrame.get(frame) ?? cameraByFrame.get(min)
    renderTrajectoryFrame(ctx, players, hidden, frame, cam)
  }, [players, hidden, frame, min, cameraByFrame])

  const toggleHidden = (pid: string) =>
    setHidden((prev) => {
      const next = new Set(prev)
      if (next.has(pid)) next.delete(pid)
      else next.add(pid)
      return next
    })

  const sec = (frame / Math.max(1, fps)).toFixed(2)
  return (
    <Panel
      title={title}
      description="Pitch-world root positions with a fading trail; the camera marker should sit opposite the action. Click a player to hide or show them."
    >
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-3 md:h-[62vh] md:flex-row">
          <div className="flex min-h-0 min-w-0 flex-[7] items-center justify-center overflow-hidden rounded-md bg-stage">
            <canvas
              ref={canvasRef}
              width={CANVAS_W}
              height={CANVAS_H}
              aria-label="Top-down pitch with player trajectories"
              className="block h-auto max-h-full w-auto max-w-full"
            />
          </div>
          <div className="flex min-w-0 flex-[3] items-center justify-center overflow-hidden rounded-md bg-stage">
            {clipSrc ? (
              <video
                ref={videoRef}
                src={clipSrc}
                muted
                playsInline
                preload="metadata"
                onLoadedMetadata={() => setVideoReady(true)}
                onError={() => setVideoReady(false)}
                className="block max-h-full w-full object-contain"
              />
            ) : (
              <span className="p-4 text-xs text-muted-foreground">No clip available</span>
            )}
          </div>
        </div>
        <TransportBar
          playing={playing}
          onToggle={toggle}
          onPrev={() => step(-1)}
          onNext={() => step(1)}
          value={frame}
          min={min}
          max={max}
          onSeek={seek}
          readout={`${frame} (${sec}s)`}
        />
        <div className="flex flex-wrap gap-1.5">
          {players.map((p) => (
            <LegendBadge key={p.pid} label={p.label} colour={p.colour} hidden={hidden.has(p.pid)} onToggle={() => toggleHidden(p.pid)} />
          ))}
        </div>
      </div>
    </Panel>
  )
}
