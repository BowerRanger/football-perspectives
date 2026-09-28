import * as React from "react"
import { useSearchParams } from "react-router"

import { Panel, PanelEmpty, PanelError, PanelSkeleton } from "@/components/panel"
import { ToneBadge } from "@/components/status"
import { useConfirm } from "@/hooks/use-dialogs"
import { errorMessage } from "@/lib/api"
import { fmt, fmtInt, fmtPct } from "@/lib/format"
import { loadBallTrack, type BallPreviewTrack } from "@/pages/ball-anchor-editor/api"
import { BallAnchorEditor } from "@/pages/ball-anchor-editor"
import { ShotSelect, useShotOptions } from "@/pages/ball-anchor-editor/shot-select"
import { Ball3D } from "./ball-3d"
import { SegmentsTable } from "./segments-table"
import { TopdownCanvas } from "./topdown-canvas"

type TrackState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; track: BallPreviewTrack | null }

function useBallTrack(shot: string): TrackState {
  const [state, setState] = React.useState<TrackState>({ kind: "loading" })
  React.useEffect(() => {
    if (!shot) return
    let cancelled = false
    setState({ kind: "loading" })
    loadBallTrack(shot)
      .then((track) => !cancelled && setState({ kind: "ready", track }))
      .catch((err: unknown) => !cancelled && setState({ kind: "error", message: errorMessage(err) }))
    return () => {
      cancelled = true
    }
  }, [shot])
  return state
}

function Summary({ track }: { track: BallPreviewTrack }) {
  const frames = track.frames ?? []
  const segs = track.flight_segments ?? []
  const states = new Map<string, number>()
  for (const f of frames) states.set(f.state, (states.get(f.state) ?? 0) + 1)
  const withSpin = segs.filter((s) => s.parabola?.spin_omega_rad_s != null).length
  return (
    <Panel title="Ball track summary">
      <div className="flex flex-col gap-3">
        <dl className="grid gap-x-8 gap-y-2 text-sm sm:grid-cols-2 xl:grid-cols-4">
          {[
            { label: "Clip", value: track.clip_id || "(unnamed)" },
            { label: "FPS", value: fmt(track.fps, 2) },
            { label: "Frames", value: fmtInt(frames.length) },
            { label: "Flight segments", value: `${segs.length} (with spin: ${withSpin})` },
          ].map((it) => (
            <div key={it.label} className="flex items-baseline justify-between gap-3 border-b border-border/50 pb-1.5">
              <dt className="text-muted-foreground">{it.label}</dt>
              <dd className="shrink-0 text-right font-medium tabular-nums">{it.value}</dd>
            </div>
          ))}
        </dl>
        <div className="flex flex-wrap items-center gap-2" aria-label="State distribution">
          <span className="text-sm text-muted-foreground">State distribution</span>
          {[...states.entries()].map(([s, n]) => (
            <ToneBadge key={s} tone={s === "flight" ? "warning" : s === "grounded" ? "success" : "muted"}>
              {s}: {n} ({fmtPct(n / Math.max(1, frames.length), 1)})
            </ToneBadge>
          ))}
        </div>
        {segs.length ? <SegmentsTable segments={segs} fps={track.fps} /> : null}
      </div>
    </Panel>
  )
}

function Trajectories({ frames, frame }: { frames: NonNullable<BallPreviewTrack["frames"]>; frame: number }) {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Panel title="Top-down trajectory" description="Follows the frame selected in the editor above.">
        <TopdownCanvas frames={frames} frame={frame} />
      </Panel>
      <Panel title="3D trajectory" description="Drag to orbit, scroll to zoom.">
        <Ball3D frames={frames} frame={frame} />
      </Panel>
    </div>
  )
}

/** Ball stage: summary, the anchor editor (single shared implementation), then trajectory views. */
export default function BallStage() {
  const [params, setParams] = useSearchParams()
  const shots = useShotOptions()
  const shot = params.get("shot") ?? shots.options[0]?.id ?? ""
  const trackState = useBallTrack(shot)
  const [frame, setFrame] = React.useState(0)
  const confirm = useConfirm()
  const [dirty, setDirty] = React.useState(false)

  const onShot = async (next: string) => {
    if (next === shot) return
    if (dirty) {
      const ok = await confirm({
        title: "Discard unsaved anchors?",
        description: `You have unsaved changes on ${shot}. Switching to ${next} will discard them.`,
        confirmLabel: "Discard and switch",
        destructive: true,
      })
      if (!ok) return
    }
    setParams(
      (prev) => {
        const p = new URLSearchParams(prev)
        p.set("shot", next)
        return p
      },
      { replace: true },
    )
  }

  const track = trackState.kind === "ready" ? trackState.track : null
  const frames = track?.frames ?? []
  const predicted = React.useMemo(() => frames.filter((f) => f.world_xyz), [frames])

  if (shots.loading) return <PanelSkeleton rows={5} media />
  if (shots.error) return <PanelError title="Could not list shots" message={shots.error} />
  if (!shots.options.length) {
    return (
      <PanelEmpty
        title="No shots yet"
        description="Run the Prepare Shots stage first; the ball stage works per shot."
      />
    )
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <ShotSelect value={shot} options={shots.options} onChange={(s) => void onShot(s)} id="ball-stage-shot" />
      </div>
      {trackState.kind === "loading" ? <PanelSkeleton rows={4} /> : null}
      {trackState.kind === "error" ? <PanelError title="Could not load the ball track" message={trackState.message} /> : null}
      {trackState.kind === "ready" && !frames.length ? (
        <PanelEmpty
          title="No ball track yet"
          description="Drop anchors below (optional), then run the Ball stage from the header. The editor only needs the clip and camera track."
        />
      ) : null}
      {track && frames.length ? <Summary track={track} /> : null}
      <Panel title="Ball anchor editor" description="Operator anchors always win over auto-detected events.">
        <BallAnchorEditor key={shot} embedded shot={shot} predicted={predicted} onFrameChange={setFrame} onDirtyChange={setDirty} />
      </Panel>
      {frames.length ? <Trajectories frames={frames} frame={frame} /> : null}
    </div>
  )
}
