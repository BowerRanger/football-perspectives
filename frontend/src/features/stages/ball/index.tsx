import * as React from "react"
import { useSearchParams } from "react-router"

import { Panel, PanelEmpty, PanelError, PanelSkeleton } from "@/components/panel"
import { ToneBadge } from "@/components/status"
import { Button } from "@/components/ui/button"
import { useConfirm } from "@/hooks/use-dialogs"
import { useResource } from "@/hooks/use-resource"
import { getJson } from "@/lib/api"
import { fmt, fmtInt, fmtPct } from "@/lib/format"
import { loadShotOptions, type BallPreviewTrack } from "@/pages/ball-anchor-editor/api"
import { BallAnchorEditor } from "@/pages/ball-anchor-editor"
import { ShotSelect } from "@/pages/ball-anchor-editor/shot-select"
import { Ball3D } from "./ball-3d"
import { SegmentsTable } from "./segments-table"
import { TopdownCanvas } from "./topdown-canvas"

function RetryButton({ onClick }: { onClick: () => void }) {
  return (
    <Button variant="outline" size="sm" className="mt-2" onClick={onClick}>
      Retry
    </Button>
  )
}

// /ball/preview answers 200 with empty frames until the stage has run, so a
// throw is a real failure (the shared loadBallTrack swallows those as null).
const loadTrack = (shot: string, signal: AbortSignal) =>
  getJson<BallPreviewTrack>(`/ball/preview?shot=${encodeURIComponent(shot)}`, { signal })

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
  const shotsRes = useResource(() => loadShotOptions(), [])
  const options = shotsRes.state.status === "ready" ? shotsRes.state.data : []
  const shot = params.get("shot") ?? options[0]?.id ?? ""
  const trackRes = useResource((signal) => (shot ? loadTrack(shot, signal) : Promise.resolve(null)), [shot])
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

  const track = trackRes.state.status === "ready" ? trackRes.state.data : null
  const frames = track?.frames ?? []
  const predicted = React.useMemo(() => frames.filter((f) => f.world_xyz), [frames])

  if (shotsRes.state.status === "loading") return <PanelSkeleton rows={5} media />
  if (shotsRes.state.status === "error") {
    return (
      <PanelError
        title="Could not list shots"
        message={shotsRes.state.error}
        action={<RetryButton onClick={shotsRes.retry} />}
      />
    )
  }
  if (!options.length) {
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
        <ShotSelect value={shot} options={options} onChange={(s) => void onShot(s)} id="ball-stage-shot" />
      </div>
      {trackRes.state.status === "loading" ? <PanelSkeleton rows={4} /> : null}
      {trackRes.state.status === "error" ? (
        <PanelError
          title="Could not load the ball track"
          message={trackRes.state.error}
          action={<RetryButton onClick={trackRes.retry} />}
        />
      ) : null}
      {trackRes.state.status === "ready" && !frames.length ? (
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
