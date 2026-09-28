import { Panel, PanelEmpty } from "@/components/panel"
import { ToneBadge } from "@/components/status"
import { cn } from "@/lib/utils"

import type { RenderCamera, RenderShotOutput } from "./camera-options"

function formatDuration(seconds: number): string {
  const s = Math.round(seconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  if (h > 0) return `${h}h ${m}m ${sec}s`
  if (m > 0) return `${m}m ${sec}s`
  return `${sec}s`
}

function CameraCard({ shotId, cam }: { shotId: string; cam: RenderCamera }) {
  const stem = cam.file.replace(/\.mp4$/, "")
  const mb = (cam.size_bytes / (1024 * 1024)).toFixed(1)
  return (
    <figure className={cn("flex min-w-0 flex-col overflow-hidden rounded-lg border bg-card", cam.vertical && "max-w-52")}>
      <div className="bg-stage">
        <video
          // #t=0.1 makes browsers paint a poster frame instead of a black box.
          src={`/api/render/video/${encodeURIComponent(shotId)}/${encodeURIComponent(stem)}#t=0.1`}
          controls
          preload="metadata"
          className="block w-full"
        />
      </div>
      <figcaption className="flex flex-wrap items-center gap-2 px-3 py-2">
        <span className="text-sm font-medium">
          {cam.id}
          {cam.vertical ? " (9:16)" : ""}
        </span>
        <ToneBadge tone="muted" className="tabular-nums">
          {mb} MB
        </ToneBadge>
      </figcaption>
    </figure>
  )
}

/** Rendered camera videos for one shot; each landscape card is followed by its 9:16 pair. */
export function CameraGrid({ shotId, output }: { shotId: string; output: RenderShotOutput | undefined }) {
  const cameras = output?.cameras ?? []
  const summary =
    output && cameras.length ? (
      <span className="flex flex-wrap items-center gap-2">
        {output.render_seconds != null ? <span>Rendered in {formatDuration(output.render_seconds)}</span> : null}
        {output.aov ? <ToneBadge tone="info">AOV passes</ToneBadge> : null}
      </span>
    ) : undefined

  if (!cameras.length) {
    return (
      <Panel title={`Rendered cameras: ${shotId}`}>
        <PanelEmpty
          title="No renders for this shot yet"
          description="Choose cameras in the selection below, then click Render."
        />
      </Panel>
    )
  }

  const landscape = cameras.filter((c) => !c.vertical)
  const verticalById = new Map(cameras.filter((c) => c.vertical).map((c) => [c.id, c]))
  const landscapeIds = new Set(landscape.map((c) => c.id))
  const orphans = cameras.filter((c) => c.vertical && !landscapeIds.has(c.id))

  return (
    <Panel title={`Rendered cameras: ${shotId}`} description={summary}>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
        {landscape.flatMap((cam) => {
          const v = verticalById.get(cam.id)
          return [
            <CameraCard key={cam.file} shotId={shotId} cam={cam} />,
            ...(v ? [<CameraCard key={v.file} shotId={shotId} cam={v} />] : []),
          ]
        })}
        {orphans.map((cam) => (
          <CameraCard key={cam.file} shotId={shotId} cam={cam} />
        ))}
      </div>
    </Panel>
  )
}
