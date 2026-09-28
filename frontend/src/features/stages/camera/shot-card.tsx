import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { StatList } from "@/components/panel"
import { fmt } from "@/lib/format"
import { CameraMetricsBlock } from "./camera-metrics"
import type { ShotCameraData } from "./types"

const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? "" : "s"}`

/** Compact per-shot summary: frames, anchors, and the honest quality metrics. */
export function ShotInfoCard({ id, colour, track, anchors }: ShotCameraData) {
  const hasTrack = !!track?.frames?.length
  const anchorN = anchors?.anchors?.length ?? 0
  const frames = track?.frames ?? []
  const anchoredFrames = frames.filter((f) => f.is_anchor).length

  return (
    <Card size="sm" className="gap-3">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <span aria-hidden className="size-2.5 shrink-0 rounded-full" style={{ backgroundColor: colour }} />
          <span className="font-mono">{id}</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {hasTrack && track ? (
          <>
            <StatList
              items={[
                { label: "Frames", value: `${frames.length} @ ${fmt(track.fps, 2)} fps` },
                { label: "Image size", value: track.image_size.join("×") },
                { label: "Anchored frames", value: `${anchoredFrames} / ${frames.length}` },
              ]}
            />
            <Separator />
            <CameraMetricsBlock shot={id} />
          </>
        ) : (
          <p className="text-sm text-muted-foreground">
            No camera track yet — run camera tracking for this shot.
          </p>
        )}
        <Separator />
        <p className="text-sm">
          {anchorN > 0 ? (
            <>
              Anchor set: <strong className="tabular-nums">{plural(anchorN, "frame")}</strong>
            </>
          ) : (
            <span className="text-muted-foreground">No anchors yet — mark landmarks in the editor below.</span>
          )}
        </p>
      </CardContent>
    </Card>
  )
}
