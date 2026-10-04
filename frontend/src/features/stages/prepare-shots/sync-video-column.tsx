import { frameAtTime } from "@/lib/frame-time"
import * as React from "react"

import { Label } from "@/components/ui/label"
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select"

interface SyncVideoColumnProps {
  role: "reference" | "active"
  title: string
  shotIds: string[]
  value: string
  /** Shot that must not be picked here (the other column's). */
  disabledId: string | null
  fps: number
  videoRef: React.RefObject<HTMLVideoElement | null>
  onChange: (shotId: string) => void
  onSeeked?: () => void
  onTimeUpdate?: () => void
  onLoadedMetadata?: () => void
  onEnded?: () => void
}

/** One labelled video with a shot picker and a frame/time readout. */
export function SyncVideoColumn({
  role,
  title,
  shotIds,
  value,
  disabledId,
  fps,
  videoRef,
  onChange,
  onSeeked,
  onTimeUpdate,
  onLoadedMetadata,
  onEnded,
}: SyncVideoColumnProps) {
  const [time, setTime] = React.useState(0)
  const selectId = `sync-${role}-shot`
  const update = () => setTime(videoRef.current?.currentTime ?? 0)

  return (
    <div className="flex min-w-0 flex-col gap-2 rounded-lg border p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Label htmlFor={selectId} className="font-semibold">
          {title}
        </Label>
        <NativeSelect
          id={selectId}
          size="sm"
          className="ml-auto"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          {shotIds.map((id) => (
            <NativeSelectOption key={id} value={id} disabled={id === disabledId && shotIds.length > 1}>
              {id}
            </NativeSelectOption>
          ))}
        </NativeSelect>
      </div>
      <div className="bg-stage">
        <video
          ref={videoRef}
          src={`/api/video/${encodeURIComponent(value)}`}
          controls
          preload="metadata"
          className="block aspect-video max-h-[60vh] w-full object-contain"
          onTimeUpdate={() => {
            update()
            onTimeUpdate?.()
          }}
          onSeeked={() => {
            update()
            onSeeked?.()
          }}
          onLoadedMetadata={() => {
            update()
            onLoadedMetadata?.()
          }}
          onEnded={onEnded}
        />
      </div>
      <p className="text-xs text-muted-foreground tabular-nums">
        Frame <span className="font-mono">{frameAtTime(time, fps)}</span> · {time.toFixed(2)}s
      </p>
    </div>
  )
}
