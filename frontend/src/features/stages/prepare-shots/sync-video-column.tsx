import { frameAtTime } from "@/lib/frame-time"
import * as React from "react"
import { CrosshairIcon } from "lucide-react"

import { ToneBadge } from "@/components/status"
import { Label } from "@/components/ui/label"
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select"
import { cn } from "@/lib/utils"

interface SyncVideoColumnProps {
  role: "reference" | "active"
  title: string
  shotIds: string[]
  value: string
  /** Shot that must not be picked here (the other column's). */
  disabledId: string | null
  fps: number
  videoRef: React.RefObject<HTMLVideoElement | null>
  /** Speed badge (member) or Reference badge, shown in the header chip row. */
  badge?: React.ReactNode
  /** Browser playbackRate to apply (1 = normal). */
  playbackRate?: number
  /** Appended to the clip URL so a retime / restore reloads the element. */
  clipVersion?: string
  /** This well receives frame-step keys. */
  focused?: boolean
  onFocusWell?: () => void
  /** Pending Match-moments mark in this clip's own frame numbers. */
  pendingMark?: number | null
  /** Reference-clock equivalent of the member's frame (member well only), e.g. "ref 137.2". */
  equivalent?: (frame: number) => string
  onChange: (shotId: string) => void
  onSeeked?: () => void
  onTimeUpdate?: () => void
  onLoadedMetadata?: () => void
  onEnded?: () => void
}

/** One labelled video with a shot picker, speed chip and a frame/time readout. */
export function SyncVideoColumn({
  role,
  title,
  shotIds,
  value,
  disabledId,
  fps,
  videoRef,
  badge,
  playbackRate = 1,
  clipVersion = "",
  focused,
  onFocusWell,
  pendingMark,
  equivalent,
  onChange,
  onSeeked,
  onTimeUpdate,
  onLoadedMetadata,
  onEnded,
}: SyncVideoColumnProps) {
  const [time, setTime] = React.useState(0)
  const selectId = `sync-${role}-shot`
  const update = () => setTime(videoRef.current?.currentTime ?? 0)

  // The element resets playbackRate on a new source; re-apply on every change.
  const applyRate = React.useCallback(() => {
    const v = videoRef.current
    if (v && v.playbackRate !== playbackRate) v.playbackRate = playbackRate
  }, [videoRef, playbackRate])
  React.useEffect(applyRate, [applyRate, value, clipVersion])

  const frame = frameAtTime(time, fps)
  return (
    <div
      className={cn("flex min-w-0 flex-col gap-2 rounded-lg border p-3", focused && "ring-2 ring-info/60")}
      onPointerDownCapture={onFocusWell}
      data-well={role}
    >
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
      {badge ? <div className="flex min-w-0 flex-wrap items-center gap-2">{badge}</div> : null}
      <div className="bg-stage">
        <video
          ref={videoRef}
          src={`/api/video/${encodeURIComponent(value)}${clipVersion ? `?v=${clipVersion}` : ""}`}
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
            applyRate()
            onLoadedMetadata?.()
          }}
          onPlay={applyRate}
          onEnded={onEnded}
        />
      </div>
      <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground tabular-nums">
        <span>
          Frame <span className="font-mono">{frame}</span> · {time.toFixed(2)}s
        </span>
        {equivalent ? <span>· {equivalent(frame)}</span> : null}
        {pendingMark != null ? (
          <ToneBadge tone="info" className="tabular-nums">
            <CrosshairIcon className="size-3" aria-hidden />
            mark {pendingMark}
          </ToneBadge>
        ) : null}
      </p>
    </div>
  )
}
