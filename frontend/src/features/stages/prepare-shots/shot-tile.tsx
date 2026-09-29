import * as React from "react"

import { ToneBadge, type Tone } from "@/components/status"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

import { fmtClock, type GroupSync, type ShotView } from "./types"

const SCALE_LABEL: Record<string, string> = { wide: "Wide", medium: "Medium", tight: "Tight" }

interface BadgeSpec {
  key: string
  tone: Tone
  text: string
  title: string
}

/** Badges for classifier scale, replay speed and the group-sync alignment. */
export function shotBadges(shot: ShotView, groupSync: GroupSync | null): BadgeSpec[] {
  const out: BadgeSpec[] = []
  const f = shot.features
  if (f?.scale) {
    out.push({
      key: "scale",
      tone: "muted",
      text: SCALE_LABEL[f.scale] ?? f.scale,
      title: `Pitch ratio ${f.pitch_ratio_median ?? "?"}`,
    })
  }
  const sf = f?.speed_factor || shot.speed_factor || 1
  if (sf >= 1.25) {
    out.push({
      key: "replay",
      tone: "warning",
      text: `Replay ×${sf.toFixed(1)}`,
      title: "Slow-motion replay, retimed to real time at extraction",
    })
  }
  const a = groupSync?.alignments.find((x) => x.shot_id === shot.id)
  if (a && groupSync) {
    if (groupSync.reference_shot === shot.id) {
      out.push({ key: "ref", tone: "info", text: "Ref", title: "Group sync reference (offset 0)" })
    } else if (a.method === "manual") {
      out.push({ key: "manual", tone: "info", text: "Manual", title: `Offset ${a.frame_offset}f, operator-set` })
    } else {
      const low = a.confidence < 0.5
      out.push({
        key: "auto",
        tone: low ? "destructive" : "success",
        text: `Auto ${a.confidence.toFixed(2)}`,
        title: `Offset ${a.frame_offset}f via ${a.method}`,
      })
    }
  }
  return out
}

interface ShotTileProps {
  shot: ShotView
  groupSync?: GroupSync | null
  /** Group identity colour (data) for the top rule. */
  accent?: string
  groupLabel?: string
  draggable?: boolean
  dimmed?: boolean
  extraBadge?: React.ReactNode
  onOpen: (shot: ShotView) => void
  /** Buttons rendered on the tile's action row. */
  actions?: React.ReactNode
}

export function ShotTile({
  shot,
  groupSync = null,
  accent,
  groupLabel,
  draggable,
  dimmed,
  extraBadge,
  onOpen,
  actions,
}: ShotTileProps) {
  const videoRef = React.useRef<HTMLVideoElement>(null)
  const [dragging, setDragging] = React.useState(false)
  const id = encodeURIComponent(shot.id)
  const seconds = shot.end_time - shot.start_time

  const play = () => {
    const v = videoRef.current
    if (!v) return
    if (!v.src) v.src = `/api/video/${id}`
    v.play().catch(() => undefined)
  }
  const stop = () => videoRef.current?.pause()

  return (
    <div
      role="group"
      aria-label={`Shot ${shot.id}`}
      draggable={draggable}
      onDragStart={(e) => {
        e.dataTransfer.setData("text/shot-id", shot.id)
        e.dataTransfer.effectAllowed = "move"
        setDragging(true)
      }}
      onDragEnd={() => setDragging(false)}
      className={cn(
        "flex min-w-0 flex-col gap-2 rounded-lg transition-opacity",
        draggable && "cursor-grab active:cursor-grabbing",
        (dragging || dimmed) && "opacity-60",
      )}
    >
      <button
        type="button"
        className="group relative block overflow-hidden rounded-lg bg-stage outline-none hover:ring-2 hover:ring-ring/40 focus-visible:ring-3 focus-visible:ring-ring/50"
        style={accent ? { borderTop: `2px solid ${accent}` } : undefined}
        aria-label={`Open ${shot.id} in the large preview`}
        onMouseEnter={play}
        onMouseLeave={stop}
        onFocus={play}
        onBlur={stop}
        onClick={() => {
          stop()
          onOpen(shot)
        }}
      >
        <video
          ref={videoRef}
          poster={`/api/shots/${id}/thumb`}
          muted
          loop
          playsInline
          preload="none"
          draggable={false}
          className="aspect-video w-full object-cover"
        />
      </button>
      <div className="flex flex-col gap-1.5 px-0.5">
        <div className="flex items-center gap-2 text-sm">
          <span className="min-w-0 truncate font-mono font-medium">{shot.id}</span>
          <span
            className="ml-auto shrink-0 text-xs text-muted-foreground tabular-nums"
            title={shot.source_start_s >= 0 ? `Source ${fmtClock(shot.source_start_s)}–${fmtClock(shot.source_end_s)}` : undefined}
          >
            {seconds.toFixed(1)}s
          </span>
        </div>
        <div className="flex min-h-5 flex-wrap gap-1">
          {extraBadge}
          {groupLabel ? (
            <Badge variant="secondary" title="Highlight group this shot belongs to">
              {groupLabel}
            </Badge>
          ) : null}
          {shotBadges(shot, groupSync).map((b) => (
            <ToneBadge key={b.key} tone={b.tone} title={b.title}>
              {b.text}
            </ToneBadge>
          ))}
        </div>
        <div className="flex items-center gap-1">{actions}</div>
      </div>
    </div>
  )
}
