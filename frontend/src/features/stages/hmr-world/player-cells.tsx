import { confidenceTone, fmt, TONE_TEXT } from "@/lib/format"
import { cn } from "@/lib/utils"

interface PlayerCellProps {
  colour: string
  playerId: string
  playerName?: string | null
}

/** Colour swatch + display name + id in mono (id only when a name exists). */
export function PlayerCell({ colour, playerId, playerName }: PlayerCellProps) {
  return (
    <span className="inline-flex items-center gap-2">
      <span aria-hidden className="size-2.5 shrink-0 rounded-full" style={{ backgroundColor: colour }} />
      <span className="font-medium">{playerName || playerId}</span>
      {playerName ? <span className="font-mono text-xs text-muted-foreground">{playerId}</span> : null}
    </span>
  )
}

const BAR_TONE = {
  success: "bg-success",
  warning: "bg-warning",
  destructive: "bg-destructive",
  info: "bg-info",
  muted: "bg-muted-foreground/40",
} as const

/** Confidence value coloured by tone with a small proportional bar. */
export function ConfidenceCell({ value }: { value: number | null | undefined }) {
  const tone = confidenceTone(value)
  const pct = value === null || value === undefined ? 0 : Math.max(0, Math.min(1, value)) * 100
  return (
    <span className="inline-flex items-center gap-2">
      <span className={cn("w-12 tabular-nums", TONE_TEXT[tone])}>{fmt(value, 3)}</span>
      <span aria-hidden className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
        <span className={cn("block h-full rounded-full", BAR_TONE[tone])} style={{ width: `${pct}%` }} />
      </span>
    </span>
  )
}
