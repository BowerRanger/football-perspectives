import {
  CircleHelpIcon,
  FastForwardIcon,
  GaugeIcon,
  HandIcon,
  HistoryIcon,
  LoaderCircleIcon,
  RulerIcon,
  TrendingUpIcon,
  TurtleIcon,
  WavesIcon,
  VideoOffIcon,
  type LucideIcon,
} from "lucide-react"

import { ToneBadge } from "@/components/status"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

import type { SpeedKind, SpeedState } from "./replay-speed"

const ICONS: Record<SpeedKind, LucideIcon> = {
  reference: GaugeIcon,
  detecting: LoaderCircleIcon,
  unmeasured: RulerIcon,
  "real-time": GaugeIcon,
  slow: TurtleIcon,
  fast: FastForwardIcon,
  ramp: TrendingUpIcon,
  "no-camera": VideoOffIcon,
  "low-confidence": CircleHelpIcon,
  manual: HandIcon,
  retimed: HistoryIcon,
  approximate: WavesIcon,
}

interface SpeedBadgeProps {
  state: SpeedState
  /** Extra tooltip line (e.g. the automatic estimate behind a manual alignment). */
  note?: string
  /** Renders the badge as a button (the "no camera" call to action). */
  onClick?: () => void
  className?: string
}

/**
 * A replay's playback speed with its source and confidence. Text first, icon
 * second, tone third: the state never relies on colour alone.
 */
export function SpeedBadge({ state, note, onClick, className }: SpeedBadgeProps) {
  if (state.kind === "reference") return null
  const Icon = ICONS[state.kind]
  const badge = (
    <ToneBadge
      tone={state.tone}
      data-speed-kind={state.kind}
      className={cn("max-w-full min-w-0 gap-1 tabular-nums", state.kind === "low-confidence" && "ring-1 ring-warning/60", className)}
    >
      <Icon className={cn("size-3 shrink-0", state.kind === "detecting" && "animate-spin")} aria-hidden />
      <span className="truncate">{state.text}</span>
    </ToneBadge>
  )
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {onClick ? (
          <button
            type="button"
            onClick={onClick}
            className="min-w-0 rounded-full outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
          >
            {badge}
          </button>
        ) : (
          <span className="inline-flex min-w-0 max-w-full" tabIndex={0}>
            {badge}
          </span>
        )}
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">
        {state.detail}
        {note ? <span className="mt-1 block opacity-80">{note}</span> : null}
      </TooltipContent>
    </Tooltip>
  )
}
