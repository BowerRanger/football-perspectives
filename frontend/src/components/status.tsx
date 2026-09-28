import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import type { LiveStageState } from "@/lib/stages"

export type StageStatus = "complete" | "partial" | "running" | "error" | "pending"

export function resolveStageStatus(
  complete: boolean | undefined,
  live: LiveStageState | undefined,
  partial?: boolean,
): StageStatus {
  if (live === "running") return "running"
  if (live === "error") return "error"
  if (complete) return "complete"
  return partial ? "partial" : "pending"
}

const STATUS_LABEL: Record<StageStatus, string> = {
  complete: "Complete",
  partial: "Partial",
  running: "Running",
  error: "Failed",
  pending: "Not run",
}

const DOT_CLASS: Record<StageStatus, string> = {
  complete: "bg-success",
  partial: "bg-[linear-gradient(90deg,var(--info)_50%,transparent_50%)] ring-1 ring-inset ring-info",
  running: "bg-warning animate-pulse",
  error: "bg-destructive",
  pending: "bg-transparent ring-1 ring-inset ring-muted-foreground/60",
}

interface StatusDotProps {
  status: StageStatus
  className?: string
}

/** Small state marker; the ring-only "pending" dot stays distinct without colour. */
export function StatusDot({ status, className }: StatusDotProps) {
  return (
    <span
      aria-hidden
      className={cn("inline-block size-2 shrink-0 rounded-full", DOT_CLASS[status], className)}
    />
  )
}

const BADGE_CLASS: Record<StageStatus, string> = {
  complete: "bg-success/15 text-success border-success/25",
  partial: "bg-info/15 text-info border-info/25",
  running: "bg-warning/15 text-warning border-warning/25",
  error: "bg-destructive/15 text-destructive border-destructive/25",
  pending: "bg-transparent text-muted-foreground border-border",
}

interface StatusBadgeProps {
  status: StageStatus
  className?: string
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  return (
    <Badge variant="outline" className={cn("gap-1.5", BADGE_CLASS[status], className)}>
      <StatusDot status={status} className={status === "pending" || status === "partial" ? "" : "bg-current"} />
      {STATUS_LABEL[status]}
    </Badge>
  )
}

export type Tone = "success" | "warning" | "destructive" | "info" | "muted"

const TONE_BADGE: Record<Tone, string> = {
  success: "bg-success/15 text-success border-success/25",
  warning: "bg-warning/15 text-warning border-warning/25",
  destructive: "bg-destructive/15 text-destructive border-destructive/25",
  info: "bg-info/15 text-info border-info/25",
  muted: "bg-muted text-muted-foreground border-transparent",
}

interface ToneBadgeProps extends React.ComponentProps<typeof Badge> {
  tone: Tone
}

/** Semantic badge for data states (ok / warn / bad / info) inside panels. */
export function ToneBadge({ tone, className, ...props }: ToneBadgeProps) {
  return <Badge variant="outline" className={cn(TONE_BADGE[tone], className)} {...props} />
}
