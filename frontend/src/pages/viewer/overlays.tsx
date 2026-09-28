import * as React from "react"
import { ChevronDownIcon, UsersIcon } from "lucide-react"

import { PanelEmpty, PanelError } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { Spinner } from "@/components/ui/spinner"
import { cn } from "@/lib/utils"
import type { LoadProgress } from "./load-scene"
import type { MatchInfo, SceneData } from "./types"

export const OVERLAY_CARD = "border-border/60 bg-background/80 py-0 shadow-sm backdrop-blur"

function minuteText(match: MatchInfo): string {
  const m = match.moment
  if (!m || m.minute === null || m.minute === undefined) return ""
  return `${m.minute}${m.added_time ? `+${m.added_time}` : ""}'`
}

/** "Liverpool FC 1–1 Chelsea FC · 63' · Anfield" */
export function MatchHeader({ match, className }: { match: MatchInfo; className?: string }) {
  const meta = [minuteText(match), match.venue].filter(Boolean)
  return (
    <Card className={cn(OVERLAY_CARD, "min-w-0 max-w-full flex-row items-center gap-2 px-3 py-1.5 text-sm", className)}>
      <span className="truncate font-medium">{match.home_team || "Home"}</span>
      <span className="shrink-0 font-mono tabular-nums text-muted-foreground">
        {match.home_score ?? 0}–{match.away_score ?? 0}
      </span>
      <span className="truncate font-medium">{match.away_team || "Away"}</span>
      {meta.length > 0 ? (
        <span className="hidden truncate text-xs text-muted-foreground sm:inline">· {meta.join(" · ")}</span>
      ) : null}
    </Card>
  )
}

interface LegendProps {
  data: SceneData
  selectedId: string | null
  onSelect: (id: string) => void
  defaultOpen: boolean
  className?: string
}

/** Collapsible player legend; clicking a row follows that player (click again to release). */
export function PlayerLegend({ data, selectedId, onSelect, defaultOpen, className }: LegendProps) {
  const [open, setOpen] = React.useState(defaultOpen)
  React.useEffect(() => setOpen(defaultOpen), [defaultOpen])
  return (
    <Collapsible open={open} onOpenChange={setOpen} className={className}>
      <Card className={cn(OVERLAY_CARD, "w-56 max-w-[calc(100vw-1rem)] gap-0")}>
        <CollapsibleTrigger asChild>
          <Button variant="ghost" size="sm" className="h-8 justify-between px-3 text-xs font-medium">
            <span className="flex items-center gap-2">
              <UsersIcon />
              Pitch-registered scene
              <span className="font-mono tabular-nums text-muted-foreground">{data.players.length}</span>
            </span>
            <ChevronDownIcon className={cn("transition-transform", open && "rotate-180")} />
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="max-h-[45vh] overflow-y-auto border-t p-1">
            {data.players.length === 0 ? (
              <p className="p-2 text-xs text-muted-foreground">No players in this shot.</p>
            ) : (
              <ul className="flex flex-col">
                {data.players.map((p) => (
                  <li key={p.id}>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      aria-pressed={selectedId === p.id}
                      onClick={() => onSelect(p.id)}
                      className={cn(
                        "h-auto w-full justify-start gap-2 px-2 py-1.5 text-left text-xs",
                        selectedId === p.id && "bg-accent text-accent-foreground ring-1 ring-info",
                      )}
                    >
                      <span
                        className="size-2.5 shrink-0 rounded-full"
                        style={{ backgroundColor: `#${p.colour.toString(16).padStart(6, "0")}` }}
                        aria-hidden
                      />
                      <span className="min-w-0 flex-1 truncate font-medium">{p.name}</span>
                      {p.name !== p.id ? (
                        <span className="shrink-0 font-mono text-[10px] text-muted-foreground">{p.id}</span>
                      ) : null}
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  )
}

export function LoadingOverlay({ progress }: { progress: LoadProgress }) {
  return (
    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-4 bg-stage/90 p-6 text-stage-foreground">
      <div className="flex items-center gap-2 text-sm" role="status" aria-live="polite">
        <Spinner />
        <span>{progress.label}</span>
      </div>
      <Progress value={progress.value} className="w-56" aria-label="Scene loading progress" />
      <div className="flex w-56 flex-col gap-2">
        <Skeleton className="h-3 w-full bg-white/10" />
        <Skeleton className="h-3 w-2/3 bg-white/10" />
      </div>
    </div>
  )
}

export function EmptyOverlay({ shot }: { shot?: string }) {
  return (
    <div className="absolute inset-0 z-20 flex items-center justify-center overflow-auto bg-stage p-4 text-stage-foreground">
      <PanelEmpty
        title="No reconstruction to show"
        description={
          <>
            {shot ? `Shot ${shot} has` : "This output has"} no player poses or ball track yet. Run the{" "}
            <strong>Refined poses</strong> stage (and <strong>Ball</strong> for the ball) from the dashboard, then reopen the viewer.
          </>
        }
      />
    </div>
  )
}

export function ErrorOverlay({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="absolute inset-0 z-20 flex items-center justify-center bg-stage p-4">
      <div className="w-full max-w-md">
        <PanelError
          title="Could not load the scene"
          message={`${message}. Check that the dashboard backend is running and the pipeline output for this shot exists.`}
          action={
            <Button variant="outline" size="sm" className="mt-3" onClick={onRetry}>
              Try again
            </Button>
          }
        />
      </div>
    </div>
  )
}
