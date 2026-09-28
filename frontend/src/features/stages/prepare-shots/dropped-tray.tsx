import * as React from "react"
import { ChevronDownIcon, Undo2Icon } from "lucide-react"

import { ToneBadge } from "@/components/status"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { cn } from "@/lib/utils"

import { ShotTile } from "./shot-tile"
import type { ShotModel, ShotView } from "./types"
import type { ShotActions } from "./use-shot-actions"

const REASON_TIPS: Record<string, string> = {
  reaction: "Classified as a crowd/bench reaction shot",
  transition: "Classified as a fade/graphic transition",
  closeup: "Classified as a player close-up (celebration or tight cut, a person dominates the frame)",
}

interface TrayProps {
  model: ShotModel
  actions: ShotActions
  onOpen: (shot: ShotView) => void
}

/** Muted collapsible tray of excluded shots; also a drop target for dragged tiles. */
export function DroppedTray({ model, actions, onOpen }: TrayProps) {
  const [over, setOver] = React.useState(false)
  const [open, setOpen] = React.useState(false)
  const reasons: Record<string, number> = {}
  for (const s of model.dropped) {
    const r = s.exclude_reason || "unknown"
    reasons[r] = (reasons[r] ?? 0) + 1
  }
  const note = Object.entries(reasons)
    .map(([r, n]) => `${n} ${r}`)
    .join(", ")

  return (
    <Collapsible
      open={open}
      onOpenChange={setOpen}
      className={cn("rounded-xl border bg-muted/40 transition-colors", over && "border-info bg-info/10")}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setOver(false)
      }}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        const sid = e.dataTransfer.getData("text/shot-id")
        if (sid) void actions.dropShots([sid], `${sid} dropped`, `shot ${sid}`)
      }}
    >
      <CollapsibleTrigger asChild>
        <button
          type="button"
          className="flex w-full items-center gap-2 rounded-xl px-4 py-2.5 text-left text-sm font-medium outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
        >
          <ChevronDownIcon className={cn("size-4 transition-transform", !open && "-rotate-90")} aria-hidden />
          Dropped shots
          <span className="font-normal text-muted-foreground">
            {model.dropped.length ? `(${model.dropped.length}: ${note})` : "(none — drag a tile here to drop it)"}
          </span>
        </button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        {model.dropped.length === 0 ? (
          <p className="px-4 pb-3 text-sm text-muted-foreground">
            Reactions, transitions and shots you discard land here. Restore any of them back into its group.
          </p>
        ) : (
          <div className="grid grid-cols-2 gap-3 px-4 pb-4 sm:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
            {model.dropped.map((shot) => (
              <ShotTile
                key={shot.id}
                shot={shot}
                dimmed
                onOpen={onOpen}
                extraBadge={
                  <ToneBadge tone="destructive" title={REASON_TIPS[shot.kind] ?? "Manually discarded"}>
                    {shot.exclude_reason || "dropped"}
                  </ToneBadge>
                }
                actions={
                  <Button
                    variant="outline"
                    size="sm"
                    title={shot.group_id ? `Restore into ${shot.group_id}` : "Restore as ungrouped"}
                    onClick={() => void actions.restore(shot.id)}
                  >
                    <Undo2Icon data-icon="inline-start" />
                    Restore
                  </Button>
                }
              />
            ))}
          </div>
        )}
      </CollapsibleContent>
    </Collapsible>
  )
}
