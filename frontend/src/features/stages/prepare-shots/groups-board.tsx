import * as React from "react"
import { GitMergeIcon, PlusIcon, RefreshCwIcon, Trash2Icon } from "lucide-react"
import { toast } from "sonner"

import { Panel } from "@/components/panel"
import { ToneBadge } from "@/components/status"
import { Button } from "@/components/ui/button"
import { errorMessage, postJson } from "@/lib/api"
import { cn } from "@/lib/utils"

import { ShotTile } from "./shot-tile"
import { TileActions } from "./tile-actions"
import { fmtClock, groupColor, type GroupView, type ShotModel, type ShotView } from "./types"
import type { ShotActions } from "./use-shot-actions"

interface BoardProps {
  model: ShotModel
  actions: ShotActions
  onOpen: (shot: ShotView, groupLabel: string) => void
  /** Refetch everything (after re-align). */
  onReload: () => Promise<void>
}

function dropHandlers(onDrop: (shotId: string) => void, setOver: (v: boolean) => void) {
  return {
    onDragOver: (e: React.DragEvent) => {
      e.preventDefault()
      setOver(true)
    },
    onDragLeave: (e: React.DragEvent) => {
      if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setOver(false)
    },
    onDrop: (e: React.DragEvent) => {
      e.preventDefault()
      setOver(false)
      const sid = e.dataTransfer.getData("text/shot-id")
      if (sid) onDrop(sid)
    },
  }
}

export function GroupsBoard({ model, actions, onOpen, onReload }: BoardProps) {
  const [newOver, setNewOver] = React.useState(false)
  const visible = model.groups.filter((g) => g.members.length > 0)
  const ungrouped: GroupView | null = model.ungrouped.length
    ? {
        id: "",
        label: "Ungrouped",
        boundary_rule: "manual",
        boundary_confidence: 1,
        members: model.ungrouped,
        sync: model.ungroupedSync,
      }
    : null

  return (
    <section className="flex flex-col gap-3" aria-label="Highlight groups">
      <div>
        <h3 className="text-sm font-semibold">
          Highlight groups{" "}
          <span className="font-normal text-muted-foreground">
            ({model.groups.length} group{model.groups.length === 1 ? "" : "s"}, {model.activeIds.length} active shots)
          </span>
        </h3>
        <p className="text-xs text-muted-foreground">
          Each card is one highlight (live action plus its replays). Drag tiles between cards to regroup, or use a
          tile&apos;s menu. Hover a thumbnail to preview, click it to open large.
        </p>
      </div>
      {[...visible, ...(ungrouped ? [ungrouped] : [])].map((g) => (
        <GroupCard key={g.id || "ungrouped"} group={g} model={model} actions={actions} onOpen={onOpen} onReload={onReload} />
      ))}
      <div
        className={cn(
          "flex items-center justify-center gap-2 rounded-lg border-2 border-dashed p-3 text-sm text-muted-foreground transition-colors",
          newOver && "border-info bg-info/10 text-foreground",
        )}
        {...dropHandlers((sid) => void actions.moveToNewGroup(sid), setNewOver)}
      >
        <PlusIcon className="size-4" aria-hidden />
        Drag a shot here to start a new group
      </div>
    </section>
  )
}

function boundaryBadge(group: GroupView) {
  if (group.boundary_confidence >= 0.9 || group.boundary_rule === "manual") return null
  return (
    <ToneBadge
      tone="warning"
      title="Low-confidence auto boundary. Check whether this group should merge with its neighbour."
    >
      Boundary {group.boundary_rule} {group.boundary_confidence.toFixed(2)}
    </ToneBadge>
  )
}

interface CardProps extends BoardProps {
  group: GroupView
}

function GroupCard({ group, model, actions, onOpen, onReload }: CardProps) {
  const [over, setOver] = React.useState(false)
  const [aligning, setAligning] = React.useState(false)
  const accent = groupColor(group.id, model.groupIds)
  const idx = model.groups.findIndex((g) => g.id === group.id)
  const prev = group.id && idx > 0 ? model.groups[idx - 1] : null
  const memberIds = group.members.map((m) => m.id)

  const starts = group.members.map((m) => m.source_start_s).filter((v) => v >= 0)
  const ends = group.members.map((m) => m.source_end_s).filter((v) => v >= 0)
  const range = starts.length ? ` · ${fmtClock(Math.min(...starts))}–${fmtClock(Math.max(...ends))}` : ""

  const realign = async () => {
    setAligning(true)
    try {
      const body = await postJson<{ aligned: number }>("/api/sync/auto", { group_id: group.id, force: true })
      toast.success(`Re-aligned ${body.aligned} shot(s) in ${group.label}`)
      await onReload()
    } catch (err) {
      toast.error("Re-align failed", { description: errorMessage(err) })
    } finally {
      setAligning(false)
    }
  }

  const title = (
    <span className="flex items-center gap-2">
      <span aria-hidden className="size-2.5 rounded-full" style={{ backgroundColor: accent }} />
      {group.label}
    </span>
  )
  const meta = (
    <span className="flex flex-wrap items-center gap-2">
      <span className="font-mono">{group.id || "—"}</span>
      <span>
        {group.members.length} shot{group.members.length === 1 ? "" : "s"}
        {range}
      </span>
      {boundaryBadge(group)}
    </span>
  )

  return (
    <div
      className={cn("rounded-xl transition-shadow", over && "ring-2 ring-info")}
      {...dropHandlers((sid) => {
        if (!group.members.some((m) => m.id === sid)) {
          void actions.moveTo([sid], group.id, `${sid} moved to ${group.label}`)
        }
      }, setOver)}
    >
      <Panel
        title={title}
        description={meta}
        actions={
          group.id ? (
            <div className="flex flex-wrap justify-end gap-1.5">
              {group.members.length >= 2 ? (
                <Button
                  variant="outline"
                  size="sm"
                  disabled={aligning}
                  title="Re-run the motion-profile auto-aligner for this group. Overwrites auto offsets; manual ones are kept unless none exist."
                  onClick={() => void realign()}
                >
                  <RefreshCwIcon data-icon="inline-start" className={aligning ? "animate-spin" : undefined} />
                  Re-align
                </Button>
              ) : null}
              <Button
                variant="outline"
                size="sm"
                disabled={!prev}
                title="Merge this group into the previous one (moves every shot)."
                onClick={() =>
                  prev && void actions.moveTo(memberIds, prev.id, `${group.label} merged into ${prev.label}`)
                }
              >
                <GitMergeIcon data-icon="inline-start" />
                Merge left
              </Button>
              <Button
                variant="destructive"
                size="sm"
                title="Exclude every shot in this group (restorable from the dropped tray)."
                onClick={() =>
                  void actions.dropShots(
                    memberIds,
                    `${group.label} discarded (${memberIds.length} shots)`,
                    `group ${group.label} (${memberIds.length} shots)`,
                  )
                }
              >
                <Trash2Icon data-icon="inline-start" />
                Discard group
              </Button>
            </div>
          ) : null
        }
      >
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
          {group.members.map((shot, i) => (
            <ShotTile
              key={shot.id}
              shot={shot}
              groupSync={group.sync}
              accent={accent}
              draggable
              onOpen={(s) => onOpen(s, group.label)}
              actions={<TileActions shot={shot} group={group} index={i} model={model} actions={actions} />}
            />
          ))}
        </div>
      </Panel>
    </div>
  )
}
