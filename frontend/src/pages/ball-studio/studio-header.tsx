import { LoaderCircleIcon, Redo2Icon, SaveIcon, Undo2Icon } from "lucide-react"

import { PageHeader } from "@/components/page-header"
import { ToneBadge } from "@/components/status"
import { Button } from "@/components/ui/button"
import { Kbd } from "@/components/ui/kbd"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { setOutcome } from "./truth-doc"
import type { GroupInfo, Outcome } from "./types"
import type { Studio } from "./use-studio"

interface HeaderProps {
  groups: GroupInfo[]
  groupId: string
  onChangeGroup: (id: string) => void
  /** Absent while the scene/truth are still loading or failed. */
  studio?: Studio
  readOnly?: boolean
}

function SolveChip({ studio }: { studio: Studio }) {
  const { solver } = studio
  const nFlags = solver.result?.flags.length ?? 0
  if (solver.status === "solving") {
    return (
      <ToneBadge tone="muted" className="gap-1.5" aria-live="polite">
        <LoaderCircleIcon className="size-3 animate-spin" aria-hidden /> Solving…
      </ToneBadge>
    )
  }
  if (solver.status === "error") {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <ToneBadge tone="destructive">Solve failed</ToneBadge>
        </TooltipTrigger>
        <TooltipContent className="max-w-xs">{solver.error}</TooltipContent>
      </Tooltip>
    )
  }
  if (solver.status === "solved" && solver.result) {
    return (
      <ToneBadge tone={nFlags ? "warning" : "success"} aria-live="polite">
        <span className={cn("size-1.5 rounded-full bg-current")} aria-hidden />
        <span className="font-mono whitespace-nowrap tabular-nums">
          Solved · {solver.ms} ms · {solver.result.stats.n_keys} keys{nFlags ? ` · ${nFlags} flag${nFlags === 1 ? "" : "s"}` : ""}
        </span>
      </ToneBadge>
    )
  }
  return <ToneBadge tone="muted">No keys yet</ToneBadge>
}

export function StudioHeader({ groups, groupId, onChangeGroup, studio, readOnly }: HeaderProps) {
  const doc = studio?.docApi.doc
  const dirty = studio?.docApi.dirty ?? false
  return (
    <PageHeader
      title="Ball studio"
      status={
        studio && doc ? (
          <>
            <ToneBadge tone={dirty ? "warning" : "success"}>{dirty ? "Unsaved changes" : studio.docApi.token ? "Saved" : "Not saved yet"}</ToneBadge>
            <ToneBadge tone={doc.meta.status === "reviewed" ? "info" : "muted"}>{doc.meta.status}</ToneBadge>
          </>
        ) : undefined
      }
      description="Author a dense 3-D ball track from every synced angle, then freeze it as ground truth."
      actions={
        <>
          <Select value={groupId} onValueChange={onChangeGroup}>
            <SelectTrigger size="sm" className="w-48" aria-label="Group">
              <SelectValue placeholder="Group" />
            </SelectTrigger>
            <SelectContent>
              {groups.map((g) => (
                <SelectItem key={g.group_id} value={g.group_id}>
                  {g.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {studio && doc && !readOnly ? (
            <>
              <Select value={doc.outcome} onValueChange={(v) => void studio.editDoc((d) => setOutcome(d, v as Outcome))}>
                <SelectTrigger size="sm" className="w-44" aria-label="Outcome">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="goal">Outcome: goal</SelectItem>
                  <SelectItem value="no_goal">Outcome: no goal</SelectItem>
                  <SelectItem value="unknown">Outcome: unknown</SelectItem>
                </SelectContent>
              </Select>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="icon-sm" aria-label="Undo" disabled={!studio.docApi.canUndo} onClick={studio.docApi.undo}>
                    <Undo2Icon />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  Undo <Kbd>Ctrl Z</Kbd>
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="icon-sm" aria-label="Redo" disabled={!studio.docApi.canRedo} onClick={studio.docApi.redo}>
                    <Redo2Icon />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  Redo <Kbd>Ctrl Shift Z</Kbd>
                </TooltipContent>
              </Tooltip>
              <SolveChip studio={studio} />
              <Button size="sm" disabled={!dirty || studio.saving} onClick={() => void studio.save()}>
                {studio.saving ? <LoaderCircleIcon className="animate-spin" /> : <SaveIcon />} Save <Kbd className="bg-primary-foreground/20 text-primary-foreground">S</Kbd>
              </Button>
            </>
          ) : studio && doc ? (
            <SolveChip studio={studio} />
          ) : null}
        </>
      }
    />
  )
}
