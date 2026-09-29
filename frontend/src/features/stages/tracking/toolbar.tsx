import type * as React from "react"
import { EyeOffIcon, GitMergeIcon, SplineIcon, Trash2Icon, Undo2Icon, UsersIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Kbd, KbdGroup } from "@/components/ui/kbd"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

interface ToolbarProps {
  selectedCount: number
  busy: string | null
  onMerge: () => void
  onMergeByName: () => void
  onIgnoreUnknown: () => void
  onDeleteSelected: () => void
  onInterpolate: () => void
  onDeleteIgnored: () => void
  /** Label of the newest undoable edit, null when the stack is empty. */
  undoLabel: string | null
  onUndo: () => void
}

interface ToolButtonProps {
  icon: React.ReactNode
  label: string
  hint: string
  variant?: "default" | "secondary" | "outline" | "destructive"
  disabled?: boolean
  onClick: () => void
}

function ToolButton({ icon, label, hint, variant = "outline", disabled, onClick }: ToolButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button size="sm" variant={variant} disabled={disabled} onClick={onClick}>
          {icon}
          {label}
        </Button>
      </TooltipTrigger>
      <TooltipContent className="max-w-64">{hint}</TooltipContent>
    </Tooltip>
  )
}

function UndoButton({ label, disabled, onClick }: { label: string | null; disabled: boolean; onClick: () => void }) {
  const off = disabled || !label
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {/* span keeps the tooltip reachable while the button is disabled */}
        <span tabIndex={off ? 0 : -1}>
          <Button size="sm" variant="outline" disabled={off} onClick={onClick} aria-label={label ? `Undo: ${label}` : "Undo"}>
            <Undo2Icon />
            Undo
          </Button>
        </span>
      </TooltipTrigger>
      <TooltipContent className="flex max-w-64 items-center gap-2">
        {label ? `Undo: ${label}` : "Nothing to undo yet — destructive edits made here can be undone."}
        <KbdGroup>
          <Kbd>⌘</Kbd>
          <Kbd>Z</Kbd>
        </KbdGroup>
      </TooltipContent>
    </Tooltip>
  )
}

const withCount = (label: string, n: number) => (n > 0 ? `${label} (${n})` : label)

export function TrackToolbar(p: ToolbarProps) {
  const idle = p.busy === null
  const some = p.selectedCount > 0
  return (
    <div className="flex flex-wrap items-center gap-2" role="toolbar" aria-label="Track tools">
      <UndoButton label={p.undoLabel} disabled={!idle} onClick={p.onUndo} />
      <>
        <ToolButton
          icon={<GitMergeIcon />}
          label={p.busy === "merge" ? "Merging…" : withCount("Merge selected", p.selectedCount)}
          hint="Merge the ticked players into one player_id (select at least two)."
          variant="secondary"
          disabled={!idle || p.selectedCount < 2}
          onClick={p.onMerge}
        />
        <ToolButton
          icon={<UsersIcon />}
          label={p.busy === "merge-by-name" ? "Merging…" : "Merge by name"}
          hint="Across every shot, unify all tracks that share a player name under one player_id."
          disabled={!idle}
          onClick={p.onMergeByName}
        />
        <ToolButton
          icon={<SplineIcon />}
          label={p.busy === "interpolate" ? "Interpolating…" : withCount("Interpolate gaps", p.selectedCount)}
          hint="Linearly interpolate bboxes for short detector dropouts inside the selected tracks."
          disabled={!idle || !some}
          onClick={p.onInterpolate}
        />
      </>
      <>
        <ToolButton
          icon={<EyeOffIcon />}
          label="Ignore unknown"
          hint="Mark every unnamed player/goalkeeper in this shot as 'ignore'."
          disabled={!idle}
          onClick={p.onIgnoreUnknown}
        />
        <ToolButton
          icon={<Trash2Icon />}
          label="Delete ignored"
          hint="Delete every track marked 'ignore' across all shots."
          disabled={!idle}
          onClick={p.onDeleteIgnored}
        />
      </>
      <ToolButton
        icon={<Trash2Icon />}
        label={withCount("Delete selected", p.selectedCount)}
        hint="Delete every track whose checkbox is currently ticked."
        variant="destructive"
        disabled={!idle || !some}
        onClick={p.onDeleteSelected}
      />
    </div>
  )
}
