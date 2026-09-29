import * as React from "react"
import { ScissorsIcon, XIcon } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { isNamed } from "./groups"
import { TEAM_COLORS, type PlayerGroup } from "./types"

interface PlayerRowProps {
  rowRef: (el: HTMLLIElement | null) => void
  group: PlayerGroup
  selected: boolean
  flash: boolean
  busy: boolean
  datalistId: string
  onToggle: (key: string, checked: boolean) => void
  onFocusChange: (key: string | null) => void
  onRename: (group: PlayerGroup, name: string) => Promise<void>
  onJump: (group: PlayerGroup) => void
  onSplit: (group: PlayerGroup) => void
  onDelete: (group: PlayerGroup) => void
}

function memberSummary(g: PlayerGroup): string {
  if (g.tracks.length <= 1) return `${g.frameCount} frames in [${g.frameRange[0]}–${g.frameRange[1]}]`
  const members = g.tracks.map((t) => `${t.track_id} [${t.frame_range[0]}–${t.frame_range[1]}]`).join(", ")
  return `${g.tracks.length} tracks merged (${members})`
}

/** One player = one row: select, jump, rename, split at current frame, delete. */
export function PlayerRow({
  rowRef,
  group,
  selected,
  flash,
  busy,
  datalistId,
  onToggle,
  onFocusChange,
  onRename,
  onJump,
  onSplit,
  onDelete,
}: PlayerRowProps) {
  const [draft, setDraft] = React.useState(group.name)
  React.useEffect(() => setDraft(group.name), [group.name])
  const named = isNamed(group.name)
  const multi = group.tracks.length > 1

  function commit() {
    onFocusChange(null)
    const next = draft.trim()
    if (next !== group.name) void onRename(group, next)
  }

  return (
    <li
      ref={rowRef}
      data-row-key={group.key}
      className={cn(
        "flex flex-col gap-1.5 border-b px-3 py-2 transition-colors last:border-b-0",
        selected && "bg-info/10",
        flash && "bg-accent",
      )}
    >
      <div className="flex items-center gap-2">
        <Checkbox
          checked={selected}
          onCheckedChange={(c) => onToggle(group.key, c === true)}
          aria-label={`Select ${group.key}`}
        />
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="link" size="xs" className="h-auto px-0 font-mono" onClick={() => onJump(group)}>
              {group.key}
            </Button>
          </TooltipTrigger>
          <TooltipContent>{memberSummary(group)} — click to jump to the middle</TooltipContent>
        </Tooltip>
        {multi ? (
          <span className="text-xs text-muted-foreground tabular-nums">×{group.tracks.length}</span>
        ) : null}
        <Badge variant={named ? "secondary" : "outline"} className="gap-1.5">
          <span
            aria-hidden
            className={cn("size-1.5 rounded-full", named ? "bg-success" : "bg-destructive")}
          />
          {group.name === "ignore" ? "Ignored" : named ? "Named" : "Unnamed"}
        </Badge>
        <span className="ml-auto flex items-center gap-1">
          <Badge
            variant="outline"
            style={{ borderColor: TEAM_COLORS[group.team] ?? TEAM_COLORS.unknown }}
            className="text-xs"
          >
            {group.team || "?"}
          </Badge>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon-xs"
                disabled={busy}
                onClick={() => onSplit(group)}
                aria-label={`Split ${group.key} at current frame`}
              >
                <ScissorsIcon />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              {multi
                ? "Split the underlying member track that contains the current video frame."
                : "Split this track at the current video frame."}
            </TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon-xs"
                className="text-destructive hover:text-destructive"
                disabled={busy}
                onClick={() => onDelete(group)}
                aria-label={`Delete ${group.key}`}
              >
                <XIcon />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              {multi ? `Delete this player and all ${group.tracks.length} underlying tracks.` : "Delete this track entirely."}
            </TooltipContent>
          </Tooltip>
        </span>
      </div>
      <Input
        value={draft}
        list={datalistId}
        placeholder="Player name"
        aria-label={`Name for ${group.key}`}
        className="h-7 text-sm"
        onChange={(e) => setDraft(e.target.value)}
        onFocus={() => onFocusChange(group.key)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") e.currentTarget.blur()
          if (e.key === "Escape") {
            setDraft(group.name)
            e.currentTarget.blur()
          }
        }}
      />
    </li>
  )
}
