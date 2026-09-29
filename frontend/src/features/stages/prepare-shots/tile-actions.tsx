import { ChevronLeftIcon, ChevronRightIcon, MoreVerticalIcon, Trash2Icon } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

import { IconButton } from "./icon-button"
import type { ShotActions } from "./use-shot-actions"
import type { GroupView, ShotModel, ShotView } from "./types"

interface TileActionsProps {
  shot: ShotView
  group: GroupView
  index: number
  model: ShotModel
  actions: ShotActions
}

/** Quick buttons (drop, move to neighbour) plus the full keyboard-friendly menu. */
export function TileActions({ shot, group, index, model, actions }: TileActionsProps) {
  const gIdx = model.groups.findIndex((g) => g.id === group.id)
  const prev = group.id && gIdx > 0 ? model.groups[gIdx - 1] : null
  const next = group.id && gIdx >= 0 && gIdx < model.groups.length - 1 ? model.groups[gIdx + 1] : null
  const otherGroups = model.groups.filter((g) => g.id !== group.id && g.members.length > 0)

  return (
    <>
      <IconButton
        label="Drop this shot (restorable from the dropped tray)"
        onClick={() => void actions.dropShots([shot.id], `${shot.id} dropped`, `shot ${shot.id}`)}
      >
        <Trash2Icon />
      </IconButton>
      {prev ? (
        <IconButton
          label={`Move to ${prev.label}`}
          onClick={() => void actions.moveTo([shot.id], prev.id, `${shot.id} moved to ${prev.label}`)}
        >
          <ChevronLeftIcon />
        </IconButton>
      ) : null}
      {next ? (
        <IconButton
          label={`Move to ${next.label}`}
          onClick={() => void actions.moveTo([shot.id], next.id, `${shot.id} moved to ${next.label}`)}
        >
          <ChevronRightIcon />
        </IconButton>
      ) : null}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="icon-sm" className="ml-auto" aria-label={`More actions for ${shot.id}`}>
            <MoreVerticalIcon />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="min-w-48">
          <DropdownMenuLabel className="font-mono">{shot.id}</DropdownMenuLabel>
          {otherGroups.length > 0 || group.id ? (
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>Move to group</DropdownMenuSubTrigger>
              <DropdownMenuSubContent>
                {otherGroups.map((g) => (
                  <DropdownMenuItem
                    key={g.id}
                    onSelect={() => void actions.moveTo([shot.id], g.id, `${shot.id} moved to ${g.label}`)}
                  >
                    {g.label}
                  </DropdownMenuItem>
                ))}
                {group.id ? (
                  <DropdownMenuItem onSelect={() => void actions.moveTo([shot.id], "", `${shot.id} ungrouped`)}>
                    Ungrouped
                  </DropdownMenuItem>
                ) : null}
              </DropdownMenuSubContent>
            </DropdownMenuSub>
          ) : null}
          <DropdownMenuItem onSelect={() => void actions.moveToNewGroup(shot.id)}>Move to new group</DropdownMenuItem>
          {group.id && index > 0 ? (
            <DropdownMenuItem
              onSelect={() =>
                void actions.splitGroupAt(
                  group.label,
                  group.members.slice(index).map((m) => m.id),
                  shot.id,
                )
              }
            >
              Split group here
            </DropdownMenuItem>
          ) : null}
          <DropdownMenuSeparator />
          <DropdownMenuItem
            variant="destructive"
            onSelect={() => void actions.dropShots([shot.id], `${shot.id} dropped`, `shot ${shot.id}`)}
          >
            Drop shot
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </>
  )
}
