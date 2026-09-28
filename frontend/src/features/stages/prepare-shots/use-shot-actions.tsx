import * as React from "react"
import { toast } from "sonner"

import { errorMessage } from "@/lib/api"
import { useConfirm } from "@/hooks/use-dialogs"

import { nextFreeGroupId, type ShotModel, type ShotUpdate } from "./types"

export interface ShotActions {
  /** Move shots to a group id ("" = ungrouped). */
  moveTo: (shotIds: string[], groupId: string, message: string) => Promise<void>
  moveToNewGroup: (shotId: string) => Promise<void>
  splitGroupAt: (groupLabel: string, tailShotIds: string[], atShot: string) => Promise<void>
  /** Exclude shots after an explicit confirm. */
  dropShots: (shotIds: string[], message: string, what: string) => Promise<void>
  restore: (shotId: string) => Promise<void>
}

/** Bulk-mutation helpers shared by tiles, cards and the dropped tray. */
export function useShotActions(
  model: ShotModel,
  patchShots: (updates: ShotUpdate[]) => Promise<void>,
): ShotActions {
  const confirm = useConfirm()

  const run = React.useCallback(
    async (updates: ShotUpdate[], okMessage: string) => {
      try {
        await patchShots(updates)
        toast.success(okMessage)
      } catch (err) {
        toast.error("Could not update shots", { description: errorMessage(err) })
      }
    },
    [patchShots],
  )

  const moveTo = React.useCallback(
    (shotIds: string[], groupId: string, message: string) =>
      run(shotIds.map((shot_id) => ({ shot_id, group_id: groupId })), message),
    [run],
  )

  const moveToNewGroup = React.useCallback(
    (shotId: string) =>
      run([{ shot_id: shotId, group_id: nextFreeGroupId(model.groupIds) }], `${shotId} moved to a new group`),
    [run, model.groupIds],
  )

  const splitGroupAt = React.useCallback(
    (groupLabel: string, tail: string[], atShot: string) => {
      const gid = nextFreeGroupId(model.groupIds)
      return run(tail.map((shot_id) => ({ shot_id, group_id: gid })), `${groupLabel} split at ${atShot}`)
    },
    [run, model.groupIds],
  )

  const dropShots = React.useCallback(
    async (shotIds: string[], message: string, what: string) => {
      const ok = await confirm({
        title: `Drop ${what}?`,
        description:
          "Dropped shots leave every stage and their sync offsets are pruned. You can restore them from the dropped tray.",
        confirmLabel: "Drop",
        destructive: true,
      })
      if (!ok) return
      await run(
        shotIds.map((shot_id) => ({ shot_id, excluded: true, exclude_reason: "manual" })),
        message,
      )
    },
    [confirm, run],
  )

  const restore = React.useCallback(
    (shotId: string) => run([{ shot_id: shotId, excluded: false, exclude_reason: "" }], `${shotId} restored`),
    [run],
  )

  return { moveTo, moveToNewGroup, splitGroupAt, dropShots, restore }
}
