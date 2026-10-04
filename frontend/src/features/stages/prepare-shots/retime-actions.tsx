import { HistoryIcon, Undo2Icon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { usePipeline } from "@/hooks/use-pipeline"

import type { SpeedState } from "./replay-speed"
import type { useRetimeActions } from "./use-retime-actions"

interface RetimeButtonsProps {
  shotId: string
  state: SpeedState
  /** Frames in the clip today (native when not retimed). */
  frames: number
  retimed: boolean
  /** Unsaved offsets in the editor: reloading would drop them. */
  dirty: boolean
  /** Marked moments not saved yet. */
  pendingPairs?: boolean
  /** Rate the clip had before it was retimed (for the restore confirm). */
  wasRate?: number
  actions: ReturnType<typeof useRetimeActions>
}

/** Per-member Retime / Restore, with the reason text whenever it is blocked. */
export function RetimeButtons({ shotId, state, frames, retimed, dirty, pendingPairs, wasRate, actions }: RetimeButtonsProps) {
  const { isRunning, runningLabel } = usePipeline()
  const busy = actions.busyId === shotId
  if (!retimed && !state.canRetime && !state.retimeBlocked) return null
  const blocked = isRunning
    ? `${runningLabel ?? "A job"} is running.`
    : pendingPairs
      ? "Save or clear the marked pairs first."
      : dirty
      ? "Save the group first."
      : !retimed
        ? state.retimeBlocked
        : ""
  return (
    <span className="flex flex-wrap items-center gap-2">
      {retimed ? (
        <Button
          variant="outline"
          size="xs"
          disabled={!!blocked || busy}
          onClick={() => void actions.restore(shotId, { confirmFirst: true, wasRate })}
        >
          <Undo2Icon data-icon="inline-start" />
          Restore native clip
        </Button>
      ) : (
        <Button
          variant="outline"
          size="xs"
          disabled={!!blocked || busy}
          onClick={() => void actions.retime({ shotId, rate: state.rate, frames })}
        >
          <HistoryIcon data-icon="inline-start" />
          Retime to real time
        </Button>
      )}
      {blocked ? <span className="text-xs text-muted-foreground">{blocked}</span> : null}
    </span>
  )
}
