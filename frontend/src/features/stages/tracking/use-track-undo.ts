import * as React from "react"
import { toast } from "sonner"

import { ApiError, errorMessage } from "@/lib/api"
import { listUndo, undoEdit, type UndoEntry } from "./api"

const TYPING = 'input, textarea, select, [contenteditable="true"], [role="dialog"], [role="menu"]'

/**
 * Track-edit undo stack (server-side snapshots). `announce` shows the
 * success toast with an Undo action; `undo` restores the newest snapshot
 * (or a specific one, refused with 409 if a later edit exists).
 */
export function useTrackUndo(onRestored: () => Promise<void>) {
  const [stack, setStack] = React.useState<UndoEntry[]>([])
  const [undoing, setUndoing] = React.useState(false)
  const restored = React.useRef(onRestored)
  React.useEffect(() => {
    restored.current = onRestored
  })

  const refresh = React.useCallback(async () => {
    try {
      setStack(await listUndo())
    } catch {
      // The stack is a convenience; a failed refresh just leaves the button stale.
    }
  }, [])

  React.useEffect(() => {
    void refresh()
  }, [refresh])

  const undo = React.useCallback(
    async (undoId?: string) => {
      setUndoing(true)
      try {
        const out = await undoEdit(undoId)
        toast.success(`Undid: ${out.undone}`)
        await restored.current()
      } catch (err) {
        if (err instanceof ApiError && err.status === 409) {
          toast.error("Can't undo that yet", {
            description: "A later edit must be undone first — use the toolbar Undo to step back in order.",
          })
        } else if (err instanceof ApiError && err.status === 404) {
          toast.info("Nothing to undo")
        } else {
          toast.error("Undo failed", { description: errorMessage(err) })
        }
      } finally {
        setUndoing(false)
        await refresh()
      }
    },
    [refresh],
  )

  /** Success toast for a destructive edit; carries an Undo action when the server made a snapshot. */
  const announce = React.useCallback(
    (message: string, undoId?: string | null) => {
      if (undoId) toast.success(message, { action: { label: "Undo", onClick: () => void undo(undoId) } })
      else toast.success(message)
      void refresh()
    },
    [undo, refresh],
  )

  const newest = stack[0] ?? null
  React.useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (!(e.metaKey || e.ctrlKey) || e.shiftKey || e.altKey || e.key.toLowerCase() !== "z") return
      if ((e.target as HTMLElement | null)?.closest(TYPING)) return
      if (!newest || undoing) return
      e.preventDefault()
      void undo()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [newest, undoing, undo])

  return { newest, undoing, undo, announce, refresh }
}
