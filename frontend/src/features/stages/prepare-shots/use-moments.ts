import * as React from "react"

import { fitMoments, sortPairs, type MomentFit, type MomentPair } from "./replay-speed"

export interface MomentsState {
  /** Committed pairs for the active member, sorted by replay frame. */
  pairs: MomentPair[]
  fit: MomentFit | null
  /** Pending marks: nothing enters the fit until both exist and Enter commits. */
  pendingRef: number | null
  pendingShot: number | null
  /** Pairs held for any member (unsaved work to protect). */
  hasUnsaved: boolean
  markRef: (frame: number) => void
  markShot: (frame: number) => void
  /** Move the pending replay mark (Alt+arrows); needs one to exist. */
  nudgeShot: (delta: number) => void
  add: () => boolean
  discardPending: () => boolean
  remove: (index: number) => void
  /** Remove the pair added most recently (not the one with the highest frame). */
  removeLatest: () => void
  clear: () => void
  /** Drop the saved member's pairs and any pending marks. */
  reset: (shotId?: string) => void
}

/**
 * Operator-marked matching moments per replay. Pairs are kept per member so
 * switching the clip to sync never silently drops them; pending marks are
 * per-session and dropped on a switch.
 */
export function useMoments(activeShot: string): MomentsState {
  const [byShot, setByShot] = React.useState<Record<string, MomentPair[]>>({})
  // Insertion order per member, so Backspace undoes the last pair ADDED.
  const [addedOrder, setAddedOrder] = React.useState<Record<string, MomentPair[]>>({})
  const [pending, setPending] = React.useState<{ shot: string; ref: number | null; replay: number | null }>({
    shot: activeShot,
    ref: null,
    replay: null,
  })
  const p = React.useMemo(
    () => (pending.shot === activeShot ? pending : { shot: activeShot, ref: null, replay: null }),
    [pending, activeShot],
  )

  const pairs = React.useMemo(() => sortPairs(byShot[activeShot] ?? []), [byShot, activeShot])
  const fit = React.useMemo(() => fitMoments(pairs), [pairs])

  const markRef = React.useCallback((frame: number) => setPending({ ...p, shot: activeShot, ref: frame }), [p, activeShot])
  const markShot = React.useCallback((frame: number) => setPending({ ...p, shot: activeShot, replay: frame }), [p, activeShot])
  const nudgeShot = React.useCallback(
    (delta: number) => {
      if (p.replay == null) return
      setPending({ ...p, replay: Math.max(0, p.replay + delta) })
    },
    [p],
  )

  const add = React.useCallback((): boolean => {
    if (p.ref == null || p.replay == null) return false
    const next: MomentPair = { reference_frame: p.ref, shot_frame: p.replay }
    setByShot((prev) => ({ ...prev, [activeShot]: [...(prev[activeShot] ?? []), next] }))
    setAddedOrder((prev) => ({ ...prev, [activeShot]: [...(prev[activeShot] ?? []), next] }))
    setPending({ shot: activeShot, ref: null, replay: null })
    return true
  }, [p, activeShot])

  const discardPending = React.useCallback((): boolean => {
    if (p.ref == null && p.replay == null) return false
    setPending({ shot: activeShot, ref: null, replay: null })
    return true
  }, [p, activeShot])

  const remove = React.useCallback(
    (index: number) => {
      const target = sortPairs(byShot[activeShot] ?? [])[index]
      if (!target) return
      setByShot((prev) => ({ ...prev, [activeShot]: (prev[activeShot] ?? []).filter((q) => q !== target) }))
      setAddedOrder((prev) => ({ ...prev, [activeShot]: (prev[activeShot] ?? []).filter((q) => q !== target) }))
    },
    [byShot, activeShot],
  )

  const removeLatest = React.useCallback(() => {
    const order = addedOrder[activeShot] ?? []
    const last = order[order.length - 1]
    if (!last) return
    setAddedOrder((prev) => ({ ...prev, [activeShot]: order.slice(0, -1) }))
    setByShot((prev) => ({ ...prev, [activeShot]: (prev[activeShot] ?? []).filter((q) => q !== last) }))
  }, [addedOrder, activeShot])

  const clear = React.useCallback(() => {
    setByShot((prev) => ({ ...prev, [activeShot]: [] }))
    setAddedOrder((prev) => ({ ...prev, [activeShot]: [] }))
  }, [activeShot])

  const reset = React.useCallback(
    (shotId?: string) => {
      const id = shotId ?? activeShot
      setByShot((prev) => ({ ...prev, [id]: [] }))
      setAddedOrder((prev) => ({ ...prev, [id]: [] }))
      setPending({ shot: activeShot, ref: null, replay: null })
    },
    [activeShot],
  )

  const hasUnsaved = Object.values(byShot).some((v) => v.length > 0)
  return { pairs, fit, pendingRef: p.ref, pendingShot: p.replay, hasUnsaved, markRef, markShot, nudgeShot, add, discardPending, remove, removeLatest, clear, reset }
}
