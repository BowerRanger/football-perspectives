import * as React from "react"

import { historyReducer, initHistory, isDirty, type HistoryState } from "./truth-doc"
import type { TruthDoc } from "./types"

export interface TruthDocApi {
  doc: TruthDoc
  dirty: boolean
  canUndo: boolean
  canRedo: boolean
  /** Concurrency token: `meta.updated_at` of the file as loaded / last saved. */
  token: string | null
  /** Apply a pure edit; returns the new document. */
  edit: (fn: (d: TruthDoc) => TruthDoc) => TruthDoc
  undo: () => void
  redo: () => void
  load: (doc: TruthDoc) => void
  markSaved: (doc: TruthDoc, updatedAt: string) => void
}

/** The document under edit with bounded undo/redo; `dirty` compares with the last saved JSON. */
export function useTruthDoc(initial: TruthDoc): TruthDocApi {
  const [state, dispatch] = React.useReducer(historyReducer, initial, initHistory) as [
    HistoryState,
    React.Dispatch<Parameters<typeof historyReducer>[1]>,
  ]
  const latest = React.useRef(state.present)
  React.useEffect(() => {
    latest.current = state.present
  }, [state.present])

  const edit = React.useCallback((fn: (d: TruthDoc) => TruthDoc) => {
    const next = fn(latest.current)
    latest.current = next
    dispatch({ type: "edit", doc: next })
    return next
  }, [])
  const undo = React.useCallback(() => dispatch({ type: "undo" }), [])
  const redo = React.useCallback(() => dispatch({ type: "redo" }), [])
  const load = React.useCallback((doc: TruthDoc) => dispatch({ type: "load", doc }), [])
  const markSaved = React.useCallback((doc: TruthDoc, updatedAt: string) => {
    latest.current = doc
    dispatch({ type: "saved", doc, updatedAt })
  }, [])

  return {
    doc: state.present,
    dirty: isDirty(state),
    canUndo: state.past.length > 0,
    canRedo: state.future.length > 0,
    token: state.token,
    edit,
    undo,
    redo,
    load,
    markSaved,
  }
}
