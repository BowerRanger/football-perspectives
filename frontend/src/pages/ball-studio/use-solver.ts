import * as React from "react"
import { toast } from "sonner"

import { errorMessage } from "@/lib/api"
import { postSolve } from "./api"
import type { SolveResult, TruthDoc } from "./types"

export type SolveStatus = "idle" | "solving" | "solved" | "error"

export interface SolverState {
  status: SolveStatus
  /** Last good result; kept (and marked stale) while a new solve runs or after a failure. */
  result: SolveResult | null
  stale: boolean
  error: string | null
  ms: number | null
}

const DEBOUNCE_MS = 150

/** Debounced solve-on-edit: aborts a superseded request, keeps the last good dense track. */
export function useSolver(groupId: string, doc: TruthDoc, enabled: boolean): SolverState {
  const [state, setState] = React.useState<SolverState>({ status: "idle", result: null, stale: false, error: null, ms: null })
  const nKeys = doc.keys.length

  React.useEffect(() => {
    if (!enabled) return
    if (nKeys === 0) {
      setState({ status: "idle", result: null, stale: false, error: null, ms: null })
      return
    }
    const ctrl = new AbortController()
    setState((s) => ({ ...s, status: "solving", stale: s.result !== null }))
    const timer = window.setTimeout(() => {
      const t0 = performance.now()
      postSolve(groupId, doc, ctrl.signal)
        .then((result) => {
          if (ctrl.signal.aborted) return
          setState({ status: "solved", result, stale: false, error: null, ms: Math.round(performance.now() - t0) })
        })
        .catch((err: unknown) => {
          if (ctrl.signal.aborted) return
          toast.error("Solve failed", { description: errorMessage(err), id: "ball-studio-solve" })
          setState((s) => ({ status: "error", result: s.result, stale: s.result !== null, error: errorMessage(err), ms: null }))
        })
    }, DEBOUNCE_MS)
    return () => {
      window.clearTimeout(timer)
      ctrl.abort()
    }
  }, [groupId, doc, enabled, nKeys])

  return state
}
