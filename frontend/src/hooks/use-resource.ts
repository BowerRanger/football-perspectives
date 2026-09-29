import * as React from "react"

import { errorMessage } from "@/lib/api"

export type ResourceState<T> =
  | { status: "loading"; data: undefined; error: undefined }
  | { status: "error"; data: undefined; error: string }
  | { status: "ready"; data: T; error: undefined }

export interface Resource<T> {
  state: ResourceState<T>
  /** Re-run the loader (Retry buttons). */
  retry: () => void
  /** Replace the loaded value locally (optimistic edits). */
  setData: (update: (prev: T) => T) => void
}

/**
 * Load-once data for a panel, with the three states kept distinct:
 * loading (skeleton), error (PanelError + Retry) and ready — where a
 * ready-but-empty value is the loader's business (e.g. getJsonOr404 → null).
 * The loader receives an AbortSignal; stale responses never land after a
 * dependency change or unmount.
 */
export function useResource<T>(load: (signal: AbortSignal) => Promise<T>, deps: React.DependencyList): Resource<T> {
  const [state, setState] = React.useState<ResourceState<T>>({ status: "loading", data: undefined, error: undefined })
  const [nonce, setNonce] = React.useState(0)
  const loadRef = React.useRef(load)
  React.useEffect(() => {
    loadRef.current = load
  })

  React.useEffect(() => {
    const ctrl = new AbortController()
    setState({ status: "loading", data: undefined, error: undefined })
    loadRef
      .current(ctrl.signal)
      .then((data) => {
        if (!ctrl.signal.aborted) setState({ status: "ready", data, error: undefined })
      })
      .catch((err: unknown) => {
        if (ctrl.signal.aborted) return
        setState({ status: "error", data: undefined, error: errorMessage(err) })
      })
    return () => ctrl.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps -- caller-supplied deps
  }, [...deps, nonce])

  const retry = React.useCallback(() => setNonce((n) => n + 1), [])
  const setData = React.useCallback((update: (prev: T) => T) => {
    setState((prev) => (prev.status === "ready" ? { ...prev, data: update(prev.data) } : prev))
  }, [])

  return { state, retry, setData }
}
