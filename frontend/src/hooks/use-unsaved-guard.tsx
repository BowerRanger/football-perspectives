import * as React from "react"
import { useBlocker } from "react-router"

import { useConfirm } from "@/hooks/use-dialogs"

interface UnsavedGuardOptions {
  /** What is unsaved, e.g. "anchor edits" — used in the dialog copy. */
  what?: string
}

/**
 * Protect unsaved operator edits: blocks in-app navigation (sidebar links,
 * back/forward) behind a confirm, and reload/close behind the browser's
 * native beforeunload prompt. Pass `dirty = false` once saved.
 */
export function useUnsavedGuard(dirty: boolean, { what = "changes" }: UnsavedGuardOptions = {}): void {
  const confirm = useConfirm()
  // Page or stage changes are guarded; ?shot= changes are not — editors
  // confirm their own shot switches, and a second dialog would double-ask.
  const blocker = useBlocker(
    ({ currentLocation, nextLocation }) =>
      dirty &&
      (currentLocation.pathname !== nextLocation.pathname ||
        new URLSearchParams(currentLocation.search).get("stage") !==
          new URLSearchParams(nextLocation.search).get("stage")),
  )

  React.useEffect(() => {
    if (blocker.state !== "blocked") return
    let cancelled = false
    void confirm({
      title: `Leave with unsaved ${what}?`,
      description: `Your ${what} haven't been saved and will be lost.`,
      confirmLabel: "Discard and leave",
      cancelLabel: "Stay",
      destructive: true,
    }).then((ok) => {
      if (cancelled) return
      if (ok) blocker.proceed()
      else blocker.reset()
    })
    return () => {
      cancelled = true
    }
  }, [blocker, confirm, what])

  React.useEffect(() => {
    if (!dirty) return
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault()
    }
    window.addEventListener("beforeunload", onBeforeUnload)
    return () => window.removeEventListener("beforeunload", onBeforeUnload)
  }, [dirty])
}
