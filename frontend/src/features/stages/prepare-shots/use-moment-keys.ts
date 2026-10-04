import * as React from "react"

export interface MomentKeyActions {
  trayOpen: boolean
  togglePlay: () => void
  /** Step the focused well by `delta` frames. */
  step: (delta: number) => void
  switchWell: () => void
  toggleTray: () => void
  markRef: () => void
  markShot: () => void
  nudgeMark: (delta: number) => void
  add: () => void
  /** Returns true when there was something to discard. */
  discard: () => boolean
  closeTray: () => void
  removeLast: () => void
  save: () => void
}

// Never steal keys from text entry, menus or a focused slider.
const IGNORE = 'input, textarea, select, [contenteditable="true"], [role="slider"], [role="combobox"], [role="menu"]'
// Space/Enter belong to the focused control.
const OWNS_ACTIVATION = 'button, a[href], [role="button"], [role="switch"], [role="tab"], summary'

/**
 * Keyboard map for the sync editor region (see the replay-speed UX spec).
 * Attach the returned handler to the region's `onKeyDown`; it only sees keys
 * from inside the region, never page-wide.
 */
export function useMomentKeys(actions: MomentKeyActions) {
  const ref = React.useRef(actions)
  React.useEffect(() => {
    ref.current = actions
  })

  return React.useCallback((e: React.KeyboardEvent) => {
    const a = ref.current
    const target = e.target as HTMLElement
    if (target.closest(IGNORE)) return
    const mod = e.metaKey || e.ctrlKey
    const stop = () => {
      e.preventDefault()
      e.stopPropagation()
    }
    if (mod && e.key.toLowerCase() === "s") {
      stop()
      a.save()
      return
    }
    if (mod) return
    const ownsActivation = !!target.closest(OWNS_ACTIVATION)
    // Timeline blocks handle their own arrows (offset slide).
    const onBlock = target.closest("[data-timeline-block]") != null
    switch (e.key) {
      case " ":
        if (ownsActivation) return
        stop()
        a.togglePlay()
        return
      case "ArrowLeft":
      case "ArrowRight": {
        if (onBlock) return
        const dir = e.key === "ArrowLeft" ? -1 : 1
        stop()
        if (e.altKey && a.trayOpen) a.nudgeMark(dir)
        else a.step(dir * (e.shiftKey ? 10 : 1))
        return
      }
      case ",":
      case ".":
        stop()
        a.step((e.key === "," ? -1 : 1) * (e.shiftKey ? 10 : 1))
        return
      case "f":
      case "F":
        stop()
        a.switchWell()
        return
      case "m":
      case "M":
        stop()
        a.toggleTray()
        return
      default:
    }
    if (!a.trayOpen) return
    switch (e.key) {
      case "1":
        stop()
        a.markRef()
        return
      case "2":
        stop()
        a.markShot()
        return
      case "Enter":
        if (ownsActivation) return
        stop()
        a.add()
        return
      case "Escape":
        stop()
        if (!a.discard()) a.closeTray()
        return
      case "Backspace":
      case "Delete":
        stop()
        a.removeLast()
        return
      default:
    }
  }, [])
}
