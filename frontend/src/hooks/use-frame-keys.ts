import * as React from "react"

export interface FrameKeysOptions {
  /** Attach the listener at all (e.g. only while this player is on screen). */
  enabled?: boolean
  frame: number
  min?: number
  max: number
  onTogglePlay?: () => void
  onSeek: (frame: number) => void
}

// Never steal keys from text entry, menus, dialogs or a focused slider.
const IGNORE_SELECTOR =
  'input, textarea, select, [contenteditable="true"], [role="menu"], [role="listbox"], [role="dialog"], [role="slider"], [role="combobox"]'
// Space additionally belongs to whatever control has focus (buttons, links,
// checkboxes, toggles activate natively) — arrows still step after a click.
const SPACE_IGNORE_SELECTOR =
  'button, a[href], [role="button"], [role="checkbox"], [role="switch"], [role="radio"], [role="tab"], summary'

/** Only one frame player owns the keyboard at a time: the most recently mounted enabled one. */
const owners: symbol[] = []

/**
 * Shared transport shortcuts: Space play/pause, ←/→ step, Shift+←/→ ±10,
 * Home/End jump. Ignored while typing or inside menus/dialogs, and when
 * several players are on one page only the newest enabled one listens.
 */
export function useFrameKeys({ enabled = true, frame, min = 0, max, onTogglePlay, onSeek }: FrameKeysOptions): void {
  const latest = React.useRef({ frame, min, max, onTogglePlay, onSeek })
  React.useEffect(() => {
    latest.current = { frame, min, max, onTogglePlay, onSeek }
  })

  React.useEffect(() => {
    if (!enabled) return
    const id = Symbol("frame-keys")
    owners.push(id)
    const onKeyDown = (e: KeyboardEvent) => {
      if (owners[owners.length - 1] !== id) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const target = e.target as HTMLElement | null
      if (target?.closest(IGNORE_SELECTOR)) return
      const { frame: f, min: lo, max: hi, onTogglePlay: toggle, onSeek: seek } = latest.current
      const clamp = (n: number) => Math.min(hi, Math.max(lo, n))
      const stepBy = e.shiftKey ? 10 : 1
      switch (e.key) {
        case " ":
          if (!toggle || target?.closest(SPACE_IGNORE_SELECTOR)) return
          e.preventDefault()
          toggle()
          break
        case "ArrowLeft":
          e.preventDefault()
          seek(clamp(f - stepBy))
          break
        case "ArrowRight":
          e.preventDefault()
          seek(clamp(f + stepBy))
          break
        case "Home":
          e.preventDefault()
          seek(lo)
          break
        case "End":
          e.preventDefault()
          seek(hi)
          break
        default:
      }
    }
    window.addEventListener("keydown", onKeyDown)
    return () => {
      window.removeEventListener("keydown", onKeyDown)
      const i = owners.indexOf(id)
      if (i >= 0) owners.splice(i, 1)
    }
  }, [enabled])
}
