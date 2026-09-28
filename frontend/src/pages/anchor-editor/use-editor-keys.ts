import * as React from "react"

import type { FramePlayer } from "./use-frame-player"
import type { ViewOptions } from "./types"

interface KeyDeps {
  rootRef: React.RefObject<HTMLElement | null>
  /** Embedded editors only react to keys while focus is inside them. */
  embedded: boolean
  player: FramePlayer
  totalFrames: number
  onEscape: () => void
  onSave: () => void
  onToggleView: (key: keyof ViewOptions) => void
}

const VIEW_KEYS: Record<string, keyof ViewOptions> = {
  s: "snap",
  p: "pitch",
  d: "detected",
  l: "labels",
  a: "anchors",
}

const TYPING = "input, textarea, select, [contenteditable=''], [contenteditable='true']"
const OVERLAYS = "[role='listbox'], [role='menu'], [role='dialog'], [role='alertdialog']"

function isTyping(el: HTMLElement): boolean {
  return el.matches(TYPING) || el.closest(OVERLAYS) !== null
}

/** Mouse-focused buttons don't own Space; keyboard-focused ones keep it. */
function ownsSpace(el: HTMLElement): boolean {
  const interactive = el.matches("button, a, [role='button'], [role='switch'], [role='checkbox']")
  return interactive && el.matches(":focus-visible")
}

/** Legacy shortcuts (arrows, Space, Esc) plus Shift+arrows, Home/End, view toggles and Cmd/Ctrl+S. */
export function useEditorKeys(deps: KeyDeps) {
  const ref = React.useRef(deps)
  React.useEffect(() => {
    ref.current = deps
  })

  React.useEffect(() => {
    const inScope = (target: HTMLElement) => {
      const d = ref.current
      return !d.embedded || (d.rootRef.current?.contains(target) ?? false)
    }

    const onKeyDown = (ev: KeyboardEvent) => {
      const target = ev.target as HTMLElement | null
      if (!target || !inScope(target)) return
      const d = ref.current
      if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "s") {
        ev.preventDefault()
        d.onSave()
        return
      }
      if (ev.metaKey || ev.ctrlKey || ev.altKey || isTyping(target)) return
      const onSlider = target.getAttribute("role") === "slider"
      const stepBy = ev.shiftKey ? 10 : 1
      if ((ev.key === "ArrowLeft" || ev.key === "ArrowRight") && !onSlider) {
        ev.preventDefault()
        d.player.step(ev.key === "ArrowLeft" ? -stepBy : stepBy)
      } else if (ev.key === " " && !ownsSpace(target)) {
        ev.preventDefault()
        d.player.togglePlay()
      } else if (ev.key === "Escape") {
        d.onEscape()
      } else if (ev.key === "Home" && !onSlider) {
        ev.preventDefault()
        d.player.seek(0)
      } else if (ev.key === "End" && !onSlider) {
        ev.preventDefault()
        d.player.seek(d.totalFrames - 1)
      } else if (!ev.shiftKey && VIEW_KEYS[ev.key.toLowerCase()]) {
        d.onToggleView(VIEW_KEYS[ev.key.toLowerCase()])
      }
    }

    // Space activates buttons on keyup; swallow it when we used it to play/pause.
    const onKeyUp = (ev: KeyboardEvent) => {
      const target = ev.target as HTMLElement | null
      if (ev.key !== " " || !target || !inScope(target) || isTyping(target) || ownsSpace(target)) return
      ev.preventDefault()
    }

    window.addEventListener("keydown", onKeyDown)
    window.addEventListener("keyup", onKeyUp)
    return () => {
      window.removeEventListener("keydown", onKeyDown)
      window.removeEventListener("keyup", onKeyUp)
    }
  }, [])
}
