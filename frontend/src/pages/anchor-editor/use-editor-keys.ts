import * as React from "react"

import type { ViewOptions } from "./types"

interface KeyDeps {
  rootRef: React.RefObject<HTMLElement | null>
  /** Embedded editors only react to keys while focus is inside them. */
  embedded: boolean
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

/**
 * Editor-only shortcuts: Cmd/Ctrl+S, Esc (cancel placement) and the overlay
 * toggles. Play / step / seek keys belong to the shared FramePlayer.
 */
export function useEditorKeys(deps: KeyDeps) {
  const ref = React.useRef(deps)
  React.useEffect(() => {
    ref.current = deps
  })

  React.useEffect(() => {
    const onKeyDown = (ev: KeyboardEvent) => {
      const target = ev.target as HTMLElement | null
      const d = ref.current
      if (!target || (d.embedded && !(d.rootRef.current?.contains(target) ?? false))) return
      if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "s") {
        ev.preventDefault()
        d.onSave()
        return
      }
      if (ev.metaKey || ev.ctrlKey || ev.altKey || isTyping(target)) return
      if (ev.key === "Escape") {
        d.onEscape()
      } else if (!ev.shiftKey && VIEW_KEYS[ev.key.toLowerCase()]) {
        d.onToggleView(VIEW_KEYS[ev.key.toLowerCase()])
      }
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [])
}
