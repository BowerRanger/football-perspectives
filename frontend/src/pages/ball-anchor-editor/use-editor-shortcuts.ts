import * as React from "react"

import { TAGS } from "./tags"
import type { EditorController } from "./use-ball-anchor-editor"

function isTypingTarget(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false
  if (t.isContentEditable) return true
  if (["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName)) return true
  return t.closest('[role="slider"], [role="combobox"], [role="listbox"]') !== null
}

/**
 * Editor-only shortcuts: digits / - / = select an anchor type, Cmd/Ctrl+S
 * saves. Play / step / seek keys belong to the shared FramePlayer.
 */
export function useEditorShortcuts(ctrl: EditorController, enabled: boolean): void {
  const ref = React.useRef(ctrl)
  React.useEffect(() => {
    ref.current = ctrl
  })

  React.useEffect(() => {
    if (!enabled) return
    const onKey = (e: KeyboardEvent) => {
      const c = ref.current
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
        e.preventDefault()
        void c.save()
        return
      }
      if (e.metaKey || e.ctrlKey || e.altKey || isTypingTarget(e.target)) return
      const tag = TAGS.find((t) => t.key === e.key)
      if (tag) c.setSelectedTag(tag.id)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [enabled])
}
