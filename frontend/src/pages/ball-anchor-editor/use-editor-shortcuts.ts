import * as React from "react"

import { TAGS } from "./tags"
import type { EditorController } from "./use-ball-anchor-editor"

function isTypingTarget(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false
  if (t.isContentEditable) return true
  if (["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName)) return true
  // Radix sliders / selects handle their own arrow + space keys.
  return t.closest('[role="slider"], [role="combobox"], [role="listbox"]') !== null
}

/**
 * Editor shortcuts: Space play/pause, arrows step (Shift = 10 frames),
 * digits / - / = select an anchor type, Cmd/Ctrl+S saves.
 */
export function useEditorShortcuts(ctrl: EditorController, enabled: boolean): void {
  const ref = React.useRef(ctrl)
  ref.current = ctrl

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
      if (e.key === " " && !(e.target instanceof HTMLButtonElement)) {
        e.preventDefault()
        c.player.toggle()
      } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault()
        c.player.step((e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 10 : 1))
      } else {
        const tag = TAGS.find((t) => t.key === e.key)
        if (tag) c.setSelectedTag(tag.id)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [enabled])
}
