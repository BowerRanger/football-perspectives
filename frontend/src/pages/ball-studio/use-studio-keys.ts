import * as React from "react"

import type { Studio } from "./use-studio"

const IGNORE =
  'input, textarea, select, [contenteditable="true"], [role="menu"], [role="listbox"], [role="dialog"], [role="slider"], [role="combobox"]'

interface Handlers {
  openEventMenu: () => void
  openHelp: () => void
}

/** Studio shortcuts (frame stepping with arrows/Space/Home/End stays with FramePlayer). */
export function useStudioKeys(studio: Studio, h: Handlers, enabled: boolean): void {
  const ref = React.useRef({ studio, h })
  React.useEffect(() => {
    ref.current = { studio, h }
  })

  React.useEffect(() => {
    if (!enabled) return
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target?.closest(IGNORE)) return
      const { studio: s, h: hh } = ref.current
      const mod = e.metaKey || e.ctrlKey
      const key = e.key

      if (mod) {
        if (key.toLowerCase() === "z") {
          e.preventDefault()
          if (e.shiftKey) s.docApi.redo()
          else s.docApi.undo()
        } else if (key.toLowerCase() === "s") {
          e.preventDefault()
          void s.save()
        }
        return
      }

      // Alt+arrows nudge the pending pick in the active view by 1 (5 with Shift) native px.
      if (e.altKey && key.startsWith("Arrow")) {
        const id = s.shots[s.activeView]?.shot_id
        if (!id || !s.pick.picks[id]) return
        e.preventDefault()
        const step = e.shiftKey ? 5 : 1
        const dx = key === "ArrowLeft" ? -step : key === "ArrowRight" ? step : 0
        const dy = key === "ArrowUp" ? -step : key === "ArrowDown" ? step : 0
        s.dispatchPick({ type: "nudge", shotId: id, dx, dy })
        return
      }
      if (e.altKey) return

      const step = (n: number) => {
        e.preventDefault()
        s.setFrame(s.frame + n)
      }
      switch (key) {
        case ",":
          return step(-1)
        case ".":
          return step(1)
        case "<":
          return step(-10)
        case ">":
          return step(10)
        case "Enter":
          if (target?.closest("button, a[href]")) return
          e.preventDefault()
          void s.commit()
          return
        case "Escape": {
          if (Object.keys(s.pick.picks).length) s.dispatchPick({ type: "clear" })
          else if (s.selection) s.setSelection(null)
          else if (s.layout === "focus") s.setLayout("compare")
          return
        }
        case "Delete":
        case "Backspace":
          e.preventDefault()
          s.deleteSelected()
          return
        case "Tab": {
          if (s.shots.length < 2) return
          e.preventDefault()
          s.setActiveView((s.activeView + (e.shiftKey ? s.shots.length - 1 : 1)) % s.shots.length)
          return
        }
        case "?":
          hh.openHelp()
          return
        default:
      }

      const lower = key.toLowerCase()
      switch (lower) {
        case "k":
          void s.commit("key")
          return
        case "o":
          void s.commit("observation")
          return
        case "s":
          void s.save()
          return
        case "g":
          s.dispatchPick({ type: "constraint", constraint: "ground" })
          return
        case "h":
          s.dispatchPick({ type: "constraint", constraint: "height" })
          return
        case "l":
          s.dispatchPick({ type: "constraint", constraint: "plane" })
          return
        case "d":
          s.dispatchPick({ type: "constraint", constraint: "depth" })
          return
        case "p":
          s.dispatchPick({ type: "constraint", constraint: "player" })
          return
        case "t":
          s.dispatchPick({ type: "mode", mode: "triangulate" })
          return
        case "e":
          e.preventDefault()
          hh.openEventMenu()
          return
        case "f":
          s.setLayout(s.layout === "focus" ? "compare" : "focus")
          return
        case "z":
          if (!e.repeat) s.setLoupe(true)
          return
        case "0":
          s.resetZoom()
          return
        case "[":
        case "]": {
          const keys = s.docApi.doc.keys
          if (!keys.length) return
          const next = lower === "]" ? keys.find((k) => k.frame > s.frame) : [...keys].reverse().find((k) => k.frame < s.frame)
          if (next) {
            s.setFrame(next.frame)
            s.setSelection({ type: "key", id: next.id })
          }
          return
        }
        case "n": {
          const list = s.attention
          if (!list.length) return
          const next = e.shiftKey ? [...list].reverse().find((f) => f < s.frame) : list.find((f) => f > s.frame)
          if (next !== undefined) s.setFrame(next)
          return
        }
        default:
      }
    }
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "z") ref.current.studio.setLoupe(false)
    }
    window.addEventListener("keydown", onKeyDown)
    window.addEventListener("keyup", onKeyUp)
    return () => {
      window.removeEventListener("keydown", onKeyDown)
      window.removeEventListener("keyup", onKeyUp)
    }
  }, [enabled])
}
