import * as React from "react"

/**
 * Whether an editor's FramePlayer should own the keyboard. Standalone pages
 * always do. An embedded editor (a stage panel that may sit beside another
 * player) only does after the operator has clicked or focused inside it, and
 * gives the keys back as soon as they click or focus anywhere else — so
 * exactly one player on the page answers Space / arrows at a time.
 */
export function useScopedKeyboard(rootRef: React.RefObject<HTMLElement | null>, scoped: boolean): boolean {
  const [active, setActive] = React.useState(false)
  React.useEffect(() => {
    if (!scoped) return
    const inside = (target: EventTarget | null) => target instanceof Node && (rootRef.current?.contains(target) ?? false)
    const onPointerDown = (e: Event) => setActive(inside(e.target))
    const onFocusIn = (e: Event) => setActive(inside(e.target))
    document.addEventListener("pointerdown", onPointerDown, true)
    document.addEventListener("focusin", onFocusIn)
    return () => {
      document.removeEventListener("pointerdown", onPointerDown, true)
      document.removeEventListener("focusin", onFocusIn)
    }
  }, [rootRef, scoped])
  return !scoped || active
}
