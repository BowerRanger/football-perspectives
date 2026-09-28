// The editable anchor document (manual anchors + shot chains + dismissed
// auto suggestions). All updates are immutable; "dirty" is derived by
// comparing against the last loaded/saved snapshot so undoing back to the
// saved state clears the indicator.

import * as React from "react"

import { makeAnchor } from "./placement"
import { dismissKey, type AutoAnchor, type BallAnchor, type DismissedAuto } from "./types"

export interface AnchorDoc {
  anchors: BallAnchor[]
  shotChains: number[][]
  dismissedAuto: DismissedAuto[]
}

export const EMPTY_DOC: AnchorDoc = { anchors: [], shotChains: [], dismissedAuto: [] }

export const docKey = (d: AnchorDoc): string => JSON.stringify(d)

const withoutFrame = (anchors: BallAnchor[], frame: number) => anchors.filter((a) => a.frame !== frame)

export interface AnchorDocApi {
  doc: AnchorDoc
  dirty: boolean
  activeChain: number[] | null
  /** Replace the doc with a freshly loaded/saved one (clears dirty). */
  reset: (doc: AnchorDoc) => void
  markSaved: () => void
  addAnchor: (anchor: BallAnchor, recordChain?: boolean) => void
  markOffScreen: (frame: number) => void
  removeAnchor: (frame: number) => void
  removeNear: (frame: number, u: number, v: number, radiusPx: number) => boolean
  setEndFrame: (frame: number, end: number | null) => void
  setLandmark: (frame: number, landmark: string) => void
  promoteAuto: (auto: AutoAnchor) => void
  dismissAuto: (auto: AutoAnchor) => void
  undoDismiss: (auto: AutoAnchor) => void
  deleteChain: (index: number) => void
  /** Start recording, or end + commit the active chain. Returns a status message. */
  toggleChain: () => { message: string; ok: boolean }
}

export function useAnchorDoc(): AnchorDocApi {
  const [doc, setDoc] = React.useState<AnchorDoc>(EMPTY_DOC)
  const [savedKey, setSavedKey] = React.useState(docKey(EMPTY_DOC))
  const [activeChain, setActiveChain] = React.useState<number[] | null>(null)
  const docRef = React.useRef(doc)
  docRef.current = doc
  const chainRef = React.useRef(activeChain)
  chainRef.current = activeChain

  const update = React.useCallback((fn: (d: AnchorDoc) => AnchorDoc) => setDoc((d) => fn(d)), [])

  const reset = React.useCallback((next: AnchorDoc) => {
    setDoc(next)
    setSavedKey(docKey(next))
    setActiveChain(null)
  }, [])

  const markSaved = React.useCallback(() => setSavedKey(docKey(docRef.current)), [])

  const addAnchor = React.useCallback(
    (anchor: BallAnchor, recordChain = true) => {
      update((d) => ({ ...d, anchors: [...withoutFrame(d.anchors, anchor.frame), anchor] }))
      if (recordChain) {
        setActiveChain((c) => (c && !c.includes(anchor.frame) ? [...c, anchor.frame] : c))
      }
    },
    [update],
  )

  const markOffScreen = React.useCallback(
    (frame: number) => addAnchor(makeAnchor(frame, null, "off_screen_flight"), false),
    [addAnchor],
  )

  const removeAnchor = React.useCallback(
    (frame: number) => update((d) => ({ ...d, anchors: withoutFrame(d.anchors, frame) })),
    [update],
  )

  const removeNear = React.useCallback(
    (frame: number, u: number, v: number, radiusPx: number) => {
      const idx = docRef.current.anchors.findIndex(
        (a) => a.frame === frame && a.image_xy && Math.hypot(a.image_xy[0] - u, a.image_xy[1] - v) < radiusPx,
      )
      if (idx < 0) return false
      update((d) => ({ ...d, anchors: d.anchors.filter((_, i) => i !== idx) }))
      return true
    },
    [update],
  )

  const patchAnchor = React.useCallback(
    (frame: number, patch: Partial<BallAnchor>) =>
      update((d) => ({ ...d, anchors: d.anchors.map((a) => (a.frame === frame ? { ...a, ...patch } : a)) })),
    [update],
  )

  const setEndFrame = React.useCallback((frame: number, end: number | null) => patchAnchor(frame, { end_frame: end }), [patchAnchor])
  const setLandmark = React.useCallback((frame: number, landmark: string) => patchAnchor(frame, { landmark }), [patchAnchor])

  const promoteAuto = React.useCallback(
    (auto: AutoAnchor) =>
      addAnchor(
        makeAnchor(auto.frame, auto.image_xy, auto.state, {
          player_id: auto.player_id,
          bone: auto.bone,
          goal_element: auto.goal_element,
          touch_type: auto.touch_type,
          confidence: 1,
        }),
        false,
      ),
    [addAnchor],
  )

  const dismissAuto = React.useCallback(
    (auto: AutoAnchor) =>
      update((d) => ({
        ...d,
        dismissedAuto: [
          ...d.dismissedAuto,
          { frame: auto.frame, state: auto.state, player_id: auto.player_id, bone: auto.bone },
        ],
      })),
    [update],
  )

  const undoDismiss = React.useCallback(
    (auto: AutoAnchor) =>
      update((d) => ({ ...d, dismissedAuto: d.dismissedAuto.filter((x) => dismissKey(x) !== dismissKey(auto)) })),
    [update],
  )

  const deleteChain = React.useCallback(
    (index: number) => update((d) => ({ ...d, shotChains: d.shotChains.filter((_, i) => i !== index) })),
    [update],
  )

  const toggleChain = React.useCallback(() => {
    const chain = chainRef.current
    if (!chain) {
      setActiveChain([])
      return { ok: true, message: "Recording a shot chain — place the strike, any deflections and the impact, then End." }
    }
    setActiveChain(null)
    if (chain.length < 2) return { ok: false, message: "A shot chain needs at least 2 anchors — discarded." }
    const sorted = [...chain].sort((a, b) => a - b)
    update((d) => ({ ...d, shotChains: [...d.shotChains, sorted] }))
    return { ok: true, message: `Shot chain saved (${chain.length} anchors).` }
  }, [update])

  return {
    doc,
    dirty: docKey(doc) !== savedKey,
    activeChain,
    reset,
    markSaved,
    addAnchor,
    markOffScreen,
    removeAnchor,
    removeNear,
    setEndFrame,
    setLandmark,
    promoteAuto,
    dismissAuto,
    undoDismiss,
    deleteChain,
    toggleChain,
  }
}
