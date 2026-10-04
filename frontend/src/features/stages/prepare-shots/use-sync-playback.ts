import { frameTime } from "@/lib/frame-time"
import * as React from "react"

import { shotFrameForRef } from "./replay-speed"

const DRIFT_S = 0.08

export interface PlaybackState {
  referenceShot: string
  activeShot: string
  /** Effective (preview-applied) frame offsets and rates. */
  offsets: Record<string, number>
  rates: Record<string, number>
  /** "fit": the member follows the time map at 1 / rate; "raw": both play untouched from where they are. */
  speedMode: "fit" | "raw"
  playing: boolean
}

interface PlaybackOptions {
  refVideo: React.RefObject<HTMLVideoElement | null>
  actVideo: React.RefObject<HTMLVideoElement | null>
  fpsByShot: Record<string, number>
  state: PlaybackState
}

/**
 * Keeps the member clip on the reference clock through the rate-aware time
 * map: frame-exact seeks while paused, continuous follow (drift re-seek only)
 * while playing in "fit" mode, and hands-off in "raw" mode so Play never
 * makes the member jump. Also owns the playhead position.
 */
export function usePlaybackSync({ refVideo, actVideo, fpsByShot, state }: PlaybackOptions) {
  const [cursorFrame, setCursorFrame] = React.useState(0)
  const latest = React.useRef(state)
  React.useEffect(() => {
    latest.current = state
  })
  const fps = React.useCallback((id: string) => fpsByShot[id] || 25, [fpsByShot])

  /** Move the active video to the instant matching the reference. */
  const syncActive = React.useCallback(
    (tolerance = 0.05) => {
      const rv = refVideo.current
      const av = actVideo.current
      if (!rv || !av) return
      const { referenceShot: r, activeShot: a, offsets, rates, playing, speedMode } = latest.current
      if (playing && speedMode === "raw") return
      const rate = rates[a] ?? 1
      // Continuous position in frames of the reference clip (frame f centre = f).
      const refPos = rv.currentTime * fps(r) - 0.5
      const shotPos = shotFrameForRef(refPos, rate, offsets[a] ?? 0)
      const target = Math.max(0, frameTime(playing ? shotPos : Math.round(shotPos), fps(a)))
      if (Math.abs(av.currentTime - target) > tolerance / Math.min(1, rate || 1)) {
        try {
          av.currentTime = target
        } catch {
          /* metadata not loaded yet; loadedmetadata re-syncs */
        }
      }
    },
    [refVideo, actVideo, fps],
  )

  const readCursor = React.useCallback(() => {
    const rv = refVideo.current
    if (rv) setCursorFrame(rv.currentTime * fps(latest.current.referenceShot))
  }, [refVideo, fps])

  const { referenceShot, activeShot, offsets, rates, playing } = state
  React.useEffect(() => {
    syncActive()
  }, [referenceShot, activeShot, offsets, rates, syncActive])

  React.useEffect(() => {
    if (!playing) return
    const rv = refVideo.current
    const av = actVideo.current
    syncActive()
    rv?.play().catch(() => undefined)
    av?.play().catch(() => undefined)
    let raf = 0
    const tick = () => {
      readCursor()
      syncActive(DRIFT_S)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
      rv?.pause()
      av?.pause()
    }
  }, [playing, refVideo, actVideo, syncActive, readCursor])

  return { syncActive, readCursor, cursorFrame }
}
