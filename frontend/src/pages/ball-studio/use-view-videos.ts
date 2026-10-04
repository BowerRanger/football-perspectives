import * as React from "react"

import { refToShot, shotVideoRange } from "./camera-model"
import type { GroupShot } from "./types"

export type ViewStatus = "loading" | "ready" | "seeking" | "out_of_range" | "error"

export interface ViewVideos {
  /** Ref callback factory: `ref={videos.register(shotId)}`. */
  register: (shotId: string) => (el: HTMLVideoElement | null) => void
  status: Record<string, ViewStatus>
  /** Shot-local frame actually on screen (set when the seek completes). */
  shown: Record<string, number | null>
  playing: boolean
  togglePlay: () => void
  pause: () => void
  markError: (shotId: string) => void
}

interface Options {
  shots: readonly GroupShot[]
  frame: number
  fps: number
  /** Last reference frame; playback stops there. */
  maxFrame: number
  onFrame: (ref: number) => void
}

/** Seek target in the middle of the frame so rounding can never land on a neighbour (verified exact on h264 30 fps). */
export const frameTime = (shotFrame: number, fps: number): number => (shotFrame + 0.5) / fps

/**
 * Owns the N <video> elements: maps the master reference frame to each
 * shot's local frame via frame_offset, seeks frame-exactly while paused, and
 * plays them in lockstep (the first in-range video is the clock).
 */
export function useViewVideos({ shots, frame, fps, maxFrame, onFrame }: Options): ViewVideos {
  const els = React.useRef(new Map<string, HTMLVideoElement>())
  const refCallbacks = React.useRef(new Map<string, (el: HTMLVideoElement | null) => void>())
  const [status, setStatus] = React.useState<Record<string, ViewStatus>>({})
  const [shown, setShown] = React.useState<Record<string, number | null>>({})
  const [playing, setPlaying] = React.useState(false)
  const [bump, setBump] = React.useState(0)
  const failed = React.useRef(new Set<string>())
  const latest = React.useRef({ frame, onFrame, maxFrame })
  React.useEffect(() => {
    latest.current = { frame, onFrame, maxFrame }
  })

  const inRange = React.useCallback(
    (s: GroupShot, ref: number) => {
      const sf = refToShot(ref, s.frame_offset)
      const [lo, hi] = shotVideoRange(s)
      return sf >= lo && sf <= hi
    },
    [],
  )

  const register = React.useCallback((shotId: string) => {
    let cb = refCallbacks.current.get(shotId)
    if (!cb) {
      cb = (el) => {
        if (el) {
          els.current.set(shotId, el)
          const onSeeked = () => {
            const sf = Math.floor(el.currentTime * fpsRef.current)
            setShown((p) => ({ ...p, [shotId]: sf }))
            setStatus((p) => (p[shotId] === "out_of_range" || p[shotId] === "error" ? p : { ...p, [shotId]: "ready" }))
          }
          el.addEventListener("seeked", onSeeked)
          el.addEventListener("loadeddata", () => setBump((n) => n + 1))
          el.addEventListener("error", () => {
            failed.current.add(shotId)
            setStatus((p) => ({ ...p, [shotId]: "error" }))
          })
        } else {
          els.current.delete(shotId)
        }
      }
      refCallbacks.current.set(shotId, cb)
    }
    return cb
  }, [])
  const fpsRef = React.useRef(fps)
  React.useEffect(() => {
    fpsRef.current = fps
  }, [fps])

  // Paused: land every video on its exact frame.
  React.useEffect(() => {
    if (playing) return
    const next: Record<string, ViewStatus> = {}
    for (const s of shots) {
      const el = els.current.get(s.shot_id)
      if (failed.current.has(s.shot_id)) {
        next[s.shot_id] = "error"
        continue
      }
      if (!inRange(s, frame)) {
        next[s.shot_id] = "out_of_range"
        el?.pause()
        setShown((p) => (p[s.shot_id] === null ? p : { ...p, [s.shot_id]: null }))
        continue
      }
      if (!el || el.readyState < 1) {
        next[s.shot_id] = "loading"
        continue
      }
      const sf = refToShot(frame, s.frame_offset)
      const t = frameTime(sf, s.fps || fps)
      if (Math.abs(el.currentTime - t) > 1e-4) {
        next[s.shot_id] = "seeking"
        el.currentTime = t
      } else {
        next[s.shot_id] = "ready"
        setShown((p) => (p[s.shot_id] === sf ? p : { ...p, [s.shot_id]: sf }))
      }
    }
    setStatus(next)
  }, [shots, frame, fps, playing, bump, inRange])

  const pause = React.useCallback(() => {
    setPlaying(false)
    els.current.forEach((el) => el.pause())
  }, [])

  const togglePlay = React.useCallback(() => {
    setPlaying((p) => !p)
  }, [])

  // Playing: videos run natively; the first in-range one drives the master frame.
  React.useEffect(() => {
    if (!playing) {
      els.current.forEach((el) => el.pause())
      return
    }
    let raf = 0
    let cancelled = false
    const tick = () => {
      if (cancelled) return
      const { frame: f, onFrame: emit, maxFrame: last } = latest.current
      let driver: { s: GroupShot; el: HTMLVideoElement } | null = null
      for (const s of shots) {
        const el = els.current.get(s.shot_id)
        if (!el || failed.current.has(s.shot_id)) continue
        if (!inRange(s, f)) {
          if (!el.paused) el.pause()
          continue
        }
        const want = frameTime(refToShot(f, s.frame_offset), s.fps || fps)
        if (el.paused) {
          el.currentTime = want
          void el.play().catch(() => undefined)
        } else if (Math.abs(el.currentTime - want) > 2.5 / fps && driver) {
          el.currentTime = want
        }
        if (!driver) driver = { s, el }
      }
      if (driver) {
        const ref = Math.floor(driver.el.currentTime * (driver.s.fps || fps)) - driver.s.frame_offset
        if (ref !== f) {
          if (ref >= last) {
            emit(last)
            setPlaying(false)
            return
          }
          emit(ref)
        }
      } else {
        // Nothing has footage here: advance on a clock until some view does.
        emit(Math.min(last, f + 1))
        if (f >= last) {
          setPlaying(false)
          return
        }
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => {
      cancelled = true
      cancelAnimationFrame(raf)
      els.current.forEach((el) => el.pause())
    }
  }, [playing, shots, fps, inRange])

  const markError = React.useCallback((shotId: string) => {
    failed.current.add(shotId)
    setStatus((p) => ({ ...p, [shotId]: "error" }))
  }, [])

  return { register, status, shown, playing, togglePlay, pause, markError }
}
