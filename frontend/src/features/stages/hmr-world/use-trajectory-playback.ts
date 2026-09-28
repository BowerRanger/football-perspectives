import * as React from "react"

interface Options {
  min: number
  max: number
  fps: number
  videoRef: React.RefObject<HTMLVideoElement | null>
  videoReady: boolean
}

/**
 * Playhead for the trajectory panel. When the picture-in-picture clip has
 * loaded it is the master clock (the two views can't drift); otherwise a
 * self-paced rAF timer advances the frame.
 */
export function useTrajectoryPlayback({ min, max, fps, videoRef, videoReady }: Options) {
  const [frame, setFrameState] = React.useState(min)
  const [playing, setPlaying] = React.useState(false)
  const frameRef = React.useRef(min)

  const setFrame = React.useCallback((f: number) => {
    frameRef.current = f
    setFrameState(f)
  }, [])

  const syncVideo = React.useCallback(
    (fi: number) => {
      const v = videoRef.current
      if (!v || !videoReady) return
      const target = fi / Math.max(1, fps)
      if (v.paused || Math.abs(v.currentTime - target) > 1.5 / fps) {
        try {
          v.currentTime = Math.max(0, Math.min(v.duration || target, target))
        } catch {
          /* metadata not loaded yet */
        }
      }
    },
    [videoRef, videoReady, fps],
  )

  React.useEffect(() => {
    setFrame(min)
    setPlaying(false)
  }, [min, max, setFrame])

  React.useEffect(() => {
    if (!playing) return
    const v = videoRef.current
    if (videoReady && v) {
      syncVideo(frameRef.current)
      void v.play().catch(() => undefined) // the click gesture satisfies autoplay policy
    }
    let raf = 0
    let lastTs = 0
    const tick = (ts: number) => {
      let next: number
      if (videoReady && v) {
        next = Math.round(v.currentTime * fps)
        if (v.ended || next > max) {
          setFrame(Math.min(next, max))
          setPlaying(false)
          return
        }
      } else {
        if (lastTs === 0) lastTs = ts
        next = frameRef.current + Math.max(1, Math.round(((ts - lastTs) / 1000) * fps))
        lastTs = ts
        if (next > max) {
          setFrame(max)
          setPlaying(false)
          return
        }
      }
      setFrame(next)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
      if (v && !v.paused) v.pause()
    }
  }, [playing, videoReady, fps, max, setFrame, syncVideo, videoRef])

  const seek = React.useCallback(
    (f: number) => {
      const clamped = Math.max(min, Math.min(max, f))
      setPlaying(false)
      setFrame(clamped)
      syncVideo(clamped)
    },
    [min, max, setFrame, syncVideo],
  )

  const toggle = React.useCallback(() => {
    if (playing) {
      setPlaying(false)
      return
    }
    if (frameRef.current >= max) setFrame(min)
    setPlaying(true)
  }, [playing, max, min, setFrame])

  return { frame, playing, seek, toggle, step: (d: number) => seek(frameRef.current + d) }
}
