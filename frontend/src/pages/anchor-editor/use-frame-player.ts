import * as React from "react"

export interface FramePlayer {
  frame: number
  playing: boolean
  /** Pause and jump to a frame (clamped). */
  seek: (frame: number) => void
  step: (delta: number) => void
  togglePlay: () => void
  reset: () => void
  videoHandlers: {
    onPlay: () => void
    onPause: () => void
    onEnded: () => void
  }
}

/**
 * Frame-accurate transport over a <video>. While playing, the frame is
 * re-derived from currentTime every animation frame; anchors are immutable
 * during playback so there is nothing to await.
 */
export function useFramePlayer(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  fps: number,
  totalFrames: number,
): FramePlayer {
  const [frame, setFrame] = React.useState(0)
  const [playing, setPlaying] = React.useState(false)
  const fpsRef = React.useRef(fps)
  const totalRef = React.useRef(totalFrames)
  const frameRef = React.useRef(0)
  React.useEffect(() => {
    fpsRef.current = fps
    totalRef.current = totalFrames
  }, [fps, totalFrames])

  const commit = React.useCallback((f: number) => {
    frameRef.current = f
    setFrame(f)
  }, [])

  const seekRaw = React.useCallback(
    (target: number) => {
      const clamped = Math.max(0, Math.min(totalRef.current - 1, Math.trunc(target)))
      commit(clamped)
      const video = videoRef.current
      if (video && Number.isFinite(clamped)) video.currentTime = clamped / fpsRef.current
    },
    [commit, videoRef],
  )

  const seek = React.useCallback(
    (target: number) => {
      videoRef.current?.pause()
      seekRaw(target)
    },
    [seekRaw, videoRef],
  )

  const step = React.useCallback((delta: number) => seek(frameRef.current + delta), [seek])

  const togglePlay = React.useCallback(() => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) void video.play().catch(() => undefined)
    else video.pause()
  }, [videoRef])

  const reset = React.useCallback(() => {
    commit(0)
    setPlaying(false)
  }, [commit])

  React.useEffect(() => {
    if (!playing) return
    let raf = 0
    const tick = () => {
      const video = videoRef.current
      if (!video || video.paused || video.ended) return
      const fi = Math.max(0, Math.min(totalRef.current - 1, Math.round(video.currentTime * fpsRef.current)))
      if (fi !== frameRef.current) commit(fi)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, commit, videoRef])

  const videoHandlers = React.useMemo(
    () => ({
      onPlay: () => setPlaying(true),
      onPause: () => setPlaying(false),
      onEnded: () => setPlaying(false),
    }),
    [],
  )

  return { frame, playing, seek, step, togglePlay, reset, videoHandlers }
}
