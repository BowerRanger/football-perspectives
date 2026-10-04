// Frame-accurate transport over a <video>: the frame index is always derived
// from video.currentTime (floor; seeks target the frame middle), so clicks place anchors on the frame the
// operator is actually looking at.

import * as React from "react"

import { frameAtTime, frameTime } from "@/lib/frame-time"

export interface FramePlayer {
  /** Callback ref — attach to the <video> element. */
  videoRef: (el: HTMLVideoElement | null) => void
  frame: number
  fps: number
  setFps: (fps: number) => void
  playing: boolean
  totalFrames: number
  videoSize: [number, number] | null
  /** Frame under the playhead right now (reads the element, not React state). */
  currentFrame: () => number
  seekTo: (frame: number) => void
  step: (delta: number) => void
  toggle: () => void
}

export function useFramePlayer(onFrameChange?: (frame: number) => void): FramePlayer {
  const [video, setVideo] = React.useState<HTMLVideoElement | null>(null)
  const [fps, setFpsState] = React.useState(30)
  const [frame, setFrame] = React.useState(0)
  const [playing, setPlaying] = React.useState(false)
  const [totalFrames, setTotalFrames] = React.useState(0)
  const [videoSize, setVideoSize] = React.useState<[number, number] | null>(null)
  const fpsRef = React.useRef(fps)
  fpsRef.current = fps
  const videoElRef = React.useRef<HTMLVideoElement | null>(null)
  videoElRef.current = video
  const totalRef = React.useRef(totalFrames)
  totalRef.current = totalFrames
  const cbRef = React.useRef(onFrameChange)
  cbRef.current = onFrameChange

  const lastFrame = () => (totalRef.current > 0 ? totalRef.current - 1 : Number.POSITIVE_INFINITY)

  const currentFrame = React.useCallback(
    () => (video ? frameAtTime(video.currentTime, fpsRef.current, lastFrame()) : 0),
    [video],
  )

  React.useEffect(() => {
    if (!video) return
    let raf = 0
    const sync = () => setFrame(frameAtTime(video.currentTime, fpsRef.current, lastFrame()))
    const loop = () => {
      sync()
      raf = requestAnimationFrame(loop)
    }
    const onMeta = () => {
      setTotalFrames(Math.round(video.duration * fpsRef.current))
      setVideoSize([video.videoWidth, video.videoHeight])
      sync()
    }
    const onPlay = () => {
      setPlaying(true)
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(loop)
    }
    const onPause = () => {
      setPlaying(false)
      cancelAnimationFrame(raf)
      sync()
    }
    video.addEventListener("loadedmetadata", onMeta)
    video.addEventListener("timeupdate", sync)
    video.addEventListener("seeked", sync)
    video.addEventListener("play", onPlay)
    video.addEventListener("pause", onPause)
    video.addEventListener("ended", onPause)
    if (video.readyState >= 1) onMeta()
    return () => {
      cancelAnimationFrame(raf)
      video.removeEventListener("loadedmetadata", onMeta)
      video.removeEventListener("timeupdate", sync)
      video.removeEventListener("seeked", sync)
      video.removeEventListener("play", onPlay)
      video.removeEventListener("pause", onPause)
      video.removeEventListener("ended", onPause)
    }
  }, [video])

  React.useEffect(() => {
    cbRef.current?.(frame)
  }, [frame])

  const setFps = React.useCallback((next: number) => {
    fpsRef.current = next
    setFpsState(next)
    const el = videoElRef.current
    if (el && Number.isFinite(el.duration)) setTotalFrames(Math.round(el.duration * next))
  }, [])

  const seekTo = React.useCallback(
    (fi: number) => {
      if (!video) return
      video.pause()
      const last = totalRef.current > 0 ? totalRef.current - 1 : Number.POSITIVE_INFINITY
      const clamped = Math.max(0, Math.min(last, Math.trunc(fi)))
      video.currentTime = frameTime(clamped, fpsRef.current)
      setFrame(clamped)
    },
    [video],
  )

  const step = React.useCallback((delta: number) => seekTo(currentFrame() + delta), [seekTo, currentFrame])

  const toggle = React.useCallback(() => {
    if (!video) return
    if (video.paused) void video.play().catch(() => undefined)
    else video.pause()
  }, [video])

  return { videoRef: setVideo, frame, fps, setFps, playing, totalFrames, videoSize, currentFrame, seekTo, step, toggle }
}
