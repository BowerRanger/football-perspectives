import * as React from "react"

import { errorMessage } from "@/lib/api"
import { ViewerEngine, type Visibility } from "./engine"
import { loadSceneData, type LoadProgress } from "./load-scene"
import type { CameraMode, SceneData } from "./types"

export type ViewerPhase = "loading" | "ready" | "empty" | "error"

export const SPEEDS = [0.25, 0.5, 1, 2] as const

export interface ViewerState {
  phase: ViewerPhase
  progress: LoadProgress
  error: string | null
  data: SceneData | null
  frame: number
  playing: boolean
  speed: number
  vis: Visibility
  cameraMode: CameraMode
  selectedId: string | null
}

const INITIAL: ViewerState = {
  phase: "loading",
  progress: { label: "Loading scene", value: 0 },
  error: null,
  data: null,
  frame: 0,
  playing: false,
  speed: 1,
  vis: { ball: true, skeleton: true, mesh: false },
  cameraMode: "overview",
  selectedId: null,
}

export interface ViewerActions {
  togglePlay: () => void
  step: (delta: number) => void
  seek: (frame: number) => void
  setSpeed: (speed: number) => void
  setCameraMode: (mode: CameraMode) => void
  setVisibility: (vis: Visibility) => void
  selectPlayer: (id: string) => void
  reload: () => void
}

/** Owns the three.js engine lifecycle + the React-visible viewer state for one shot. */
export function useViewer(shot: string | undefined) {
  const containerRef = React.useRef<HTMLDivElement>(null)
  const engineRef = React.useRef<ViewerEngine | null>(null)
  const [state, setState] = React.useState<ViewerState>(INITIAL)
  const [reloadKey, setReloadKey] = React.useState(0)
  const visRef = React.useRef(INITIAL.vis)
  const speedRef = React.useRef(INITIAL.speed)
  const playingRef = React.useRef(false)
  const selectedRef = React.useRef<string | null>(null)

  React.useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const engine = new ViewerEngine(container, (frame) => setState((s) => ({ ...s, frame })))
    engine.setVisibility(visRef.current)
    engine.setSpeed(speedRef.current)
    engineRef.current = engine
    return () => {
      engine.dispose()
      engineRef.current = null
    }
  }, [])

  React.useEffect(() => {
    const controller = new AbortController()
    setState((s) => ({
      ...INITIAL,
      vis: s.vis,
      speed: s.speed,
      progress: { label: "Loading scene", value: 0 },
    }))
    loadSceneData(shot, (progress) => setState((s) => ({ ...s, progress })), controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return
        const engine = engineRef.current
        engine?.load(data)
        engine?.setPlaying(false)
        playingRef.current = false
        selectedRef.current = null
        const cameraMode: CameraMode = data.track && data.track.size > 0 ? "tracked" : "overview"
        setState((s) => ({ ...s, data, cameraMode, frame: 0, phase: data.totalFrames > 0 ? "ready" : "empty" }))
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return
        setState((s) => ({ ...s, phase: "error", error: errorMessage(err) }))
      })
    return () => controller.abort()
  }, [shot, reloadKey])

  const actions = React.useMemo<ViewerActions>(
    () => ({
      togglePlay: () => {
        const next = !playingRef.current
        playingRef.current = next
        engineRef.current?.setPlaying(next)
        setState((s) => ({ ...s, playing: next }))
      },
      step: (delta) => {
        const engine = engineRef.current
        if (!engine) return
        playingRef.current = false
        engine.setPlaying(false)
        engine.setFrame(engine.currentFrame + delta)
        setState((s) => ({ ...s, playing: false }))
      },
      seek: (frame) => engineRef.current?.setFrame(frame),
      setSpeed: (speed) => {
        speedRef.current = speed
        engineRef.current?.setSpeed(speed)
        setState((s) => ({ ...s, speed }))
      },
      setCameraMode: (cameraMode) => {
        selectedRef.current = null
        engineRef.current?.setCameraMode(cameraMode)
        setState((s) => ({ ...s, cameraMode, selectedId: null }))
      },
      setVisibility: (vis) => {
        visRef.current = vis
        engineRef.current?.setVisibility(vis)
        setState((s) => ({ ...s, vis }))
      },
      selectPlayer: (id) => {
        const next = selectedRef.current === id ? null : id
        selectedRef.current = next
        engineRef.current?.setSelected(next)
        setState((s) => ({ ...s, selectedId: next }))
      },
      reload: () => setReloadKey((k) => k + 1),
    }),
    [],
  )

  return { containerRef, state, actions }
}
