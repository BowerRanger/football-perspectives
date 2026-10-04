import * as React from "react"

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { Scene } from "./types"
import { StudioEngine, type CameraPreset, type EngineInput } from "./scene-engine"

const PRESETS: { id: CameraPreset; label: string }[] = [
  { id: "fit", label: "Fit track" },
  { id: "overview", label: "Overview" },
  { id: "view-a", label: "Through view A" },
  { id: "view-b", label: "Through view B" },
  { id: "top", label: "Top down" },
  { id: "goal-near", label: "Near goal" },
  { id: "goal-far", label: "Far goal" },
  { id: "follow", label: "Follow ball" },
  { id: "free", label: "Free" },
]

interface Scene3DProps {
  scene: Scene
  input: EngineInput
  hasViewB: boolean
  onDepth?: (m: number) => void
  className?: string
}

/** Thin React shell around StudioEngine; all GPU state is released on unmount. */
export function Scene3D({ scene, input, hasViewB, onDepth, className }: Scene3DProps) {
  const hostRef = React.useRef<HTMLDivElement | null>(null)
  const engineRef = React.useRef<StudioEngine | null>(null)
  const [preset, setPreset] = React.useState<CameraPreset>("fit")
  const [failed, setFailed] = React.useState(false)
  const depthRef = React.useRef(onDepth)
  React.useEffect(() => {
    depthRef.current = onDepth
  })

  React.useEffect(() => {
    const host = hostRef.current
    if (!host) return
    let engine: StudioEngine
    try {
      engine = new StudioEngine(host)
    } catch {
      setFailed(true)
      return
    }
    engine.setCallbacks({ onPresetChange: setPreset, onDepth: (m) => depthRef.current?.(m) })
    engineRef.current = engine
    return () => {
      engine.dispose()
      engineRef.current = null
    }
  }, [])

  React.useEffect(() => {
    engineRef.current?.setScene(scene)
  }, [scene])

  React.useEffect(() => {
    engineRef.current?.update(input)
  }, [input])

  return (
    <div className={className}>
      <div ref={hostRef} className="absolute inset-0" role="img" aria-label="3-D scene: pitch, cameras, rays, solved ball track and players" />
      {failed ? (
        <p className="absolute inset-0 flex items-center justify-center p-4 text-center text-sm text-stage-foreground">
          WebGL is not available in this browser, so the 3-D well cannot draw.
        </p>
      ) : null}
      <div className="absolute top-2 left-2 rounded-md border bg-background/80 shadow-sm backdrop-blur">
        <Select
          value={preset}
          onValueChange={(v) => {
            const p = v as CameraPreset
            if (p !== "free") engineRef.current?.setPreset(p)
          }}
        >
          <SelectTrigger size="sm" className="h-7 w-40 border-0 bg-transparent text-xs shadow-none" aria-label="3-D camera preset">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {PRESETS.filter((p) => hasViewB || p.id !== "view-b").map((p) => (
              <SelectItem key={p.id} value={p.id} disabled={p.id === "free"}>
                {p.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  )
}
