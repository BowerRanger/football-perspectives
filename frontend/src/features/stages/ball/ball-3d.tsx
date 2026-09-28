import * as React from "react"

import { Skeleton } from "@/components/ui/skeleton"
import { errorMessage } from "@/lib/api"
import type { BallPreviewTrack } from "@/pages/ball-anchor-editor/api"
import { buildBall3D, type Ball3DHandle } from "./ball-3d-scene"

interface Ball3DProps {
  frames: NonNullable<BallPreviewTrack["frames"]>
  frame: number
}

/** Lazy three.js ball-only viewer in a stage well; disposes its GL context on unmount. */
export function Ball3D({ frames, frame }: Ball3DProps) {
  const containerRef = React.useRef<HTMLDivElement | null>(null)
  const handleRef = React.useRef<Ball3DHandle | null>(null)
  const frameRef = React.useRef(frame)
  frameRef.current = frame
  const [state, setState] = React.useState<"loading" | "ready" | "error">("loading")
  const [error, setError] = React.useState("")

  React.useEffect(() => {
    const container = containerRef.current
    if (!container) return
    let disposed = false
    setState("loading")
    void (async () => {
      try {
        const THREE = await import("three")
        const { OrbitControls } = await import("three/addons/controls/OrbitControls.js")
        if (disposed) return
        handleRef.current = buildBall3D(THREE, OrbitControls, container, frames)
        handleRef.current.setFrame(frameRef.current)
        setState("ready")
      } catch (err) {
        if (disposed) return
        setError(errorMessage(err))
        setState("error")
      }
    })()
    return () => {
      disposed = true
      handleRef.current?.dispose()
      handleRef.current = null
    }
  }, [frames])

  React.useEffect(() => {
    handleRef.current?.setFrame(frame)
  }, [frame, state])

  return (
    <div ref={containerRef} className="relative aspect-[16/10] w-full overflow-hidden rounded-lg bg-stage">
      {state === "loading" ? <Skeleton className="absolute inset-0 rounded-none" /> : null}
      {state === "error" ? (
        <p className="absolute inset-0 flex items-center justify-center p-4 text-center text-sm text-stage-foreground">
          3D viewer unavailable ({error})
        </p>
      ) : null}
    </div>
  )
}
