import * as React from "react"

import { useResource } from "@/hooks/use-resource"
import { getJson } from "@/lib/api"
import type { CameraSample, CameraTrackResponse } from "./types"

const DEFAULT_FPS = 30

export interface CameraTrackInfo {
  fps: number
  cameraByFrame: ReadonlyMap<number, CameraSample>
}

const EMPTY: CameraTrackInfo = { fps: DEFAULT_FPS, cameraByFrame: new Map() }

/** Camera centre C = -R^T t per frame (OpenCV convention), plus the configured fps. */
export function buildCameraSamples(ct: CameraTrackResponse): CameraTrackInfo {
  const tWorld = ct.t_world && ct.t_world.length === 3 ? ct.t_world : null
  const map = new Map<number, CameraSample>()
  for (const f of ct.frames ?? []) {
    const t = f.t && f.t.length === 3 ? f.t : tWorld
    const R = f.R
    if (!t || !R) continue
    const C = [0, 1, 2].map((k) => -(R[0][k] * t[0] + R[1][k] * t[1] + R[2][k] * t[2]))
    // Forward in world = third row of R, projected to the pitch plane.
    map.set(f.frame, { pos: [C[0], C[1]], z: C[2], fwd: [R[2][0], R[2][1]] })
  }
  return { fps: ct.fps || DEFAULT_FPS, cameraByFrame: map }
}

/**
 * Fetches /camera/track once (the server answers 200 with empty frames when the
 * camera stage hasn't run, so any throw is a real failure). The camera is
 * secondary here: on error the panel still works at 30 fps with no marker, and
 * `error` lets it say so instead of pretending the camera was never solved.
 */
export function useCameraTrack(): CameraTrackInfo & { error: string | null } {
  const { state } = useResource((signal) => getJson<CameraTrackResponse>("/camera/track", { signal }), [])
  return React.useMemo(
    () => ({ ...(state.status === "ready" ? buildCameraSamples(state.data) : EMPTY), error: state.status === "error" ? state.error : null }),
    [state],
  )
}
