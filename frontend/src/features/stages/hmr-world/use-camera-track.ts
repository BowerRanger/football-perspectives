import * as React from "react"

import { getJsonOrNull } from "@/lib/api"
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

/** Fetches /camera/track once; falls back to 30 fps and no marker when absent. */
export function useCameraTrack(): CameraTrackInfo {
  const [info, setInfo] = React.useState<CameraTrackInfo>(EMPTY)
  React.useEffect(() => {
    let alive = true
    void getJsonOrNull<CameraTrackResponse>("/camera/track").then((ct) => {
      if (alive && ct) setInfo(buildCameraSamples(ct))
    })
    return () => {
      alive = false
    }
  }, [])
  return info
}
