import type { CameraTrack, IndexedShot, PitchCameraPose, ShotCameraData } from "./types"

/**
 * Camera centre per frame: C = -R^T t (falling back to the shot's t_world
 * when a frame has no per-frame t), forward = third row of R projected to the
 * pitch plane. For a static rig the position is constant per shot; for a
 * moving-camera shot it genuinely translates as you scrub.
 */
export function poseFromFrame(R: number[][], t: number[], frame: number, isAnchor: boolean): PitchCameraPose {
  const C = [0, 1, 2].map((j) => -(R[0][j] * t[0] + R[1][j] * t[1] + R[2][j] * t[2]))
  let fx = R[2][0]
  let fy = R[2][1]
  const mag = Math.hypot(fx, fy) || 1
  fx /= mag
  fy /= mag
  return { frame, pos: [C[0], C[1]], z: C[2], fwd: [fx, fy], isAnchor }
}

function poses(track: CameraTrack): PitchCameraPose[] {
  const tWorld = track.t_world && track.t_world.length === 3 ? track.t_world : null
  const out: PitchCameraPose[] = []
  for (const f of track.frames) {
    const t = f.t && f.t.length === 3 ? f.t : tWorld
    if (!t || !f.R) continue
    out.push(poseFromFrame(f.R, t, f.frame, !!f.is_anchor))
  }
  return out.sort((a, b) => a.frame - b.frame)
}

export function indexShots(perShot: readonly ShotCameraData[]): IndexedShot[] {
  const out: IndexedShot[] = []
  for (const { id, track, colour } of perShot) {
    if (!track?.frames?.length) continue
    const frames = poses(track)
    if (frames.length === 0) continue
    out.push({
      id,
      colour,
      fps: track.fps || 30,
      frames,
      byFrame: new Map(frames.map((p) => [p.frame, p])),
      minFrame: frames[0].frame,
      maxFrame: frames[frames.length - 1].frame,
    })
  }
  return out
}

/** Pose at `fi`; shots that don't cover it hold at their nearest end. */
export function nearestInShot(shot: IndexedShot, fi: number): PitchCameraPose {
  const exact = shot.byFrame.get(fi)
  if (exact) return exact
  if (fi <= shot.minFrame) return shot.frames[0]
  if (fi >= shot.maxFrame) return shot.frames[shot.frames.length - 1]
  let best = shot.frames[0]
  for (const e of shot.frames) {
    if (Math.abs(e.frame - fi) < Math.abs(best.frame - fi)) best = e
  }
  return best
}
