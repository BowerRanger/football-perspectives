// Payload shapes for /camera/track, /anchors/{shot}, /api/camera/metrics.

export interface CameraFrame {
  frame: number
  R?: number[][]
  t?: number[]
  is_anchor?: boolean
  confidence?: number | null
}

export interface CameraTrack {
  fps: number
  image_size: number[]
  t_world?: number[]
  frames: CameraFrame[]
}

export interface AnchorsPayload {
  anchors?: unknown[]
}

export interface CameraMetrics {
  available?: boolean
  covered: number
  clip_frames: number
  line_rms_mean?: number | null
  jitter_p95?: number | null
  circle?: { misfit: number | null; frames: number } | null
  vs_manual?: { rotation: number; centre: number } | null
}

export interface ShotCameraData {
  id: string
  colour: string
  track: CameraTrack | null
  anchors: AnchorsPayload | null
}

/** Per-frame camera pose on the pitch, used by the top-down map. */
export interface PitchCameraPose {
  frame: number
  pos: [number, number]
  z: number
  fwd: [number, number]
  isAnchor: boolean
}

export interface IndexedShot {
  id: string
  colour: string
  fps: number
  frames: PitchCameraPose[]
  byFrame: Map<number, PitchCameraPose>
  minFrame: number
  maxFrame: number
}
