// Stage vocabulary shared by the sidebar, headers, run controls and the
// dependency gate. Order mirrors `STAGE_ORDER` in src/web/server.py.

export type StageName =
  | "prepare_shots"
  | "tracking"
  | "camera"
  | "hmr_world"
  | "refined_poses"
  | "ball"
  | "appearance"
  | "export"
  | "render"
  | "shorts"

/** One row of `GET /api/stages`. */
export interface StageInfo {
  name: StageName
  /** 1-based position in the pipeline. */
  index: number
  complete: boolean
  /** Not complete, but some output exists on disk. */
  partial?: boolean
}

/** Transient state of a stage inside the current job (absent = idle). */
export type LiveStageState = "running" | "error"

export const STAGE_LABELS: Record<StageName, string> = {
  prepare_shots: "Prepare Shots",
  tracking: "Tracking",
  camera: "Camera Tracking",
  hmr_world: "HMR World",
  refined_poses: "Refined Poses",
  ball: "Ball",
  appearance: "Appearance",
  export: "Export",
  render: "Render",
  shorts: "Shorts",
}

export const STAGE_DESCRIPTIONS: Record<StageName, string> = {
  prepare_shots: "Split, classify, group and sync the input into shots.",
  tracking: "Detect and track players and the ball per shot.",
  camera: "Solve the broadcast camera from pitch anchors, then propagate.",
  hmr_world: "GVHMR body pose per player, foot-anchored to the pitch.",
  refined_poses: "Cross-shot fusion, cleanup, foot-lock and physics takeover.",
  ball: "Physically solved 3D ball trajectory from detections and anchors.",
  appearance: "Team and kit colour clustering from the footage; suggestions only, tracks are never edited.",
  export: "glTF scene for the viewer and FBX for Unreal/Blender.",
  render: "Headless Blender toon renders from broadcast and virtual cameras.",
  shorts: "Vertical highlight shorts cut from each goal shot's renders.",
}

/** Stages that must have (at least partial) output before this one can run. */
export const STAGE_DEPS: Record<StageName, StageName[]> = {
  prepare_shots: [],
  tracking: ["prepare_shots"],
  camera: ["prepare_shots"],
  hmr_world: ["tracking", "camera"],
  refined_poses: ["hmr_world"],
  ball: ["camera", "refined_poses"],
  appearance: ["tracking", "refined_poses"],
  export: ["camera"],
  render: ["camera"],
  shorts: ["ball", "appearance"],
}

/** Display label for a stage id; unknown ids pass through, empty shows an em dash. */
export function humanizeStageName(name: string | null | undefined): string {
  return name ? (STAGE_LABELS[name as StageName] ?? name) : "—"
}

export function isStageName(name: string | null | undefined): name is StageName {
  return !!name && name in STAGE_LABELS
}
