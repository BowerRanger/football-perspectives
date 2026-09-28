import * as React from "react"

import type { StageName } from "@/lib/stages"

// Each stage panel is its own lazily-loaded chunk: the dashboard only pays
// for three.js / canvas editors when the operator opens that stage.
export const STAGE_PANELS: Record<StageName, React.LazyExoticComponent<React.ComponentType>> = {
  prepare_shots: React.lazy(() => import("./prepare-shots")),
  tracking: React.lazy(() => import("./tracking")),
  camera: React.lazy(() => import("./camera")),
  hmr_world: React.lazy(() => import("./hmr-world")),
  refined_poses: React.lazy(() => import("./refined-poses")),
  ball: React.lazy(() => import("./ball")),
  export: React.lazy(() => import("./export")),
  render: React.lazy(() => import("./render")),
}
