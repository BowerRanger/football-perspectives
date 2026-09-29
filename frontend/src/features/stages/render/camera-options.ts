// Fixed (non-player) render cameras the operator can pick. Ids match the
// server's RenderSelection schema; pov:/ots: entries are built per player.

export interface CameraOption {
  id: string
  name: string
  description: string
}

export const CAMERA_OPTIONS: readonly CameraOption[] = [
  { id: "broadcast", name: "Original broadcast", description: "Matches the source footage" },
  { id: "drone", name: "Action drone", description: "Elevated view following the action" },
  { id: "tactical", name: "Full pitch", description: "Steady overhead view of team shape" },
  { id: "sideline:near", name: "Main sideline", description: "Elevated near-touchline replay" },
  { id: "sideline:far", name: "Reverse sideline", description: "The action from the opposite stand" },
  { id: "corner:left", name: "Left corner", description: "Elevated diagonal view toward play" },
  { id: "corner:right", name: "Right corner", description: "Elevated diagonal view toward play" },
  { id: "goal:left", name: "Behind left goal", description: "Low view through the goal net" },
  { id: "goal:right", name: "Behind right goal", description: "Low view through the goal net" },
  { id: "goalline:left", name: "Left goal line", description: "Close view inside the goal mouth" },
  { id: "goalline:right", name: "Right goal line", description: "Close view inside the goal mouth" },
  { id: "orbit", name: "Orbit", description: "Sweeps around the action" },
  { id: "chase", name: "Ball chase", description: "Trails the ball's direction of travel" },
  { id: "dolly", name: "Touchline dolly", description: "Low tracking shot along the near touchline" },
]

export interface RenderCamera {
  id: string
  file: string
  size_bytes: number
  mtime: number
  vertical: boolean
}

export interface RenderShotOutput {
  cameras: RenderCamera[]
  render_seconds: number | null
  aov: boolean
}

export interface RenderOutputs {
  shots: Record<string, RenderShotOutput>
}

export interface RenderSelection {
  shot_id: string
  cameras: string[]
  vertical_variant: boolean | null
}

export interface AvailablePlayer {
  player_id: string
  display_name?: string
}
