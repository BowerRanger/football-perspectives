import type * as React from "react"
import { BoxIcon, CircleDotIcon, VideoIcon, WorkflowIcon } from "lucide-react"

import { FramePlayer } from "@/components/frame-player"
import { Card } from "@/components/ui/card"
import { Kbd } from "@/components/ui/kbd"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import type { Visibility } from "./engine"
import { OVERLAY_CARD } from "./overlays"
import type { CameraMode, SceneData } from "./types"
import { SPEEDS, type ViewerActions, type ViewerState } from "./use-viewer"

function Hint({ label, keys, children }: { label: string; keys?: string; children: React.ReactElement }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent side="top">
        <span className="flex items-center gap-2">
          {label}
          {keys ? <Kbd>{keys}</Kbd> : null}
        </span>
      </TooltipContent>
    </Tooltip>
  )
}

const TOGGLES: { key: keyof Visibility; label: string; icon: React.ReactNode }[] = [
  { key: "ball", label: "Ball", icon: <CircleDotIcon /> },
  { key: "skeleton", label: "Skeleton", icon: <WorkflowIcon /> },
  { key: "mesh", label: "Mesh", icon: <BoxIcon /> },
]

function VisibilityToggles({ data, vis, onChange }: { data: SceneData; vis: Visibility; onChange: (v: Visibility) => void }) {
  const active = TOGGLES.filter((t) => vis[t.key]).map((t) => t.key)
  const meshMissing = data.smpl === null
  return (
    <ToggleGroup
      type="multiple"
      variant="outline"
      size="sm"
      spacing={1}
      value={active}
      onValueChange={(next) =>
        onChange({ ball: next.includes("ball"), skeleton: next.includes("skeleton"), mesh: next.includes("mesh") })
      }
      aria-label="Scene layers"
    >
      {TOGGLES.map((t) => {
        const disabled = t.key === "mesh" && meshMissing
        const item = (
          <ToggleGroupItem key={t.key} value={t.key} aria-label={t.label} disabled={disabled}>
            {t.icon}
            <span className="hidden sm:inline">{t.label}</span>
          </ToggleGroupItem>
        )
        return disabled ? (
          <Hint key={t.key} label="SMPL model unavailable (run scripts/extract_smpl_neutral.py)">
            <span tabIndex={0}>{item}</span>
          </Hint>
        ) : (
          item
        )
      })}
    </ToggleGroup>
  )
}

const CAMERA_LABELS: Record<Exclude<CameraMode, "tracked">, string> = {
  overview: "Overview",
  tactical: "Tactical (top)",
  "behind-goal": "Behind goal",
}

function CameraSelect({ data, mode, onChange }: { data: SceneData; mode: CameraMode; onChange: (m: CameraMode) => void }) {
  return (
    <Select value={mode} onValueChange={(v) => onChange(v as CameraMode)}>
      <SelectTrigger size="sm" className="w-auto min-w-32" aria-label="Camera">
        <VideoIcon className="text-muted-foreground" />
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {data.track ? <SelectItem value="tracked">Broadcast (solved)</SelectItem> : null}
        {(Object.keys(CAMERA_LABELS) as (keyof typeof CAMERA_LABELS)[]).map((k) => (
          <SelectItem key={k} value={k}>
            {CAMERA_LABELS[k]}
          </SelectItem>
        ))}
        
      </SelectContent>
    </Select>
  )
}

interface TransportProps {
  data: SceneData
  state: ViewerState
  actions: ViewerActions
  /** Whether this player owns Space / arrows (false while another player on the page does). */
  keyboard: boolean
  className?: string
}

export function Transport({ data, state, actions, keyboard, className }: TransportProps) {
  const last = Math.max(0, data.totalFrames - 1)
  return (
    <Card className={cn(OVERLAY_CARD, "px-3 py-2", className)}>
      <FramePlayer
        keyboard={keyboard}
        frame={state.frame}
        max={last}
        fps={data.fps}
        playing={state.playing}
        onTogglePlay={actions.togglePlay}
        onSeek={actions.seek}
        label="Frame"
      >
        <div className="flex flex-wrap items-center gap-2">
        <Select value={String(state.speed)} onValueChange={(v) => actions.setSpeed(Number(v))}>
          <SelectTrigger size="sm" className="w-auto min-w-20" aria-label="Playback speed">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SPEEDS.map((s) => (
              <SelectItem key={s} value={String(s)}>
                {s}x
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <CameraSelect data={data} mode={state.cameraMode} onChange={actions.setCameraMode} />
        <VisibilityToggles data={data} vis={state.vis} onChange={actions.setVisibility} />
        </div>
      </FramePlayer>
    </Card>
  )
}
