import * as React from "react"
import { CheckIcon, XIcon } from "lucide-react"

import { ToneBadge } from "@/components/status"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Kbd } from "@/components/ui/kbd"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { Label } from "@/components/ui/label"
import { RESIDUAL_BAD_PX, RESIDUAL_REJECT_PX, residualSeverity, viewColour, viewLetter } from "./palette"
import { naturalCommit } from "./pick-session"
import type { ConstraintMode } from "./types"
import type { Studio } from "./use-studio"

const CONSTRAINTS: { id: ConstraintMode; label: string; key: string }[] = [
  { id: "ground", label: "Ground", key: "G" },
  { id: "height", label: "Height", key: "H" },
  { id: "plane", label: "Goal line", key: "L" },
  { id: "depth", label: "Depth", key: "D" },
  { id: "player", label: "Player", key: "P" },
]

function Hint({ children, k }: { children: React.ReactNode; k: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      {children}
      <Kbd>{k}</Kbd>
    </span>
  )
}

export function ModeBar({ studio }: { studio: Studio }) {
  const { pick, dispatchPick, layers, setLayers, layout, setLayout } = studio
  const single = studio.cams.length < 2
  const mode = single && pick.mode === "triangulate" ? "constraint" : pick.mode

  return (
    <div className="flex flex-col gap-2 rounded-lg border bg-card px-3 py-2">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Click</span>
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            spacing={0}
            value={mode}
            onValueChange={(v) => v && dispatchPick({ type: "mode", mode: v as typeof pick.mode })}
            aria-label="What a click does"
          >
            {single ? null : (
              <ToggleGroupItem value="triangulate" aria-label="Triangulate across views">
                <Hint k="T">Triangulate</Hint>
              </ToggleGroupItem>
            )}
            <ToggleGroupItem value="constraint" aria-label="Single-view ray constraint">
              Ray constraint
            </ToggleGroupItem>
            <ToggleGroupItem value="observation" aria-label="Soft observation">
              <Hint k="O">Observation</Hint>
            </ToggleGroupItem>
          </ToggleGroup>
        </div>

        {mode === "constraint" ? (
          <div className="flex flex-wrap items-center gap-2">
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              spacing={0}
              value={pick.constraint}
              onValueChange={(v) => v && dispatchPick({ type: "constraint", constraint: v as ConstraintMode })}
              aria-label="Ray constraint"
            >
              {CONSTRAINTS.map((c) => (
                <ToggleGroupItem key={c.id} value={c.id}>
                  <Hint k={c.key}>{c.label}</Hint>
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
            <ConstraintParams studio={studio} />
          </div>
        ) : null}

        <div className="ml-auto flex flex-wrap items-center gap-x-4 gap-y-2">
          {single ? null : (
            <>
              <Option id="snap" label="Snap to epipolar" checked={pick.snap} onChange={(snap) => dispatchPick({ type: "prefs", snap })} />
              <Option id="auto" label="Auto-commit" checked={pick.autoCommit} onChange={(autoCommit) => dispatchPick({ type: "prefs", autoCommit })} />
            </>
          )}
          <Option id="rays" label="Rays" checked={layers.rays} onChange={(rays) => setLayers({ ...layers, rays })} />
          <Option id="pipe" label="Pipeline" checked={layers.pipeline} onChange={(pipeline) => setLayers({ ...layers, pipeline })} />
          <Option id="res" label="Residuals" checked={layers.residuals} onChange={(residuals) => setLayers({ ...layers, residuals })} />
          {single ? null : (
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              spacing={0}
              value={layout}
              onValueChange={(v) => v && setLayout(v as typeof layout)}
              aria-label="Layout"
            >
              <ToggleGroupItem value="compare">Compare</ToggleGroupItem>
              <ToggleGroupItem value="focus">
                <Hint k="F">Focus</Hint>
              </ToggleGroupItem>
            </ToggleGroup>
          )}
        </div>
      </div>
      <PendingReadout studio={studio} />
    </div>
  )
}

function Option({ id, label, checked, onChange }: { id: string; label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex items-center gap-1.5">
      <Switch id={`opt-${id}`} size="sm" checked={checked} onCheckedChange={onChange} />
      <Label htmlFor={`opt-${id}`} className="text-xs font-normal">
        {label}
      </Label>
    </div>
  )
}

function ConstraintParams({ studio }: { studio: Studio }) {
  const { pick, dispatchPick, planes, scene } = studio
  if (pick.constraint === "height") {
    return (
      <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Input
          type="number"
          step="0.1"
          min={0}
          aria-label="Height in metres"
          className="h-7 w-20 font-mono text-xs"
          value={pick.params.heightM}
          onChange={(e) => dispatchPick({ type: "params", params: { heightM: Number.parseFloat(e.target.value) || 0 } })}
        />
        m above the pitch
      </span>
    )
  }
  if (pick.constraint === "plane") {
    return (
      <Select value={String(pick.params.planeIndex)} onValueChange={(v) => dispatchPick({ type: "params", params: { planeIndex: Number(v) } })}>
        <SelectTrigger size="sm" className="h-7 w-44 text-xs" aria-label="Goal-line plane">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {planes.map((p, i) => (
            <SelectItem key={p.id} value={String(i)}>
              {p.id.replace(/_/g, " ")} ({p.axis} = {p.value})
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    )
  }
  if (pick.constraint === "depth") {
    return (
      <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Input
          type="number"
          step="1"
          min={1}
          aria-label="Depth along the ray in metres"
          className="h-7 w-20 font-mono text-xs"
          value={pick.params.depthM}
          onChange={(e) => dispatchPick({ type: "params", params: { depthM: Math.max(1, Number.parseFloat(e.target.value) || 1) } })}
        />
        m along ray (drag the handle in 3-D)
      </span>
    )
  }
  if (pick.constraint === "player") {
    return (
      <span className="flex items-center gap-2">
        <Select
          value={pick.params.playerId ?? ""}
          onValueChange={(v) => dispatchPick({ type: "params", params: { playerId: v || null } })}
        >
          <SelectTrigger size="sm" className="h-7 w-28 text-xs" aria-label="Player">
            <SelectValue placeholder="Player" />
          </SelectTrigger>
          <SelectContent>
            {scene.players.map((p) => (
              <SelectItem key={p.player_id} value={p.player_id}>
                {p.player_id}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={pick.params.bone} onValueChange={(v) => dispatchPick({ type: "params", params: { bone: v } })}>
          <SelectTrigger size="sm" className="h-7 w-32 text-xs" aria-label="Bone">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {scene.bones.map((b) => (
              <SelectItem key={b} value={b}>
                {b}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </span>
    )
  }
  return null
}

/** What is pending right now and what Enter will do about it. */
function PendingReadout({ studio }: { studio: Studio }) {
  const { pick, live, liveBusy, cams, dispatchPick } = studio
  const ids = Object.keys(pick.picks)
  const kind = naturalCommit(pick)
  if (!ids.length) {
    return (
      <p className="text-xs text-muted-foreground">
        {cams.length > 1
          ? "Click the ball in a view to start a pending pick. Nothing is saved until you commit."
          : "One angle: depth comes from constraints, so residuals are not available. Click the ball, choose a constraint, press Enter."}
      </p>
    )
  }
  const worst = live?.max_residual_px ?? null
  const label = kind === "triangulated" ? "Triangulate" : kind === "constraint" ? "Commit key" : kind === "observation" ? "Add observation" : null
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs" aria-live="polite">
      <span className="font-medium">Pending</span>
      {ids.map((id) => {
        const i = cams.findIndex((c) => c.shotId === id)
        const uv = pick.picks[id]
        return (
          <span key={id} className="inline-flex items-center gap-1.5">
            <span className="inline-flex size-4 items-center justify-center rounded-sm font-mono text-[10px] font-semibold text-black" style={{ backgroundColor: viewColour(i) }}>
              {viewLetter(i)}
            </span>
            <span className="font-mono tabular-nums">
              {uv[0].toFixed(0)}, {uv[1].toFixed(0)}
            </span>
          </span>
        )
      })}
      {liveBusy ? <span className="text-muted-foreground">checking...</span> : null}
      {live?.xyz ? (
        <span className="font-mono tabular-nums">
          → {live.xyz[0].toFixed(2)}, {live.xyz[1].toFixed(2)}, <span className="font-medium">{live.xyz[2].toFixed(2)} m</span>
        </span>
      ) : null}
      {worst !== null ? (
        <ToneBadge tone={residualSeverity(worst)} className="font-mono">
          {worst.toFixed(1)} px
        </ToneBadge>
      ) : null}
      {live?.skew_gap_cm != null ? <span className="text-muted-foreground">gap {live.skew_gap_cm.toFixed(0)} cm</span> : null}
      {live?.ray_angle_deg != null ? <span className="text-muted-foreground">rays {live.ray_angle_deg.toFixed(0)}°</span> : null}
      {live && !live.ok && live.reason ? (
        <ToneBadge tone="destructive">{live.reason.replace(/_/g, " ")}</ToneBadge>
      ) : null}
      {worst !== null && worst > RESIDUAL_BAD_PX ? (
        <span className="text-warning">
          {worst > RESIDUAL_REJECT_PX ? "Over the 15 px limit. " : ""}Re-click the ball, or check the time sync in Prepare Shots.
        </span>
      ) : null}
      {live?.flags.map((f) => (
        <span key={f.code} className="text-warning">
          {f.message.charAt(0).toUpperCase() + f.message.slice(1)}
        </span>
      ))}
      {live?.observations_used?.some((o) => o.repeat) ? (
        <span className="text-warning">A picked frame repeats the previous image; pick on a fresh frame.</span>
      ) : null}
      <span className="ml-auto flex items-center gap-1.5">
        {label ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button size="xs" onClick={() => void studio.commit()}>
                <CheckIcon /> {label}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              Commit <Kbd>Enter</Kbd>
            </TooltipContent>
          </Tooltip>
        ) : (
          <span className="text-muted-foreground">Pick the other view, or choose a constraint.</span>
        )}
        <Button size="xs" variant="ghost" onClick={() => dispatchPick({ type: "clear" })}>
          <XIcon /> Discard <Kbd>Esc</Kbd>
        </Button>
      </span>
    </div>
  )
}
