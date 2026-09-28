import * as React from "react"

import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { playerLabelWithId } from "@/lib/format"
import { AUTO, GOAL_ELEMENTS, SPIN_OPTIONS, TOUCH_BONES, TOUCH_TYPES } from "./tags"
import type { EditorController } from "./use-ball-anchor-editor"

function LabelledSelect(props: {
  id: string
  label: string
  value: string
  onChange: (v: string) => void
  options: { id: string; label: string }[]
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label htmlFor={props.id} className="text-xs">
        {props.label}
      </Label>
      <Select value={props.value} onValueChange={props.onChange}>
        <SelectTrigger id={props.id} size="sm" className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {props.options.map((o) => (
            <SelectItem key={o.id} value={o.id}>
              {o.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

function Help({ children }: { children: React.ReactNode }) {
  return <p className="text-xs leading-relaxed text-muted-foreground">{children}</p>
}

function TouchAuthoring({ ctrl }: { ctrl: EditorController }) {
  const { authoring: a, patchAuthoring, players, touchSuggestion } = ctrl
  const playerOptions = [
    { id: AUTO, label: "Auto (nearest joint)" },
    ...players.map((p) => ({ id: p.player_id, label: playerLabelWithId(p) })),
  ]
  const shotLike = a.touchType === "shot" || a.touchType === "volley"
  return (
    <div className="flex flex-col gap-3">
      {players.length ? (
        <LabelledSelect id="touch-player" label="Player" value={a.player} onChange={(v) => patchAuthoring({ player: v })} options={playerOptions} />
      ) : (
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="touch-player-input" className="text-xs">Player</Label>
          <Input
            id="touch-player-input"
            placeholder="auto / P001"
            value={a.player === AUTO ? "" : a.player}
            onChange={(e) => patchAuthoring({ player: e.target.value.trim() || AUTO })}
          />
        </div>
      )}
      <LabelledSelect
        id="touch-bone"
        label="Body part"
        value={a.bone}
        onChange={(v) => patchAuthoring({ bone: v })}
        options={TOUCH_BONES}
      />
      <LabelledSelect id="touch-type" label="Type" value={a.touchType} onChange={(v) => patchAuthoring({ touchType: v })} options={TOUCH_TYPES} />
      {shotLike ? <LabelledSelect id="touch-spin" label="Spin" value={a.spin} onChange={(v) => patchAuthoring({ spin: v })} options={SPIN_OPTIONS} /> : null}
      <div className="flex flex-col gap-2">
        <Label htmlFor="touch-confidence" className="justify-between text-xs">
          Confidence <span className="font-mono tabular-nums">{a.confidence.toFixed(2)}</span>
        </Label>
        <Slider id="touch-confidence" min={0} max={1} step={0.05} value={[a.confidence]} onValueChange={(v) => patchAuthoring({ confidence: v[0] ?? 1 })} />
      </div>
      <Help>
        Click the ball — the nearest reconstructed joint is auto-filled. With Player on Auto, the suggested player and body
        part are used.
      </Help>
      {touchSuggestion ? (
        <Help>
          Last suggestion: <span className="font-mono text-foreground">{touchSuggestion.player} / {touchSuggestion.bone}</span>
        </Help>
      ) : null}
    </div>
  )
}

function GoalAuthoring({ ctrl }: { ctrl: EditorController }) {
  const { authoring: a, patchAuthoring } = ctrl
  return (
    <div className="flex flex-col gap-3">
      <LabelledSelect
        id="goal-element"
        label="Element"
        value={a.goalElement}
        onChange={(v) => patchAuthoring({ goalElement: v })}
        options={[{ id: AUTO, label: "Auto (suggest from click)" }, ...GOAL_ELEMENTS]}
      />
      <Help>Click where the ball strikes the goal — the nearest element is picked by ray residual.</Help>
    </div>
  )
}

function PitchFixAuthoring({ ctrl }: { ctrl: EditorController }) {
  const { pitchFixes, docApi } = ctrl
  const anchor = pitchFixes ? docApi.doc.anchors.find((x) => x.frame === pitchFixes.frame) : undefined
  return (
    <div className="flex flex-col gap-3">
      <Help>Click the ball — nearby pitch features are suggested (nearest first). The nearest is applied; pick another below to change it.</Help>
      {pitchFixes && anchor ? (
        <div className="flex flex-col gap-1.5">
          <Label className="text-xs">Feature at frame {pitchFixes.frame}</Label>
          <ToggleGroup
            type="single"
            orientation="vertical"
            spacing={1}
            className="w-full flex-col items-stretch"
            value={anchor.landmark ?? ""}
            onValueChange={(v) => v && docApi.setLandmark(pitchFixes.frame, v)}
            aria-label="Pitch feature"
          >
            {pitchFixes.items.map((s) => (
              <ToggleGroupItem key={s.name} value={s.name} className="h-auto justify-between px-2 py-1 text-xs">
                <span>{s.name}</span>
                <span className="font-mono tabular-nums text-muted-foreground">{s.distance_m.toFixed(2)} m</span>
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </div>
      ) : null}
    </div>
  )
}

const TITLES: Record<string, string> = {
  player_touch: "Touch authoring",
  goal_impact: "Goal impact",
  pitch_fix: "Pitch fix",
}

/** Per-tag authoring controls; renders nothing for plain tags. */
export function AuthoringPanel({ ctrl }: { ctrl: EditorController }) {
  const title = TITLES[ctrl.selectedTag]
  if (!title) return null
  return (
    <section className="flex flex-col gap-3 border-t pt-3" aria-label={title}>
      <h3 className="text-sm font-medium">{title}</h3>
      {ctrl.selectedTag === "player_touch" ? <TouchAuthoring ctrl={ctrl} /> : null}
      {ctrl.selectedTag === "goal_impact" ? <GoalAuthoring ctrl={ctrl} /> : null}
      {ctrl.selectedTag === "pitch_fix" ? <PitchFixAuthoring ctrl={ctrl} /> : null}
    </section>
  )
}
