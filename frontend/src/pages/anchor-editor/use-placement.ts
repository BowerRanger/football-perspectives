import * as React from "react"

import { postJson } from "@/lib/api"
import { upsertLine, upsertPoint } from "./anchor-ops"
import type { AnchorMap, Landmark, PaletteMode, PitchLine, SnapResult, Vec2 } from "./types"

export interface Flash {
  text: string
  tone: "muted" | "warning" | "destructive"
}

interface PlacementDeps {
  shot: string
  frame: number
  snapEnabled: boolean
  disabled: boolean
  landmarks: readonly Landmark[]
  pitchLines: readonly PitchLine[]
  edit: (fn: (m: AnchorMap) => AnchorMap) => void
  flash: (f: Flash | null) => void
}

export interface Placement {
  mode: PaletteMode
  setMode: (mode: PaletteMode) => void
  selected: string | null
  pendingLineStart: Vec2 | null
  select: (name: string) => void
  clear: () => void
  place: (click: Vec2) => Promise<void>
  hint: string | null
}

async function snapClick(deps: PlacementDeps, frame: number, click: Vec2, mode: string): Promise<SnapResult> {
  const raw: SnapResult = { xy: click, snapped: false, mode_used: "off", confidence: 0 }
  if (!deps.snapEnabled) return raw
  try {
    return await postJson<SnapResult>("/api/anchor/snap", { shot_id: deps.shot, frame, click, mode })
  } catch {
    return raw
  }
}

/** Mode, armed landmark/line and the click -> observation logic. */
export function usePlacement(deps: PlacementDeps): Placement {
  const [mode, setModeState] = React.useState<PaletteMode>("points")
  const [selected, setSelected] = React.useState<string | null>(null)
  const [pending, setPending] = React.useState<Vec2 | null>(null)
  const busy = React.useRef(false)

  const clear = React.useCallback(() => {
    setSelected(null)
    setPending(null)
  }, [])

  const setMode = React.useCallback(
    (next: PaletteMode) => {
      setModeState(next)
      clear()
    },
    [clear],
  )

  const select = React.useCallback((name: string) => {
    setSelected(name)
    setPending(null)
  }, [])

  // A half-drawn line belongs to the frame it was started on.
  React.useEffect(() => {
    setPending(null)
  }, [deps.frame, deps.shot])

  React.useEffect(() => {
    clear()
  }, [deps.shot, clear])

  const placePoint = async (click: Vec2, frame: number) => {
    const lm = deps.landmarks.find((l) => l.name === selected)
    if (!lm) return
    const snap = await snapClick(deps, frame, click, "auto")
    if (snap.snapped) {
      deps.flash({ text: `Snapped to ${snap.mode_used} (confidence ${snap.confidence.toFixed(2)})`, tone: "muted" })
    }
    const obs = { name: lm.name, image_xy: snap.xy, world_xyz: lm.world_xyz }
    deps.edit((m) => upsertPoint(m, frame, obs))
    clear()
  }

  const placeLine = async (click: Vec2, frame: number) => {
    const ln = deps.pitchLines.find((l) => l.name === selected)
    if (!ln) return
    const snap = await snapClick(deps, frame, click, "line_endpoint")
    if (pending === null) {
      setPending(snap.xy)
      deps.flash({ text: `Now click the other end of "${ln.name}"`, tone: "warning" })
      return
    }
    const obs = {
      name: ln.name,
      image_segment: [pending, snap.xy] as const,
      world_segment: ln.world_segment ?? null,
      world_direction: ln.world_direction ?? null,
    }
    deps.edit((m) => upsertLine(m, frame, obs))
    clear()
  }

  const place = async (click: Vec2) => {
    if (deps.disabled || !selected || busy.current) return
    busy.current = true
    try {
      if (mode === "points") await placePoint(click, deps.frame)
      else await placeLine(click, deps.frame)
    } finally {
      busy.current = false
    }
  }

  let hint: string | null = null
  if (selected) {
    if (mode === "points") hint = `Click the frame to place "${selected}"`
    else hint = pending ? `Click the other end of "${selected}"` : `Click one end of "${selected}", then the other`
  }

  return { mode, setMode, selected, pendingLineStart: pending, select, clear, place, hint }
}
