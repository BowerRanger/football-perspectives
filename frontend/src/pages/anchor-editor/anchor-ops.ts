// Pure, immutable operations over the frame -> anchor map.
import { EMPTY_ANCHOR } from "./types"
import type { AnchorFrame, AnchorMap, AnchorsResponse, LineObs, PointObs } from "./types"

function withFrame(map: AnchorMap, frame: number, next: AnchorFrame): AnchorMap {
  const out = new Map(map)
  out.set(frame, next)
  return out
}

export function addEmptyAnchor(map: AnchorMap, frame: number): AnchorMap {
  return map.has(frame) ? map : withFrame(map, frame, EMPTY_ANCHOR)
}

export function removeAnchor(map: AnchorMap, frame: number): AnchorMap {
  if (!map.has(frame)) return map
  const out = new Map(map)
  out.delete(frame)
  return out
}

export function upsertPoint(map: AnchorMap, frame: number, obs: PointObs): AnchorMap {
  const cur = map.get(frame) ?? EMPTY_ANCHOR
  const idx = cur.points.findIndex((p) => p.name === obs.name)
  const points = idx >= 0 ? cur.points.map((p, i) => (i === idx ? obs : p)) : [...cur.points, obs]
  return withFrame(map, frame, { ...cur, points })
}

/** Vanishing-direction lines may repeat under one name; segment lines replace by name. */
export function upsertLine(map: AnchorMap, frame: number, obs: LineObs): AnchorMap {
  const cur = map.get(frame) ?? EMPTY_ANCHOR
  const idx = obs.world_direction !== null ? -1 : cur.lines.findIndex((l) => l.name === obs.name)
  const lines = idx >= 0 ? cur.lines.map((l, i) => (i === idx ? obs : l)) : [...cur.lines, obs]
  return withFrame(map, frame, { ...cur, lines })
}

export function removePoint(map: AnchorMap, frame: number, name: string): AnchorMap {
  const cur = map.get(frame)
  if (!cur) return map
  return withFrame(map, frame, { ...cur, points: cur.points.filter((p) => p.name !== name) })
}

export function removeLine(map: AnchorMap, frame: number, index: number): AnchorMap {
  const cur = map.get(frame)
  if (!cur) return map
  return withFrame(map, frame, { ...cur, lines: cur.lines.filter((_, i) => i !== index) })
}

export function anchorFromResponse(data: AnchorsResponse): AnchorMap {
  const out = new Map<number, AnchorFrame>()
  for (const a of data.anchors ?? []) {
    out.set(a.frame, { points: a.landmarks ?? [], lines: a.lines ?? [] })
  }
  return out
}

/** Payload shape for POST /anchors/{shot}. */
export function serialiseAnchors(map: AnchorMap) {
  return [...map.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([frame, a]) => ({
      frame,
      landmarks: a.points.map((p) => ({ name: p.name, image_xy: p.image_xy, world_xyz: p.world_xyz })),
      lines: a.lines.map((l) => ({
        name: l.name,
        image_segment: l.image_segment,
        world_segment: l.world_segment ?? null,
        world_direction: l.world_direction ?? null,
      })),
    }))
}

export function anchorSummary(a: AnchorFrame): string {
  const np = a.points.length
  const nl = a.lines.length
  const pts = `${np} pt${np === 1 ? "" : "s"}`
  return nl ? `${pts} + ${nl} line${nl === 1 ? "" : "s"}` : pts
}

export function fmtCoord(v: readonly number[]): string {
  return v.map((n) => n.toFixed(1)).join(", ")
}
