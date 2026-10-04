import { describe, expect, it } from "vitest"

import {
  addEvent,
  addKey,
  addObservation,
  canonical,
  historyReducer,
  initHistory,
  isDirty,
  nextKeyId,
  removeEvent,
  removeKey,
  removeSegment,
  setOutcome,
  setSegmentKind,
  updateKey,
  withResolvedKeys,
  HISTORY_LIMIT,
} from "./truth-doc"
import { EVENT_KINDS, EVENT_STYLE, KEY_SOURCES, KEY_SOURCE_STYLE, SEGMENT_KINDS, SEGMENT_STYLE, residualSeverity } from "./palette"
import type { TruthDoc } from "./types"

const base: TruthDoc = {
  version: 1,
  group_id: "origi01",
  reference_shot: "origi01",
  fps: 30,
  shots: [
    { shot_id: "origi01", frame_offset: 0 },
    { shot_id: "origi02", frame_offset: -142 },
  ],
  outcome: "unknown",
  keys: [],
  segments: [],
  observations: [],
  events: [],
  meta: { authored_by: "operator", updated_at: null, notes: "", status: "draft" },
}

const draft = (frame: number) => ({
  frame,
  xyz: [1, 2, 0.11] as const,
  source: "ray_ground" as const,
  observations: [{ shot_id: "origi01", shot_frame: frame, uv: [10, 20] as const }],
})

describe("keys", () => {
  it("assigns increasing ids and keeps keys sorted by frame", () => {
    let d = addKey(base, draft(50)).doc
    d = addKey(d, draft(20)).doc
    d = addKey(d, draft(80)).doc
    expect(d.keys.map((k) => k.frame)).toEqual([20, 50, 80])
    expect(d.keys.map((k) => k.id)).toEqual(["k20", "k50", "k80"])
    expect(nextKeyId(d)).toBe("k81")
  })

  it("falls back to max+1 when the frame-named id is taken by a key that moved", () => {
    let d = addKey(base, draft(10)).doc
    d = updateKey(d, "k10", { frame: 99 })
    const r = addKey(d, draft(10))
    expect(r.id).toBe("k11")
    expect(new Set(r.doc.keys.map((k) => k.id)).size).toBe(2)
  })

  it("replaces the key already on a frame instead of duplicating", () => {
    const first = addKey(base, draft(50))
    const second = addKey(first.doc, { ...draft(50), source: "manual" })
    expect(second.doc.keys).toHaveLength(1)
    expect(second.id).toBe(first.id)
    expect(second.doc.keys[0].source).toBe("manual")
  })

  it("does not mutate its input", () => {
    const frozen = Object.freeze({ ...base, keys: Object.freeze([]) as never })
    expect(() => addKey(frozen, draft(5))).not.toThrow()
    expect(base.keys).toHaveLength(0)
  })

  it("drops segments that straddle a newly inserted key", () => {
    let d = addKey(base, draft(10)).doc
    d = addKey(d, draft(40)).doc
    d = setSegmentKind(d, d.keys[0].id, d.keys[1].id, "flight")
    expect(d.segments).toHaveLength(1)
    d = addKey(d, draft(25)).doc
    expect(d.segments).toHaveLength(0)
  })

  it("removing a key removes its segments", () => {
    let d = addKey(base, draft(10)).doc
    d = addKey(d, draft(40)).doc
    d = setSegmentKind(d, d.keys[0].id, d.keys[1].id, "roll")
    d = removeKey(d, d.keys[1].id)
    expect(d.keys).toHaveLength(1)
    expect(d.segments).toHaveLength(0)
  })

  it("updateKey re-sorts when the frame changes", () => {
    let d = addKey(base, draft(10)).doc
    d = addKey(d, draft(40)).doc
    const id = d.keys[0].id
    d = updateKey(d, id, { frame: 60 })
    expect(d.keys.map((k) => k.frame)).toEqual([40, 60])
  })
})

describe("segments, observations, events", () => {
  it("sets and clears a segment kind", () => {
    let d = addKey(base, draft(10)).doc
    d = addKey(d, draft(40)).doc
    const [a, b] = d.keys
    d = setSegmentKind(d, a.id, b.id, "flight")
    d = setSegmentKind(d, a.id, b.id, "roll")
    expect(d.segments).toHaveLength(1)
    expect(d.segments[0].kind).toBe("roll")
    d = removeSegment(d, a.id, b.id)
    expect(d.segments).toHaveLength(0)
  })

  it("replaces a soft observation on the same shot frame", () => {
    let d = addObservation(base, { shot_id: "origi01", shot_frame: 5, uv: [1, 1] })
    d = addObservation(d, { shot_id: "origi01", shot_frame: 5, uv: [2, 2] })
    d = addObservation(d, { shot_id: "origi02", shot_frame: 5, uv: [3, 3] })
    expect(d.observations).toHaveLength(2)
    expect(d.observations[0].uv).toEqual([2, 2])
  })

  it("adds events sorted by frame and returns the new index", () => {
    let r = addEvent(base, { frame: 90, kind: "bounce", player_id: null, bone: null })
    r = addEvent(r.doc, { frame: 30, kind: "touch", player_id: "P023", bone: "r_foot" })
    expect(r.doc.events.map((e) => e.frame)).toEqual([30, 90])
    expect(r.index).toBe(0)
    expect(removeEvent(r.doc, 0).events).toHaveLength(1)
  })

  it("adopts server-resolved positions except for manual keys", () => {
    let d = addKey(base, draft(10)).doc
    d = addKey(d, { ...draft(20), source: "manual" }).doc
    const out = withResolvedKeys(d, [
      { id: d.keys[0].id, frame: 10, xyz: [9, 9, 9], source: "ray_ground", residual_px: { origi01: 1 }, status: "ok", messages: [] },
      { id: d.keys[1].id, frame: 20, xyz: [8, 8, 8], source: "manual", residual_px: {}, status: "ok", messages: [] },
    ])
    expect(out.keys[0].xyz).toEqual([9, 9, 9])
    expect(out.keys[1].xyz).toEqual([1, 2, 0.11])
  })
})

describe("history", () => {
  it("undo and redo walk the edit stack and dirty follows the saved baseline", () => {
    let s = initHistory(base)
    expect(isDirty(s)).toBe(false)
    s = historyReducer(s, { type: "edit", doc: setOutcome(s.present, "goal") })
    expect(isDirty(s)).toBe(true)
    s = historyReducer(s, { type: "undo" })
    expect(s.present.outcome).toBe("unknown")
    expect(isDirty(s)).toBe(false)
    s = historyReducer(s, { type: "redo" })
    expect(s.present.outcome).toBe("goal")
  })

  it("a new edit clears the redo stack", () => {
    let s = initHistory(base)
    s = historyReducer(s, { type: "edit", doc: setOutcome(s.present, "goal") })
    s = historyReducer(s, { type: "undo" })
    s = historyReducer(s, { type: "edit", doc: setOutcome(s.present, "no_goal") })
    expect(s.future).toHaveLength(0)
  })

  it("saving moves the baseline but keeps undo available", () => {
    let s = initHistory(base)
    s = historyReducer(s, { type: "edit", doc: setOutcome(s.present, "goal") })
    s = historyReducer(s, { type: "saved", doc: s.present, updatedAt: "2026-10-04T12:00:00Z" })
    expect(isDirty(s)).toBe(false)
    expect(s.token).toBe("2026-10-04T12:00:00Z")
    expect(s.past).toHaveLength(1)
    s = historyReducer(s, { type: "undo" })
    expect(isDirty(s)).toBe(true)
  })

  it("is bounded", () => {
    let s = initHistory(base)
    for (let i = 0; i < HISTORY_LIMIT + 20; i++) s = historyReducer(s, { type: "edit", doc: { ...s.present, fps: 30 + i + 1 } })
    expect(s.past.length).toBe(HISTORY_LIMIT)
  })

  it("canonical form ignores the server timestamp", () => {
    expect(canonical({ ...base, meta: { ...base.meta, updated_at: "x" } })).toBe(canonical(base))
  })
})

describe("palette", () => {
  it("defines every key source, segment kind and event kind", () => {
    for (const s of KEY_SOURCES) expect(KEY_SOURCE_STYLE[s]).toBeDefined()
    for (const k of SEGMENT_KINDS) expect(SEGMENT_STYLE[k].colour).toMatch(/^#/)
    for (const e of EVENT_KINDS) expect(EVENT_STYLE[e].glyph).toBeTruthy()
  })

  it("grades residuals on the 3 / 8 px bands", () => {
    expect(residualSeverity(2.9)).toBe("success")
    expect(residualSeverity(5)).toBe("warning")
    expect(residualSeverity(9)).toBe("destructive")
  })
})
