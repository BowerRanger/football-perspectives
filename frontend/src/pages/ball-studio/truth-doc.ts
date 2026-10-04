// Pure operations over the truth document plus a bounded undo/redo history.
// Everything returns new objects; nothing mutates its input.
import type {
  EventKind,
  KeyConstraint,
  Observation,
  Outcome,
  SegmentKind,
  SegmentParams,
  SolvedKey,
  TruthDoc,
  TruthEvent,
  TruthKey,
  TruthSegment,
  TruthStatus,
} from "./types"

export const EMPTY_CONSTRAINT: KeyConstraint = {
  height_m: null,
  plane: null,
  depth_m: null,
  player_id: null,
  bone: null,
  offset: null,
}

export const DEFAULT_SEGMENT_PARAMS: SegmentParams = { drag: true, cd: null, magnus: "auto", player_id: null, bone: null }

export const HISTORY_LIMIT = 100

// ---- document operations --------------------------------------------------

export type KeyDraft = Omit<TruthKey, "id" | "note" | "residual_px" | "constraint"> &
  Partial<Pick<TruthKey, "note" | "residual_px" | "constraint">>

export function nextKeyId(doc: TruthDoc): string {
  let n = 0
  for (const k of doc.keys) {
    const m = /^k(\d+)$/.exec(k.id)
    if (m) n = Math.max(n, Number(m[1]))
  }
  return `k${n + 1}`
}

const byFrame = (a: { frame: number }, b: { frame: number }) => a.frame - b.frame

/**
 * Add a key (replacing any key already on that frame - two keys on one
 * frame are invalid). Segments that now straddle the new key are dropped so
 * the solver proposes fresh ones between consecutive keys.
 */
export function addKey(doc: TruthDoc, draft: KeyDraft): { doc: TruthDoc; id: string } {
  const existing = doc.keys.find((k) => k.frame === draft.frame)
  // New keys are named after their frame ("k449" is frame 449) when that id is free.
  const frameId = `k${draft.frame}`
  const id = existing?.id ?? (doc.keys.some((k) => k.id === frameId) || draft.frame < 0 ? nextKeyId(doc) : frameId)
  const key: TruthKey = {
    id,
    frame: draft.frame,
    xyz: draft.xyz,
    source: draft.source,
    constraint: { ...EMPTY_CONSTRAINT, ...(draft.constraint ?? {}) },
    observations: draft.observations,
    residual_px: draft.residual_px ?? {},
    note: draft.note ?? existing?.note ?? "",
  }
  const keys = [...doc.keys.filter((k) => k.id !== id), key].sort(byFrame)
  return { doc: { ...doc, keys, segments: pruneSegments(keys, doc.segments, existing ? null : id) }, id }
}

/** Drop segments that no longer join two consecutive keys (or whose ends vanished). */
function pruneSegments(keys: readonly TruthKey[], segments: readonly TruthSegment[], justAdded: string | null): TruthSegment[] {
  const order = new Map(keys.map((k, i) => [k.id, i]))
  return segments.filter((s) => {
    const a = order.get(s.from)
    const b = order.get(s.to)
    if (a === undefined || b === undefined) return false
    if (justAdded !== null && (s.from === justAdded || s.to === justAdded)) return false
    return b === a + 1
  })
}

export function removeKey(doc: TruthDoc, id: string): TruthDoc {
  const keys = doc.keys.filter((k) => k.id !== id)
  return { ...doc, keys, segments: pruneSegments(keys, doc.segments, null) }
}

export function updateKey(doc: TruthDoc, id: string, patch: Partial<Omit<TruthKey, "id">>): TruthDoc {
  const keys = doc.keys.map((k) => (k.id === id ? { ...k, ...patch } : k)).sort(byFrame)
  return { ...doc, keys, segments: pruneSegments(keys, doc.segments, null) }
}

/** Create or change the kind of the segment between two consecutive keys. */
export function setSegmentKind(doc: TruthDoc, from: string, to: string, kind: SegmentKind): TruthDoc {
  const has = doc.segments.some((s) => s.from === from && s.to === to)
  const segments: TruthSegment[] = has
    ? doc.segments.map((s) => (s.from === from && s.to === to ? { ...s, kind } : s))
    : [...doc.segments, { from, to, kind, params: DEFAULT_SEGMENT_PARAMS }]
  return { ...doc, segments }
}

export function setSegmentParams(doc: TruthDoc, from: string, to: string, params: Partial<SegmentParams>): TruthDoc {
  return {
    ...doc,
    segments: doc.segments.map((s) => (s.from === from && s.to === to ? { ...s, params: { ...s.params, ...params } } : s)),
  }
}

/** Back to the solver's auto-proposal. */
export function removeSegment(doc: TruthDoc, from: string, to: string): TruthDoc {
  return { ...doc, segments: doc.segments.filter((s) => !(s.from === from && s.to === to)) }
}

export function addObservation(doc: TruthDoc, ob: Observation): TruthDoc {
  const dup = doc.observations.some((o) => o.shot_id === ob.shot_id && o.shot_frame === ob.shot_frame)
  const observations = dup
    ? doc.observations.map((o) => (o.shot_id === ob.shot_id && o.shot_frame === ob.shot_frame ? ob : o))
    : [...doc.observations, ob]
  return { ...doc, observations }
}

export function removeObservation(doc: TruthDoc, index: number): TruthDoc {
  return { ...doc, observations: doc.observations.filter((_, i) => i !== index) }
}

export function addEvent(doc: TruthDoc, ev: Omit<TruthEvent, "note"> & { note?: string }): { doc: TruthDoc; index: number } {
  const event: TruthEvent = { note: "", ...ev }
  const events = [...doc.events, event].sort(byFrame)
  return { doc: { ...doc, events }, index: events.indexOf(event) }
}

export function updateEvent(doc: TruthDoc, index: number, patch: Partial<TruthEvent>): TruthDoc {
  return { ...doc, events: doc.events.map((e, i) => (i === index ? { ...e, ...patch } : e)).sort(byFrame) }
}

export function removeEvent(doc: TruthDoc, index: number): TruthDoc {
  return { ...doc, events: doc.events.filter((_, i) => i !== index) }
}

export const setOutcome = (doc: TruthDoc, outcome: Outcome): TruthDoc => ({ ...doc, outcome })
export const setNotes = (doc: TruthDoc, notes: string): TruthDoc => ({ ...doc, meta: { ...doc.meta, notes } })
export const setStatus = (doc: TruthDoc, status: TruthStatus): TruthDoc => ({ ...doc, meta: { ...doc.meta, status } })

export function eventKindLabel(kind: EventKind): string {
  return kind.replace(/_/g, " ")
}

/**
 * Adopt the server-resolved positions of non-manual keys (the contract: the
 * client "should adopt the resolved value ... for the next save").
 */
export function withResolvedKeys(doc: TruthDoc, solved: readonly SolvedKey[]): TruthDoc {
  const byId = new Map(solved.map((k) => [k.id, k]))
  return {
    ...doc,
    keys: doc.keys.map((k) => {
      const s = byId.get(k.id)
      if (!s || k.source === "manual") return k
      return { ...k, xyz: s.xyz, residual_px: { ...s.residual_px } }
    }),
  }
}

// ---- dirty tracking -------------------------------------------------------

/** Canonical JSON for dirty comparison (the server owns `meta.updated_at`). */
export function canonical(doc: TruthDoc): string {
  return JSON.stringify({ ...doc, meta: { ...doc.meta, updated_at: null } })
}

// ---- undo / redo history --------------------------------------------------

export interface HistoryState {
  present: TruthDoc
  past: readonly TruthDoc[]
  future: readonly TruthDoc[]
  saved: string
  /** `meta.updated_at` the file had when loaded or last saved (concurrency token). */
  token: string | null
}

export type HistoryAction =
  | { type: "edit"; doc: TruthDoc }
  | { type: "undo" }
  | { type: "redo" }
  | { type: "load"; doc: TruthDoc }
  | { type: "saved"; doc: TruthDoc; updatedAt: string }

export function initHistory(doc: TruthDoc): HistoryState {
  return { present: doc, past: [], future: [], saved: canonical(doc), token: doc.meta.updated_at }
}

export function historyReducer(state: HistoryState, action: HistoryAction): HistoryState {
  switch (action.type) {
    case "edit": {
      if (action.doc === state.present) return state
      const past = [...state.past, state.present].slice(-HISTORY_LIMIT)
      return { ...state, present: action.doc, past, future: [] }
    }
    case "undo": {
      if (!state.past.length) return state
      const present = state.past[state.past.length - 1]
      return { ...state, present, past: state.past.slice(0, -1), future: [state.present, ...state.future] }
    }
    case "redo": {
      if (!state.future.length) return state
      const [present, ...future] = state.future
      return { ...state, present, past: [...state.past, state.present], future }
    }
    case "load":
      return initHistory(action.doc)
    case "saved":
      // Undo history survives a save; the saved baseline moves to what was written.
      return { ...state, present: action.doc, saved: canonical(action.doc), token: action.updatedAt }
    default:
      return state
  }
}

export const isDirty = (s: HistoryState): boolean => canonical(s.present) !== s.saved
