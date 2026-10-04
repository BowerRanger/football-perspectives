// Replay-speed maths and badge-state derivation for the group sync editor.
//
// Time map (matches src/schemas/sync_map.py): ref_frame = rate * shot_frame - frame_offset.
// At rate 1 this is the editor's existing "frame_offset = active - reference".

import type { ReplaySyncMember, Shot, SyncAlignment } from "./types"

/** Within this of 1.0 a replay counts as real time (config replay_sync.retime_tolerance). */
export const REAL_TIME_TOLERANCE = 0.08
/** Confidence below which an estimate is shown as "check". */
export const LOW_CONFIDENCE = 0.5
/** Mirror of the backend's ramp test (replay_speed.py). */
const RAMP_FACTOR = 1.2
const RAMP_MIN_RESIDUAL = 1.5

export interface MomentPair {
  /** Frame in the reference (live) clip. */
  reference_frame: number
  /** Frame in the replay clip. */
  shot_frame: number
}

export interface MomentFit {
  rate: number
  /** Reference frame at replay frame 0 (ref = offset + rate * shot). */
  offset: number
  /** Integer offset as stored in the sync map: round(-offset). */
  frameOffset: number
  /** Worst |residual| in reference frames. */
  residualFrames: number
  intervalRates: number[]
  ramp: boolean
  n: number
}

export type MomentIssue = "order" | "duplicate-shot" | "rate"

export function sortPairs(pairs: MomentPair[]): MomentPair[] {
  return [...pairs].sort((a, b) => a.shot_frame - b.shot_frame || a.reference_frame - b.reference_frame)
}

/** Indices (in sorted order) of pairs that run backwards or repeat a frame. */
export function pairIssues(sorted: MomentPair[]): Map<number, MomentIssue> {
  const out = new Map<number, MomentIssue>()
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].shot_frame <= sorted[i - 1].shot_frame) out.set(i, "duplicate-shot")
    else if (sorted[i].reference_frame <= sorted[i - 1].reference_frame) out.set(i, "order")
  }
  return out
}

/**
 * Least-squares rate + offset from marked moments (mirrors rate_from_moments).
 * Returns null for fewer than two pairs or pairs that don't run forwards in both clips.
 */
export function fitMoments(pairs: MomentPair[]): MomentFit | null {
  const pts = sortPairs(pairs)
  if (pts.length < 2 || pairIssues(pts).size > 0) return null
  const n = pts.length
  const mx = pts.reduce((s, p) => s + p.shot_frame, 0) / n
  const my = pts.reduce((s, p) => s + p.reference_frame, 0) / n
  let sxx = 0
  let sxy = 0
  for (const p of pts) {
    sxx += (p.shot_frame - mx) ** 2
    sxy += (p.shot_frame - mx) * (p.reference_frame - my)
  }
  if (sxx === 0) return null
  const rate = sxy / sxx
  const offset = my - rate * mx
  const residualFrames = Math.max(...pts.map((p) => Math.abs(p.reference_frame - (offset + rate * p.shot_frame))))
  const intervalRates = pts.slice(1).map((p, i) => (p.reference_frame - pts[i].reference_frame) / (p.shot_frame - pts[i].shot_frame))
  const ramp =
    intervalRates.length >= 2 &&
    Math.min(...intervalRates) > 0 &&
    Math.max(...intervalRates) / Math.min(...intervalRates) > RAMP_FACTOR &&
    residualFrames > RAMP_MIN_RESIDUAL
  return { rate, offset, frameOffset: Math.round(-offset), residualFrames, intervalRates, ramp, n }
}

const MIN_SAVE_RATE = 0.02
const MAX_SAVE_RATE = 4

/** Why a set of pairs can't be saved, or "" when it can. */
export function momentsProblem(pairs: MomentPair[], fit: MomentFit | null): string {
  if (pairs.length < 2) return "Add at least two pairs to save."
  if (pairIssues(pairs).size > 0) return "Pairs must run forwards in both clips."
  if (!fit) return "These pairs don't give a rate."
  if (!(fit.rate >= MIN_SAVE_RATE && fit.rate <= MAX_SAVE_RATE)) {
    return `A rate of ${fit.rate.toFixed(2)}× is outside ${MIN_SAVE_RATE} to ${MAX_SAVE_RATE}×.`
  }
  return ""
}

/** Reference frame shown at `shotFrame`: rate * shot - frame_offset. */
export function refFrameForShot(shotFrame: number, rate: number, frameOffset: number): number {
  return rate * shotFrame - frameOffset
}

/** Shot frame that matches `refFrame`: (ref + frame_offset) / rate. */
export function shotFrameForRef(refFrame: number, rate: number, frameOffset: number): number {
  return (refFrame + frameOffset) / (rate > 0 ? rate : 1)
}

/** Severity of a moment-fit residual (reference frames), shared with Ball Studio's bands. */
export function residualTone(frames: number): "success" | "warning" | "destructive" {
  return frames <= 3 ? "success" : frames <= 8 ? "warning" : "destructive"
}

export function isRealTime(rate: number): boolean {
  return Math.abs(rate - 1) <= REAL_TIME_TOLERANCE
}

export function fmtRate(rate: number): string {
  return `${rate.toFixed(2)}×`
}

// ---------------------------------------------------------------- badge state

export type SpeedKind =
  | "reference"
  | "detecting"
  | "unmeasured"
  | "real-time"
  | "slow"
  | "fast"
  | "ramp"
  | "no-camera"
  | "low-confidence"
  | "manual"
  | "retimed"

export type SpeedTone = "success" | "warning" | "info" | "muted"

export interface SpeedState {
  kind: SpeedKind
  tone: SpeedTone
  /** Badge text. */
  text: string
  /** One-sentence tooltip. */
  detail: string
  /** Rate the member plays at relative to live (1 when retimed or unknown). */
  rate: number
  /** Operator can retime this member to real time. */
  canRetime: boolean
  /** Why Retime is unavailable (shown as text), or "" when it is available / not relevant. */
  retimeBlocked: string
}

export interface SpeedInput {
  isReference: boolean
  shot: Pick<Shot, "retimed" | "speed_factor" | "native_frames"> | undefined
  alignment: Pick<SyncAlignment, "method" | "confidence" | "playback_rate"> | undefined
  /** `shots/replay_sync.json` entry for this member, when the stage ran. */
  member: ReplaySyncMember | undefined | null
  /** The replay_sync stage is running right now. */
  detecting: boolean
}

const pct = (c: number) => `${Math.round(c * 100)} %`
const SOURCE_AUTO = "matched on players"

function base(partial: Partial<SpeedState> & Pick<SpeedState, "kind" | "tone" | "text" | "detail">): SpeedState {
  return { rate: 1, canRetime: false, retimeBlocked: "", ...partial }
}

export function deriveSpeedState({ isReference, shot, alignment, member, detecting }: SpeedInput): SpeedState {
  if (isReference) {
    return base({ kind: "reference", tone: "info", text: "reference", detail: "The live clip every replay is aligned to." })
  }
  if (shot?.retimed) {
    const was = shot.speed_factor > 0 ? 1 / shot.speed_factor : 1
    return base({
      kind: "retimed",
      tone: "success",
      text: `retimed to real time (was ${fmtRate(was)})`,
      detail: "This clip was re-encoded to real time; the native clip is kept and can be restored.",
    })
  }
  const rate = alignment?.playback_rate && alignment.playback_rate > 0 ? alignment.playback_rate : 1
  const est = member?.estimate ?? null
  const decision = member?.decision

  if (alignment?.method === "manual") {
    if (!isRealTime(rate)) {
      return slowOrFast(rate, "set by you", `Set from moments you marked${decisionNote(est)}.`, "info", true, "")
    }
    if (est && decision === "kept_manual") {
      return base({
        kind: "manual",
        tone: "info",
        text: `real time · your offset kept · measured ${fmtRate(est.rate)}`,
        detail: "Your manual alignment was kept; the automatic estimate is shown for comparison only.",
        rate,
      })
    }
  }

  if (alignment?.method !== "manual") {
    if (decision === "ramp_not_applied" && est) {
      const lo = est.rate_first ?? est.rate
      const hi = est.rate_second ?? est.rate
      return base({
        kind: "ramp",
        tone: "warning",
        text: `speed ramp: ${lo.toFixed(2)}→${hi.toFixed(2)}×, not applied`,
        detail: "The replay changes speed part-way through. Mark matching moments to see where.",
      })
    }
    if (decision === "no_camera" || decision === "no_tracks") {
      return base({
        kind: "no-camera",
        tone: "warning",
        text: decision === "no_camera" ? "no camera: mark moments" : "no tracks: mark moments",
        detail: member?.reason || "Automatic speed needs tracks and a camera on both shots.",
      })
    }
    if (decision === "low_confidence" || (est && est.confidence < LOW_CONFIDENCE && alignment?.method !== "manual")) {
      const r = est?.rate ?? rate
      return base({
        kind: "low-confidence",
        tone: "warning",
        text: `${r.toFixed(2)}×? · ${SOURCE_AUTO} · ${pct(est?.confidence ?? 0)}, check`,
        detail: member?.reason || "The match is weak. Confirm it by marking moments.",
        rate,
        retimeBlocked: "Confidence is below 60 %: confirm by marking moments first.",
      })
    }
    if (alignment && alignment.method === "player_formation") {
      const conf = alignment.confidence
      if (conf < LOW_CONFIDENCE) {
        return base({
          kind: "low-confidence",
          tone: "warning",
          text: `${rate.toFixed(2)}×? · ${SOURCE_AUTO} · ${pct(conf)}, check`,
          detail: "The match is weak. Confirm it by marking moments.",
          rate,
          retimeBlocked: "Confidence is below 60 %: confirm by marking moments first.",
        })
      }
      if (isRealTime(rate)) {
        return base({
          kind: "real-time",
          tone: "success",
          text: rate === 1 ? "real time" : `real time (${rate.toFixed(2)}×)`,
          detail: `Matched on players, ${pct(conf)} confident.`,
          rate,
        })
      }
      return slowOrFast(rate, `${SOURCE_AUTO} · ${pct(conf)}`, `Matched on players, ${pct(conf)} confident.`, "info", conf >= 0.6, conf >= 0.6 ? "" : "Confidence is below 60 %: confirm by marking moments first.")
    }
  }

  if (detecting && !member) {
    return base({ kind: "detecting", tone: "muted", text: "Detecting speed…", detail: "The replay_sync stage is running." })
  }
  if (!isRealTime(rate)) {
    return slowOrFast(rate, alignment?.method === "manual" ? "set by you" : "set", "Playback rate saved in the sync map.", "info", true, "")
  }
  return base({
    kind: "unmeasured",
    tone: "muted",
    text: "speed not measured",
    detail: "Assumed real time. Run replay_sync, or mark matching moments to measure it.",
    rate,
  })
}

function decisionNote(est: ReplaySyncMember["estimate"] | null): string {
  return est ? `; the automatic estimate was ${fmtRate(est.rate)}` : ""
}

function slowOrFast(rate: number, source: string, detail: string, tone: SpeedTone, canRetime: boolean, blocked: string): SpeedState {
  if (rate > 1 + REAL_TIME_TOLERANCE) {
    return base({ kind: "fast", tone: "warning", text: `${fmtRate(rate)} faster than live · ${source}, check`, detail, rate })
  }
  return base({ kind: "slow", tone, text: `${fmtRate(rate)} slow motion · ${source}`, detail, rate, canRetime, retimeBlocked: blocked })
}
