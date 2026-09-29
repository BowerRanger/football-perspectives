// Turns a click on the frame into a new manual anchor. Suggestion endpoints
// (joints-near, goal-element-suggest, pitch-fix-suggest) are best-effort:
// the operator's own choices always win over a suggestion.

import { errorMessage } from "@/lib/api"
import { goalElementSuggest, jointsNear, pitchFixSuggest, type PitchFixSuggestion } from "./api"
import { AUTO } from "./tags"
import type { BallAnchor } from "./types"

export interface Authoring {
  /** AUTO or a player id. */
  player: string
  bone: string
  /** "none" | "shot" | "volley" */
  touchType: string
  spin: string
  confidence: number
  /** AUTO or a goal element id. */
  goalElement: string
}

export const DEFAULT_AUTHORING: Authoring = {
  player: AUTO,
  bone: "r_foot",
  touchType: "none",
  spin: "none",
  confidence: 1,
  goalElement: AUTO,
}

export type Placement =
  | { ok: true; anchor: BallAnchor; message?: string; touchSuggestion?: { player: string; bone: string }; pitchFixes?: PitchFixSuggestion[] }
  | { ok: false; message: string }

export function makeAnchor(frame: number, xy: [number, number] | null, state: string, extra: Partial<BallAnchor> = {}): BallAnchor {
  return {
    frame,
    image_xy: xy,
    state,
    player_id: null,
    bone: null,
    goal_element: null,
    touch_type: null,
    spin: null,
    confidence: 1,
    end_frame: null,
    landmark: null,
    ...extra,
  }
}

/** Run a suggestion lookup; a failure becomes a warning, never a silent "no suggestion". */
async function suggest<T>(fn: () => Promise<T>, fallback: T): Promise<{ value: T; failed: string | null }> {
  try {
    return { value: await fn(), failed: null }
  } catch (err) {
    return { value: fallback, failed: errorMessage(err) }
  }
}

async function placeTouch(shot: string, frame: number, xy: [number, number], a: Authoring): Promise<Placement> {
  const { value: hits, failed } = await suggest(() => jointsNear(shot, frame, xy[0], xy[1]), [])
  const forced = a.player !== AUTO
  const player = forced ? a.player : hits[0]?.player_id
  const bone = forced ? a.bone : hits[0]?.bone
  if (!player || !bone) {
    if (failed) {
      return { ok: false, message: `Joint suggestions failed (${failed}). Pick a player and body part in the panel, then click again.` }
    }
    return { ok: false, message: "Touch needs a player and body part — no joint under the cursor. Pick both in the panel, or click closer to a player." }
  }
  const shotLike = a.touchType === "shot" || a.touchType === "volley"
  return {
    ok: true,
    message: failed ? `Joint suggestions failed (${failed}); used your player and body part.` : undefined,
    touchSuggestion: hits[0] ? { player: hits[0].player_id, bone: hits[0].bone } : undefined,
    anchor: makeAnchor(frame, xy, "player_touch", {
      player_id: player,
      bone,
      touch_type: a.touchType === "none" ? null : a.touchType,
      spin: shotLike && a.spin !== "none" ? a.spin : null,
      confidence: a.confidence,
    }),
  }
}

async function placeGoal(shot: string, frame: number, xy: [number, number], a: Authoring): Promise<Placement> {
  let element = a.goalElement
  let failed: string | null = null
  if (element === AUTO) {
    const res = await suggest(() => goalElementSuggest(shot, frame, xy[0], xy[1]), [] as string[])
    failed = res.failed
    element = res.value[0] ?? AUTO
  }
  if (element === AUTO) {
    if (failed) return { ok: false, message: `Goal element suggestions failed (${failed}). Pick an element manually.` }
    return { ok: false, message: "Goal impact needs an element — no goal under the cursor. Pick one manually." }
  }
  return {
    ok: true,
    message: a.goalElement === AUTO ? `Goal impact anchored: ${element} (auto-suggest stays on)` : undefined,
    anchor: makeAnchor(frame, xy, "goal_impact", { goal_element: element }),
  }
}

async function placePitchFix(shot: string, frame: number, xy: [number, number]): Promise<Placement> {
  const { value: fixes, failed } = await suggest(() => pitchFixSuggest(shot, frame, xy[0], xy[1]), [] as PitchFixSuggestion[])
  if (failed) return { ok: false, message: `Pitch feature suggestions failed (${failed}). Try again.` }
  if (!fixes.length) return { ok: false, message: "No pitch feature within range of that click." }
  return { ok: true, pitchFixes: fixes, anchor: makeAnchor(frame, xy, "grounded", { landmark: fixes[0].name }) }
}

export async function placeAnchor(
  tag: string,
  shot: string,
  frame: number,
  xy: [number, number],
  authoring: Authoring,
): Promise<Placement> {
  if (tag === "off_screen_flight") return { ok: false, message: "Off-screen flight has no pixel — use “Mark off-screen flight”." }
  if (tag === "player_touch") return placeTouch(shot, frame, xy, authoring)
  if (tag === "goal_impact") return placeGoal(shot, frame, xy, authoring)
  if (tag === "pitch_fix") return placePitchFix(shot, frame, xy)
  return { ok: true, anchor: makeAnchor(frame, xy, tag) }
}
