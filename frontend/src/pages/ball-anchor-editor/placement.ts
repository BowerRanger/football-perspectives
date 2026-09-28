// Turns a click on the frame into a new manual anchor. Suggestion endpoints
// (joints-near, goal-element-suggest, pitch-fix-suggest) are best-effort:
// the operator's own choices always win over a suggestion.

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

async function placeTouch(shot: string, frame: number, xy: [number, number], a: Authoring): Promise<Placement> {
  const hits = await jointsNear(shot, frame, xy[0], xy[1])
  const forced = a.player !== AUTO
  const player = forced ? a.player : hits[0]?.player_id
  const bone = forced ? a.bone : hits[0]?.bone
  if (!player || !bone) {
    return { ok: false, message: "Touch needs a player and body part — no joint under the cursor. Pick both in the panel, or click closer to a player." }
  }
  const shotLike = a.touchType === "shot" || a.touchType === "volley"
  return {
    ok: true,
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
  if (element === AUTO) {
    element = (await goalElementSuggest(shot, frame, xy[0], xy[1]))[0] ?? AUTO
  }
  if (element === AUTO) {
    return { ok: false, message: "Goal impact needs an element — no goal under the cursor. Pick one manually." }
  }
  return {
    ok: true,
    message: a.goalElement === AUTO ? `Goal impact anchored: ${element} (auto-suggest stays on)` : undefined,
    anchor: makeAnchor(frame, xy, "goal_impact", { goal_element: element }),
  }
}

async function placePitchFix(shot: string, frame: number, xy: [number, number]): Promise<Placement> {
  const fixes = await pitchFixSuggest(shot, frame, xy[0], xy[1])
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
