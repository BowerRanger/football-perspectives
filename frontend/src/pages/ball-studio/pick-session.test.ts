import { describe, expect, it } from "vitest"

import { project, ShotCameras, type FrameCam } from "./camera-model"
import {
  computePreview,
  constraintPayload,
  initialPickState,
  naturalCommit,
  pickReducer,
  snapClick,
  type PickState,
} from "./pick-session"
import type { SceneShot, Vec3 } from "./types"

function lookAtRows(C: Vec3, target: Vec3): { K: number[]; R: number[]; t: number[] } {
  const f = [target[0] - C[0], target[1] - C[1], target[2] - C[2]]
  const fn = Math.hypot(...f)
  const z = f.map((v) => v / fn)
  let r = [z[1], -z[0], 0]
  const rn = Math.hypot(...r)
  r = r.map((v) => v / rn)
  const d = [z[1] * r[2] - z[2] * r[1], z[2] * r[0] - z[0] * r[2], z[0] * r[1] - z[1] * r[0]]
  const R = [...r, ...d, ...z]
  const t = [0, 1, 2].map((i) => -(R[i * 3] * C[0] + R[i * 3 + 1] * C[1] + R[i * 3 + 2] * C[2]))
  return { K: [1800, 1800, 960, 540], R, t }
}

function shot(id: string, offset: number, C: Vec3, target: Vec3): SceneShot {
  const rows = lookAtRows(C, target)
  const frames = Array.from({ length: 400 }, (_, i) => i)
  return {
    shot_id: id,
    frame_offset: offset,
    n_frames: 400,
    fps: 30,
    image_size: [1920, 1080],
    excluded: false,
    video_url: "",
    frame_url: "",
    distortion: [0, 0],
    camera_centre: C,
    frames,
    K: frames.map(() => rows.K),
    R: frames.map(() => rows.R),
    t: frames.map(() => rows.t),
    confidence: frames.map(() => 1),
  }
}

const cams = [
  new ShotCameras(shot("origi01", 0, [52.5, -30, 22], [40, 34, 0])),
  new ShotCameras(shot("origi02", -142, [-20, 34, 15], [20, 34, 0])),
]
const frame = 300
const X: Vec3 = [30, 31, 1.4]
const fcOf = (i: number): FrameCam => cams[i].atRef(frame)!
const uvA = project(fcOf(0), X)!.uv
const uvB = project(fcOf(1), X)!.uv
const noJoint = () => null

const pick = (s: PickState, shotId: string, uv: readonly [number, number], f = frame) =>
  pickReducer(s, { type: "pick", frame: f, shotId, uv })

describe("pick reducer", () => {
  it("accumulates picks at one instant and restarts on another frame", () => {
    let s = pick(initialPickState, "origi01", uvA)
    s = pick(s, "origi02", uvB)
    expect(Object.keys(s.picks)).toEqual(["origi01", "origi02"])
    s = pick(s, "origi01", uvA, frame + 1)
    expect(Object.keys(s.picks)).toEqual(["origi01"])
  })

  it("scrubbing drops pending picks", () => {
    const s = pickReducer(pick(initialPickState, "origi01", uvA), { type: "frame", frame: frame + 5 })
    expect(Object.keys(s.picks)).toHaveLength(0)
    expect(s.frame).toBeNull()
  })

  it("re-picking the same view replaces its pick; nudge moves it", () => {
    let s = pick(initialPickState, "origi01", [10, 10])
    s = pick(s, "origi01", [20, 20])
    s = pickReducer(s, { type: "nudge", shotId: "origi01", dx: 1, dy: -1 })
    expect(s.picks.origi01).toEqual([21, 19])
  })

  it("decides the natural commit", () => {
    expect(naturalCommit(initialPickState)).toBe("none")
    const one = pick(initialPickState, "origi01", uvA)
    expect(naturalCommit(one)).toBe("none")
    expect(naturalCommit(pickReducer(one, { type: "constraint", constraint: "ground" }))).toBe("constraint")
    expect(naturalCommit(pickReducer(one, { type: "mode", mode: "observation" }))).toBe("observation")
    expect(naturalCommit(pick(one, "origi02", uvB))).toBe("triangulated")
  })

  it("builds constraint payloads from the active chip", () => {
    const planes = [{ id: "goal_line_near", axis: "x" as const, value: 0 }]
    expect(constraintPayload(initialPickState, planes)).toBeNull()
    let s = pickReducer(initialPickState, { type: "constraint", constraint: "height" })
    s = pickReducer(s, { type: "params", params: { heightM: 2.2 } })
    expect(constraintPayload(s, planes)).toEqual({ mode: "height", height_m: 2.2 })
    s = pickReducer(s, { type: "constraint", constraint: "plane" })
    expect(constraintPayload(s, planes)).toEqual({ mode: "plane", plane: { axis: "x", value: 0 } })
    s = pickReducer(s, { type: "constraint", constraint: "player" })
    expect(constraintPayload(s, planes)).toBeNull()
  })
})

describe("preview geometry", () => {
  it("two picks triangulate to the true point", () => {
    let s = pick(initialPickState, "origi01", uvA)
    s = pick(s, "origi02", uvB)
    const pv = computePreview({ frame, picks: s.picks, cams, state: s, planes: [], joint: noJoint })
    expect(pv.ghost![0]).toBeCloseTo(X[0], 3)
    expect(pv.ghost![2]).toBeCloseTo(X[2], 3)
    expect(pv.skew!.gap_m).toBeLessThan(1e-3)
    expect(pv.ghostSource).toBe("triangulated")
  })

  it("one pick draws an epipolar line in the other view, and a constraint gives a ghost", () => {
    let s = pick(initialPickState, "origi01", uvA)
    s = pickReducer(s, { type: "constraint", constraint: "height" })
    s = pickReducer(s, { type: "params", params: { heightM: X[2] } })
    const pv = computePreview({ frame, picks: s.picks, cams, state: s, planes: [], joint: noJoint })
    expect(pv.epipolar.origi02).toHaveLength(1)
    expect(pv.epipolar.origi01).toBeUndefined()
    expect(pv.ghost![0]).toBeCloseTo(X[0], 3)
    expect(pv.ghost![1]).toBeCloseTo(X[1], 3)
  })

  it("snaps a near click onto the epipolar line and leaves a far click alone", () => {
    const s = pick(initialPickState, "origi01", uvA)
    const pv = computePreview({ frame, picks: s.picks, cams, state: s, planes: [], joint: noJoint })
    const lines = pv.epipolar.origi02
    const off: [number, number] = [uvB[0], uvB[1] + 4]
    const hit = snapClick(off, lines, true, 8, false)
    expect(hit.snapped).toBe(true)
    expect(hit.dist).toBeGreaterThan(0)
    expect(snapClick(off, lines, true, 8, true).snapped).toBe(false)
    expect(snapClick(off, lines, false, 8, false).snapped).toBe(false)
    expect(snapClick([uvB[0], uvB[1] + 300], lines, true, 8, false).snapped).toBe(false)
  })
})
