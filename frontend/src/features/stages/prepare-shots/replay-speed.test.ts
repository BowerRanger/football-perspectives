import { describe, expect, it } from "vitest"

import {
  deriveSpeedState,
  isCallToAction,
  fitMoments,
  pairIssues,
  refFrameForShot,
  residualTone,
  shotFrameForRef,
  sortPairs,
} from "./replay-speed"

// Hand-labelled ground truth, Liverpool g11 s042 (live) / s043 (replay), rate 0.341.
const HAND = [
  { reference_frame: 137, shot_frame: 28 },
  { reference_frame: 168, shot_frame: 120 },
  { reference_frame: 182, shot_frame: 160 },
]

describe("fitMoments", () => {
  it("two pairs fix rate and offset exactly", () => {
    const fit = fitMoments(HAND.slice(0, 2))!
    expect(fit.rate).toBeCloseTo(31 / 92, 6)
    expect(fit.residualFrames).toBeCloseTo(0, 9)
    expect(fit.ramp).toBe(false)
    expect(fit.n).toBe(2)
  })

  it("three consistent pairs give ~0.34 with small residual and no ramp", () => {
    const fit = fitMoments(HAND)!
    expect(fit.rate).toBeCloseTo(0.341, 1)
    expect(fit.residualFrames).toBeLessThan(3)
    expect(fit.ramp).toBe(false)
    expect(residualTone(fit.residualFrames)).toBe("success")
  })

  it("stores the integer offset as round(-offset)", () => {
    const fit = fitMoments([
      { reference_frame: 10, shot_frame: 0 },
      { reference_frame: 20, shot_frame: 10 },
    ])!
    expect(fit.rate).toBe(1)
    expect(fit.offset).toBeCloseTo(10)
    expect(fit.frameOffset).toBe(-10)
  })

  it("reveals a ramp from interval rates", () => {
    const fit = fitMoments([
      { reference_frame: 0, shot_frame: 0 },
      { reference_frame: 27, shot_frame: 100 },
      { reference_frame: 68, shot_frame: 200 },
      { reference_frame: 150, shot_frame: 400 },
    ])!
    expect(fit.intervalRates[0]).toBeCloseTo(0.27)
    expect(fit.intervalRates[1]).toBeCloseTo(0.41)
    expect(fit.ramp).toBe(true)
  })

  it("rejects one pair, repeated and backwards moments", () => {
    expect(fitMoments(HAND.slice(0, 1))).toBeNull()
    expect(fitMoments([HAND[0], { ...HAND[0], reference_frame: 140 }])).toBeNull()
    expect(fitMoments([HAND[0], { reference_frame: 100, shot_frame: 90 }])).toBeNull()
  })

  it("sorts by replay frame and flags inverted order", () => {
    const sorted = sortPairs([HAND[2], HAND[0]])
    expect(sorted[0].shot_frame).toBe(28)
    const bad = sortPairs([HAND[0], { reference_frame: 100, shot_frame: 90 }])
    expect(bad[0].shot_frame).toBe(28)
    expect(pairIssues([HAND[0], { reference_frame: 100, shot_frame: 40 }]).get(1)).toBe("order")
  })
})

describe("time map", () => {
  it("round-trips ref and shot frames", () => {
    for (const [rate, off] of [[1, -5], [0.341, -127], [1.24, 30]] as const) {
      for (const f of [0, 17, 250]) {
        expect(shotFrameForRef(refFrameForShot(f, rate, off), rate, off)).toBeCloseTo(f, 9)
      }
    }
  })

  it("is the plain offset at rate 1", () => {
    expect(refFrameForShot(100, 1, 12)).toBe(88)
  })
})

describe("deriveSpeedState", () => {
  const shot = { retimed: false, speed_factor: 1, native_frames: 0 }
  const member = (decision: string, est: object | null = null, reason = "") =>
    ({ shot_id: "s043", estimate: est, decision, reason }) as never

  it("labels a confident slow replay", () => {
    const s = deriveSpeedState({
      isReference: false,
      shot,
      alignment: { method: "player_formation", confidence: 0.81, playback_rate: 0.34 },
      member: null,
      detecting: false,
    })
    expect(s.kind).toBe("slow")
    expect(s.text).toBe("0.34× slow motion · matched on players · 81 %")
    expect(s.canRetime).toBe(true)
  })

  it("real time, ramp, no camera, low confidence", () => {
    const base = { isReference: false, shot, detecting: false }
    expect(
      deriveSpeedState({ ...base, alignment: { method: "player_formation", confidence: 0.9, playback_rate: 1 }, member: null }).text,
    ).toBe("real time")
    const ramp = deriveSpeedState({
      ...base,
      alignment: undefined,
      member: member("ramp_not_applied", { rate: 0.34, confidence: 0.7, rate_first: 0.27, rate_second: 0.41 }),
    })
    expect(ramp.kind).toBe("ramp")
    expect(ramp.text).toBe("speed ramp: 0.27→0.41×, not applied")
    expect(deriveSpeedState({ ...base, alignment: undefined, member: member("no_camera") }).text).toBe("no camera: mark moments")
    const low = deriveSpeedState({
      ...base,
      alignment: undefined,
      member: member("low_confidence", { rate: 0.34, confidence: 0.38 }),
    })
    expect(low.kind).toBe("low-confidence")
    expect(low.canRetime).toBe(false)
  })

  it("manual rate wins over an auto decision; retimed wins over everything", () => {
    const manual = deriveSpeedState({
      isReference: false,
      shot,
      alignment: { method: "manual", confidence: 1, playback_rate: 0.34 },
      member: member("kept_manual", { rate: 0.99, confidence: 0.8 }),
      detecting: false,
    })
    expect(manual.kind).toBe("slow")
    expect(manual.text).toContain("set by you")
    const retimed = deriveSpeedState({
      isReference: false,
      shot: { retimed: true, speed_factor: 1 / 0.34, native_frames: 187 },
      alignment: { method: "manual", confidence: 1, playback_rate: 1 },
      member: null,
      detecting: false,
    })
    expect(retimed.kind).toBe("retimed")
    expect(retimed.text).toBe("retimed to real time (was 0.34×)")
  })

  it("approximate slow rate is a call to action and says so", () => {
    const st = deriveSpeedState({
      isReference: false,
      shot,
      alignment: { method: "player_formation", confidence: 0.81, playback_rate: 0.34 },
      member: { shot_id: "s043", estimate: { rate: 0.34, confidence: 0.81, rate_uncertainty: 0.08 }, decision: "applied", reason: "", approximate: true } as never,
      detecting: false,
    })
    expect(st.kind).toBe("approximate")
    expect(st.text).toBe("≈0.34× slow motion · ±8 % · confirm with moments")
    expect(isCallToAction(st)).toBe(true)
  })

  it("unmeasured vs detecting", () => {
    const base = { isReference: false, shot, alignment: { method: "manual", confidence: 1, playback_rate: 1 }, member: null }
    expect(deriveSpeedState({ ...base, detecting: false }).kind).toBe("unmeasured")
    expect(deriveSpeedState({ ...base, detecting: true }).kind).toBe("detecting")
  })
})
