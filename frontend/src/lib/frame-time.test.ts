import { describe, expect, it } from "vitest"

import { frameAtTime, frameTime } from "./frame-time"

describe("frame-time", () => {
  it("round-trips every frame at common rates", () => {
    for (const fps of [24, 25, 29.97, 30, 60]) {
      for (let f = 0; f < 600; f++) expect(frameAtTime(frameTime(f, fps), fps)).toBe(f)
    }
  })

  it("reads the frame the playhead is inside, not the nearest boundary", () => {
    // 0.9 of the way through frame 10 is still frame 10 (Math.round would say 11).
    expect(frameAtTime((10 + 0.9) / 30, 30)).toBe(10)
    expect(frameAtTime(10 / 30, 30)).toBe(10)
  })

  it("clamps to the valid range", () => {
    expect(frameAtTime(-1, 30)).toBe(0)
    expect(frameAtTime(100, 30, 205)).toBe(205)
  })
})
