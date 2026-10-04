// Frame <-> video-time convention shared by every frame-exact transport.
//
// A frame f of a constant-fps video occupies [f/fps, (f+1)/fps). Seeking to
// the frame START lands on the previous frame when the decoder rounds down,
// and Math.round(currentTime*fps) reads the NEXT frame from the second half of
// the interval - the one-frame-late anchor bug. So seek to the frame MIDDLE and
// read with floor.

/** Seek target (seconds) that lands inside frame `frame`. */
export function frameTime(frame: number, fps: number): number {
  return (frame + 0.5) / fps
}

/** Frame index under the playhead; `last` clamps to the final frame when known. */
export function frameAtTime(time: number, fps: number, last = Number.POSITIVE_INFINITY): number {
  const f = Math.floor(time * fps + 1e-6)
  return Math.max(0, Math.min(last, f))
}
