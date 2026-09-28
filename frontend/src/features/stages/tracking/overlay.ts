import { IGNORE_NAME, TEAM_COLORS, type FrameBox } from "./types"

const SELECT_COLOR = "#f97316" // orange: selected / focused row
const NAMED_COLOR = "#22c55e"
const IGNORED_COLOR = "#64748b"

/**
 * Draw bboxes for one frame. Named tracks are green with the player name,
 * unnamed use the team colour with the raw track id, ignored are slate, and
 * selected/focused tracks override to orange with a thicker stroke.
 */
export function drawTrackOverlay(
  canvas: HTMLCanvasElement,
  boxes: readonly FrameBox[],
  selectedTrackIds: ReadonlySet<string>,
  nameByTrack: ReadonlyMap<string, string>,
): void {
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.font = "bold 12px ui-sans-serif, system-ui, sans-serif"
  for (const b of boxes) {
    const [x1, y1, x2, y2] = b.bbox
    const name = nameByTrack.get(b.track_id) ?? b.player_name ?? ""
    const named = name !== "" && name !== IGNORE_NAME
    const ignored = name === IGNORE_NAME
    const selected = selectedTrackIds.has(b.track_id)
    const base = ignored ? IGNORED_COLOR : named ? NAMED_COLOR : (TEAM_COLORS[b.team ?? ""] ?? TEAM_COLORS.unknown)
    const color = selected ? SELECT_COLOR : base
    const tag = named ? name : ignored ? IGNORE_NAME : b.track_id
    ctx.lineWidth = selected ? 3 : 2
    ctx.strokeStyle = color
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
    const tw = ctx.measureText(tag).width + 6
    const labelY = y1 - 4 < 14 ? y1 + 16 : y1 - 4
    ctx.fillStyle = "rgba(0,0,0,0.65)"
    ctx.fillRect(x1, labelY - 12, tw, 14)
    ctx.fillStyle = color
    ctx.fillText(tag, x1 + 3, labelY)
  }
}

export function hitTestBoxes(boxes: readonly FrameBox[], x: number, y: number): FrameBox | null {
  for (const b of boxes) {
    const [x1, y1, x2, y2] = b.bbox
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) return b
  }
  return null
}
