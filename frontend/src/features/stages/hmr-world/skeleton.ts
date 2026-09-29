// COCO-17 skeleton overlay used by the kp2d viewer.

export const COCO_SKELETON: ReadonlyArray<readonly [number, number]> = [
  [5, 7], [7, 9], [6, 8], [8, 10], // arms
  [11, 13], [13, 15], [12, 14], [14, 16], // legs
  [5, 6], [11, 12], [5, 11], [6, 12], // torso
  [0, 1], [0, 2], [1, 3], [2, 4], // head
]

const KP_CONF_MIN = 0.3

export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  kps: number[][] | null | undefined,
  colour: string,
  label?: string,
): void {
  if (!kps || !kps.length) return
  ctx.lineWidth = 2
  ctx.strokeStyle = colour
  for (const [a, b] of COCO_SKELETON) {
    if (a >= kps.length || b >= kps.length) continue
    const ka = kps[a]
    const kb = kps[b]
    if (!ka || !kb) continue
    if ((ka[2] || 0) < KP_CONF_MIN || (kb[2] || 0) < KP_CONF_MIN) continue
    ctx.beginPath()
    ctx.moveTo(ka[0], ka[1])
    ctx.lineTo(kb[0], kb[1])
    ctx.stroke()
  }
  ctx.fillStyle = colour
  for (const k of kps) {
    if (!k || (k[2] || 0) < KP_CONF_MIN) continue
    ctx.beginPath()
    ctx.arc(k[0], k[1], 3, 0, Math.PI * 2)
    ctx.fill()
  }
  // Name tag above the nose so overlapping skeletons stay legible.
  const nose = kps[0]
  if (label && nose && (nose[2] || 0) >= KP_CONF_MIN) {
    ctx.font = "bold 11px sans-serif"
    const tw = ctx.measureText(label).width + 6
    ctx.fillStyle = "rgba(0,0,0,0.7)"
    ctx.fillRect(nose[0] - tw / 2, nose[1] - 22, tw, 14)
    ctx.fillStyle = colour
    ctx.fillText(label, nose[0] - tw / 2 + 3, nose[1] - 11)
  }
}
