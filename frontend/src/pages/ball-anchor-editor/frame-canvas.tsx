import * as React from "react"

import { videoUrl } from "./api"
import { drawOverlay } from "./overlay-draw"
import type { EditorController } from "./use-ball-anchor-editor"

/** Video + overlay canvas in a `bg-stage` well. Left-click places, right-click deletes. */
export function FrameCanvas({ ctrl }: { ctrl: EditorController }) {
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
  const { player, docApi, autoAnchors, layers, predictedByFrame, previewByFrame } = ctrl
  const size = player.videoSize

  React.useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !size) return
    if (canvas.width !== size[0]) canvas.width = size[0]
    if (canvas.height !== size[1]) canvas.height = size[1]
    drawOverlay(canvas, {
      width: size[0],
      height: size[1],
      frame: player.frame,
      anchors: docApi.doc.anchors,
      autoAnchors,
      layers,
      predictedByFrame,
      previewByFrame,
    })
  }, [size, player.frame, docApi.doc.anchors, autoAnchors, layers, predictedByFrame, previewByFrame])

  const toVideoPx = (e: React.MouseEvent<HTMLCanvasElement>): [number, number] | null => {
    const canvas = canvasRef.current
    if (!canvas || !canvas.width) return null
    const rect = canvas.getBoundingClientRect()
    return [
      (e.clientX - rect.left) * (canvas.width / rect.width),
      (e.clientY - rect.top) * (canvas.height / rect.height),
    ]
  }

  return (
    <div className="relative overflow-hidden rounded-lg bg-stage">
      <video
        ref={player.videoRef}
        src={videoUrl(ctrl.shot)}
        preload="metadata"
        muted
        playsInline
        className="block h-auto w-full"
        aria-label={`Shot ${ctrl.shot} video`}
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 size-full cursor-crosshair"
        aria-label="Frame overlay — click to place an anchor, right-click to delete"
        onClick={(e) => {
          const p = toVideoPx(e)
          if (p) void ctrl.placeAt(p[0], p[1])
        }}
        onContextMenu={(e) => {
          e.preventDefault()
          const p = toVideoPx(e)
          if (p) ctrl.removeAt(p[0], p[1])
        }}
      />
    </div>
  )
}
