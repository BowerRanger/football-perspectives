import * as React from "react"
import { VideoOffIcon } from "lucide-react"

import { videoUrl } from "./api"
import { drawOverlay } from "./overlay-draw"
import type { EditorController } from "./use-ball-anchor-editor"

/** Video + overlay canvas in a `bg-stage` well. Left-click places, right-click deletes. */
export function FrameCanvas({ ctrl }: { ctrl: EditorController }) {
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
  const { player, docApi, autoAnchors, layers, predictedByFrame, previewByFrame } = ctrl
  const size = player.videoSize
  const [failed, setFailed] = React.useState(false)
  React.useEffect(() => setFailed(false), [ctrl.shot])

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
    <div className={`relative overflow-hidden rounded-lg bg-stage ${failed ? "min-h-48" : ""}`}>
      <video
        ref={player.videoRef}
        src={videoUrl(ctrl.shot)}
        preload="metadata"
        muted
        playsInline
        className="block h-auto w-full"
        aria-label={`Shot ${ctrl.shot} video`}
        onError={() => setFailed(true)}
      />
      {failed ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-stage p-4 text-center text-stage-foreground">
          <VideoOffIcon className="size-6 opacity-70" aria-hidden />
          <p className="text-sm font-medium">Could not load the video for {ctrl.shot}</p>
          <p className="text-xs opacity-70">Run prepare_shots to (re)create shots/{ctrl.shot}.mp4, then reload.</p>
        </div>
      ) : null}
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
