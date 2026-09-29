import { FlagIcon, VideoOffIcon } from "lucide-react"
import { toast } from "sonner"

import { FramePlayer } from "@/components/frame-player"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import type { EditorController } from "./use-ball-anchor-editor"

/** The shared FramePlayer plus the ball editor's chain / off-screen actions and layer toggles. */
export function Transport({ ctrl }: { ctrl: EditorController }) {
  const { player, docApi, selectedTag } = ctrl
  const chain = docApi.activeChain

  const onChain = () => {
    const res = docApi.toggleChain()
    if (res.ok) toast.message(res.message)
    else toast.warning(res.message)
  }

  return (
    <FramePlayer
      label="Seek frame"
      frame={player.frame}
      max={Math.max(0, player.totalFrames - 1)}
      fps={player.fps}
      frameInput
      playing={player.playing}
      onTogglePlay={player.toggle}
      onSeek={player.seekTo}
    >
      <div className="flex w-full flex-wrap items-center gap-2">
        {selectedTag === "off_screen_flight" ? (
          <Button size="sm" variant="secondary" onClick={() => docApi.markOffScreen(player.currentFrame())}>
            <VideoOffIcon /> Mark off-screen flight
          </Button>
        ) : null}
        <Button size="sm" variant={chain ? "default" : "outline"} onClick={onChain}>
          <FlagIcon /> {chain ? `End shot chain (${chain.length})` : "Start shot chain"}
        </Button>
        <LayerToggles ctrl={ctrl} />
      </div>
    </FramePlayer>
  )
}

function LayerToggles({ ctrl }: { ctrl: EditorController }) {
  const { layers, patchLayers, predictedByFrame, previewResult } = ctrl
  const items: { key: "anchors" | "predicted" | "preview"; label: string; show: boolean }[] = [
    { key: "anchors", label: "Anchors", show: true },
    { key: "predicted", label: "Predicted ball path", show: predictedByFrame.size > 0 },
    { key: "preview", label: "Solve preview ring", show: previewResult !== null },
  ]
  return (
    <div className="ml-auto flex flex-wrap items-center gap-3">
      {items
        .filter((i) => i.show)
        .map((i) => (
          <div key={i.key} className="flex items-center gap-1.5">
            <Checkbox
              id={`layer-${i.key}`}
              checked={layers[i.key]}
              onCheckedChange={(c) => patchLayers({ [i.key]: c === true })}
            />
            <Label htmlFor={`layer-${i.key}`} className="text-xs font-normal">
              {i.label}
            </Label>
          </div>
        ))}
    </div>
  )
}
