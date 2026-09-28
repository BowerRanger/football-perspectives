import * as React from "react"
import { ChevronLeftIcon, ChevronRightIcon, FlagIcon, PauseIcon, PlayIcon, VideoOffIcon } from "lucide-react"
import { toast } from "sonner"

import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Kbd } from "@/components/ui/kbd"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { EditorController } from "./use-ball-anchor-editor"

function IconTip({ label, keys, children }: { label: string; keys?: string; children: React.ReactElement }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent>
        {label} {keys ? <Kbd>{keys}</Kbd> : null}
      </TooltipContent>
    </Tooltip>
  )
}

export function Transport({ ctrl }: { ctrl: EditorController }) {
  const { player, docApi, selectedTag } = ctrl
  const chain = docApi.activeChain

  const onChain = () => {
    const res = docApi.toggleChain()
    if (res.ok) toast.message(res.message)
    else toast.warning(res.message)
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <IconTip label={player.playing ? "Pause" : "Play"} keys="Space">
        <Button size="icon-sm" variant="secondary" aria-label={player.playing ? "Pause" : "Play"} onClick={player.toggle}>
          {player.playing ? <PauseIcon /> : <PlayIcon />}
        </Button>
      </IconTip>
      <IconTip label="Previous frame" keys="←">
        <Button size="icon-sm" variant="outline" aria-label="Previous frame" onClick={() => player.step(-1)}>
          <ChevronLeftIcon />
        </Button>
      </IconTip>
      <IconTip label="Next frame" keys="→">
        <Button size="icon-sm" variant="outline" aria-label="Next frame" onClick={() => player.step(1)}>
          <ChevronRightIcon />
        </Button>
      </IconTip>
      <Slider
        className="min-w-40 flex-1"
        aria-label="Seek frame"
        min={0}
        max={Math.max(1, player.totalFrames)}
        step={1}
        value={[Math.min(player.frame, Math.max(1, player.totalFrames))]}
        onValueChange={(v) => player.seekTo(v[0] ?? 0)}
      />
      <span className="min-w-24 text-right font-mono text-xs tabular-nums text-muted-foreground" aria-live="off">
        Frame {player.frame}
      </span>
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
    </div>
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
