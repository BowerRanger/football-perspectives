import * as React from "react"
import { InfoIcon } from "lucide-react"

import { FramePlayer } from "@/components/frame-player"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Inspector } from "./inspector"
import { Studio3D, StudioView } from "./studio-views"
import { Timeline } from "./timeline"
import type { Studio } from "./use-studio"

/** Read-only review for phones: scrub, compare angles, check residuals. No commit controls. */
export function MobileReview({ studio, onHelp }: { studio: Studio; onHelp: () => void }) {
  const [well, setWell] = React.useState<string>("0")
  return (
    <div className="flex flex-col gap-3 p-4">
      <Alert>
        <InfoIcon />
        <AlertTitle>Read-only on this screen</AlertTitle>
        <AlertDescription>Authoring needs a desktop screen. Here you can scrub, compare angles and check residuals.</AlertDescription>
      </Alert>
      <div className="sticky top-[60px] z-20 -mx-4 border-b bg-background/95 px-4 py-2 backdrop-blur">
        <FramePlayer
          frame={studio.frame}
          min={studio.range[0]}
          max={studio.range[1]}
          fps={studio.scene.fps}
          playing={studio.videos.playing}
          onTogglePlay={studio.videos.togglePlay}
          onSeek={studio.setFrame}
        />
      </div>
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        spacing={0}
        value={well}
        onValueChange={(v) => v && setWell(v)}
        aria-label="Which view to show"
        className="w-full [&>*]:flex-1"
      >
        {studio.cams.map((c, i) => (
          <ToggleGroupItem key={c.shotId} value={String(i)}>
            {["A", "B", "C"][i]} · {c.shotId}
          </ToggleGroupItem>
        ))}
        <ToggleGroupItem value="3d">3-D</ToggleGroupItem>
      </ToggleGroup>
      {well === "3d" ? (
        <div className="relative h-72 overflow-hidden rounded-lg bg-stage ring-1 ring-border">
          <Studio3D studio={studio} className="absolute inset-0" />
        </div>
      ) : (
        <StudioView studio={studio} index={Math.min(Number(well), studio.cams.length - 1)} interactive={false} canFocus={false} />
      )}
      <Timeline studio={studio} />
      <Inspector studio={studio} onHelp={onHelp} readOnly />
    </div>
  )
}
