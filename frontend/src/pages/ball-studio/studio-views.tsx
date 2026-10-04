import { refToShot, epipolarVisible } from "./camera-model"
import { Scene3D } from "./scene-3d"
import type { Studio } from "./use-studio"
import { ViewWell } from "./view-well"
import { viewLetter } from "./palette"

interface StudioViewProps {
  studio: Studio
  index: number
  interactive?: boolean
  focused?: boolean
  canFocus?: boolean
  maxHeight?: string
  className?: string
}

/** One ViewWell wired to the studio controller. */
export function StudioView({ studio, index, interactive = true, focused = false, canFocus = true, maxHeight, className }: StudioViewProps) {
  const shot = studio.shots[index]
  const cam = studio.cams[index]
  const shotFrame = refToShot(studio.frame, cam.frameOffset)
  const status = studio.videos.status[shot.shot_id] ?? "loading"
  const hasCamera = cam.at(shotFrame) !== null
  const lines = studio.preview.epipolar[shot.shot_id]
  const hidden = lines?.length && !lines.some((l) => epipolarVisible(l.epi, shot.image_size))
  const from = lines?.[0] ? studio.cams.findIndex((c) => c.shotId === lines[0].from) : 0

  return (
    <ViewWell
      className={className}
      maxHeight={maxHeight}
      index={index}
      shot={shot}
      shotFrame={shotFrame}
      status={status}
      videoRef={studio.videos.register(shot.shot_id)}
      draw={studio.drawPropsFor(index)}
      active={studio.activeView === index}
      interactive={interactive && hasCamera}
      focused={focused}
      loupe={studio.loupe && studio.activeView === index}
      resetSignal={studio.resetSignal}
      resolve={(uv, radius, alt) => studio.resolveSnap(index, uv, radius, alt)}
      onPick={(uv) => {
        studio.setActiveView(index)
        studio.dispatchPick({ type: "pick", frame: studio.frame, shotId: shot.shot_id, uv })
      }}
      onHover={(uv) => studio.setHover(uv ? { view: index, uv } : null)}
      onActivate={() => studio.setActiveView(index)}
      onToggleFocus={canFocus && studio.cams.length > 1 ? () => studio.setLayout(focused ? "compare" : "focus") : undefined}
      onVideoError={() => studio.videos.markError(shot.shot_id)}
      isRepeat={studio.repeatSets[index]?.has(shotFrame) ?? false}
      note={
        status !== "out_of_range" && !hasCamera
          ? "No solved camera for this frame, so picks are disabled here"
          : hidden
            ? `${viewLetter(from)}'s ray does not reach this view`
            : null
      }
    />
  )
}

export function Studio3D({ studio, className }: { studio: Studio; className?: string }) {
  const { dispatchPick } = studio
  return (
    <Scene3D
      className={className}
      scene={studio.scene}
      input={studio.engineInput}
      hasViewB={studio.cams.length > 1}
      onDepth={(m) => dispatchPick({ type: "params", params: { depthM: m } })}
    />
  )
}
