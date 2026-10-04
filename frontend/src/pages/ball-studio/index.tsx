import * as React from "react"
import { useSearchParams } from "react-router"
import { SparklesIcon } from "lucide-react"

import { FramePlayer } from "@/components/frame-player"
import { PanelEmpty, PanelError, PanelSkeleton } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { useSidebar } from "@/components/ui/sidebar"
import { useConfirm } from "@/hooks/use-dialogs"
import { useIsMobile } from "@/hooks/use-mobile"
import { useResource } from "@/hooks/use-resource"
import { useUnsavedGuard } from "@/hooks/use-unsaved-guard"
import { getGroups, getScene, getTruth } from "./api"
import { EventMenu } from "./event-menu"
import { HelpSheet } from "./help-sheet"
import { Inspector } from "./inspector"
import { MobileReview } from "./mobile-review"
import { ModeBar } from "./mode-bar"
import { Studio3D, StudioView } from "./studio-views"
import { StudioHeader } from "./studio-header"
import { Timeline } from "./timeline"
import type { GroupInfo, Scene, TruthDoc } from "./types"
import { useStudio } from "./use-studio"
import { useStudioKeys } from "./use-studio-keys"

function readOnboarded(): boolean {
  try {
    return window.localStorage.getItem("ball-studio.onboarded") === "1"
  } catch {
    return false
  }
}

/** Collapse the app sidebar to its icon rail while the studio is open, then restore it. */
function useCollapsedSidebar(): void {
  const { open, setOpen } = useSidebar()
  const initial = React.useRef(open)
  React.useEffect(() => {
    const was = initial.current
    setOpen(false)
    return () => setOpen(was)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mount/unmount only
  }, [])
}

interface ReadyProps {
  group: GroupInfo
  groups: GroupInfo[]
  scene: Scene
  truth: TruthDoc
  onChangeGroup: (id: string) => void
}

function StudioReady({ group, groups, scene, truth, onChangeGroup }: ReadyProps) {
  const isMobile = useIsMobile()
  const confirm = useConfirm()
  const studio = useStudio(group, scene, truth)
  const [eventOpen, setEventOpen] = React.useState(false)
  const [helpOpen, setHelpOpen] = React.useState(false)
  const [onboarded] = React.useState(readOnboarded)

  useUnsavedGuard(studio.docApi.dirty, { what: "ball truth edits" })
  useStudioKeys(studio, { openEventMenu: () => setEventOpen(true), openHelp: () => setHelpOpen(true) }, !isMobile)

  const changeGroup = async (next: string) => {
    if (next === group.group_id) return
    if (studio.docApi.dirty) {
      const ok = await confirm({
        title: `Discard unsaved keys on ${group.group_id}?`,
        description: `Switching to ${next} will discard your unsaved edits.`,
        confirmLabel: "Discard and switch",
        destructive: true,
      })
      if (!ok) return
    }
    onChangeGroup(next)
  }

  const focus = studio.layout === "focus" && studio.cams.length > 1
  const focusIdx = Math.min(studio.activeView, studio.cams.length - 1)
  const pipIdx = focusIdx === 0 ? 1 : 0

  const transport = (
    <FramePlayer
      frame={studio.frame}
      min={studio.range[0]}
      max={studio.range[1]}
      fps={scene.fps}
      frameInput
      playing={studio.videos.playing}
      onTogglePlay={studio.videos.togglePlay}
      onSeek={studio.setFrame}
      label="Reference frame"
    >
      <EventMenu open={eventOpen} onOpenChange={setEventOpen} frame={studio.frame} onPick={studio.addEventAtPlayhead} />
    </FramePlayer>
  )

  const empty = studio.docApi.doc.keys.length === 0 && !onboarded
  const onboarding = empty ? (
    <p className="flex items-center gap-1.5 truncate text-xs text-muted-foreground">
      <SparklesIcon className="size-3.5 shrink-0" aria-hidden />
      Scrub to a frame where the ball is visible, then click it in a view.
    </p>
  ) : null

  return (
    <>
      <StudioHeader groups={groups} groupId={group.group_id} onChangeGroup={(g) => void changeGroup(g)} studio={studio} readOnly={isMobile} />
      {isMobile ? (
        <MobileReview studio={studio} onHelp={() => setHelpOpen(true)} />
      ) : (
        <div className="flex min-w-0 flex-col gap-3 p-4">
          <ModeBar studio={studio} />
          <div
            className={
              focus
                ? "grid items-start gap-3 md:grid-cols-[minmax(0,1fr)_320px] xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_360px]"
                : "grid items-stretch gap-3 md:grid-cols-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_360px]"
            }
          >
            {focus ? (
              <>
                <StudioView studio={studio} index={focusIdx} focused className="md:col-start-1 xl:col-span-2" />
                <div className="flex flex-col gap-3 md:col-start-2 xl:col-start-3">
                  <StudioView studio={studio} index={pipIdx} canFocus={false} />
                  <div className="relative h-64 overflow-hidden rounded-lg bg-stage ring-1 ring-border">
                    <Studio3D studio={studio} className="absolute inset-0" />
                  </div>
                </div>
              </>
            ) : (
              <>
                <StudioView studio={studio} index={0} />
                {studio.cams.length > 1 ? <StudioView studio={studio} index={1} /> : null}
                <div className="relative h-72 overflow-hidden rounded-lg bg-stage ring-1 ring-border md:col-span-2 xl:col-span-1 xl:h-auto xl:min-h-64">
                  <Studio3D studio={studio} className="absolute inset-0" />
                </div>
              </>
            )}
          </div>
          <div className="grid items-start gap-3 xl:grid-cols-[minmax(0,2fr)_360px]">
            <div className="flex min-w-0 flex-col gap-3">
              {transport}
              <Timeline studio={studio} onboarding={onboarding} />
            </div>
            <Inspector studio={studio} onHelp={() => setHelpOpen(true)} />
          </div>
        </div>
      )}
      <HelpSheet open={helpOpen} onOpenChange={setHelpOpen} />
    </>
  )
}

interface LoaderProps {
  group: GroupInfo
  groups: GroupInfo[]
  onChangeGroup: (id: string) => void
}

function GroupLoader({ group, groups, onChangeGroup }: LoaderProps) {
  const scene = useResource((signal) => getScene(group.group_id, signal), [group.group_id])
  const truth = useResource((signal) => getTruth(group.group_id, signal), [group.group_id])

  if (scene.state.status === "error") {
    return (
      <>
        <StudioHeader groups={groups} groupId={group.group_id} onChangeGroup={onChangeGroup} />
        <div className="p-4">
          <PanelError
            title={`Could not load the scene for ${group.group_id}`}
            message={`${scene.state.error}. Editing is disabled until the cameras and players load.`}
            action={
              <Button variant="outline" size="sm" className="mt-2" onClick={scene.retry}>
                Retry
              </Button>
            }
          />
        </div>
      </>
    )
  }
  if (truth.state.status === "error") {
    return (
      <>
        <StudioHeader groups={groups} groupId={group.group_id} onChangeGroup={onChangeGroup} />
        <div className="p-4">
          <PanelError
            title={`Could not load ball truth for ${group.group_id}`}
            message={`${truth.state.error}. Saving is disabled so the existing file can't be overwritten.`}
            action={
              <Button variant="outline" size="sm" className="mt-2" onClick={truth.retry}>
                Retry
              </Button>
            }
          />
        </div>
      </>
    )
  }
  if (scene.state.status === "loading" || truth.state.status === "loading") {
    return (
      <>
        <StudioHeader groups={groups} groupId={group.group_id} onChangeGroup={onChangeGroup} />
        <div className="grid gap-3 p-4 md:grid-cols-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_360px]">
          <PanelSkeleton rows={2} media />
          <PanelSkeleton rows={2} media />
          <PanelSkeleton rows={4} />
        </div>
      </>
    )
  }
  return (
    <StudioReady
      key={group.group_id}
      group={group}
      groups={groups}
      scene={scene.state.data}
      truth={truth.state.data.truth}
      onChangeGroup={onChangeGroup}
    />
  )
}

export default function BallStudioPage() {
  useCollapsedSidebar()
  const [params, setParams] = useSearchParams()
  const groups = useResource((signal) => getGroups(signal), [])
  const wanted = params.get("group")
  const list = groups.state.status === "ready" ? groups.state.data : []

  const setGroup = React.useCallback(
    (id: string, replace = false) =>
      setParams(
        (prev) => {
          const p = new URLSearchParams(prev)
          p.set("group", id)
          return p
        },
        { replace },
      ),
    [setParams],
  )

  React.useEffect(() => {
    if (!wanted && list.length) setGroup(list[0].group_id, true)
  }, [wanted, list, setGroup])

  if (groups.state.status === "error") {
    return (
      <>
        <StudioHeader groups={[]} groupId="" onChangeGroup={() => undefined} />
        <div className="p-4">
          <PanelError
            title="Could not list groups"
            message={groups.state.error}
            action={
              <Button variant="outline" size="sm" className="mt-2" onClick={groups.retry}>
                Retry
              </Button>
            }
          />
        </div>
      </>
    )
  }
  if (groups.state.status === "loading") {
    return (
      <>
        <StudioHeader groups={[]} groupId="" onChangeGroup={() => undefined} />
        <div className="p-4">
          <PanelSkeleton rows={3} media />
        </div>
      </>
    )
  }
  if (!list.length) {
    return (
      <>
        <StudioHeader groups={[]} groupId="" onChangeGroup={() => undefined} />
        <div className="p-4">
          <PanelEmpty
            title="No synced groups yet"
            description="Run Prepare Shots (group and align) from the dashboard; the studio lists every group in sync_map.json that has solved cameras."
          />
        </div>
      </>
    )
  }
  const group = list.find((g) => g.group_id === wanted) ?? list[0]
  return <GroupLoader group={group} groups={list} onChangeGroup={(id) => setGroup(id)} />
}
