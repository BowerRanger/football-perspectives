import * as React from "react"
import { Link } from "react-router"
import { CameraIcon, MaximizeIcon } from "lucide-react"

import { Panel, PanelEmpty, PanelError, PanelSkeleton } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { errorMessage, getJson, getJsonOrNull, qs } from "@/lib/api"
import { playerColour } from "@/lib/format"
import { AnchorEditor } from "@/pages/anchor-editor"
import { OverlaidPitchMap } from "./overlaid-pitch-map"
import { indexShots } from "./pitch-shots"
import { ShotInfoCard } from "./shot-card"
import type { AnchorsPayload, CameraTrack, ShotCameraData } from "./types"

type LoadState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; shots: ShotCameraData[] }

// Per-shot track + anchors are fetched in parallel so more shots don't
// lengthen the load serially. A missing track/anchor file is normal (stage
// not run yet), so those resolve to null; the shot list itself must succeed.
async function loadShots(): Promise<ShotCameraData[]> {
  const { shots } = await getJson<{ shots?: string[] }>("/api/output/shots")
  return Promise.all(
    (shots ?? []).map(async (id, i) => {
      const [track, anchors] = await Promise.all([
        getJsonOrNull<CameraTrack>(`/camera/track${qs({ shot: id })}`),
        getJsonOrNull<AnchorsPayload>(`/anchors/${encodeURIComponent(id)}`),
      ])
      return { id, colour: playerColour(i), track, anchors }
    }),
  )
}

function AnchorEditorPanel() {
  return (
    <Panel
      title="Anchor editor"
      description="Mark pitch landmarks on key frames; the camera stage solves anchors first, then propagates between them."
      actions={
        <Button asChild variant="outline" size="sm">
          <Link to="/anchor_editor">
            <MaximizeIcon />
            Open full screen
          </Link>
        </Button>
      }
      flush
    >
      <div className="h-[min(80vh,820px)] min-h-[520px] overflow-hidden">
        <AnchorEditor embedded />
      </div>
    </Panel>
  )
}

export default function CameraStage() {
  const [state, setState] = React.useState<LoadState>({ status: "loading" })

  React.useEffect(() => {
    let live = true
    loadShots()
      .then((shots) => live && setState({ status: "ready", shots }))
      .catch((err: unknown) => live && setState({ status: "error", message: errorMessage(err) }))
    return () => {
      live = false
    }
  }, [])

  const indexed = React.useMemo(() => (state.status === "ready" ? indexShots(state.shots) : []), [state])

  if (state.status === "loading") return <PanelSkeleton rows={4} media />
  if (state.status === "error") {
    return (
      <Panel title="Camera tracks">
        <PanelError title="Could not load camera data" message={state.message} />
      </Panel>
    )
  }
  if (state.shots.length === 0) {
    return (
      <div className="flex flex-col gap-4">
        <Panel title="Camera tracks">
          <PanelEmpty
            icon={<CameraIcon />}
            title="No shots yet"
            description="Run prepare_shots first — camera tracking works on the shots it produces."
          />
        </Panel>
        <AnchorEditorPanel />
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      <Panel
        title="Camera tracks (all shots)"
        description="Top-down pitch with every shot's camera position and view direction; scrub or play to watch moving cameras."
      >
        <div className="grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          <OverlaidPitchMap shots={indexed} />
          <div className="grid content-start gap-3 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
            {state.shots.map((s) => (
              <ShotInfoCard key={s.id} {...s} />
            ))}
          </div>
        </div>
      </Panel>
      <AnchorEditorPanel />
    </div>
  )
}
