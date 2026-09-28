import * as React from "react"
import { useSearchParams } from "react-router"
import { UsersIcon } from "lucide-react"

import { Panel, PanelEmpty, PanelError, PanelSkeleton } from "@/components/panel"
import { Label } from "@/components/ui/label"
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select"
import { errorMessage } from "@/lib/api"
import { listTrackedShots } from "./api"
import { TrackEditor } from "./track-editor"

type ShotsState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; shots: string[] }

export default function TrackingStage() {
  const [state, setState] = React.useState<ShotsState>({ status: "loading" })
  const [params, setParams] = useSearchParams()

  React.useEffect(() => {
    let live = true
    listTrackedShots()
      .then((d) => live && setState({ status: "ready", shots: d.shots ?? [] }))
      .catch((err: unknown) => live && setState({ status: "error", message: errorMessage(err) }))
    return () => {
      live = false
    }
  }, [])

  if (state.status === "loading") return <PanelSkeleton rows={3} media />
  if (state.status === "error") {
    return (
      <Panel title="Track editor">
        <PanelError title="Could not list tracked shots" message={state.message} />
      </Panel>
    )
  }
  if (state.shots.length === 0) {
    return (
      <Panel title="Track editor">
        <PanelEmpty
          icon={<UsersIcon />}
          title="No tracks yet"
          description="Run tracking to detect players and the ball, then come back to name and merge the tracks."
        />
      </Panel>
    )
  }

  const requested = params.get("shot")
  const shot = requested && state.shots.includes(requested) ? requested : state.shots[0]

  function pickShot(id: string) {
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        next.set("shot", id)
        return next
      },
      { replace: true },
    )
  }

  return (
    <Panel
      title="Track editor"
      description="Name players, merge fragments, split identity swaps and fill detector dropouts. Names propagate to hmr_world."
      actions={
        state.shots.length > 1 ? (
          <div className="flex items-center gap-2">
            <Label htmlFor="tracking-shot" className="text-xs text-muted-foreground">
              Shot
            </Label>
            <NativeSelect id="tracking-shot" size="sm" value={shot} onChange={(e) => pickShot(e.target.value)}>
              {state.shots.map((id) => (
                <NativeSelectOption key={id} value={id}>
                  {id}
                </NativeSelectOption>
              ))}
            </NativeSelect>
          </div>
        ) : undefined
      }
    >
      <TrackEditor key={shot} shotId={shot} />
    </Panel>
  )
}
