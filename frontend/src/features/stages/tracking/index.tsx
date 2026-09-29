import { useSearchParams } from "react-router"
import { UsersIcon } from "lucide-react"

import { Panel, PanelEmpty, PanelError, PanelSkeleton } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select"
import { useResource } from "@/hooks/use-resource"
import { listTrackedShots } from "./api"
import { TrackEditor } from "./track-editor"

export default function TrackingStage() {
  const { state, retry } = useResource(async () => (await listTrackedShots()).shots ?? [], [])
  const [params, setParams] = useSearchParams()

  if (state.status === "loading") return <PanelSkeleton rows={3} media />
  if (state.status === "error") {
    return (
      <Panel title="Track editor">
        <PanelError title="Could not list tracked shots" message={state.error}
          action={
            <Button size="sm" variant="outline" className="mt-2" onClick={retry}>
              Retry
            </Button>
          }
        />
      </Panel>
    )
  }
  if (state.data.length === 0) {
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
  const shot = requested && state.data.includes(requested) ? requested : state.data[0]

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
        state.data.length > 1 ? (
          <div className="flex items-center gap-2">
            <Label htmlFor="tracking-shot" className="text-xs text-muted-foreground">
              Shot
            </Label>
            <NativeSelect id="tracking-shot" size="sm" value={shot} onChange={(e) => pickShot(e.target.value)}>
              {state.data.map((id) => (
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
