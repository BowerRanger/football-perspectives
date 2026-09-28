import * as React from "react"
import { ClapperboardIcon } from "lucide-react"

import { Panel, PanelEmpty, PanelError, PanelSkeleton } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select"
import { toast } from "sonner"

import { errorMessage, getJson, getJsonOrNull, postJson } from "@/lib/api"
import { usePipeline } from "@/hooks/use-pipeline"

import { CameraGrid } from "./camera-grid"
import type { RenderOutputs } from "./camera-options"
import { SelectionEditor } from "./selection-editor"

interface LoadedRender {
  outputs: RenderOutputs["shots"]
  shotIds: string[]
}

/** Render stage: rendered-camera grid, per-shot camera selection, Render button. */
export default function RenderStage() {
  const { startRun, isRunning, attachToJob } = usePipeline()
  const [data, setData] = React.useState<LoadedRender | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [shotId, setShotId] = React.useState("")

  const load = React.useCallback(async () => {
    setError(null)
    try {
      const [outputs, allShots] = await Promise.all([
        getJson<RenderOutputs>("/api/render/outputs"),
        getJsonOrNull<{ shots?: string[] }>("/api/output/shots"),
      ])
      const shots = outputs.shots ?? {}
      // Union: shots with renders plus shots that exist but were never rendered,
      // so the selection editor works before the first Render run.
      const shotIds = Array.from(new Set([...Object.keys(shots), ...(allShots?.shots ?? [])])).sort()
      setData({ outputs: shots, shotIds })
      setShotId((cur) => (cur && shotIds.includes(cur) ? cur : (shotIds[0] ?? "")))
    } catch (err) {
      setError(errorMessage(err))
    }
  }, [])

  React.useEffect(() => {
    void load()
  }, [load])

  // Per-shot render: the render stage honours shot_filter via /api/run-shot.
  const renderShot = async () => {
    try {
      const { job_id } = await postJson<{ job_id: string }>("/api/run-shot", { stage: "render", shot_id: shotId })
      attachToJob(job_id, "render")
    } catch (err) {
      toast.error(`Could not start the ${shotId} render`, { description: errorMessage(err) })
    }
  }

  if (error) {
    return (
      <PanelError
        title="Could not load render outputs"
        message={error}
        action={
          <Button variant="outline" size="sm" className="mt-2" onClick={() => void load()}>
            Retry
          </Button>
        }
      />
    )
  }
  if (!data) return <PanelSkeleton rows={4} media />
  if (data.shotIds.length === 0) {
    return (
      <Panel title="Render">
        <PanelEmpty
          title="No shots yet"
          description="Run Prepare Shots first, then choose cameras here and render."
        />
      </Panel>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      <Panel
        title="Render"
        description="Renders use each shot's saved camera selection below."
        actions={
          <>
            <Button variant="outline" disabled={isRunning} onClick={() => void startRun("render")}>
              Render all shots
            </Button>
            <Button disabled={isRunning || !shotId} onClick={() => void renderShot()}>
              <ClapperboardIcon data-icon="inline-start" />
              Render {shotId || "shot"}
            </Button>
          </>
        }
      >
        <div className="flex flex-wrap items-center gap-2">
          <Label htmlFor="render-shot">Shot</Label>
          <NativeSelect id="render-shot" value={shotId} onChange={(e) => setShotId(e.target.value)}>
            {data.shotIds.map((id) => (
              <NativeSelectOption key={id} value={id}>
                {id}
              </NativeSelectOption>
            ))}
          </NativeSelect>
          <span className="text-xs text-muted-foreground">Picks the shot to preview, configure and render.</span>
        </div>
      </Panel>
      {shotId ? (
        <>
          <CameraGrid shotId={shotId} output={data.outputs[shotId]} />
          <SelectionEditor key={shotId} shotId={shotId} />
        </>
      ) : null}
    </div>
  )
}
