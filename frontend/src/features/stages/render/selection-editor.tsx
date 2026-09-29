import * as React from "react"
import { toast } from "sonner"

import { Panel, PanelEmpty, PanelError } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { errorMessage, getJson, putJson, qs } from "@/lib/api"

import {
  CAMERA_OPTIONS,
  type AvailablePlayer,
  type RenderSelection,
} from "./camera-options"

type VerticalChoice = "default" | "on" | "off"

const toChoice = (v: boolean | null | undefined): VerticalChoice => (v === true ? "on" : v === false ? "off" : "default")
const fromChoice = (c: VerticalChoice): boolean | null => (c === "on" ? true : c === "off" ? false : null)

interface Loaded {
  chosen: string[]
  vertical: VerticalChoice
  players: AvailablePlayer[]
}

/** Camera selection for one shot. Every change saves immediately (PUT /api/render/selection). */
export function SelectionEditor({ shotId }: { shotId: string }) {
  const [state, setState] = React.useState<Loaded | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [status, setStatus] = React.useState<string>("")
  const [attempt, setAttempt] = React.useState(0)

  React.useEffect(() => {
    let cancelled = false
    setState(null)
    setError(null)
    setStatus("")
    void Promise.all([
      getJson<RenderSelection>(`/api/render/selection${qs({ shot: shotId })}`),
      // 200 {players: []} when hmr_world hasn't run; a failure is an error, not "no players".
      getJson<{ players?: AvailablePlayer[] }>(`/api/export/available-players${qs({ shot: shotId })}`),
    ])
      .then(([sel, avail]) => {
        if (cancelled) return
        const chosen = sel.cameras ?? []
        setState({ chosen, vertical: toChoice(sel.vertical_variant), players: avail.players ?? [] })
        if (chosen.length) setStatus(`${chosen.length} camera(s) saved for ${shotId}. Click Render to generate them.`)
      })
      .catch((err) => !cancelled && setError(errorMessage(err)))
    return () => {
      cancelled = true
    }
  }, [shotId, attempt])

  const save = async (chosen: string[], vertical: VerticalChoice) => {
    setStatus("Saving…")
    try {
      await putJson(`/api/render/selection${qs({ shot: shotId })}`, {
        shot_id: shotId,
        cameras: chosen,
        vertical_variant: fromChoice(vertical),
      })
      setStatus(
        chosen.length
          ? `Saved ${chosen.length} camera(s) for ${shotId}. Click Render to generate them.`
          : `No cameras selected for ${shotId}.`,
      )
    } catch (err) {
      setStatus("")
      toast.error("Could not save the camera selection", { description: errorMessage(err) })
    }
  }

  if (error) {
    return (
      <Panel title="Camera selection">
        <PanelError
          title="Could not load the camera selection"
          message={error}
          action={
            <Button variant="outline" size="sm" className="mt-2" onClick={() => setAttempt((a) => a + 1)}>
              Retry
            </Button>
          }
        />
      </Panel>
    )
  }
  if (!state) {
    return (
      <Panel title="Camera selection">
        <Skeleton className="h-24 w-full" />
      </Panel>
    )
  }

  const toggle = (camId: string, on: boolean) => {
    const chosen = on ? [...state.chosen.filter((c) => c !== camId), camId] : state.chosen.filter((c) => c !== camId)
    setState({ ...state, chosen })
    void save(chosen, state.vertical)
  }
  const setVertical = (choice: VerticalChoice) => {
    setState({ ...state, vertical: choice })
    void save(state.chosen, choice)
  }

  return (
    <Panel title="Camera selection" description={`Cameras to render for ${shotId}. Changes save automatically.`}>
      <div className="flex flex-col gap-5">
        <fieldset>
          <legend className="mb-2 text-sm font-medium">Stadium and action cameras</legend>
          <div className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2 xl:grid-cols-3">
            {CAMERA_OPTIONS.map((c) => (
              <div key={c.id} className="flex items-start gap-2">
                <Checkbox
                  id={`cam-${c.id}`}
                  checked={state.chosen.includes(c.id)}
                  onCheckedChange={(v) => toggle(c.id, v === true)}
                  className="mt-0.5"
                />
                <Label htmlFor={`cam-${c.id}`} className="flex flex-col items-start gap-0.5 font-normal">
                  <span className="text-sm">{c.name}</span>
                  <span className="text-xs text-muted-foreground">{c.description}</span>
                </Label>
              </div>
            ))}
          </div>
        </fieldset>

        <div>
          <h4 className="mb-2 text-sm font-medium">Player cameras</h4>
          {state.players.length ? (
            <div className="max-h-72 overflow-y-auto rounded-md border">
            <Table>
              <TableHeader className="sticky top-0 bg-card">
                <TableRow>
                  <TableHead>Player</TableHead>
                  <TableHead className="w-20">POV</TableHead>
                  <TableHead className="w-20">OTS</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {state.players.map((p) => (
                  <TableRow key={p.player_id}>
                    <TableCell>{p.display_name || p.player_id}</TableCell>
                    {(["pov", "ots"] as const).map((rig) => {
                      const camId = `${rig}:${p.player_id}`
                      return (
                        <TableCell key={rig}>
                          <Checkbox
                            aria-label={`${rig.toUpperCase()} camera for ${p.display_name || p.player_id}`}
                            checked={state.chosen.includes(camId)}
                            onCheckedChange={(v) => toggle(camId, v === true)}
                          />
                        </TableCell>
                      )
                    })}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            </div>
          ) : (
            <PanelEmpty
              className="py-6"
              title="No player cameras yet"
              description="Run HMR World first: POV and over-the-shoulder cameras need players with SMPL data for this shot."
            />
          )}
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <span id="vertical-label" className="text-sm font-medium">
            9:16 vertical variant
          </span>
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            aria-labelledby="vertical-label"
            value={state.vertical}
            onValueChange={(v) => v && setVertical(v as VerticalChoice)}
          >
            <ToggleGroupItem value="default">Config default</ToggleGroupItem>
            <ToggleGroupItem value="on">Always</ToggleGroupItem>
            <ToggleGroupItem value="off">Never</ToggleGroupItem>
          </ToggleGroup>
        </div>

        <p role="status" aria-live="polite" className="min-h-5 text-sm text-muted-foreground">
          {status}
        </p>
      </div>
    </Panel>
  )
}
