import * as React from "react"
import { PlayIcon } from "lucide-react"
import { toast } from "sonner"

import { Panel, PanelEmpty, PanelSkeleton } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { Field, FieldLabel } from "@/components/ui/field"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { usePipeline } from "@/hooks/use-pipeline"
import { errorMessage, getJsonOrNull, postJson } from "@/lib/api"
import { HmrShotBody } from "./shot-body"

const ALL = "__all__"

interface PlayerOption {
  id: string
  name: string
}

interface TrackPreview {
  tracks?: { track_id: string; class_name?: string; player_id?: string | null; player_name?: string | null }[]
}

/** Every player tracking saw in the shot, so a not-yet-processed one can be targeted. */
async function loadPlayerOptions(shotId: string): Promise<PlayerOption[]> {
  const data = await getJsonOrNull<TrackPreview>(`/tracking/preview?shot_id=${encodeURIComponent(shotId)}`)
  const seen = new Map<string, string>()
  for (const t of data?.tracks ?? []) {
    if (t.class_name !== "player" && t.class_name !== "goalkeeper") continue
    if (t.player_name === "ignore") continue
    const pid = t.player_id || `${shotId}_T${t.track_id}`
    if (!seen.has(pid)) seen.set(pid, t.player_name || "")
  }
  return [...seen.entries()].sort().map(([id, name]) => ({ id, name }))
}

export default function HmrWorldStage() {
  const { attachToJob, isRunning } = usePipeline()
  const [shots, setShots] = React.useState<string[] | null>(null)
  const [shot, setShot] = React.useState("")
  const [options, setOptions] = React.useState<PlayerOption[]>([])
  const [player, setPlayer] = React.useState(ALL)
  const [dispatching, setDispatching] = React.useState(false)

  React.useEffect(() => {
    let alive = true
    void getJsonOrNull<{ shots?: string[] }>("/api/output/shots").then((d) => {
      if (!alive) return
      const ids = d?.shots ?? []
      setShots(ids)
      if (ids.length) setShot(ids[0])
    })
    return () => {
      alive = false
    }
  }, [])

  React.useEffect(() => {
    if (!shot) return
    let alive = true
    setPlayer(ALL)
    void loadPlayerOptions(shot).then((o) => {
      if (alive) setOptions(o)
    })
    return () => {
      alive = false
    }
  }, [shot])

  const run = async () => {
    setDispatching(true)
    try {
      const forAll = player === ALL
      const { job_id } = forAll
        ? await postJson<{ job_id: string }>("/api/run-shot", { stage: "hmr_world", shot_id: shot })
        : await postJson<{ job_id: string }>("/api/run-shot-player", { shot_id: shot, player_id: player })
      toast.info(forAll ? `hmr_world running for ${shot}` : `hmr_world running for ${shot}__${player}`, {
        description: `Job ${job_id} — follow it in the log dock.`,
      })
      attachToJob(job_id, "hmr_world", () => setDispatching(false))
    } catch (err) {
      setDispatching(false)
      toast.error("Could not start hmr_world", { description: errorMessage(err) })
    }
  }

  if (shots === null) return <PanelSkeleton rows={2} />
  if (shots.length === 0) {
    return (
      <PanelEmpty
        title="No shots yet"
        description="Run prepare_shots first — hmr_world works per shot."
      />
    )
  }

  return (
    <div className="flex flex-col gap-4">
      <Panel title="Run selection" description="Re-run hmr_world for one shot, or a single player for fast iteration.">
        <div className="flex flex-wrap items-end gap-3">
          <Field className="w-44">
            <FieldLabel>Shot</FieldLabel>
            <Select value={shot} onValueChange={setShot}>
              <SelectTrigger className="w-full font-mono" aria-label="Shot">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {shots.map((id) => (
                  <SelectItem key={id} value={id} className="font-mono">
                    {id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          <Field className="w-56">
            <FieldLabel>Player</FieldLabel>
            <Select value={player} onValueChange={setPlayer}>
              <SelectTrigger className="w-full" aria-label="Player">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL}>All players</SelectItem>
                {options.map((o) => (
                  <SelectItem key={o.id} value={o.id}>
                    {o.name ? `${o.name} (${o.id})` : o.id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="ml-auto inline-flex">
                <Button onClick={() => void run()} disabled={isRunning || dispatching || !shot}>
                  <PlayIcon data-icon="inline-start" />
                  Run for selection
                </Button>
              </span>
            </TooltipTrigger>
            <TooltipContent className="max-w-64">
              Run hmr_world for the selected shot. With a player chosen the run is filtered to that one player_id.
            </TooltipContent>
          </Tooltip>
        </div>
      </Panel>
      <HmrShotBody key={shot} shotId={shot} />
    </div>
  )
}
