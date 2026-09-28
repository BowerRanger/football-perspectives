import * as React from "react"
import { ExternalLinkIcon } from "lucide-react"
import { Link } from "react-router"

import { Panel, PanelEmpty, PanelSkeleton } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { getJsonOrNull } from "@/lib/api"
import { Viewer } from "@/pages/viewer"
import { CameraPicker } from "./camera-picker"
import { StatusTable, type ShotExportRow } from "./status-table"

interface Loaded {
  shots: string[]
  rows: ShotExportRow[]
  defaultShot: string
}

interface ExportMetadata {
  players?: unknown[]
}

async function loadRow(shotId: string, exported: boolean): Promise<ShotExportRow> {
  const [meta, hmr] = await Promise.all([
    getJsonOrNull<ExportMetadata>(`/api/export/metadata?shot=${encodeURIComponent(shotId)}`),
    getJsonOrNull<{ players?: unknown[] }>(`/hmr_world/players?shot=${encodeURIComponent(shotId)}`),
  ])
  return {
    shotId,
    exported: exported || meta !== null,
    playersInScene: meta ? (meta.players?.length ?? 0) : null,
    hmrCount: hmr?.players?.length ?? 0,
  }
}

async function load(): Promise<Loaded> {
  const [exp, all] = await Promise.all([
    getJsonOrNull<{ shots?: string[] }>("/api/export/shots"),
    getJsonOrNull<{ shots?: string[] }>("/api/output/shots"),
  ])
  const exportShots = exp?.shots ?? []
  // Union: every shot the viewer can inspect, so single-player iterations
  // show up before export has caught up.
  const shots = Array.from(new Set([...exportShots, ...(all?.shots ?? [])]))
  const rows = await Promise.all(shots.map((s) => loadRow(s, exportShots.includes(s))))
  // Default to the latest exported scene, else the shot with the most HMR players.
  const byHmr = [...rows].sort((a, b) => b.hmrCount - a.hmrCount)
  const defaultShot = exportShots[exportShots.length - 1] ?? byHmr[0]?.shotId ?? ""
  return { shots, rows, defaultShot }
}

function ViewerPanel({ shots, shot, onShot }: { shots: string[]; shot: string; onShot: (s: string) => void }) {
  return (
    <Panel
      title="3D viewer"
      description="Live SMPL skeletons and ball trajectory in pitch-world; renders from camera and hmr_world data, so it works with a single player."
      actions={
        <>
          <Select value={shot} onValueChange={onShot}>
            <SelectTrigger size="sm" className="w-40 font-mono" aria-label="Viewer shot">
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
          <Button asChild variant="outline" size="sm">
            <Link to={`/viewer?shot=${encodeURIComponent(shot)}`}>
              <ExternalLinkIcon data-icon="inline-start" />
              Open full screen
            </Link>
          </Button>
        </>
      }
      flush
    >
      <div className="h-[70vh] min-h-[420px] overflow-hidden bg-stage">
        <Viewer key={shot} embedded shot={shot} />
      </div>
    </Panel>
  )
}

export default function ExportStage() {
  const [data, setData] = React.useState<Loaded | null>(null)
  const [shot, setShot] = React.useState("")

  React.useEffect(() => {
    let alive = true
    void load().then((d) => {
      if (!alive) return
      setData(d)
      setShot(d.defaultShot)
    })
    return () => {
      alive = false
    }
  }, [])

  if (!data) return <PanelSkeleton rows={5} />
  if (data.shots.length === 0) {
    return <PanelEmpty title="No shots yet" description="Run prepare_shots first — export works per shot." />
  }
  return (
    <div className="flex flex-col gap-4">
      <StatusTable rows={data.rows} />
      {shot ? <CameraPicker shotId={shot} /> : null}
      {shot ? <ViewerPanel shots={data.shots} shot={shot} onShot={setShot} /> : null}
    </div>
  )
}
