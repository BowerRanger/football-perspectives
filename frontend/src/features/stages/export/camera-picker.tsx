import * as React from "react"
import { toast } from "sonner"

import { Panel, PanelEmpty, PanelSkeleton } from "@/components/panel"
import { ToneBadge } from "@/components/status"
import { Checkbox } from "@/components/ui/checkbox"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { errorMessage, getJsonOrNull, putJson } from "@/lib/api"

type Rig = "pov" | "ots"
const RIGS: readonly Rig[] = ["pov", "ots"]

interface Selection {
  player_id: string
  rigs: string[]
}

interface AvailablePlayer {
  player_id: string
  display_name?: string
}

// Last-known selection per shot. Module-level so the picker survives the
// stage-panel re-mount after an export job: if the re-mount's GET fails we
// restore the ticked boxes from here rather than show a blank picker.
const lastKnown: Record<string, Selection[]> = {}

type Chosen = ReadonlyMap<string, ReadonlySet<string>>

function toChosen(selections: readonly Selection[]): Chosen {
  return new Map(selections.map((s) => [s.player_id, new Set(s.rigs)]))
}

function toSelections(players: readonly AvailablePlayer[], chosen: Chosen): Selection[] {
  return players.flatMap((p) => {
    const rigs = RIGS.filter((r) => chosen.get(p.player_id)?.has(r))
    return rigs.length ? [{ player_id: p.player_id, rigs }] : []
  })
}

function countRigs(chosen: Chosen): number {
  let n = 0
  chosen.forEach((s) => (n += s.size))
  return n
}

function withToggle(chosen: Chosen, playerId: string, rig: Rig, on: boolean): Chosen {
  const next = new Map(chosen)
  const rigs = new Set(next.get(playerId) ?? [])
  if (on) rigs.add(rig)
  else rigs.delete(rig)
  next.set(playerId, rigs)
  return next
}

/** Every checkbox toggle saves immediately; the status line mirrors what is on disk. */
export function CameraPicker({ shotId }: { shotId: string }) {
  const [players, setPlayers] = React.useState<AvailablePlayer[] | null>(null)
  const [chosen, setChosen] = React.useState<Chosen>(new Map())
  const [saving, setSaving] = React.useState(false)

  React.useEffect(() => {
    let alive = true
    setPlayers(null)
    void Promise.all([
      getJsonOrNull<{ players?: AvailablePlayer[] }>(`/api/export/available-players?shot=${encodeURIComponent(shotId)}`),
      getJsonOrNull<{ selections?: Selection[] }>(`/api/export/camera-selection?shot=${encodeURIComponent(shotId)}`),
    ]).then(([avail, sel]) => {
      if (!alive) return
      const persisted = Array.isArray(sel?.selections) ? sel.selections : (lastKnown[shotId] ?? [])
      lastKnown[shotId] = persisted
      setChosen(toChosen(persisted))
      setPlayers(avail?.players ?? [])
    })
    return () => {
      alive = false
    }
  }, [shotId])

  const toggle = async (playerId: string, rig: Rig, on: boolean) => {
    if (!players) return
    const previous = chosen
    const next = withToggle(chosen, playerId, rig, on)
    const selections = toSelections(players, next)
    setChosen(next)
    setSaving(true)
    try {
      await putJson(`/api/export/camera-selection?shot=${encodeURIComponent(shotId)}`, { shot_id: shotId, selections })
      lastKnown[shotId] = selections
    } catch (err) {
      setChosen(previous)
      toast.error("Could not save camera selection", { description: errorMessage(err) })
    } finally {
      setSaving(false)
    }
  }

  if (players === null) return <PanelSkeleton rows={3} />
  const count = countRigs(chosen)
  return (
    <Panel
      title="Perspective cameras"
      description={`Per-player POV / over-the-shoulder rigs for "${shotId}". Changes save immediately; re-run Export to generate them.`}
      actions={
        saving ? (
          <ToneBadge tone="muted">Saving…</ToneBadge>
        ) : (
          <ToneBadge tone={count ? "success" : "muted"}>{count ? `${count} saved` : "None selected"}</ToneBadge>
        )
      }
      flush={players.length > 0}
    >
      {players.length === 0 ? (
        <PanelEmpty title="No players with SMPL data for this shot" description="Run hmr_world for this shot, then pick the players you want POV or OTS cameras for." />
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Player</TableHead>
              <TableHead className="w-20 text-center">POV</TableHead>
              <TableHead className="w-20 text-center">OTS</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {players.map((p) => (
              <TableRow key={p.player_id}>
                <TableCell>
                  <span className="font-medium">{p.display_name || p.player_id}</span>
                  {p.display_name && p.display_name !== p.player_id ? (
                    <span className="ml-2 font-mono text-xs text-muted-foreground">{p.player_id}</span>
                  ) : null}
                </TableCell>
                {RIGS.map((rig) => (
                  <TableCell key={rig} className="text-center">
                    <Checkbox
                      checked={chosen.get(p.player_id)?.has(rig) ?? false}
                      onCheckedChange={(v) => void toggle(p.player_id, rig, v === true)}
                      aria-label={`${rig.toUpperCase()} camera for ${p.display_name || p.player_id}`}
                    />
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </Panel>
  )
}
