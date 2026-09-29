import * as React from "react"

import { Panel, PanelEmpty, PanelError, PanelSkeleton } from "@/components/panel"
import { Button } from "@/components/ui/button"
import { useResource } from "@/hooks/use-resource"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { getJson, getJsonOr404 } from "@/lib/api"
import { fmtInt, playerColour } from "@/lib/format"
import { Kp2dPanel } from "./kp2d-panel"
import { ConfidenceCell, PlayerCell } from "./player-cells"
import { TrajectoryPanel } from "./trajectory-panel"
import type { Coloured, PlayerRef, PlayersResponse, PosePreview } from "./types"

type LoadedPlayer = Coloured<PlayerRef> & { data: PosePreview | null }

// /hmr_world/players answers 200 with [] when nothing ran; /preview 404s for a
// listed-but-missing player (row shows "—"). Anything else is a real failure.
async function loadShot(shotId: string, signal: AbortSignal): Promise<LoadedPlayer[]> {
  const list = await getJson<PlayersResponse>(`/hmr_world/players?shot=${encodeURIComponent(shotId)}`, { signal })
  return Promise.all(
    (list.players ?? []).map(async (p, i) => ({
      ...p,
      colour: playerColour(i),
      data: await getJsonOr404<PosePreview>(
        `/hmr_world/preview?shot=${encodeURIComponent(shotId)}&player_id=${encodeURIComponent(p.player_id)}`,
        { signal },
      ),
    })),
  )
}

function meanConfidence(conf: readonly number[] | undefined): number | null {
  return conf && conf.length ? conf.reduce((s, v) => s + v, 0) / conf.length : null
}

function PlayersTable({ shotId, players }: { shotId: string; players: readonly LoadedPlayer[] }) {
  return (
    <Panel title={`HMR World — ${shotId} (${players.length} player${players.length === 1 ? "" : "s"})`} flush>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Player</TableHead>
            <TableHead className="text-right">Frames</TableHead>
            <TableHead>Mean confidence</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {players.map((p) => (
            <TableRow key={p.player_id}>
              <TableCell>
                <PlayerCell colour={p.colour} playerId={p.player_id} playerName={p.player_name} />
              </TableCell>
              <TableCell className="text-right">{p.data ? fmtInt(p.data.frames?.length ?? 0) : "—"}</TableCell>
              <TableCell>
                <ConfidenceCell value={p.data ? meanConfidence(p.data.confidence) : null} />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Panel>
  )
}

/** One shot's hmr_world output: player table, trajectories, keypoints. */
export function HmrShotBody({ shotId }: { shotId: string }) {
  const { state, retry } = useResource((signal) => loadShot(shotId, signal), [shotId])
  const players = state.status === "ready" ? state.data : null

  const withRoot = React.useMemo(
    () =>
      (players ?? []).flatMap((p) =>
        p.data && Array.isArray(p.data.root_t) && p.data.root_t.length > 0 ? [{ ...p, data: p.data }] : [],
      ),
    [players],
  )

  if (state.status === "loading") return <PanelSkeleton rows={5} />
  if (state.status === "error") {
    return (
      <PanelError
        title={`Could not load hmr_world output for ${shotId}`}
        message={state.error}
        action={
          <Button variant="outline" size="sm" className="mt-2" onClick={retry}>
            Retry
          </Button>
        }
      />
    )
  }
  if (!players || players.length === 0) {
    return (
      <Panel title="HMR World">
        <PanelEmpty
          title={`No hmr_world tracks for ${shotId} yet`}
          description="Pick a player above (or All players) and choose Run for selection."
        />
      </Panel>
    )
  }
  return (
    <>
      <PlayersTable shotId={shotId} players={players} />
      {withRoot.length > 0 ? <TrajectoryPanel players={withRoot} shotId={shotId} title="Top-down player trajectories (pitch-world)" keyboard={false} /> : null}
      <Kp2dPanel shotId={shotId} />
    </>
  )
}
