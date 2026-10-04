import * as React from "react"

import { usePipeline } from "@/hooks/use-pipeline"
import { toast } from "sonner"

import { errorMessage, getJsonOr404 } from "@/lib/api"

import type { ReplaySyncMember, ReplaySyncReport } from "./types"

export interface ReplaySyncLookup {
  /** replay_sync.json entry per member shot id. */
  byShot: Record<string, ReplaySyncMember>
  /** The replay_sync stage is running right now. */
  detecting: boolean
}

/**
 * Loads `GET /api/replay-sync` and refreshes it whenever a pipeline job
 * finishes. A missing report (404) means the stage has not run: an empty
 * lookup. Any other failure is toasted; the editor still works without it.
 */
export function useReplaySync(): ReplaySyncLookup {
  const { outputVersion, liveState } = usePipeline()
  const [report, setReport] = React.useState<ReplaySyncReport | null>(null)

  React.useEffect(() => {
    let live = true
    getJsonOr404<ReplaySyncReport>("/api/replay-sync")
      .then((r) => {
        if (live) setReport(r)
      })
      .catch((err: unknown) => {
        if (live) toast.error("Could not read the replay speed report", { description: errorMessage(err) })
      })
    return () => {
      live = false
    }
  }, [outputVersion])

  return React.useMemo(() => {
    const byShot: Record<string, ReplaySyncMember> = {}
    for (const g of report?.groups ?? []) for (const m of g.members ?? []) byShot[m.shot_id] = m
    return { byShot, detecting: liveState.replay_sync === "running" }
  }, [report, liveState.replay_sync])
}
