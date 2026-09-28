import * as React from "react"

import { PanelEmpty, PanelError, PanelSkeleton } from "@/components/panel"
import { Button } from "@/components/ui/button"

import { DroppedTray } from "./dropped-tray"
import { GroupSync } from "./group-sync"
import { GroupsBoard } from "./groups-board"
import { IngestPanel } from "./ingest-panel"
import MatchDetails from "./match-details"
import { MultiShotStatus } from "./multi-shot-status"
import { ShotPreview } from "./shot-preview"
import { buildShotModel, type ShotView } from "./types"
import { useShotActions } from "./use-shot-actions"
import { useShotsData } from "./use-shots-data"

/** Prepare Shots stage: ingest, highlight-group board, dropped tray, status, group sync, match details. */
export default function PrepareShotsStage() {
  const data = useShotsData()
  const [preview, setPreview] = React.useState<{ shot: ShotView; groupLabel?: string } | null>(null)

  const model = React.useMemo(
    () => (data.manifest ? buildShotModel(data.manifest, data.features, data.sync) : null),
    [data.manifest, data.features, data.sync],
  )
  const emptyModel = React.useMemo(
    () => buildShotModel({ shots: [] }, {}, { groups: [] }),
    [],
  )
  const actions = useShotActions(model ?? emptyModel, data.patchShots)

  if (data.loading) return <PanelSkeleton rows={5} media />

  return (
    <div className="flex flex-col gap-4">
      <IngestPanel onChanged={data.reload} />
      {data.error ? (
        <PanelError
          title="Could not load the shots manifest"
          message={data.error}
          action={
            <Button variant="outline" size="sm" className="mt-2" onClick={() => void data.reload()}>
              Retry
            </Button>
          }
        />
      ) : null}
      {model && data.manifest && data.manifest.shots.length > 0 ? (
        <>
          <GroupsBoard
            model={model}
            actions={actions}
            onOpen={(shot, groupLabel) => setPreview({ shot, groupLabel })}
            onReload={data.reload}
          />
          <DroppedTray model={model} actions={actions} onOpen={(shot) => setPreview({ shot })} />
          <MultiShotStatus shotIds={model.activeIds} />
          <GroupSync model={model} revision={data.revision} onSaved={() => void data.reloadSyncQuiet()} />
        </>
      ) : !data.error ? (
        <PanelEmpty
          title="No shots yet"
          description="Drop a full highlights reel above to auto-split it, use Add shots for pre-trimmed clips, or pass --input to recon.py."
        />
      ) : null}
      <MatchDetails />
      <ShotPreview shot={preview?.shot ?? null} groupLabel={preview?.groupLabel} onClose={() => setPreview(null)} />
    </div>
  )
}
