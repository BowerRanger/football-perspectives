import * as React from "react"

import { Panel } from "@/components/panel"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"

import { SyncEditor } from "./sync-editor"
import { CLEAN_GATE, type RetimeGateInputs } from "./retime-gate"
import { useReplaySync } from "./use-replay-sync"
import type { GroupView, ShotModel } from "./types"

interface GroupSyncProps {
  model: ShotModel
  /** Changes when the manifest/sync were replaced wholesale; reseeds editors. */
  revision: number
  onSaved: () => void
  /** Full reload (manifest + sync); the editor reseeds. */
  onReload: () => Promise<void>
}

/** Group-scoped sync editor: offsets are only meaningful inside one highlight. */
export function GroupSync({ model, revision, onSaved, onReload }: GroupSyncProps) {
  const replaySync = useReplaySync()
  const gateRef = React.useRef<RetimeGateInputs>(CLEAN_GATE)
  const editable: GroupView[] = [
    ...model.groups.filter((g) => g.members.length >= 2),
    ...(model.ungrouped.length >= 2
      ? [
          {
            id: "",
            label: "Ungrouped",
            boundary_rule: "manual",
            boundary_confidence: 1,
            members: model.ungrouped,
            sync: model.ungroupedSync,
          },
        ]
      : []),
  ]
  const [activeGid, setActiveGid] = React.useState<string>(editable[0]?.id ?? "")
  if (editable.length === 0) return null
  const group = editable.find((g) => g.id === activeGid) ?? editable[0]

  return (
    <Panel
      title="Group sync"
      description="Align shots within a highlight. Scrub both videos to the same instant and lock the offset, drag clips on the timeline, or mark matching moments to set speed and offset. Edits become manual and survive re-alignment."
    >
      <div className="flex flex-col gap-4">
        <Tabs value={group.id} onValueChange={setActiveGid}>
          <TabsList className="h-auto w-full max-w-full justify-start overflow-x-auto md:w-fit md:flex-wrap md:overflow-visible">
            {editable.map((g) => (
              <TabsTrigger key={g.id || "ungrouped"} value={g.id}>
                {g.label} ({g.members.length})
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
        <SyncEditor
          key={`${group.id}:${group.members.map((m) => m.id).join(",")}:${revision}`}
          group={group}
          onSaved={onSaved}
          onReload={onReload}
          replaySync={replaySync}
          gateRef={gateRef}
        />
      </div>
    </Panel>
  )
}
