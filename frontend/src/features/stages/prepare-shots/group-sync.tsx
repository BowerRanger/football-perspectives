import * as React from "react"

import { Panel } from "@/components/panel"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"

import { SyncEditor } from "./sync-editor"
import type { GroupView, ShotModel } from "./types"

interface GroupSyncProps {
  model: ShotModel
  /** Changes when the manifest/sync were replaced wholesale; reseeds editors. */
  revision: number
  onSaved: () => void
}

/** Group-scoped sync editor: offsets are only meaningful inside one highlight. */
export function GroupSync({ model, revision, onSaved }: GroupSyncProps) {
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
      description="Align shots within a highlight. Scrub both videos to the same instant and lock the offset, or drag clips on the timeline. Edits become manual and survive re-alignment."
    >
      <div className="flex flex-col gap-4">
        <Tabs value={group.id} onValueChange={setActiveGid}>
          <TabsList className="h-auto flex-wrap justify-start">
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
        />
      </div>
    </Panel>
  )
}
