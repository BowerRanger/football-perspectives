import { PageHeader } from "@/components/page-header"
import { PanelEmpty } from "@/components/panel"

// STUB — replaced by the anchor-editor port.
export function AnchorEditor(_props: { embedded?: boolean; shot?: string }) {
  return <PanelEmpty title="Not ported yet" />
}

export default function AnchorEditorPage() {
  return (
    <>
      <PageHeader title="Pitch anchors" />
      <div className="p-4"><AnchorEditor /></div>
    </>
  )
}
