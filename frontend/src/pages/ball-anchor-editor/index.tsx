import { PageHeader } from "@/components/page-header"
import { PanelEmpty } from "@/components/panel"

// STUB — replaced by the ball-anchor-editor port.
export function BallAnchorEditor(_props: { embedded?: boolean; shot?: string }) {
  return <PanelEmpty title="Not ported yet" />
}

export default function BallAnchorEditorPage() {
  return (
    <>
      <PageHeader title="Ball anchors" />
      <div className="p-4"><BallAnchorEditor /></div>
    </>
  )
}
