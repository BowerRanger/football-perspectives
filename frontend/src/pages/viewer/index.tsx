import { PageHeader } from "@/components/page-header"
import { PanelEmpty } from "@/components/panel"

// STUB — replaced by the viewer port.
export function Viewer(_props: { embedded?: boolean; shot?: string }) {
  return <PanelEmpty title="Not ported yet" />
}

export default function ViewerPage() {
  return (
    <>
      <PageHeader title="3D viewer" />
      <div className="p-4"><Viewer /></div>
    </>
  )
}
