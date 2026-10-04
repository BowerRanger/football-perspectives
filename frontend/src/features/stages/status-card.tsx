import { LayersIcon } from "lucide-react"

import { Panel, PanelEmpty } from "@/components/panel"
import { usePipeline } from "@/hooks/use-pipeline"
import { humanizeStageName, STAGE_DEPS, type StageName } from "@/lib/stages"

interface StatusCardProps {
  stage: StageName
  /** Where the stage writes, relative to the output dir. */
  outputs: string
  /** What the operator does next (until a full panel lands). */
  nextStep: string
}

/**
 * Minimal stage panel: output location and state. Stages without a bespoke
 * panel yet render this so the dashboard never dead-ends; the run controls
 * live in the page header.
 */
export function StatusCard({ stage, outputs, nextStep }: StatusCardProps) {
  const { stages } = usePipeline()
  const info = stages.find((s) => s.name === stage)
  const deps = STAGE_DEPS[stage].map(humanizeStageName).join(", ")
  const state = info?.complete ? "Complete" : info?.partial ? "Partial" : "Not run yet"
  return (
    <Panel title={`${humanizeStageName(stage)} output`} description={`Writes ${outputs}`}>
      <PanelEmpty icon={<LayersIcon />} title={state} description={nextStep}>
        {deps ? <p className="text-xs text-muted-foreground">Runs after {deps}.</p> : null}
      </PanelEmpty>
    </Panel>
  )
}
