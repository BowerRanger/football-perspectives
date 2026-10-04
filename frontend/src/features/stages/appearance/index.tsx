import { StatusCard } from "../status-card"

/** Appearance stage: kit/team clustering status (full panel to follow). */
export default function AppearanceStage() {
  return (
    <StatusCard
      stage="appearance"
      outputs="appearance/kits.json"
      nextStep="Clusters team and kit colours from the footage. Suggestions never edit tracks; operator kit labels always win."
    />
  )
}
