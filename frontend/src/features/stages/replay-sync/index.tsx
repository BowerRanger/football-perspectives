import { StatusCard } from "../status-card"

/** Replay sync stage: playback-speed detection status. Per-replay controls live in Prepare Shots, Group sync. */
export default function ReplaySyncStage() {
  return (
    <StatusCard
      stage="replay_sync"
      outputs="shots/replay_sync.json"
      nextStep="Measures each replay's speed from the players on the pitch and retimes confident slow motion to real time. Review and override per replay in Prepare Shots, Group sync."
    />
  )
}
