import { StatusCard } from "../status-card"

/** Shorts stage: vertical highlight cuts status (full panel to follow). */
export default function ShortsStage() {
  return (
    <StatusCard
      stage="shorts"
      outputs="shorts/<shot>_*.mp4"
      nextStep="Cuts a vertical short for every goal shot that has a strike and an impact."
    />
  )
}
