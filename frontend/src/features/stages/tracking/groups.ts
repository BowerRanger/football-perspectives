import { IGNORE_NAME, type PlayerGroup, type TrackSummary } from "./types"

function mostCommonTeam(tracks: readonly TrackSummary[]): string {
  const counts = new Map<string, number>()
  for (const t of tracks) {
    const team = t.team || "unknown"
    counts.set(team, (counts.get(team) ?? 0) + 1)
  }
  let best = "unknown"
  let bestN = -1
  for (const [team, n] of counts) {
    if (n > bestN) {
      best = team
      bestN = n
    }
  }
  return best
}

function canonicalName(tracks: readonly TrackSummary[]): string {
  const named = tracks.find((t) => t.player_name && t.player_name !== IGNORE_NAME)
  if (named?.player_name) return named.player_name
  return tracks.some((t) => t.player_name === IGNORE_NAME) ? IGNORE_NAME : ""
}

/**
 * Group tracks by player_id (falling back to track_id) — hmr_world groups by
 * the same key, so one row mirrors one GVHMR player.
 */
export function buildPlayerGroups(editable: readonly TrackSummary[]): PlayerGroup[] {
  const byKey = new Map<string, TrackSummary[]>()
  for (const t of editable) {
    const key = t.player_id || t.track_id
    byKey.set(key, [...(byKey.get(key) ?? []), t])
  }
  return [...byKey.entries()].map(([key, tracks]) => {
    const ranged = tracks.filter((t) => t.frame_range.length === 2)
    const lo = ranged.length ? Math.min(...ranged.map((t) => t.frame_range[0])) : 0
    const hi = ranged.length ? Math.max(...ranged.map((t) => t.frame_range[1])) : 0
    return {
      key,
      tracks,
      frameCount: tracks.reduce((n, t) => n + t.frame_count, 0),
      frameRange: [lo, hi] as [number, number],
      name: canonicalName(tracks),
      team: mostCommonTeam(tracks),
    }
  })
}

export function isNamed(name: string): boolean {
  return name !== "" && name !== IGNORE_NAME
}

export function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`
}
