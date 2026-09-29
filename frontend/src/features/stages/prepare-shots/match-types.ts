// Match details form model: API payload <-> editable form state, plus the
// season helpers the football-data lookup needs. Server contract:
//   GET /api/match, PUT /api/match, POST /api/match/lookup

export interface RosterEntryPayload {
  name: string
  team: string
  position: string
  shirt_number: number | null
}

export interface KitsPayload {
  home_primary: string
  away_primary: string
  home_goalkeeper: string
  away_goalkeeper: string
  referee: string
}

export interface MomentPayload {
  minute: number
  added_time: number
  event_type: string
  description: string
}

export interface MatchPayload {
  home_team: string
  away_team: string
  home_score: number
  away_score: number
  venue: string
  competition: string
  date: string
  moment: MomentPayload | null
  kits: KitsPayload | null
  roster: RosterEntryPayload[]
}

export interface MatchCandidate {
  match: Partial<MatchPayload>
  filled_fields: string[]
}

export interface RosterRow {
  key: string
  name: string
  position: string
  shirt: string
}

export interface MatchForm {
  home_team: string
  away_team: string
  home_score: string
  away_score: string
  venue: string
  competition: string
  date: string
  minute: string
  added_time: string
  description: string
  kits: KitsPayload
  home: RosterRow[]
  away: RosterRow[]
}

/** Mirrors the viewer's hard-coded team colours so a fresh form has sensible swatches. */
export const DEFAULT_KITS: KitsPayload = {
  home_primary: "#3b82f6",
  away_primary: "#ef4444",
  home_goalkeeper: "",
  away_goalkeeper: "",
  referee: "#f59e0b",
}

let rowCounter = 0
export function newRosterRow(entry?: Partial<RosterEntryPayload>): RosterRow {
  rowCounter += 1
  return {
    key: `r${rowCounter}`,
    name: entry?.name ?? "",
    position: entry?.position ?? "",
    shirt: entry?.shirt_number == null ? "" : String(entry.shirt_number),
  }
}

export function toRows(roster: RosterEntryPayload[] | undefined, team: "A" | "B"): RosterRow[] {
  return (roster ?? []).filter((r) => r.team === team).map((r) => newRosterRow(r))
}

export function formFromMatch(m: MatchPayload | null): MatchForm {
  return {
    home_team: m?.home_team ?? "",
    away_team: m?.away_team ?? "",
    home_score: String(m?.home_score ?? 0),
    away_score: String(m?.away_score ?? 0),
    venue: m?.venue ?? "",
    competition: m?.competition ?? "",
    date: m?.date ?? "",
    minute: m?.moment?.minute == null ? "" : String(m.moment.minute),
    added_time: String(m?.moment?.added_time ?? 0),
    description: m?.moment?.description ?? "",
    kits: {
      home_primary: m?.kits?.home_primary || DEFAULT_KITS.home_primary,
      away_primary: m?.kits?.away_primary || DEFAULT_KITS.away_primary,
      home_goalkeeper: m?.kits?.home_goalkeeper || "#000000",
      away_goalkeeper: m?.kits?.away_goalkeeper || "#000000",
      referee: m?.kits?.referee || DEFAULT_KITS.referee,
    },
    home: toRows(m?.roster, "A"),
    away: toRows(m?.roster, "B"),
  }
}

function rosterPayload(rows: RosterRow[], team: "A" | "B"): RosterEntryPayload[] {
  return rows
    .filter((r) => r.name.trim())
    .map((r) => ({
      name: r.name.trim(),
      team,
      position: r.position.trim(),
      shirt_number: r.shirt.trim() === "" ? null : Number(r.shirt),
    }))
}

export function payloadFromForm(f: MatchForm): MatchPayload {
  const minuteRaw = f.minute.trim()
  return {
    home_team: f.home_team.trim(),
    away_team: f.away_team.trim(),
    home_score: Number(f.home_score || 0),
    away_score: Number(f.away_score || 0),
    venue: f.venue.trim(),
    competition: f.competition.trim(),
    date: f.date,
    moment:
      minuteRaw === ""
        ? null
        : { minute: Number(minuteRaw), added_time: Number(f.added_time || 0), event_type: "goal", description: f.description },
    kits: { ...f.kits },
    roster: [...rosterPayload(f.home, "A"), ...rosterPayload(f.away, "B")],
  }
}

/** Overlay a lookup candidate onto the form (fields the provider lacked stay as typed). */
export function applyCandidate(form: MatchForm, cand: MatchCandidate): MatchForm {
  const m = cand.match
  const roster = Array.isArray(m.roster) ? m.roster : null
  return {
    ...form,
    home_team: m.home_team || form.home_team,
    away_team: m.away_team || form.away_team,
    home_score: String(m.home_score ?? form.home_score),
    away_score: String(m.away_score ?? form.away_score),
    venue: m.venue || "",
    date: m.date || form.date,
    competition: m.competition || "",
    kits: {
      ...form.kits,
      ...(m.kits?.home_primary ? { home_primary: m.kits.home_primary } : {}),
      ...(m.kits?.away_primary ? { away_primary: m.kits.away_primary } : {}),
    },
    home: roster ? toRows(roster, "A") : form.home,
    away: roster ? toRows(roster, "B") : form.away,
  }
}

const seasonLabel = (start: number) => `${start}-${String((start + 1) % 100).padStart(2, "0")}`

/** "YYYY-YY" options from the upcoming season back ~60 years (seasons start in July). */
export function seasonOptions(now = new Date()): { options: string[]; defaultSeason: string } {
  const currentStart = now.getMonth() >= 6 ? now.getFullYear() : now.getFullYear() - 1
  const options: string[] = []
  for (let s = currentStart + 1; s >= currentStart - 60; s--) options.push(seasonLabel(s))
  return { options, defaultSeason: seasonLabel(currentStart) }
}

/** Season string for an ISO match date, or "" when missing/malformed. */
export function seasonFromIsoDate(iso: string | undefined): string {
  const m = typeof iso === "string" ? iso.match(/^(\d{4})-(\d{2})-\d{2}$/) : null
  if (!m) return ""
  const y = Number(m[1])
  return seasonLabel(Number(m[2]) >= 7 ? y : y - 1)
}
