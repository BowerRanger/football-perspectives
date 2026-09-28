import * as React from "react"
import { SearchIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select"
import { Spinner } from "@/components/ui/spinner"
import { errorMessage, postJson } from "@/lib/api"
import { cn } from "@/lib/utils"

import { seasonFromIsoDate, seasonOptions, type MatchCandidate } from "./match-types"

interface LookupProps {
  initialHome: string
  initialAway: string
  initialDate: string
  onApply: (cand: MatchCandidate) => void
}

type Tone = "muted" | "success" | "warning" | "destructive"
const TONE: Record<Tone, string> = {
  muted: "text-muted-foreground",
  success: "text-success",
  warning: "text-warning",
  destructive: "text-destructive",
}
const CHECKED_FIELDS = ["score", "venue", "competition", "kits", "roster"]

/** Search-to-assign: pulls match + squad from football-data.org (kits from Wikidata). */
export function MatchLookup({ initialHome, initialAway, initialDate, onApply }: LookupProps) {
  const { options, defaultSeason } = React.useMemo(() => seasonOptions(), [])
  const guess = seasonFromIsoDate(initialDate) || defaultSeason
  const [season, setSeason] = React.useState(options.includes(guess) ? guess : defaultSeason)
  const [home, setHome] = React.useState(initialHome)
  const [away, setAway] = React.useState(initialAway)
  const [busy, setBusy] = React.useState(false)
  const [status, setStatus] = React.useState<{ tone: Tone; text: string } | null>(null)
  const [candidates, setCandidates] = React.useState<MatchCandidate[]>([])

  const apply = (cand: MatchCandidate) => {
    setCandidates([])
    onApply(cand)
    const missing = CHECKED_FIELDS.filter((f) => !cand.filled_fields.includes(f))
    if (missing.length) {
      setStatus({
        tone: "warning",
        text: `Filled ${cand.filled_fields.join(", ") || "nothing"}. Not found: ${missing.join(", ")}. Fill the rest by hand.`,
      })
    } else setStatus({ tone: "success", text: "All fields filled from the provider." })
  }

  const lookup = async () => {
    const payload = { season, home_team: home.trim(), away_team: away.trim(), provider: "football-data" }
    if (!payload.season || !payload.home_team || !payload.away_team) {
      setStatus({ tone: "destructive", text: "Lookup needs a season, home team and away team." })
      return
    }
    setBusy(true)
    setCandidates([])
    setStatus({ tone: "muted", text: "Querying football-data.org…" })
    try {
      const cands = await postJson<MatchCandidate[]>("/api/match/lookup", payload)
      if (!cands || cands.length === 0) setStatus({ tone: "warning", text: "No matches found. Fill in manually." })
      else if (cands.length === 1) apply(cands[0])
      else {
        setCandidates(cands)
        setStatus({ tone: "muted", text: `Found ${cands.length} candidates. Pick one:` })
      }
    } catch (err) {
      setStatus({ tone: "destructive", text: `Lookup failed: ${errorMessage(err)}` })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-col gap-3 rounded-lg border bg-muted/30 p-3">
      <div className="flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="lookup-season">Season</Label>
          <NativeSelect id="lookup-season" value={season} onChange={(e) => setSeason(e.target.value)}>
            {options.map((s) => (
              <NativeSelectOption key={s} value={s}>
                {s}
              </NativeSelectOption>
            ))}
          </NativeSelect>
        </div>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="lookup-home">Home team</Label>
          <Input id="lookup-home" value={home} onChange={(e) => setHome(e.target.value)} />
        </div>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="lookup-away">Away team</Label>
          <Input id="lookup-away" value={away} onChange={(e) => setAway(e.target.value)} />
        </div>
        <Button
          variant="secondary"
          disabled={busy}
          title="Pulls match and squad from football-data.org and kit colours from Wikidata. Needs the FOOTBALL_DATA_ORG_API_KEY env var on the server."
          onClick={() => void lookup()}
        >
          {busy ? <Spinner data-icon="inline-start" /> : <SearchIcon data-icon="inline-start" />}
          Look up match
        </Button>
      </div>
      {status ? (
        <p role="status" className={cn("text-xs", TONE[status.tone])}>
          {status.text}
        </p>
      ) : null}
      {candidates.length > 0 ? (
        <ul className="flex flex-col gap-1.5">
          {candidates.map((c, i) => (
            <li key={i}>
              <Button variant="outline" size="sm" className="w-full justify-start" onClick={() => apply(c)}>
                {c.match.date || "?"} · {c.match.competition || "?"} · {c.match.venue || "?"}
              </Button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}
