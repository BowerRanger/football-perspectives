// Data colours and number formatting. Hex lives here only because these are
// DATA colours (per-player series); UI chrome uses theme tokens.

export const PALETTE = [
  "#60a5fa",
  "#f87171",
  "#34d399",
  "#fbbf24",
  "#c084fc",
  "#fb7185",
  "#22d3ee",
  "#facc15",
  "#a78bfa",
  "#f472b6",
  "#4ade80",
  "#fb923c",
  "#38bdf8",
  "#e879f9",
  "#a3e635",
  "#facc15",
]

/** Stable data colour for series index `i` (wraps, handles negatives). */
export function playerColour(i: number): string {
  return PALETTE[((i % PALETTE.length) + PALETTE.length) % PALETTE.length]
}

interface PlayerLike {
  player_id?: string | null
  player_name?: string | null
}

export function playerLabel(p: PlayerLike | null | undefined): string {
  return p?.player_name ? p.player_name : p?.player_id || "?"
}

export function playerLabelWithId(p: PlayerLike | null | undefined): string {
  return p?.player_name && p.player_id ? `${p.player_name} (${p.player_id})` : playerLabel(p)
}

/** "#rgb"/"#rrggbb" -> "rgba(r,g,b,alpha)". */
export function withAlpha(hex: string, alpha: number): string {
  const h = hex.replace("#", "")
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h
  return `rgba(${parseInt(full.slice(0, 2), 16)},${parseInt(full.slice(2, 4), 16)},${parseInt(full.slice(4, 6), 16)},${alpha})`
}

const isMissing = (v: number | null | undefined): v is null | undefined => v == null || Number.isNaN(Number(v))

export function fmt(v: number | null | undefined, digits = 1): string {
  return isMissing(v) ? "—" : Number(v).toFixed(digits)
}

export function fmtInt(v: number | null | undefined): string {
  return isMissing(v) ? "—" : Math.round(Number(v)).toLocaleString()
}

/** Fraction (0..1) -> "NN%". */
export function fmtPct(v: number | null | undefined, digits = 0): string {
  return isMissing(v) ? "—" : `${(Number(v) * 100).toFixed(digits)}%`
}

export type Tone = "success" | "warning" | "destructive" | "info" | "muted"

/** Map a 0..1 confidence to a status tone. */
export function confidenceTone(v: number | null | undefined, good = 0.7, ok = 0.4): Tone {
  if (isMissing(v)) return "muted"
  return v >= good ? "success" : v >= ok ? "warning" : "destructive"
}

export const TONE_TEXT: Record<Tone, string> = {
  success: "text-success",
  warning: "text-warning",
  destructive: "text-destructive",
  info: "text-info",
  muted: "text-muted-foreground",
}

/** Resolve a CSS custom property at runtime (for canvas drawing); `fallback` when unavailable. */
export function cssVar(name: string, fallback = "#888"): string {
  if (typeof window === "undefined") return fallback
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback
}
