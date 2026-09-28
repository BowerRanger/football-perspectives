import { getJsonOrNull, qs } from "@/lib/api"
import { playerColour } from "@/lib/format"
import { loadSmplModel } from "./smpl"
import type {
  BallFrameRaw,
  CameraTrackRaw,
  KitColours,
  MatchInfo,
  PlayerPreview,
  PlayerRow,
  PlayerTrack,
  SceneData,
  SceneMetadata,
  TrackedPose,
  Vec3,
} from "./types"

// Data colours (team identity), overridden by match.kits when present.
const DEFAULT_COLOURS: KitColours = { A: 0x3b82f6, B: 0xef4444, referee: 0xf59e0b, unknown: 0x6b7280 }
const DEFAULT_BALL_RADIUS_M = 0.11
const PREVIEW_CONCURRENCY = 4

export interface LoadProgress {
  label: string
  /** 0..100 */
  value: number
}
type ProgressFn = (p: LoadProgress) => void

function hexToInt(s: string | null | undefined): number | null {
  if (typeof s !== "string") return null
  const m = s.trim().toLowerCase().match(/^#?([0-9a-f]{6})$/)
  return m ? parseInt(m[1], 16) : null
}

function kitColours(match: MatchInfo | null): KitColours {
  const kits = match?.kits
  return {
    ...DEFAULT_COLOURS,
    A: hexToInt(kits?.home_primary) ?? DEFAULT_COLOURS.A,
    B: hexToInt(kits?.away_primary) ?? DEFAULT_COLOURS.B,
    referee: hexToInt(kits?.referee) ?? DEFAULT_COLOURS.referee,
  }
}

/** Kit colour by team; unassigned players get the shared per-player palette so they stay distinguishable. */
function teamColour(team: string, colours: KitColours, index: number): number {
  if (team === "A") return colours.A
  if (team === "B") return colours.B
  if (team === "referee") return colours.referee
  return hexToInt(playerColour(index)) ?? colours.unknown
}

async function loadMatch(shot: string | undefined, signal: AbortSignal): Promise<MatchInfo | null> {
  const meta = await getJsonOrNull<SceneMetadata>(`/api/export/metadata${qs({ shot })}`, { signal })
  if (meta?.match) return meta.match
  return getJsonOrNull<MatchInfo>("/api/match", { signal })
}

function buildTrack(raw: CameraTrackRaw | null): Map<number, TrackedPose> | null {
  if (!raw?.frames?.length) return null
  const fallbackT: Vec3 = raw.t_world ?? [0, 0, 0]
  const out = new Map<number, TrackedPose>()
  for (const cf of raw.frames) {
    if (cf.K && cf.R) out.set(cf.frame, { K: cf.K, R: cf.R, t: cf.t ?? fallbackT })
  }
  return out.size > 0 ? out : null
}

interface PlayerListing {
  source: SceneData["playerSource"]
  refs: { id: string; shot: string; name: string }[]
}

/** Prefer refined_poses; fall back to hmr_world when it has nothing for this shot. */
async function listPlayers(shot: string | undefined, signal: AbortSignal): Promise<PlayerListing> {
  const refined = await getJsonOrNull<{ players?: PlayerRow[] }>("/refined_poses/players", { signal })
  const matching = (refined?.players ?? []).filter((row) => {
    if (!shot) return true
    const cs = row.contributing_shots ?? []
    // Empty contributing_shots means single-shot legacy data; keep it.
    return cs.length === 0 || cs.includes(shot)
  })
  if (matching.length > 0) {
    return {
      source: "refined_poses",
      refs: matching.map((r) => ({ id: r.player_id, shot: shot ?? "", name: r.player_name ?? "" })),
    }
  }
  const list = await getJsonOrNull<{ players?: (string | PlayerRow)[] }>(`/hmr_world/players${qs({ shot })}`, { signal })
  const refs: PlayerListing["refs"] = []
  for (const r of list?.players ?? []) {
    if (typeof r === "string") refs.push({ id: r, shot: shot ?? "", name: "" })
    else if (r?.player_id) refs.push({ id: r.player_id, shot: r.shot_id ?? shot ?? "", name: r.player_name ?? "" })
  }
  return { source: "hmr_world", refs }
}

async function fetchPreviews(
  listing: PlayerListing,
  signal: AbortSignal,
  onStep: (done: number) => void,
): Promise<{ ref: PlayerListing["refs"][number]; preview: PlayerPreview }[]> {
  const endpoint = listing.source === "refined_poses" ? "/refined_poses/preview" : "/hmr_world/preview"
  const results: { ref: PlayerListing["refs"][number]; preview: PlayerPreview }[] = []
  let done = 0
  for (let i = 0; i < listing.refs.length; i += PREVIEW_CONCURRENCY) {
    const chunk = listing.refs.slice(i, i + PREVIEW_CONCURRENCY)
    const loaded = await Promise.all(
      chunk.map(async (ref) => {
        const preview = await getJsonOrNull<PlayerPreview>(
          `${endpoint}${qs({ player_id: ref.id, include_pose: 1, shot: ref.shot })}`,
          { signal },
        )
        done += 1
        onStep(done)
        return preview ? { ref, preview } : null
      }),
    )
    for (const l of loaded) if (l) results.push(l)
  }
  return results
}

function toTrack(ref: PlayerListing["refs"][number], p: PlayerPreview, colours: KitColours, index: number): PlayerTrack {
  const frames = p.frames ?? []
  const team = p.team ?? "unknown"
  return {
    id: p.player_id,
    name: ref.name || p.player_id,
    team,
    colour: teamColour(team, colours, index),
    betas: p.betas ?? [],
    frames,
    frameIndex: new Map(frames.map((f, i) => [f, i])),
    rootT: p.root_t ?? [],
    rootR: p.root_R ?? [],
    thetas: p.thetas ?? [],
  }
}

async function loadBall(shot: string | undefined, signal: AbortSignal) {
  const cfg = await getJsonOrNull<{ ball?: { ball_radius_m?: number } }>("/api/config", { signal })
  const r = cfg?.ball?.ball_radius_m
  const radius = typeof r === "number" && r > 0 ? r : DEFAULT_BALL_RADIUS_M
  const ball = await getJsonOrNull<{ frames?: BallFrameRaw[] }>(`/ball/preview${qs({ shot })}`, { signal })
  const frames = ball?.frames ?? []
  const byFrame = new Map<number, Vec3>()
  for (const f of frames) if (f.world_xyz) byFrame.set(f.frame, f.world_xyz)
  const last = frames.length > 0 ? frames[frames.length - 1].frame + 1 : 0
  return { radius, byFrame, hasBall: frames.length > 0, frameCount: last }
}

/** Fetch everything the viewer needs. Throws on abort; missing stage output yields empty data. */
export async function loadSceneData(
  shot: string | undefined,
  onProgress: ProgressFn,
  signal: AbortSignal,
): Promise<SceneData> {
  onProgress({ label: "Reading match and camera", value: 5 })
  const [match, camRaw] = await Promise.all([
    loadMatch(shot, signal),
    getJsonOrNull<CameraTrackRaw>(`/camera/track${qs({ shot })}`, { signal }),
  ])
  const track = buildTrack(camRaw)
  let totalFrames = camRaw?.frames?.length ?? 0

  onProgress({ label: "Listing players", value: 15 })
  const listing = await listPlayers(shot, signal)
  const smplPromise = loadSmplModel(signal)

  const total = Math.max(1, listing.refs.length)
  const previews = await fetchPreviews(listing, signal, (done) =>
    onProgress({ label: `Loading poses (${done}/${total})`, value: 15 + Math.round((done / total) * 60) }),
  )
  const colours = kitColours(match)
  const players = previews.map(({ ref, preview }, i) => toTrack(ref, preview, colours, i))
  for (const p of players) {
    if (p.frames.length > 0) totalFrames = Math.max(totalFrames, p.frames[p.frames.length - 1] + 1)
  }

  onProgress({ label: "Loading ball and body model", value: 80 })
  const [ball, smpl] = await Promise.all([loadBall(shot, signal), smplPromise])
  totalFrames = Math.max(totalFrames, ball.frameCount)

  return {
    fps: camRaw?.fps && camRaw.fps > 0 ? camRaw.fps : 25,
    totalFrames,
    players,
    ball: ball.byFrame,
    hasBall: ball.hasBall,
    ballRadius: ball.radius,
    track,
    trackClipId: camRaw?.clip_id ?? shot ?? "tracked",
    trackImageHeight: camRaw?.image_size?.[1] ?? 1080,
    smpl,
    match,
    colours,
    playerSource: listing.source,
  }
}
