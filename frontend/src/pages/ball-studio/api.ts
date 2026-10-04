import { getJson, postJson, putJson } from "@/lib/api"
import type {
  GroupInfo,
  PutTruthResponse,
  Scene,
  SolveResult,
  TriangulateRequest,
  TriangulateResult,
  TruthDoc,
  TruthResponse,
} from "./types"

const base = (g: string) => `/api/ball-studio/groups/${encodeURIComponent(g)}`

export const getGroups = (signal?: AbortSignal) =>
  getJson<{ groups: GroupInfo[] }>("/api/ball-studio/groups", { signal }).then((r) => r.groups)

export const getScene = (g: string, signal?: AbortSignal) => getJson<Scene>(`${base(g)}/scene`, { signal })

export const getTruth = (g: string, signal?: AbortSignal) => getJson<TruthResponse>(`${base(g)}/truth`, { signal })

/** `expectedUpdatedAt` is the optimistic-concurrency token (null = file did not exist at load). */
export const putTruth = (g: string, truth: TruthDoc, expectedUpdatedAt: string | null) =>
  putJson<PutTruthResponse>(`${base(g)}/truth`, { truth, expected_updated_at: expectedUpdatedAt })

export async function postSolve(g: string, truth: TruthDoc, signal?: AbortSignal): Promise<SolveResult> {
  const res = await fetch(`${base(g)}/solve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(truth),
    signal,
  })
  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as { detail?: unknown } | null
    throw new Error(describeDetail(body?.detail) || `Solve failed (HTTP ${res.status})`)
  }
  return (await res.json()) as SolveResult
}

export const postTriangulate = (g: string, req: TriangulateRequest) =>
  postJson<TriangulateResult>(`${base(g)}/triangulate`, req)

/** Readable text for a FastAPI `detail` (string, validation `{errors}` list, or `{message}`). */
export function describeDetail(detail: unknown): string {
  if (typeof detail === "string") return detail
  if (detail && typeof detail === "object") {
    const d = detail as { errors?: { path: string; message: string }[]; message?: string }
    if (d.errors?.length) return d.errors.map((e) => `${e.path}: ${e.message}`).join("; ")
    if (d.message) return d.message
  }
  return ""
}
