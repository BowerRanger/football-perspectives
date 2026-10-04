// Thin fetch helpers shared by every panel. Errors surface as `ApiError`
// carrying the HTTP status and the server's `detail` so callers can branch on
// 404/409 and `errorMessage` can print something readable.

export class ApiError extends Error {
  status: number
  detail: unknown

  constructor(status: number, detail: unknown, url: string) {
    super(ApiError.describe(status, detail, url))
    this.name = "ApiError"
    this.status = status
    this.detail = detail
  }

  static describe(status: number, detail: unknown, url: string): string {
    if (typeof detail === "string" && detail) return detail
    if (detail && typeof detail === "object") {
      const d = detail as { message?: unknown }
      return typeof d.message === "string" ? d.message : JSON.stringify(detail)
    }
    return `${url} failed (HTTP ${status})`
  }
}

async function toApiError(res: Response, url: string): Promise<ApiError> {
  let detail: unknown = null
  try {
    const body = await res.json()
    detail = body?.detail ?? body
  } catch {
    try {
      detail = await res.text()
    } catch {
      detail = null
    }
  }
  return new ApiError(res.status, detail, url)
}

/** GET JSON; throws `ApiError` on any non-2xx. */
export async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) throw await toApiError(res, url)
  return (await res.json()) as T
}

/** GET JSON; a 404 resolves to `null` (other failures still throw). */
export async function getJsonOr404<T>(url: string, init?: RequestInit): Promise<T | null> {
  const res = await fetch(url, init)
  if (res.status === 404) return null
  if (!res.ok) throw await toApiError(res, url)
  return (await res.json()) as T
}

/** GET JSON; any failure (network or non-2xx) resolves to `null`. */
export async function getJsonOrNull<T>(url: string, init?: RequestInit): Promise<T | null> {
  try {
    const res = await fetch(url, init)
    return res.ok ? ((await res.json()) as T) : null
  } catch {
    return null
  }
}

async function sendJson<T>(method: string, url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!res.ok) throw await toApiError(res, url)
  const text = await res.text()
  return (text ? JSON.parse(text) : null) as T
}

export const postJson = <T = unknown>(url: string, body?: unknown) => sendJson<T>("POST", url, body)
export const putJson = <T = unknown>(url: string, body?: unknown) => sendJson<T>("PUT", url, body)
export const deleteJson = <T = unknown>(url: string, body?: unknown) => sendJson<T>("DELETE", url, body)

/** POST multipart form data (uploads); the browser sets the boundary header. */
export async function postForm<T = unknown>(url: string, form: FormData): Promise<T> {
  const res = await fetch(url, { method: "POST", body: form })
  if (!res.ok) throw await toApiError(res, url)
  return (await res.json()) as T
}

export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

/** Query string from a record; null/undefined/"" entries are dropped. Returns "" or "?a=b". */
export const qs = (params: Record<string, string | number | boolean | null | undefined>): string => {
  const sp = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v != null && v !== "") sp.set(k, String(v))
  }
  const s = sp.toString()
  return s ? `?${s}` : ""
}
