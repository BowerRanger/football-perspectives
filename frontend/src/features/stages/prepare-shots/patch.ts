// PATCH is not exported by lib/api, so this feature carries its own thin
// wrapper (same error semantics: the server `detail` becomes the message).
import { ApiError } from "@/lib/api"

export async function patchJson<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    let detail: unknown = null
    try {
      const j = await res.json()
      detail = j?.detail ?? j
    } catch {
      detail = res.statusText
    }
    throw new ApiError(res.status, detail, url)
  }
  return (await res.json()) as T
}
