import * as React from "react"

import { toast } from "sonner"

import { errorMessage, getJson } from "@/lib/api"

import { patchJson } from "./patch"
import type { FeaturesMap, Manifest, ShotUpdate, SyncMap } from "./types"

const EMPTY_SYNC: SyncMap = { groups: [] }

export interface ShotsData {
  manifest: Manifest | null
  features: FeaturesMap
  sync: SyncMap
  loading: boolean
  error: string | null
  /** Bumps when state was replaced wholesale; sync editors reseed on it. */
  revision: number
  reload: () => Promise<void>
  /** Refetch only the sync map (no editor reseed). */
  reloadSyncQuiet: () => Promise<void>
  patchShots: (updates: ShotUpdate[]) => Promise<void>
}

/**
 * Loads manifest + features + sync and keeps them fresh after mutations.
 * PATCH /api/shots/bulk returns the updated manifest, so a mutation only
 * needs one extra fetch (the pruned sync map).
 */
export function useShotsData(): ShotsData {
  const [manifest, setManifest] = React.useState<Manifest | null>(null)
  const [features, setFeatures] = React.useState<FeaturesMap>({})
  const [sync, setSync] = React.useState<SyncMap>(EMPTY_SYNC)
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  const [revision, setRevision] = React.useState(0)

  const reload = React.useCallback(async () => {
    try {
      const [m, f, s] = await Promise.all([
        getJson<Manifest>("/api/shots/manifest"),
        getJson<FeaturesMap>("/api/shots/features"),
        getJson<SyncMap>("/api/sync"),
      ])
      setManifest(m)
      setFeatures(f)
      setSync(s)
      setError(null)
      setRevision((r) => r + 1)
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setLoading(false)
    }
  }, [])

  React.useEffect(() => {
    void reload()
  }, [reload])

  const reloadSyncQuiet = React.useCallback(async () => {
    try {
      setSync(await getJson<SyncMap>("/api/sync"))
    } catch (err) {
      toast.error("Could not refresh the sync map", { description: errorMessage(err) })
    }
  }, [])

  const patchShots = React.useCallback(async (updates: ShotUpdate[]) => {
    const updated = await patchJson<Manifest>("/api/shots/bulk", { updates })
    setManifest(updated)
    try {
      setSync(await getJson<SyncMap>("/api/sync"))
    } catch (err) {
      // The edit itself saved; only the pruned sync map is stale.
      toast.error("Saved, but could not refresh the sync map", { description: errorMessage(err) })
    }
    setRevision((r) => r + 1)
  }, [])

  return { manifest, features, sync, loading, error, revision, reload, reloadSyncQuiet, patchShots }
}
