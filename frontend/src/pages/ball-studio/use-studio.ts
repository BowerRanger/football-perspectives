import * as React from "react"
import { toast } from "sonner"

import { ApiError, errorMessage } from "@/lib/api"
import { useConfirm } from "@/hooks/use-dialogs"
import { describeDetail, postTriangulate, putTruth } from "./api"
import { ShotCameras, refToShot } from "./camera-model"
import {
  computePreview,
  constraintPayload,
  constraintSource,
  initialPickState,
  naturalCommit,
  pickReducer,
  projectInto,
  snapClick,
  type GoalPlaneRef,
  type Preview,
  type SnapResult,
} from "./pick-session"
import { EMPTY_PREVIEW } from "./pick-session"
import { RESIDUAL_BAD_PX, RESIDUAL_REJECT_PX } from "./palette"
import {
  addEvent,
  addKey,
  addObservation,
  removeEvent,
  removeKey,
  removeObservation,
  setStatus,
  withResolvedKeys,
  EMPTY_CONSTRAINT,
} from "./truth-doc"
import type {
  EventKind,
  GroupInfo,
  Observation,
  Scene,
  Selection,
  SegmentKind,
  TriangulateResult,
  TruthDoc,
  Vec2,
  Vec3,
} from "./types"
import { useSolver } from "./use-solver"
import { useTruthDoc } from "./use-truth-doc"
import { useViewVideos } from "./use-view-videos"
import type { EngineInput } from "./scene-engine"
import type { DrawKey, DrawTrack } from "./view-overlay-draw"
import type { ViewDrawProps } from "./view-well"

export type LayoutMode = "compare" | "focus"

export interface Layers {
  pipeline: boolean
  rays: boolean
  residuals: boolean
}

const JOINT_PICK_RADIUS_M = 1.6

/** Index of `frame` in a sorted frame list (or -1). */
function indexOfFrame(frames: readonly number[], frame: number): number {
  let lo = 0
  let hi = frames.length - 1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (frames[mid] === frame) return mid
    if (frames[mid] < frame) lo = mid + 1
    else hi = mid - 1
  }
  return -1
}

export function useStudio(group: GroupInfo, scene: Scene, initialTruth: TruthDoc) {
  const confirm = useConfirm()
  const docApi = useTruthDoc(initialTruth)
  const { doc } = docApi

  const cams = React.useMemo(() => {
    const ref = scene.shots.find((s) => s.shot_id === scene.reference_shot)
    const ordered = ref ? [ref, ...scene.shots.filter((s) => s !== ref)] : scene.shots
    return ordered.map((s) => new ShotCameras(s))
  }, [scene])
  const shots = React.useMemo(() => cams.map((c) => scene.shots.find((s) => s.shot_id === c.shotId)!), [cams, scene])

  const [minF, maxF] = scene.ref_frame_range
  const [frame, setFrameRaw] = React.useState(() => Math.min(maxF, Math.max(minF, doc.keys[0]?.frame ?? Math.round((minF + maxF) / 2))))
  const setFrame = React.useCallback((f: number) => setFrameRaw(Math.min(maxF, Math.max(minF, Math.round(f)))), [minF, maxF])

  const videos = useViewVideos({ shots, frame, fps: scene.fps, maxFrame: maxF, onFrame: setFrame })

  const solver = useSolver(group.group_id, doc, true)
  const solved = solver.result

  // ---- UI state ----------------------------------------------------------
  const [layout, setLayout] = React.useState<LayoutMode>("compare")
  const [activeView, setActiveView] = React.useState(0)
  const [layers, setLayers] = React.useState<Layers>({ pipeline: true, rays: true, residuals: true })
  const [loupe, setLoupe] = React.useState(false)
  const [resetSignal, setResetSignal] = React.useState(0)
  const [selection, setSelection] = React.useState<Selection>(null)
  const [saving, setSaving] = React.useState(false)
  const [pick, dispatchPick] = React.useReducer(pickReducer, initialPickState)
  const [hover, setHover] = React.useState<{ view: number; uv: Vec2 } | null>(null)
  const [live, setLive] = React.useState<TriangulateResult | null>(null)
  const [liveBusy, setLiveBusy] = React.useState(false)

  // Scrubbing drops pending picks (they belong to one instant).
  React.useEffect(() => {
    dispatchPick({ type: "frame", frame })
  }, [frame])

  // ---- scene lookups -----------------------------------------------------
  const planes: GoalPlaneRef[] = React.useMemo(
    () =>
      scene.goals.goal_planes?.length
        ? scene.goals.goal_planes.map((p) => ({ id: p.id, axis: p.axis, value: p.value }))
        : [
            { id: "goal_line_near", axis: "x" as const, value: scene.goals.goal_line_x_near },
            { id: "goal_line_far", axis: "x" as const, value: scene.goals.goal_line_x_far },
          ],
    [scene.goals],
  )

  const playerRows = React.useMemo(() => scene.players.map((p) => ({ p, idx: new Map(p.frames.map((f, i) => [f, i])) })), [scene.players])
  const joint = React.useCallback(
    (pid: string, bone: string, ref: number): Vec3 | null => {
      const row = playerRows.find((r) => r.p.player_id === pid)
      const i = row?.idx.get(ref)
      const j = i === undefined ? undefined : row?.p.joints[bone]?.[i]
      return j ? [j[0], j[1], j[2]] : null
    },
    [playerRows],
  )

  // ---- pending-pick geometry --------------------------------------------
  const preview: Preview = React.useMemo(() => {
    const picks = { ...pick.picks }
    if (hover && !picks[cams[hover.view]?.shotId] && (pick.mode === "constraint" || Object.keys(picks).length)) {
      picks[cams[hover.view].shotId] = hover.uv
    }
    if (!Object.keys(picks).length) return EMPTY_PREVIEW
    return computePreview({ frame, picks, cams, state: pick, planes, joint })
  }, [pick, hover, cams, frame, planes, joint])

  // Authoritative numbers from the server for the committed picks.
  const pickKey = JSON.stringify([frame, pick.picks, pick.mode, pick.constraint, pick.params])
  React.useEffect(() => {
    const ids = Object.keys(pick.picks)
    if (!ids.length || pick.mode === "observation") {
      setLive(null)
      setLiveBusy(false)
      return
    }
    const ctrl = new AbortController()
    setLiveBusy(true)
    const timer = window.setTimeout(() => {
      const req = buildRequest()
      postTriangulate(group.group_id, req)
        .then((r) => {
          if (!ctrl.signal.aborted) {
            setLive(r)
            setLiveBusy(false)
          }
        })
        .catch(() => {
          if (!ctrl.signal.aborted) {
            setLive(null)
            setLiveBusy(false)
          }
        })
    }, 80)
    return () => {
      window.clearTimeout(timer)
      ctrl.abort()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- pickKey serialises the inputs
  }, [pickKey, group.group_id])

  function buildRequest() {
    return {
      frame,
      observations: pickObservations(),
      constraint: constraintPayload(pick, planes),
    }
  }

  function pickObservations(): Observation[] {
    return Object.entries(pick.picks).map(([shotId, uv]) => {
      const cam = cams.find((c) => c.shotId === shotId)!
      return { shot_id: shotId, shot_frame: refToShot(frame, cam.frameOffset), uv }
    })
  }

  // ---- editing wrappers --------------------------------------------------
  const reviewedAsked = React.useRef(false)
  const editDoc = React.useCallback(
    async (fn: (d: TruthDoc) => TruthDoc): Promise<boolean> => {
      if (docApi.doc.meta.status === "reviewed" && !reviewedAsked.current) {
        const ok = await confirm({
          title: "Edit a reviewed group?",
          description: "This group is marked reviewed. Editing returns it to draft.",
          confirmLabel: "Edit and return to draft",
        })
        if (!ok) return false
        reviewedAsked.current = true
        docApi.edit((d) => setStatus(fn(d), "draft"))
        return true
      }
      docApi.edit(fn)
      return true
    },
    [confirm, docApi],
  )

  const commit = React.useCallback(
    async (forced?: "key" | "observation"): Promise<void> => {
      let kind = naturalCommit(pick)
      if (forced === "observation" && Object.keys(pick.picks).length) kind = "observation"
      if (forced === "key" && kind === "none") {
        toast.info("A key needs a second view or a constraint", {
          description: "Click the ball in the other view, or choose Ground, Height, Plane, Depth or Player first.",
        })
        return
      }
      if (kind === "none") {
        if (Object.keys(pick.picks).length) {
          toast.info("Click the ball in the other view or pick a constraint", {
            description: "Enter commits a triangulation (two views) or a constrained single-view key.",
          })
        }
        return
      }
      const obs = pickObservations()
      if (kind === "observation") {
        const ok = await editDoc((d) => obs.reduce((acc, o) => addObservation(acc, o), d))
        if (ok) dispatchPick({ type: "clear" })
        return
      }
      let result: TriangulateResult
      try {
        result = await postTriangulate(group.group_id, buildRequest())
      } catch (err) {
        toast.error("Could not triangulate", { description: errorMessage(err) })
        return
      }
      if (!result.xyz || (!result.ok && result.reason !== "residual_exceeds_limit")) {
        toast.error("This pick cannot become a key", { description: result.reason?.replace(/_/g, " ") ?? "The solver rejected it." })
        return
      }
      const worst = result.max_residual_px ?? Math.max(0, ...Object.values(result.residual_px))
      if (!result.ok || worst > RESIDUAL_REJECT_PX) {
        const go = await confirm({
          title: `Commit with residual ${worst.toFixed(1)} px?`,
          description: "The two picks disagree by more than the 15 px limit. A sync error or a mis-click is the usual cause.",
          confirmLabel: "Commit anyway",
          destructive: true,
        })
        if (!go) return
      }
      const source = kind === "triangulated" ? "triangulated" : constraintSource(pick.constraint)
      const constraint = { ...EMPTY_CONSTRAINT }
      if (kind === "constraint") {
        const c = constraintPayload(pick, planes)
        if (c?.height_m !== undefined) constraint.height_m = c.height_m
        if (c?.plane) constraint.plane = c.plane
        if (c?.depth_m !== undefined) constraint.depth_m = c.depth_m
        if (c?.player_id) constraint.player_id = c.player_id
        if (c?.bone) constraint.bone = c.bone
      }
      let newId = ""
      const ok = await editDoc((d) => {
        const r = addKey(d, { frame, xyz: result.xyz as Vec3, source, constraint, observations: obs, residual_px: result.residual_px })
        newId = r.id
        return r.doc
      })
      if (ok) {
        dispatchPick({ type: "clear" })
        setSelection({ type: "key", id: newId })
        if (!window.localStorage) return
        try {
          window.localStorage.setItem("ball-studio.onboarded", "1")
        } catch {
          /* private mode */
        }
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reads latest pick/doc through closures
    [pick, frame, cams, planes, group.group_id, editDoc, confirm],
  )

  // Auto-commit a clean two-view triangulation.
  const committing = React.useRef("")
  React.useEffect(() => {
    if (!pick.autoCommit || pick.mode !== "triangulate" || liveBusy || !live) return
    if (Object.keys(pick.picks).length < 2 || !live.ok || live.source !== "triangulated") return
    if ((live.max_residual_px ?? 99) > RESIDUAL_BAD_PX) return
    if (committing.current === pickKey) return
    committing.current = pickKey
    void commit()
  }, [live, liveBusy, pick, pickKey, commit])

  const deleteSelected = React.useCallback(() => {
    const sel = selection
    if (!sel) return
    void editDoc((d) => {
      if (sel.type === "key") return removeKey(d, sel.id)
      if (sel.type === "event") return removeEvent(d, sel.index)
      if (sel.type === "observation") return removeObservation(d, sel.index)
      return d
    }).then((ok) => ok && setSelection(null))
  }, [selection, editDoc])

  // ---- derived solver views ---------------------------------------------
  const denseTrack: DrawTrack | null = React.useMemo(
    () => (solved ? { frames: solved.dense.frames, xyz: solved.dense.xyz, kind: solved.dense.kind } : null),
    [solved],
  )

  const keyKinds = React.useMemo(() => {
    const m = new Map<string, SegmentKind>()
    if (!solved) return m
    for (const s of solved.segments) {
      m.set(s.from, s.kind)
      if (!m.has(s.to)) m.set(s.to, s.kind)
    }
    return m
  }, [solved])

  const solvedKeyById = React.useMemo(() => new Map((solved?.keys ?? []).map((k) => [k.id, k])), [solved])

  const keyXyz = React.useCallback((id: string): Vec3 => solvedKeyById.get(id)?.xyz ?? doc.keys.find((k) => k.id === id)!.xyz, [solvedKeyById, doc.keys])

  const denseAt = React.useCallback(
    (f: number): Vec3 | null => {
      if (!denseTrack) return null
      const i = indexOfFrame(denseTrack.frames, f)
      const p = i < 0 ? null : denseTrack.xyz[i]
      return p ? [p[0], p[1], p[2]] : null
    },
    [denseTrack],
  )

  const pipelineFor = React.useCallback(
    (shotId: string): DrawTrack | null => {
      const t = scene.pipeline_tracks.find((p) => p.shot_id === shotId) ?? scene.pipeline_tracks[0]
      return t ? { frames: t.frames, xyz: t.xyz } : null
    },
    [scene.pipeline_tracks],
  )

  const resolveSnap = React.useCallback(
    (viewIndex: number, uv: Vec2, radius: number, alt: boolean): SnapResult => {
      const lines = preview.epipolar[cams[viewIndex].shotId]
      // Only snap to lines from picks in other views.
      return snapClick(uv, lines, pick.snap, radius, alt)
    },
    [preview.epipolar, cams, pick.snap],
  )

  /** Everything the canvas needs for one view. */
  const drawPropsFor = (vi: number): ViewDrawProps => {
    const cam = cams[vi]
    const shot = shots[vi]
    const shotFrame = refToShot(frame, cam.frameOffset)
    const fc = cam.at(shotFrame)
    const keys: DrawKey[] = doc.keys.map((k) => {
      const ob = k.observations.find((o) => o.shot_id === shot.shot_id && o.shot_frame === refToShot(k.frame, cam.frameOffset))
      const sObs = solved?.observations.find((o) => o.kind === "key" && o.key_id === k.id && o.shot_id === shot.shot_id)
      return {
        id: k.id,
        frame: k.frame,
        xyz: keyXyz(k.id),
        source: k.source,
        selected: selection?.type === "key" && selection.id === k.id,
        pickedUv: ob ? ob.uv : null,
        projectedUv: sObs?.projected_uv ?? null,
        kind: keyKinds.get(k.id) ?? "flight",
      }
    })
    const soft = doc.observations
      .map((o, i) => ({ o, i }))
      .filter(({ o }) => o.shot_id === shot.shot_id && o.shot_frame === shotFrame)
      .map(({ o, i }) => ({
        uv: o.uv,
        projectedUv: solved?.observations.find((s) => s.kind === "soft" && s.index === i)?.projected_uv ?? null,
        selected: selection?.type === "observation" && selection.index === i,
      }))
    const epipolar = (preview.epipolar[shot.shot_id] ?? []).map((e) => ({
      fromView: cams.findIndex((c) => c.shotId === e.from),
      epi: e.epi,
    }))
    const anchors = scene.pipeline_anchors.filter((a) => a.shot_id === shot.shot_id && a.shot_frame === shotFrame).map((a) => a.uv)
    return {
      viewIndex: vi,
      cam: fc,
      frame,
      dense: denseTrack,
      stale: solver.stale,
      pipeline: pipelineFor(shot.shot_id),
      showPipeline: layers.pipeline,
      showResiduals: layers.residuals,
      showRays: layers.rays,
      keys,
      soft,
      pipelineAnchors: anchors,
      pending: pick.picks[shot.shot_id] ?? null,
      ghost: preview.ghost,
      epipolar,
      selectedFrame: null,
    }
  }

  const attention = React.useMemo(() => {
    if (!solved) return [] as number[]
    const fs = new Set<number>()
    for (const f of solved.flags) {
      if (f.frame !== undefined) fs.add(f.frame)
      else if (f.ref?.segment !== undefined) {
        const s = solved.segments[f.ref.segment]
        if (s) fs.add(s.frame_range[0])
      }
    }
    for (const s of solved.segments) if (s.n_soft_obs === 0 && s.kind === "flight") fs.add(s.frame_range[0])
    return [...fs].sort((a, b) => a - b)
  }, [solved])

  const ball = denseAt(frame)

  const engineViews = cams.map((c, i) => ({ index: i, cam: c.atRef(frame), imageSize: c.imageSize }))
  const engineKeys = React.useMemo(
    () =>
      doc.keys.map((k) => ({
        id: k.id,
        xyz: keyXyz(k.id),
        kind: keyKinds.get(k.id) ?? ("flight" as SegmentKind),
        selected: selection?.type === "key" && selection.id === k.id,
      })),
    [doc.keys, keyXyz, keyKinds, selection],
  )
  const pipelineMain = React.useMemo(() => pipelineFor(scene.reference_shot), [pipelineFor, scene.reference_shot])
  const depthRay = pick.mode === "constraint" && pick.constraint === "depth" ? preview.rays[0] : undefined
  const engineInput: EngineInput = {
    frame,
    views: engineViews,
    dense: denseTrack,
    stale: solver.stale,
    keys: engineKeys,
    pipeline: pipelineMain,
    showPipeline: layers.pipeline,
    showRays: layers.rays,
    showFrusta: true,
    rays: preview.rays.map((r) => ({
      viewIndex: cams.findIndex((c) => c.shotId === r.shotId),
      origin: r.ray.origin,
      dir: r.ray.dir,
      reach: preview.ghost ? Math.hypot(preview.ghost[0] - r.ray.origin[0], preview.ghost[1] - r.ray.origin[1], preview.ghost[2] - r.ray.origin[2]) : null,
    })),
    ghost: preview.ghost,
    skew: preview.skew,
    ball,
    depthHandle: depthRay
      ? { viewIndex: cams.findIndex((c) => c.shotId === depthRay.shotId), origin: depthRay.ray.origin, dir: depthRay.ray.dir, depth: pick.params.depthM }
      : null,
  }

  // ---- events ------------------------------------------------------------
  const addEventAtPlayhead = React.useCallback(
    (kind: EventKind) => {
      let player: string | null = null
      let bone: string | null = null
      if ((kind === "touch" || kind === "keeper_save") && ball) {
        let best = JOINT_PICK_RADIUS_M
        for (const { p, idx } of playerRows) {
          const i = idx.get(frame)
          if (i === undefined) continue
          for (const [b, arr] of Object.entries(p.joints)) {
            if (b === "pelvis") continue
            const j = arr[i]
            if (!j) continue
            const d = Math.hypot(j[0] - ball[0], j[1] - ball[1], j[2] - ball[2])
            if (d < best) {
              best = d
              player = p.player_id
              bone = b
            }
          }
        }
      }
      let index = 0
      void editDoc((d) => {
        const r = addEvent(d, { frame, kind, player_id: player, bone })
        index = r.index
        return r.doc
      }).then((ok) => ok && setSelection({ type: "event", index }))
    },
    [ball, frame, playerRows, editDoc],
  )

  // ---- save --------------------------------------------------------------
  const save = React.useCallback(async (): Promise<boolean> => {
    if (saving) return false
    setSaving(true)
    try {
      const resolved = solved && !solver.stale && solver.status === "solved" ? withResolvedKeys(docApi.doc, solved.keys) : docApi.doc
      let token = docApi.token
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          const res = await putTruth(group.group_id, resolved, token)
          docApi.markSaved({ ...resolved, meta: { ...resolved.meta, updated_at: res.updated_at } }, res.updated_at)
          toast.success("Ball truth saved", {
            description: res.solve_ok ? `${resolved.keys.length} keys, ${res.n_flags} flags` : "Saved, but the solver could not fully solve it.",
          })
          return true
        } catch (err) {
          if (err instanceof ApiError && err.status === 409) {
            const d = err.detail as { current_updated_at?: string | null } | null
            const go = await confirm({
              title: "The saved truth changed since you opened it",
              description: "Another tab or session wrote this file. Overwrite it with your edits?",
              confirmLabel: "Overwrite theirs",
              cancelLabel: "Keep editing",
              destructive: true,
            })
            if (!go) return false
            token = d?.current_updated_at ?? null
            continue
          }
          if (err instanceof ApiError && err.status === 422) {
            toast.error("The server rejected this truth", { description: describeDetail(err.detail) || err.message })
            return false
          }
          throw err
        }
      }
      return false
    } catch (err) {
      toast.error("Could not save ball truth", { description: errorMessage(err) })
      return false
    } finally {
      setSaving(false)
    }
  }, [saving, solved, solver.stale, solver.status, docApi, group.group_id, confirm])

  return {
    group,
    scene,
    cams,
    shots,
    frame,
    setFrame,
    range: [minF, maxF] as const,
    videos,
    docApi,
    solver,
    layout,
    setLayout,
    activeView,
    setActiveView,
    layers,
    setLayers,
    loupe,
    setLoupe,
    resetSignal,
    resetZoom: () => setResetSignal((n) => n + 1),
    selection,
    setSelection,
    saving,
    save,
    pick,
    dispatchPick,
    hover,
    setHover,
    preview,
    live,
    liveBusy,
    commit,
    deleteSelected,
    editDoc,
    addEventAtPlayhead,
    planes,
    joint,
    resolveSnap,
    drawPropsFor,
    engineInput,
    attention,
    denseAt,
    keyKinds,
    solvedKeyById,
    projectInto,
  }
}

export type Studio = ReturnType<typeof useStudio>
