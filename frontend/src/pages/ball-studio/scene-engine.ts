// Plain three.js 3-D well for Ball Studio (same imperative pattern as the
// viewer's ViewerEngine; pitch drawing is reused, players are drawn as
// joint dots + stick limbs from the scene payload - no SMPL loader).
import * as THREE from "three"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"

import { buildPitch, disposeTree } from "@/pages/viewer/pitch"
import { pitchToThree } from "@/pages/viewer/smpl"
import { unproject, rayAt, type FrameCam } from "./camera-model"
import { PIPELINE_GHOST, SEGMENT_STYLE, viewColour } from "./palette"
import type { Scene, ScenePlayer, SegmentKind, Vec2, Vec3 } from "./types"

export type CameraPreset = "overview" | "top" | "view-a" | "view-b" | "goal-near" | "goal-far" | "follow" | "free"

export interface EngineTrack {
  frames: readonly number[]
  xyz: readonly (readonly number[] | null)[]
  kind?: readonly SegmentKind[]
}

export interface EngineView {
  index: number
  cam: FrameCam | null
  imageSize: Vec2
}

export interface EngineInput {
  frame: number
  views: readonly EngineView[]
  dense: EngineTrack | null
  stale: boolean
  keys: readonly { id: string; xyz: Vec3; kind: SegmentKind; selected: boolean }[]
  pipeline: EngineTrack | null
  showPipeline: boolean
  showRays: boolean
  showFrusta: boolean
  rays: readonly { viewIndex: number; origin: Vec3; dir: Vec3; reach: number | null }[]
  ghost: Vec3 | null
  skew: { a: Vec3; b: Vec3; gap_m: number } | null
  ball: Vec3 | null
  depthHandle: { viewIndex: number; origin: Vec3; dir: Vec3; depth: number } | null
}

const v3 = (p: readonly number[]): THREE.Vector3 => {
  const t = pitchToThree([p[0], p[1], p[2]])
  return new THREE.Vector3(t[0], t[1], t[2])
}

function labelSprite(): { sprite: THREE.Sprite; set: (text: string, colour: string) => void } {
  const canvas = document.createElement("canvas")
  canvas.width = 256
  canvas.height = 64
  const tex = new THREE.CanvasTexture(canvas)
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }))
  sprite.scale.set(6, 1.5, 1)
  sprite.renderOrder = 10
  let last = ""
  return {
    sprite,
    set(text, colour) {
      const key = text + colour
      if (key === last) return
      last = key
      const ctx = canvas.getContext("2d")!
      ctx.clearRect(0, 0, 256, 64)
      ctx.font = '600 30px "Geist Mono Variable", ui-monospace, monospace'
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.lineWidth = 6
      ctx.strokeStyle = "rgba(0,0,0,0.8)"
      ctx.strokeText(text, 128, 32)
      ctx.fillStyle = colour
      ctx.fillText(text, 128, 32)
      tex.needsUpdate = true
    },
  }
}

const LIMBS: readonly (readonly [string, string])[] = [
  ["pelvis", "chest"],
  ["chest", "head"],
  ["pelvis", "l_knee"],
  ["l_knee", "l_foot"],
  ["pelvis", "r_knee"],
  ["r_knee", "r_foot"],
  ["chest", "l_shoulder"],
  ["chest", "r_shoulder"],
  ["l_shoulder", "l_hand"],
  ["r_shoulder", "r_hand"],
]

export class StudioEngine {
  private readonly renderer: THREE.WebGLRenderer
  private readonly scene = new THREE.Scene()
  private readonly camera = new THREE.PerspectiveCamera(45, 1, 0.1, 600)
  private readonly controls: OrbitControls
  private readonly ro: ResizeObserver
  private readonly pitch = buildPitch()
  private readonly dyn = new THREE.Group()
  private readonly trackGroup = new THREE.Group()
  private readonly keyGroup = new THREE.Group()
  private readonly pipeLine: THREE.Line
  private readonly playersGroup = new THREE.Group()
  private readonly frusta = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true }))
  private readonly rayGroup = new THREE.Group()
  private readonly ballMesh: THREE.Mesh
  private readonly stem: THREE.Line
  private readonly dropRing: THREE.Mesh
  private readonly heightLabel = labelSprite()
  private readonly gapLabel = labelSprite()
  private readonly ghostMesh: THREE.Mesh
  private readonly skewLine: THREE.Line
  private readonly handle: THREE.Mesh
  private readonly raycaster = new THREE.Raycaster()
  private players: ScenePlayer[] = []
  private playerIndex: Map<number, number>[] = []
  private playerObjs: { dots: THREE.Points; limbs: THREE.LineSegments }[] = []
  private lastDense: unknown = null
  private lastDenseStale = false
  private lastKeys: unknown = null
  private lastPipeline: unknown = null
  private preset: CameraPreset = "overview"
  private input: EngineInput | null = null
  private dirty = true
  private raf = 0
  private disposed = false
  private dragging = false
  private onDepth: ((m: number) => void) | null = null
  private onPresetChange: ((p: CameraPreset) => void) | null = null
  private followPrev: THREE.Vector3 | null = null
  private pristine = true

  private readonly container: HTMLElement

  constructor(container: HTMLElement) {
    this.container = container
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    this.renderer.setClearColor(0x000000, 0)
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    this.renderer.domElement.className = "block size-full touch-none"
    container.appendChild(this.renderer.domElement)

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.8))
    const dir = new THREE.DirectionalLight(0xffffff, 0.5)
    dir.position.set(20, 40, 20)
    this.scene.add(dir, this.pitch, this.dyn)
    this.dyn.add(this.trackGroup, this.keyGroup, this.playersGroup, this.frusta, this.rayGroup)

    const pipeMat = new THREE.LineDashedMaterial({
      color: PIPELINE_GHOST.colour,
      transparent: true,
      opacity: PIPELINE_GHOST.alpha,
      dashSize: 0.6,
      gapSize: 0.4,
    })
    this.pipeLine = new THREE.Line(new THREE.BufferGeometry(), pipeMat)
    this.dyn.add(this.pipeLine)

    this.ballMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.3, 16, 16),
      new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x444444 }),
    )
    this.stem = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 }))
    this.dropRing = new THREE.Mesh(
      new THREE.RingGeometry(0.35, 0.5, 24),
      new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.8 }),
    )
    this.dropRing.rotation.x = -Math.PI / 2
    this.ghostMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.4, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.8 }),
    )
    this.skewLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xf87171, depthTest: false }))
    this.handle = new THREE.Mesh(
      new THREE.SphereGeometry(0.7, 14, 14),
      new THREE.MeshBasicMaterial({ color: 0xfbbf24, depthTest: false, transparent: true, opacity: 0.9 }),
    )
    this.handle.renderOrder = 9
    for (const o of [this.ballMesh, this.stem, this.dropRing, this.ghostMesh, this.skewLine, this.handle, this.heightLabel.sprite, this.gapLabel.sprite]) {
      o.visible = false
      this.dyn.add(o)
    }

    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.addEventListener("change", this.markDirty)
    this.controls.addEventListener("start", () => {
      this.pristine = false
      if (this.preset !== "follow") this.setPreset("free", false)
    })
    this.setPreset("overview")

    const el = this.renderer.domElement
    el.addEventListener("pointerdown", this.onPointerDown)
    el.addEventListener("pointermove", this.onPointerMove)
    window.addEventListener("pointerup", this.onPointerUp)

    this.ro = new ResizeObserver(this.resize)
    this.ro.observe(container)
    this.resize()
    this.raf = requestAnimationFrame(this.tick)
  }

  setCallbacks(cb: { onDepth?: (m: number) => void; onPresetChange?: (p: CameraPreset) => void }): void {
    this.onDepth = cb.onDepth ?? null
    this.onPresetChange = cb.onPresetChange ?? null
  }

  private readonly markDirty = () => {
    this.dirty = true
  }

  private readonly resize = () => {
    const w = Math.max(1, this.container.clientWidth)
    const h = Math.max(1, this.container.clientHeight)
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(w, h, false)
    if (this.preset === "overview" && this.pristine) this.applyPreset("overview")
    this.dirty = true
  }

  private readonly tick = () => {
    if (this.disposed) return
    this.raf = requestAnimationFrame(this.tick)
    const moved = this.controls.update()
    if (!this.dirty && !moved) return
    this.dirty = false
    this.renderer.render(this.scene, this.camera)
  }

  setScene(scene: Scene): void {
    this.players = scene.players
    this.playerIndex = scene.players.map((p) => new Map(p.frames.map((f, i) => [f, i])))
    this.playersGroup.clear()
    this.playerObjs = scene.players.map(() => {
      const dots = new THREE.Points(
        new THREE.BufferGeometry(),
        new THREE.PointsMaterial({ color: 0x94a3b8, size: 0.45, sizeAttenuation: true }),
      )
      const limbs = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.75 }))
      this.playersGroup.add(dots, limbs)
      return { dots, limbs }
    })
    this.dirty = true
  }

  setPreset(p: CameraPreset, apply = true): void {
    this.preset = p
    this.pristine = p === "overview"
    this.followPrev = null
    this.onPresetChange?.(p)
    if (apply) this.applyPreset(p)
    this.dirty = true
  }

  private applyPreset(p: CameraPreset): void {
    const c = this.controls
    const set = (pos: [number, number, number], target: [number, number, number]) => {
      this.camera.position.set(...pos)
      c.target.set(...target)
      this.camera.fov = 45
      this.camera.updateProjectionMatrix()
      c.update()
    }
    switch (p) {
      case "overview": {
        // Fit the whole pitch to the well: distance from the narrower of width / height at a 40 degree tilt.
        const half = (this.camera.fov * Math.PI) / 360
        const d = Math.max(62 / (Math.tan(half) * Math.max(0.5, this.camera.aspect)), 45 / Math.tan(half)) * 1.12
        set([52.5, d * 0.64, -34 + d * 0.77], [52.5, 0, -34])
        break
      }
      case "top":
        set([52.5, 120, -33.9], [52.5, 0, -34])
        break
      case "goal-near":
        set([-18, 10, -34], [8, 1, -34])
        break
      case "goal-far":
        set([123, 10, -34], [97, 1, -34])
        break
      case "view-a":
      case "view-b": {
        const view = this.input?.views[p === "view-a" ? 0 : 1]
        if (view?.cam) this.lookThrough(view.cam, view.imageSize)
        break
      }
      case "follow": {
        const b = this.input?.ball
        const t = b ? v3(b) : new THREE.Vector3(52.5, 0, -34)
        set([t.x - 14, t.y + 10, t.z + 14], [t.x, t.y, t.z])
        this.followPrev = t.clone()
        break
      }
      default:
    }
  }

  private lookThrough(cam: FrameCam, size: Vec2): void {
    const ray = unproject(cam, size[0] / 2, size[1] / 2)
    const pos = v3(cam.C)
    const target = v3(rayAt(ray, 40))
    this.camera.position.copy(pos)
    this.controls.target.copy(target)
    this.camera.fov = Math.min(100, Math.max(15, (2 * Math.atan(size[1] / 2 / cam.fy) * 180) / Math.PI))
    this.camera.updateProjectionMatrix()
    this.controls.update()
  }

  update(inp: EngineInput): void {
    this.input = inp
    this.syncTrack(inp)
    this.syncKeys(inp)
    this.syncPipeline(inp)
    this.syncPlayers(inp)
    this.syncFrusta(inp)
    this.syncRays(inp)
    this.syncBall(inp)
    if (this.preset === "follow" && inp.ball) {
      const t = v3(inp.ball)
      if (this.followPrev) {
        const d = t.clone().sub(this.followPrev)
        this.camera.position.add(d)
      }
      this.controls.target.copy(t)
      this.followPrev = t
    } else if ((this.preset === "view-a" || this.preset === "view-b") && !this.dragging) {
      const view = inp.views[this.preset === "view-a" ? 0 : 1]
      if (view?.cam) this.lookThrough(view.cam, view.imageSize)
    }
    this.dirty = true
  }

  private syncTrack(inp: EngineInput): void {
    if (inp.dense === this.lastDense && inp.stale === this.lastDenseStale) return
    this.lastDense = inp.dense
    this.lastDenseStale = inp.stale
    disposeTree(this.trackGroup)
    this.trackGroup.clear()
    const d = inp.dense
    if (!d || d.frames.length < 2) return
    // Runs of one segment kind, each sharing its boundary point with the next.
    let run: THREE.Vector3[] = []
    let kind: SegmentKind | null = null
    const flush = () => {
      if (run.length >= 2 && kind) {
        const st = SEGMENT_STYLE[kind]
        if (st.pattern !== "ring") {
          const curve = new THREE.CatmullRomCurve3(run)
          const geo = new THREE.TubeGeometry(curve, Math.max(2, run.length * 2), 0.07 * (st.width / 2), 6, false)
          const mat = new THREE.MeshBasicMaterial({ color: st.colour, transparent: inp.stale, opacity: inp.stale ? 0.45 : 1 })
          this.trackGroup.add(new THREE.Mesh(geo, mat))
        }
      }
    }
    for (let i = 0; i < d.frames.length; i++) {
      const p = d.xyz[i]
      if (!p) continue
      const k = d.kind?.[i] ?? "flight"
      const pt = v3(p)
      if (kind !== null && k !== kind) {
        run.push(pt)
        flush()
        run = []
      }
      kind = k
      run.push(pt)
    }
    flush()
  }

  private syncKeys(inp: EngineInput): void {
    if (inp.keys === this.lastKeys) return
    this.lastKeys = inp.keys
    disposeTree(this.keyGroup)
    this.keyGroup.clear()
    for (const k of inp.keys) {
      const colour = SEGMENT_STYLE[k.kind].colour
      const m = new THREE.Mesh(new THREE.OctahedronGeometry(k.selected ? 0.55 : 0.4), new THREE.MeshBasicMaterial({ color: colour }))
      m.position.copy(v3(k.xyz))
      this.keyGroup.add(m)
      if (k.selected) {
        const ring = new THREE.Mesh(new THREE.RingGeometry(0.8, 0.95, 24), new THREE.MeshBasicMaterial({ color: 0x60a5fa, side: THREE.DoubleSide, depthTest: false }))
        ring.position.copy(m.position)
        ring.lookAt(this.camera.position)
        this.keyGroup.add(ring)
      }
    }
  }

  private syncPipeline(inp: EngineInput): void {
    this.pipeLine.visible = inp.showPipeline && !!inp.pipeline
    if (inp.pipeline === this.lastPipeline) return
    this.lastPipeline = inp.pipeline
    const pts: THREE.Vector3[] = []
    for (const p of inp.pipeline?.xyz ?? []) if (p) pts.push(v3(p))
    this.pipeLine.geometry.dispose()
    this.pipeLine.geometry = new THREE.BufferGeometry().setFromPoints(pts)
    this.pipeLine.computeLineDistances()
  }

  private syncPlayers(inp: EngineInput): void {
    this.players.forEach((p, i) => {
      const row = this.playerIndex[i].get(inp.frame)
      const obj = this.playerObjs[i]
      if (row === undefined) {
        obj.dots.visible = false
        obj.limbs.visible = false
        return
      }
      obj.dots.visible = true
      obj.limbs.visible = true
      const names = Object.keys(p.joints)
      const pos: number[] = []
      const at = (name: string): THREE.Vector3 | null => (p.joints[name]?.[row] ? v3(p.joints[name][row]) : null)
      for (const n of names) {
        const j = at(n)
        if (j) pos.push(j.x, j.y, j.z)
      }
      obj.dots.geometry.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3))
      const seg: number[] = []
      for (const [a, b] of LIMBS) {
        const ja = at(a)
        const jb = at(b)
        if (ja && jb) seg.push(ja.x, ja.y, ja.z, jb.x, jb.y, jb.z)
      }
      obj.limbs.geometry.setAttribute("position", new THREE.Float32BufferAttribute(seg, 3))
    })
  }

  private syncFrusta(inp: EngineInput): void {
    this.frusta.visible = inp.showFrusta
    const pos: number[] = []
    const col: number[] = []
    for (const v of inp.views) {
      if (!v.cam) continue
      const c = new THREE.Color(viewColour(v.index))
      const C = v3(v.cam.C)
      const corners = [[0, 0], [v.imageSize[0], 0], [v.imageSize[0], v.imageSize[1]], [0, v.imageSize[1]]].map(([u, w]) =>
        v3(rayAt(unproject(v.cam!, u, w), 14)),
      )
      const push = (a: THREE.Vector3, b: THREE.Vector3) => {
        pos.push(a.x, a.y, a.z, b.x, b.y, b.z)
        col.push(c.r, c.g, c.b, c.r, c.g, c.b)
      }
      corners.forEach((p, i) => {
        push(C, p)
        push(p, corners[(i + 1) % 4])
      })
    }
    this.frusta.geometry.dispose()
    const g = new THREE.BufferGeometry()
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3))
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3))
    this.frusta.geometry = g
  }

  private syncRays(inp: EngineInput): void {
    disposeTree(this.rayGroup)
    this.rayGroup.clear()
    if (inp.showRays) {
      for (const r of inp.rays) {
        const colour = viewColour(r.viewIndex)
        const reach = r.reach ?? 60
        const a = v3(r.origin)
        const b = v3(rayAt({ origin: r.origin, dir: r.dir }, reach))
        const c = v3(rayAt({ origin: r.origin, dir: r.dir }, 200))
        this.rayGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]), new THREE.LineBasicMaterial({ color: colour })))
        this.rayGroup.add(
          new THREE.Line(new THREE.BufferGeometry().setFromPoints([b, c]), new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: 0.25 })),
        )
      }
    }
    // skew gap
    const sk = inp.skew
    this.skewLine.visible = !!sk && sk.gap_m > 0.02
    this.gapLabel.sprite.visible = this.skewLine.visible
    if (sk && this.skewLine.visible) {
      const a = v3(sk.a)
      const b = v3(sk.b)
      this.skewLine.geometry.dispose()
      this.skewLine.geometry = new THREE.BufferGeometry().setFromPoints([a, b])
      this.gapLabel.set(`gap ${Math.round(sk.gap_m * 100)} cm`, "#f87171")
      this.gapLabel.sprite.position.copy(a.clone().lerp(b, 0.5)).add(new THREE.Vector3(0, 1.2, 0))
    }
    // ghost
    this.ghostMesh.visible = !!inp.ghost
    if (inp.ghost) this.ghostMesh.position.copy(v3(inp.ghost))
    // depth handle
    const h = inp.depthHandle
    this.handle.visible = !!h
    if (h) this.handle.position.copy(v3(rayAt({ origin: h.origin, dir: h.dir }, h.depth)))
  }

  private syncBall(inp: EngineInput): void {
    const b = inp.ball
    this.ballMesh.visible = !!b
    this.stem.visible = !!b
    this.dropRing.visible = !!b
    this.heightLabel.sprite.visible = !!b
    if (!b) return
    const p = v3(b)
    this.ballMesh.position.copy(p)
    const ground = new THREE.Vector3(p.x, 0.02, p.z)
    this.stem.geometry.dispose()
    this.stem.geometry = new THREE.BufferGeometry().setFromPoints([p, ground])
    this.dropRing.position.copy(ground)
    this.heightLabel.set(`${b[2].toFixed(2)} m`, "#ffffff")
    this.heightLabel.sprite.position.copy(p).add(new THREE.Vector3(0, 1.4, 0))
  }

  // ---- depth handle drag --------------------------------------------------

  private ndc(e: PointerEvent): THREE.Vector2 {
    const r = this.renderer.domElement.getBoundingClientRect()
    return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
  }

  private readonly onPointerDown = (e: PointerEvent) => {
    if (!this.handle.visible || e.button !== 0) return
    this.raycaster.setFromCamera(this.ndc(e), this.camera)
    if (this.raycaster.intersectObject(this.handle).length) {
      this.dragging = true
      this.controls.enabled = false
      this.renderer.domElement.setPointerCapture(e.pointerId)
    }
  }

  private readonly onPointerMove = (e: PointerEvent) => {
    const h = this.input?.depthHandle
    if (!this.dragging || !h) return
    this.raycaster.setFromCamera(this.ndc(e), this.camera)
    const a = v3(h.origin)
    const b = v3(rayAt({ origin: h.origin, dir: h.dir }, 250))
    const onSeg = new THREE.Vector3()
    this.raycaster.ray.distanceSqToSegment(a, b, undefined, onSeg)
    this.onDepth?.(Math.max(1, Math.round(onSeg.distanceTo(a) * 10) / 10))
  }

  private readonly onPointerUp = () => {
    if (!this.dragging) return
    this.dragging = false
    this.controls.enabled = true
  }

  dispose(): void {
    this.disposed = true
    cancelAnimationFrame(this.raf)
    this.ro.disconnect()
    window.removeEventListener("pointerup", this.onPointerUp)
    this.controls.dispose()
    disposeTree(this.scene)
    this.renderer.dispose()
    this.renderer.domElement.remove()
  }
}
