import * as THREE from "three"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"

import { buildPitch, disposeTree } from "./pitch"
import { SMPL_BONES, SMPL_J_REST, buildSmplSkinnedMesh, pitchToThree, smplFK } from "./smpl"
import type { CameraMode, Mat3, PlayerTrack, SceneData, TrackedPose, Vec3 } from "./types"

const DEFAULT_FOV = 50
const PITCH_CENTRE = new THREE.Vector3(52.5, 0, -34)
const PRESETS: Record<Exclude<CameraMode, "tracked">, Vec3> = {
  broadcast: [52.5, 25, 50],
  tactical: [52.5, 80, -34],
  "behind-goal": [-10, 10, -34],
}

export interface Visibility {
  ball: boolean
  skeleton: boolean
  mesh: boolean
}

interface PlayerObject {
  track: PlayerTrack
  skel: THREE.LineSegments
  joints: THREE.Points
  posBuf: Float32Array
  jointBuf: Float32Array
  mesh: THREE.SkinnedMesh | null
  bones: THREE.Bone[] | null
}

/**
 * Imperative three.js scene for the viewer. React owns the chrome; this class
 * owns the renderer, the RAF loop and every GPU resource, and `dispose()`
 * releases all of them (the Export panel mounts and unmounts the viewer).
 */
export class ViewerEngine {
  private readonly renderer: THREE.WebGLRenderer
  private readonly scene = new THREE.Scene()
  private readonly camera = new THREE.PerspectiveCamera(DEFAULT_FOV, 1, 0.1, 500)
  private readonly controls: OrbitControls
  private readonly resizeObserver: ResizeObserver
  private readonly players: PlayerObject[] = []
  private readonly pitch = buildPitch()
  private ball: THREE.Mesh | null = null
  private data: SceneData | null = null
  private raf = 0
  private lastTime = 0
  private accumulator = 0
  private dirty = true
  private disposed = false
  private frame = 0
  private playing = false
  private speed = 1
  private cameraMode: CameraMode = "broadcast"
  private selectedId: string | null = null
  private vis: Visibility = { ball: true, skeleton: true, mesh: false }
  private readonly tmpAxis = new THREE.Vector3()
  private readonly tmpMat4 = new THREE.Matrix4()
  private readonly basis = { m: new THREE.Matrix4(), r: new THREE.Vector3(), u: new THREE.Vector3(), b: new THREE.Vector3() }

  private readonly container: HTMLElement
  private readonly onFrame: (frame: number) => void

  constructor(container: HTMLElement, onFrame: (frame: number) => void) {
    this.container = container
    this.onFrame = onFrame
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    this.renderer.setClearColor(0x000000, 0)
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    this.renderer.domElement.className = "block size-full"
    container.appendChild(this.renderer.domElement)

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
    const dir = new THREE.DirectionalLight(0xffffff, 0.6)
    dir.position.set(20, 40, 20)
    this.scene.add(dir)
    this.scene.add(this.pitch)

    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.addEventListener("change", this.markDirty)
    this.applyPreset("broadcast")

    this.resizeObserver = new ResizeObserver(this.resize)
    this.resizeObserver.observe(container)
    this.resize()
    this.raf = requestAnimationFrame(this.tick)
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
    this.dirty = true
  }

  get currentFrame(): number {
    return this.frame
  }

  get totalFrames(): number {
    return this.data?.totalFrames ?? 0
  }

  load(data: SceneData): void {
    this.clearContent()
    this.data = data
    for (const track of data.players) this.players.push(this.addPlayer(track, data))
    if (data.hasBall) {
      const mat = new THREE.MeshStandardMaterial({ color: 0xf2f2f2, roughness: 0.65, metalness: 0 })
      this.ball = new THREE.Mesh(new THREE.SphereGeometry(data.ballRadius, 16, 16), mat)
      this.ball.visible = false
      this.scene.add(this.ball)
    }
    this.frame = Math.min(this.frame, Math.max(0, data.totalFrames - 1))
    this.applyPreset("broadcast")
    this.updateFrame(this.frame)
  }

  private addPlayer(track: PlayerTrack, data: SceneData): PlayerObject {
    const colour = track.colour
    const posBuf = new Float32Array(SMPL_BONES.length * 6)
    const skelGeom = new THREE.BufferGeometry()
    skelGeom.setAttribute("position", new THREE.BufferAttribute(posBuf, 3))
    const skel = new THREE.LineSegments(
      skelGeom,
      new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: 0.95 }),
    )
    const jointBuf = new Float32Array(SMPL_J_REST.length * 3)
    const jointGeom = new THREE.BufferGeometry()
    jointGeom.setAttribute("position", new THREE.BufferAttribute(jointBuf, 3))
    const joints = new THREE.Points(
      jointGeom,
      new THREE.PointsMaterial({ color: colour, size: 6, sizeAttenuation: false, transparent: true, opacity: 0.9 }),
    )
    for (const o of [skel, joints]) {
      o.visible = false
      o.frustumCulled = false
      this.scene.add(o)
    }
    const built = data.smpl ? buildSmplSkinnedMesh(data.smpl, track.betas, colour) : null
    if (built) this.scene.add(built.mesh)
    return { track, skel, joints, posBuf, jointBuf, mesh: built?.mesh ?? null, bones: built?.bones ?? null }
  }

  private clearContent(): void {
    for (const p of this.players) {
      for (const o of [p.skel, p.joints, p.mesh]) {
        if (!o) continue
        this.scene.remove(o)
        disposeTree(o)
      }
    }
    this.players.length = 0
    if (this.ball) {
      this.scene.remove(this.ball)
      disposeTree(this.ball)
      this.ball = null
    }
    this.selectedId = null
  }

  setVisibility(vis: Visibility): void {
    this.vis = vis
    this.updateFrame(this.frame)
  }

  setPlaying(playing: boolean): void {
    this.playing = playing
    this.accumulator = 0
  }

  setSpeed(speed: number): void {
    this.speed = speed
  }

  setFrame(fi: number): void {
    const max = Math.max(0, this.totalFrames - 1)
    this.updateFrame(Math.min(max, Math.max(0, Math.round(fi))))
  }

  setSelected(id: string | null): void {
    this.selectedId = id
    this.updateFrame(this.frame)
  }

  setCameraMode(mode: CameraMode): void {
    this.cameraMode = mode
    this.selectedId = null
    if (mode === "tracked") {
      // Orbit input is disabled: every frame's pose comes from the solver.
      this.controls.enabled = false
      this.applyTrackedCamera(this.frame)
    } else {
      this.applyPreset(mode)
    }
    this.dirty = true
  }

  private applyPreset(mode: Exclude<CameraMode, "tracked">): void {
    this.controls.enabled = true
    if (this.camera.fov !== DEFAULT_FOV) {
      this.camera.fov = DEFAULT_FOV
      this.camera.updateProjectionMatrix()
    }
    const [x, y, z] = PRESETS[mode]
    this.camera.position.set(x, y, z)
    this.controls.target.copy(PITCH_CENTRE)
    this.controls.update()
  }

  /**
   * Pose convention: X_cam = R X_world + t (x right, y down, z forward),
   * centre C = -R^T t. Rows of R are the camera axes in pitch-world; y and z
   * are negated for three.js' y-up / z-back, then routed through pitchToThree.
   */
  private applyTrackedCamera(fi: number): void {
    const cf: TrackedPose | undefined = this.data?.track?.get(fi)
    if (!cf) return // no data on this frame: hold the previous pose
    const { K, R, t } = cf
    const c = cameraCentre(R, t)
    if (!c.every(Number.isFinite)) return
    this.camera.position.set(...pitchToThree(c))
    const { m, r, u, b } = this.basis
    r.set(...pitchToThree([R[0][0], R[0][1], R[0][2]]))
    u.set(...pitchToThree([-R[1][0], -R[1][1], -R[1][2]]))
    b.set(...pitchToThree([-R[2][0], -R[2][1], -R[2][2]]))
    m.makeBasis(r, u, b)
    this.camera.quaternion.setFromRotationMatrix(m)
    const fy = K[1][1]
    if (fy > 0) {
      const fov = (2 * Math.atan((this.data?.trackImageHeight ?? 1080) / (2 * fy)) * 180) / Math.PI
      if (Math.abs(this.camera.fov - fov) > 1e-4) {
        this.camera.fov = fov
        this.camera.updateProjectionMatrix()
      }
    }
  }

  private updateFrame(fi: number): void {
    const changed = fi !== this.frame
    this.frame = fi
    for (const p of this.players) this.poseObject(p, fi)
    this.poseBall(fi)
    if (this.cameraMode === "tracked") this.applyTrackedCamera(fi)
    else this.followSelected(fi)
    this.dirty = true
    if (changed) this.onFrame(fi)
  }

  private poseObject(p: PlayerObject, fi: number): void {
    const idx = p.track.frameIndex.get(fi)
    const t = idx !== undefined ? p.track.rootT[idx] : null
    const R = idx !== undefined ? p.track.rootR[idx] : null
    const theta = idx !== undefined ? p.track.thetas[idx] : null
    if (!t || !R || !theta || Number.isNaN(t[0])) {
      p.skel.visible = false
      p.joints.visible = false
      if (p.mesh) p.mesh.visible = false
      return
    }
    const pos = smplFK(theta, R, t)
    SMPL_BONES.forEach(([a, b], i) => {
      p.posBuf.set(pitchToThree(pos[a]), i * 6)
      p.posBuf.set(pitchToThree(pos[b]), i * 6 + 3)
    })
    pos.forEach((v, i) => p.jointBuf.set(pitchToThree(v), i * 3))
    p.skel.geometry.attributes.position.needsUpdate = true
    p.joints.geometry.attributes.position.needsUpdate = true
    p.skel.visible = this.vis.skeleton
    p.joints.visible = this.vis.skeleton
    if (p.mesh && p.bones && this.vis.mesh) this.poseMesh(p.mesh, p.bones, t, R, theta)
    else if (p.mesh) p.mesh.visible = false
  }

  private poseMesh(
    mesh: THREE.SkinnedMesh,
    bones: THREE.Bone[],
    t: Vec3,
    R: Mat3,
    theta: readonly (readonly number[])[],
  ): void {
    bones[0].position.set(...pitchToThree(t))
    // R_three = M_p2t @ R with M_p2t rows [1,0,0], [0,0,1], [0,-1,0], inlined.
    this.tmpMat4.set(
      R[0][0], R[0][1], R[0][2], 0,
      R[2][0], R[2][1], R[2][2], 0,
      -R[1][0], -R[1][1], -R[1][2], 0,
      0, 0, 0, 1,
    )
    bones[0].quaternion.setFromRotationMatrix(this.tmpMat4)
    for (let i = 1; i < 24; i++) {
      const aa = theta[i]
      const [ax, ay, az] = [aa?.[0] ?? 0, aa?.[1] ?? 0, aa?.[2] ?? 0]
      const ang = Math.hypot(ax, ay, az)
      if (ang < 1e-9) bones[i].quaternion.set(0, 0, 0, 1)
      else bones[i].quaternion.setFromAxisAngle(this.tmpAxis.set(ax / ang, ay / ang, az / ang), ang)
    }
    mesh.visible = true
  }

  private poseBall(fi: number): void {
    if (!this.ball || !this.data) return
    const xyz = this.data.ball.get(fi)
    if (this.vis.ball && xyz) {
      this.ball.position.set(...pitchToThree(xyz))
      this.ball.visible = true
    } else {
      this.ball.visible = false
    }
  }

  /** Retarget the orbit focus to the selected player's pelvis; position stays sticky. */
  private followSelected(fi: number): void {
    const p = this.players.find((o) => o.track.id === this.selectedId)
    const idx = p?.track.frameIndex.get(fi)
    const t = p && idx !== undefined ? p.track.rootT[idx] : null
    if (t && !Number.isNaN(t[0])) {
      this.controls.target.set(...pitchToThree(t))
      this.controls.update()
    }
  }

  private readonly tick = (time: number): void => {
    if (this.disposed) return
    this.raf = requestAnimationFrame(this.tick)
    const dt = (time - this.lastTime) / 1000
    this.lastTime = time
    if (this.playing && this.totalFrames > 0) this.advance(dt)
    // OrbitControls.update() would overwrite the per-frame tracked quaternion.
    const moved = this.cameraMode !== "tracked" && this.controls.update()
    if (this.dirty || moved) {
      this.dirty = false
      this.renderer.render(this.scene, this.camera)
    }
  }

  private advance(dt: number): void {
    this.accumulator += Math.min(dt, 0.25) * this.speed
    const frameDt = 1 / Math.max(1, this.data?.fps ?? 25)
    let next = this.frame
    while (this.accumulator >= frameDt) {
      this.accumulator -= frameDt
      next = (next + 1) % this.totalFrames
    }
    if (next !== this.frame) this.updateFrame(next)
  }

  dispose(): void {
    this.disposed = true
    cancelAnimationFrame(this.raf)
    this.resizeObserver.disconnect()
    this.controls.removeEventListener("change", this.markDirty)
    this.controls.dispose()
    this.clearContent()
    disposeTree(this.pitch)
    this.scene.clear()
    this.renderer.dispose()
    this.renderer.forceContextLoss()
    this.renderer.domElement.remove()
  }
}

function cameraCentre(R: Mat3, t: Vec3): Vec3 {
  return [
    -(R[0][0] * t[0] + R[1][0] * t[1] + R[2][0] * t[2]),
    -(R[0][1] * t[0] + R[1][1] * t[1] + R[2][1] * t[2]),
    -(R[0][2] * t[0] + R[1][2] * t[1] + R[2][2] * t[2]),
  ]
}
