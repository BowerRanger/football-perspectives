// three.js scene for the ball-only 3D trajectory view. Built lazily (three is
// dynamically imported by the caller) and rendered on demand — on frame change,
// orbit change and resize — so it never runs an idle animation loop. `build`
// returns a cleanup that disposes the renderer, controls, geometries and the
// resize observer (the legacy view leaked a GL context per visit).

import type { BallPreviewTrack } from "@/pages/ball-anchor-editor/api"

type Three = typeof import("three")
type OrbitControlsCtor = typeof import("three/addons/controls/OrbitControls.js").OrbitControls

type TrackFrame = NonNullable<BallPreviewTrack["frames"]>[number]

export interface Ball3DHandle {
  setFrame: (frame: number) => void
  dispose: () => void
}

const PITCH_L = 105
const PITCH_W = 68
const COLOUR_GROUND = 0x4ade80
const COLOUR_FLIGHT = 0xfb923c

// Pitch metres (x along touchline, y across, z up) -> three.js (y up).
const toThree = (p: number[]): [number, number, number] => [p[0], p[2] ?? 0, -(p[1] ?? 0)]

function addPitch(THREE: Three, scene: import("three").Scene) {
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(PITCH_L, PITCH_W),
    new THREE.MeshStandardMaterial({ color: 0x1a5e1a, side: THREE.DoubleSide }),
  )
  plane.rotation.x = -Math.PI / 2
  plane.position.set(PITCH_L / 2, 0, -PITCH_W / 2)
  scene.add(plane)
  const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 })
  const line = (x1: number, z1: number, x2: number, z2: number) => {
    const pts = [new THREE.Vector3(x1, 0.01, -z1), new THREE.Vector3(x2, 0.01, -z2)]
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), lineMat))
  }
  line(0, 0, PITCH_L, 0)
  line(PITCH_L, 0, PITCH_L, PITCH_W)
  line(PITCH_L, PITCH_W, 0, PITCH_W)
  line(0, PITCH_W, 0, 0)
  line(PITCH_L / 2, 0, PITCH_L / 2, PITCH_W)
  const circle: import("three").Vector3[] = []
  for (let a = 0; a <= Math.PI * 2; a += 0.1) {
    circle.push(new THREE.Vector3(PITCH_L / 2 + 9.15 * Math.cos(a), 0.01, -(PITCH_W / 2 + 9.15 * Math.sin(a))))
  }
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(circle), lineMat))
  for (const [x0, sign] of [[0, 1], [PITCH_L, -1]] as const) {
    line(x0, 13.84, x0 + sign * 16.5, 13.84)
    line(x0 + sign * 16.5, 13.84, x0 + sign * 16.5, 54.16)
    line(x0 + sign * 16.5, 54.16, x0, 54.16)
  }
}

function addRun(THREE: Three, scene: import("three").Scene, frames: TrackFrame[], state: string, colour: number) {
  const seg: import("three").Vector3[] = []
  let prev: import("three").Vector3 | null = null
  for (const f of frames) {
    if (!f.world_xyz || f.state !== state) {
      prev = null
      continue
    }
    const [x, y, z] = toThree(f.world_xyz)
    const v = new THREE.Vector3(x, y, z)
    if (prev) seg.push(prev, v)
    prev = v
  }
  if (!seg.length) return
  scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(seg), new THREE.LineBasicMaterial({ color: colour })))
}

export function buildBall3D(
  THREE: Three,
  OrbitControls: OrbitControlsCtor,
  container: HTMLElement,
  frames: TrackFrame[],
): Ball3DHandle {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x0a0a0a)
  const w = container.clientWidth || 640
  const h = container.clientHeight || 400
  const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 500)
  camera.position.set(52.5, 35, 50)
  const renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(w, h)
  container.appendChild(renderer.domElement)
  const controls = new OrbitControls(camera, renderer.domElement)
  controls.target.set(52.5, 0, -34)
  controls.update()
  const render = () => renderer.render(scene, camera)
  controls.addEventListener("change", render)

  scene.add(new THREE.AmbientLight(0xffffff, 0.7))
  const sun = new THREE.DirectionalLight(0xffffff, 0.6)
  sun.position.set(20, 40, 20)
  scene.add(sun)
  addPitch(THREE, scene)
  addRun(THREE, scene, frames, "grounded", COLOUR_GROUND)
  addRun(THREE, scene, frames, "flight", COLOUR_FLIGHT)

  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(0.4, 24, 16),
    new THREE.MeshStandardMaterial({ color: 0xfafafa, emissive: 0x222222 }),
  )
  ball.visible = false
  scene.add(ball)
  const drop = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
    new THREE.LineDashedMaterial({ color: 0x94a3b8, dashSize: 0.3, gapSize: 0.3, transparent: true, opacity: 0.7 }),
  )
  drop.visible = false
  scene.add(drop)

  const byFrame = new Map(frames.map((f) => [f.frame, f]))
  const setFrame = (fi: number) => {
    const f = byFrame.get(fi)
    if (!f?.world_xyz) {
      ball.visible = false
      drop.visible = false
    } else {
      const [x, y, z] = toThree(f.world_xyz)
      ball.position.set(x, y, z)
      ;(ball.material as import("three").MeshStandardMaterial).color.setHex(f.state === "flight" ? COLOUR_FLIGHT : 0xfafafa)
      ball.visible = true
      drop.geometry.setFromPoints([new THREE.Vector3(x, y, z), new THREE.Vector3(x, 0.01, z)])
      drop.computeLineDistances()
      drop.visible = f.state === "flight" && y > 0.2
    }
    render()
  }

  const ro = new ResizeObserver(() => {
    const nw = container.clientWidth
    const nh = container.clientHeight
    if (!nw || !nh) return
    renderer.setSize(nw, nh)
    camera.aspect = nw / nh
    camera.updateProjectionMatrix()
    render()
  })
  ro.observe(container)
  render()

  const dispose = () => {
    ro.disconnect()
    controls.removeEventListener("change", render)
    controls.dispose()
    scene.traverse((obj) => {
      const m = obj as import("three").Mesh
      m.geometry?.dispose()
      const mat = m.material
      if (Array.isArray(mat)) mat.forEach((x) => x.dispose())
      else mat?.dispose()
    })
    renderer.dispose()
    renderer.domElement.remove()
  }
  return { setFrame, dispose }
}
