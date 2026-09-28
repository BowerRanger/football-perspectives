import * as THREE from "three"

import { pitchToThree } from "./smpl"

export const PITCH_L = 105
export const PITCH_W = 68

function line(group: THREE.Group, mat: THREE.LineBasicMaterial, x1: number, z1: number, x2: number, z2: number) {
  const pts = [new THREE.Vector3(x1, 0.01, -z1), new THREE.Vector3(x2, 0.01, -z2)]
  group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat))
}

function box18(group: THREE.Group, mat: THREE.LineBasicMaterial, xs: number, sign: number) {
  line(group, mat, xs, 13.84, xs + sign * 16.5, 13.84)
  line(group, mat, xs + sign * 16.5, 13.84, xs + sign * 16.5, 54.16)
  line(group, mat, xs + sign * 16.5, 54.16, xs, 54.16)
}

function goalFrame(group: THREE.Group, mat: THREE.LineBasicMaterial, xs: number) {
  // FIFA: 7.32 m wide, 2.44 m tall, centred on the pitch midline (post, post, crossbar).
  const half = 7.32 / 2
  const h = 2.44
  const y1 = PITCH_W / 2 - half
  const y2 = PITCH_W / 2 + half
  const pts = [
    pitchToThree([xs, y1, 0]), pitchToThree([xs, y1, h]),
    pitchToThree([xs, y2, 0]), pitchToThree([xs, y2, h]),
    pitchToThree([xs, y1, h]), pitchToThree([xs, y2, h]),
  ].map((p) => new THREE.Vector3(p[0], p[1], p[2]))
  group.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), mat))
}

/** FIFA pitch plane, markings and goal frames (data-drawing colours). */
export function buildPitch(): THREE.Group {
  const group = new THREE.Group()
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(PITCH_L, PITCH_W),
    new THREE.MeshStandardMaterial({ color: 0x1a5e1a, side: THREE.DoubleSide }),
  )
  plane.rotation.x = -Math.PI / 2
  plane.position.set(PITCH_L / 2, 0, -PITCH_W / 2)
  group.add(plane)

  const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 })
  line(group, lineMat, 0, 0, PITCH_L, 0)
  line(group, lineMat, PITCH_L, 0, PITCH_L, PITCH_W)
  line(group, lineMat, PITCH_L, PITCH_W, 0, PITCH_W)
  line(group, lineMat, 0, PITCH_W, 0, 0)
  line(group, lineMat, PITCH_L / 2, 0, PITCH_L / 2, PITCH_W)
  const circle: THREE.Vector3[] = []
  for (let a = 0; a <= Math.PI * 2; a += 0.1) {
    circle.push(new THREE.Vector3(PITCH_L / 2 + 9.15 * Math.cos(a), 0.01, -(PITCH_W / 2 + 9.15 * Math.sin(a))))
  }
  group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(circle), lineMat))
  box18(group, lineMat, 0, 1)
  box18(group, lineMat, PITCH_L, -1)

  const goalMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85 })
  goalFrame(group, goalMat, 0)
  goalFrame(group, goalMat, PITCH_L)
  return group
}

/** Dispose every geometry/material under an object. */
export function disposeTree(root: THREE.Object3D): void {
  root.traverse((obj) => {
    const o = obj as THREE.Mesh
    o.geometry?.dispose()
    const mat = o.material as THREE.Material | THREE.Material[] | undefined
    if (Array.isArray(mat)) mat.forEach((m) => m.dispose())
    else mat?.dispose()
  })
}
