import * as THREE from "three"

import { getJson } from "@/lib/api"
import type { Mat3, SmplModel, Vec3 } from "./types"

// 24-joint SMPL hierarchy. parent[0] = -1 (root).
export const SMPL_PARENTS: readonly number[] = [
  -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
]

export const SMPL_BONES: readonly (readonly [number, number])[] = SMPL_PARENTS.map(
  (p, i) => [p, i] as const,
).filter((pair) => pair[0] >= 0)

// T-pose joints, standard SMPL canonical (y-up) space, neutral shape, pelvis at origin.
// root_R maps this canonical space straight to pitch z-up, so FK runs in
// canonical space and pitch positions only appear once root_R is applied.
export const SMPL_J_REST: readonly Vec3[] = [
  [0.0, 0.0, 0.0], [0.06, -0.087, -0.013], [-0.06, -0.087, -0.013], [0.001, 0.108, -0.027],
  [0.099, -0.494, -0.001], [-0.099, -0.494, -0.001], [0.002, 0.246, 0.018],
  [0.087, -0.882, -0.038], [-0.087, -0.882, -0.038], [0.0, 0.3, 0.06],
  [0.117, -0.939, 0.071], [-0.117, -0.939, 0.071], [0.0, 0.518, 0.013],
  [0.084, 0.474, 0.008], [-0.084, 0.474, 0.008], [0.0, 0.609, 0.052],
  [0.184, 0.427, -0.012], [-0.184, 0.427, -0.012], [0.448, 0.426, -0.039],
  [-0.448, 0.426, -0.039], [0.711, 0.42, -0.046], [-0.711, 0.42, -0.046],
  [0.789, 0.418, -0.034], [-0.789, 0.418, -0.034],
]

const SMPL_LOCAL_OFFSET: readonly Vec3[] = SMPL_J_REST.map((p, i) => {
  const par = SMPL_PARENTS[i]
  if (par < 0) return [0, 0, 0] as const
  const q = SMPL_J_REST[par]
  return [p[0] - q[0], p[1] - q[1], p[2] - q[2]] as const
})

const IDENTITY: Mat3 = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
]

/** Rodrigues: axis-angle 3-vector to 3x3 rotation. */
export function aaToMat(aa: readonly number[]): Mat3 {
  const [x, y, z] = [aa[0] ?? 0, aa[1] ?? 0, aa[2] ?? 0]
  const theta = Math.hypot(x, y, z)
  if (theta < 1e-9) return IDENTITY
  const [ux, uy, uz] = [x / theta, y / theta, z / theta]
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  const C = 1 - c
  return [
    [c + ux * ux * C, ux * uy * C - uz * s, ux * uz * C + uy * s],
    [uy * ux * C + uz * s, c + uy * uy * C, uy * uz * C - ux * s],
    [uz * ux * C - uy * s, uz * uy * C + ux * s, c + uz * uz * C],
  ]
}

function matMul3(A: Mat3, B: Mat3): Mat3 {
  return A.map((row) => [0, 1, 2].map((j) => row[0] * B[0][j] + row[1] * B[1][j] + row[2] * B[2][j]))
}

function matVec3(M: Mat3, v: Vec3): Vec3 {
  return [
    M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
    M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
    M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
  ]
}

/**
 * SMPL forward kinematics for one frame; joint world positions in the pitch frame.
 * thetas[0] is intentionally ignored: root_R already carries the root orientation.
 */
export function smplFK(thetas: readonly (readonly number[])[], rootR: Mat3, rootT: Vec3): Vec3[] {
  const n = SMPL_J_REST.length
  const rot: Mat3[] = new Array<Mat3>(n)
  const pos: Vec3[] = new Array<Vec3>(n)
  rot[0] = rootR
  pos[0] = [rootT[0], rootT[1], rootT[2]]
  for (let i = 1; i < n; i++) {
    const par = SMPL_PARENTS[i]
    rot[i] = matMul3(rot[par], aaToMat(thetas[i] ?? [0, 0, 0]))
    const off = matVec3(rot[par], SMPL_LOCAL_OFFSET[i])
    pos[i] = [pos[par][0] + off[0], pos[par][1] + off[1], pos[par][2] + off[2]]
  }
  return pos
}

/** Pitch (x, y, z-up) to three.js (x, z, -y). */
export function pitchToThree(p: Vec3): Vec3 {
  return [p[0], p[2] || 0, -(p[1] || 0)]
}

interface RawSmplModel {
  v_template: number[][]
  faces: number[][]
  skin_index: number[][]
  skin_weight: number[][]
  joint_positions: number[][]
  parents: number[]
  shapedirs?: number[][][]
  joint_shapedirs?: number[][][]
}

function flatten2(arr: number[][], cols: number): Float32Array {
  const out = new Float32Array(arr.length * cols)
  for (let i = 0; i < arr.length; i++) for (let j = 0; j < cols; j++) out[i * cols + j] = arr[i][j]
  return out
}

function flatten3(arr: number[][][]): Float32Array {
  const d2 = arr[0].length
  const k = arr[0][0].length
  const out = new Float32Array(arr.length * d2 * k)
  for (let i = 0; i < arr.length; i++)
    for (let j = 0; j < d2; j++) for (let m = 0; m < k; m++) out[i * d2 * k + j * k + m] = arr[i][j][m]
  return out
}

/** Fetch /api/smpl_model. Resolves to null when the optional endpoint is unavailable. */
export async function loadSmplModel(signal?: AbortSignal): Promise<SmplModel | null> {
  let raw: RawSmplModel
  try {
    raw = await getJson<RawSmplModel>("/api/smpl_model", { signal })
  } catch {
    return null
  }
  const nVerts = raw.v_template.length
  const faces = new Uint32Array(raw.faces.length * 3)
  raw.faces.forEach((f, i) => faces.set(f.slice(0, 3), i * 3))
  const skinIndex = new Uint16Array(nVerts * 4)
  raw.skin_index.forEach((s, i) => skinIndex.set(s.slice(0, 4), i * 4))
  return {
    vTemplate: flatten2(raw.v_template, 3),
    faces,
    skinIndex,
    skinWeight: flatten2(raw.skin_weight, 4),
    jointPositions: flatten2(raw.joint_positions, 3),
    parents: raw.parents,
    nVerts,
    nBetas: raw.shapedirs ? raw.shapedirs[0][0].length : 0,
    shapedirs: raw.shapedirs ? flatten3(raw.shapedirs) : undefined,
    jointShapedirs: raw.joint_shapedirs ? flatten3(raw.joint_shapedirs) : undefined,
  }
}

function addShape(base: Float32Array, dirs: Float32Array, count: number, nBetas: number, betas: readonly number[]): void {
  const k = Math.min(nBetas, betas.length)
  for (let v = 0; v < count; v++) {
    for (let c = 0; c < 3; c++) {
      const start = v * 3 * nBetas + c * nBetas
      let sum = 0
      for (let i = 0; i < k; i++) sum += dirs[start + i] * betas[i]
      base[v * 3 + c] += sum
    }
  }
}

/** Beta-adjusted vertices + joints (mean shape unchanged when betas/shapedirs are absent). */
function applyBetas(model: SmplModel, betas: readonly number[]) {
  const vShaped = new Float32Array(model.vTemplate)
  const jShaped = new Float32Array(model.jointPositions)
  if (betas.length && model.shapedirs && model.nBetas > 0) {
    addShape(vShaped, model.shapedirs, model.nVerts, model.nBetas, betas)
    if (model.jointShapedirs) addShape(jShaped, model.jointShapedirs, 24, model.nBetas, betas)
  }
  return { vShaped, jShaped }
}

export interface SkinnedPlayer {
  mesh: THREE.SkinnedMesh
  bones: THREE.Bone[]
}

/** SkinnedMesh + 24-bone skeleton with the player's body shape pre-applied. */
export function buildSmplSkinnedMesh(model: SmplModel, betas: readonly number[], color: number): SkinnedPlayer {
  const { vShaped, jShaped } = applyBetas(model, betas)
  const geom = new THREE.BufferGeometry()
  geom.setAttribute("position", new THREE.BufferAttribute(vShaped, 3))
  geom.setIndex(new THREE.BufferAttribute(model.faces, 1))
  geom.setAttribute("skinIndex", new THREE.BufferAttribute(new Uint16Array(model.skinIndex), 4))
  geom.setAttribute("skinWeight", new THREE.BufferAttribute(new Float32Array(model.skinWeight), 4))
  geom.computeVertexNormals()
  const material = new THREE.MeshStandardMaterial({ color, roughness: 0.65, metalness: 0, side: THREE.DoubleSide })

  const bones = Array.from({ length: 24 }, () => new THREE.Bone())
  for (let i = 0; i < 24; i++) {
    const p = model.parents[i]
    if (p < 0) {
      bones[i].position.set(jShaped[i * 3], jShaped[i * 3 + 1], jShaped[i * 3 + 2])
    } else {
      bones[i].position.set(
        jShaped[i * 3] - jShaped[p * 3],
        jShaped[i * 3 + 1] - jShaped[p * 3 + 1],
        jShaped[i * 3 + 2] - jShaped[p * 3 + 2],
      )
      bones[p].add(bones[i])
    }
  }
  const mesh = new THREE.SkinnedMesh(geom, material)
  mesh.add(bones[0])
  // Bone matrixWorld must be populated before Skeleton computes inverses,
  // otherwise skinning collapses to T(root_t) * v_template.
  mesh.updateMatrixWorld(true)
  mesh.bind(new THREE.Skeleton(bones))
  mesh.frustumCulled = false
  mesh.visible = false
  return { mesh, bones }
}
