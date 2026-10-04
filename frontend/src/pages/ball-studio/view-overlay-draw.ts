// Canvas painters for one video view: solved track, pipeline ghost, keys,
// soft observations, pending picks, epipolar polylines with height ticks and
// residual vectors. Pure drawing - no React, no state.
import { cssVar } from "@/lib/format"
import { project, type Epipolar, type FrameCam } from "./camera-model"
import { KEY_SOURCE_STYLE, PIPELINE_GHOST, SEGMENT_STYLE, SEVERITY_CANVAS, residualSeverity, viewColour } from "./palette"
import type { KeySource, SegmentKind, Vec2, Vec3 } from "./types"

const MONO = '"Geist Mono Variable", ui-monospace, monospace'

export interface DrawKey {
  id: string
  frame: number
  xyz: Vec3
  source: KeySource
  selected: boolean
  /** Pixel the operator picked in THIS view at this key's instant. */
  pickedUv: Vec2 | null
  /** Where the solver reprojects it in this view (residual vector end). */
  projectedUv: Vec2 | null
  kind: SegmentKind
}

export interface DrawTrack {
  frames: readonly number[]
  xyz: readonly (readonly number[] | null)[]
  kind?: readonly SegmentKind[]
}

export interface DrawEpipolar {
  fromView: number
  epi: Epipolar
}

export interface ViewDrawInput {
  width: number
  height: number
  dpr: number
  imageSize: Vec2
  zoom: number
  tx: number
  ty: number
  viewIndex: number
  cam: FrameCam | null
  frame: number
  dense: DrawTrack | null
  stale: boolean
  pipeline: DrawTrack | null
  showPipeline: boolean
  showResiduals: boolean
  showRays: boolean
  keys: readonly DrawKey[]
  soft: readonly { uv: Vec2; projectedUv: Vec2 | null; selected: boolean }[]
  pipelineAnchors: readonly Vec2[]
  /** Solver projection of the ball at this view's own instant. */
  nowUv: Vec2 | null
  pending: Vec2 | null
  ghost: Vec3 | null
  epipolar: readonly DrawEpipolar[]
  hover: { uv: Vec2; snapped: boolean } | null
  selectedFrame: number | null
}

const TRAIL = 12

function setup(ctx: CanvasRenderingContext2D, inp: ViewDrawInput) {
  const base = inp.width / inp.imageSize[0]
  const k = base * inp.zoom
  const toScreen = (uv: Vec2): Vec2 => [uv[0] * k + inp.tx, uv[1] * k + inp.ty]
  ctx.setTransform(inp.dpr, 0, 0, inp.dpr, 0, 0)
  ctx.clearRect(0, 0, inp.width, inp.height)
  return { toScreen, k }
}

function halo(ctx: CanvasRenderingContext2D, draw: () => void, colour = "rgba(0,0,0,0.65)", extra = 2.5) {
  const w = ctx.lineWidth
  ctx.strokeStyle = colour
  ctx.lineWidth = w + extra
  draw()
  ctx.lineWidth = w
}

function dashFor(pattern: string): number[] {
  return pattern === "dotted" ? [1.5, 4] : pattern === "dashed" ? [6, 4] : []
}

function drawTrack(ctx: CanvasRenderingContext2D, inp: ViewDrawInput, toScreen: (uv: Vec2) => Vec2) {
  const { dense, cam } = inp
  if (!dense || !cam) return
  ctx.save()
  ctx.globalAlpha = inp.stale ? 0.45 : 1
  ctx.lineCap = "round"
  ctx.lineJoin = "round"
  let prev: Vec2 | null = null
  let prevKind: SegmentKind | null = null
  let prevFrame = 0
  for (let i = 0; i < dense.frames.length; i++) {
    const p = dense.xyz[i]
    const pr = p ? project(cam, [p[0], p[1], p[2]]) : null
    if (!pr) {
      prev = null
      continue
    }
    const kind = dense.kind?.[i] ?? "flight"
    const st = SEGMENT_STYLE[kind]
    const cur = toScreen(pr.uv)
    if (prev && prevKind) {
      const near = Math.abs(dense.frames[i] - inp.frame) <= TRAIL || Math.abs(prevFrame - inp.frame) <= TRAIL
      const style = SEGMENT_STYLE[prevKind]
      if (style.pattern !== "ring") {
        ctx.beginPath()
        ctx.moveTo(prev[0], prev[1])
        ctx.lineTo(cur[0], cur[1])
        ctx.setLineDash(dashFor(style.pattern))
        ctx.lineWidth = near ? style.width + 1 : style.width
        halo(ctx, () => ctx.stroke())
        ctx.strokeStyle = style.colour
        ctx.globalAlpha = (inp.stale ? 0.45 : 1) * (near ? 1 : 0.7)
        ctx.stroke()
        ctx.globalAlpha = inp.stale ? 0.45 : 1
      }
    }
    // gravity-arc ticks every 5 frames on flights
    if (kind === "flight" && dense.frames[i] % 5 === 0) {
      ctx.setLineDash([])
      ctx.fillStyle = st.colour
      ctx.beginPath()
      ctx.arc(cur[0], cur[1], 2, 0, Math.PI * 2)
      ctx.fill()
    }
    prev = cur
    prevKind = kind
    prevFrame = dense.frames[i]
  }
  ctx.restore()
}

function drawPipeline(ctx: CanvasRenderingContext2D, inp: ViewDrawInput, toScreen: (uv: Vec2) => Vec2) {
  const { pipeline, cam } = inp
  if (!pipeline || !cam || !inp.showPipeline) return
  ctx.save()
  ctx.strokeStyle = PIPELINE_GHOST.colour
  ctx.globalAlpha = PIPELINE_GHOST.alpha
  ctx.lineWidth = PIPELINE_GHOST.width
  ctx.setLineDash([5, 4])
  ctx.beginPath()
  let pen = false
  for (let i = 0; i < pipeline.frames.length; i++) {
    const p = pipeline.xyz[i]
    const pr = p ? project(cam, [p[0], p[1], p[2]]) : null
    if (!pr) {
      pen = false
      continue
    }
    const s = toScreen(pr.uv)
    if (pen) ctx.lineTo(s[0], s[1])
    else ctx.moveTo(s[0], s[1])
    pen = true
  }
  ctx.stroke()
  ctx.restore()
}

function diamond(ctx: CanvasRenderingContext2D, x: number, y: number, r: number) {
  ctx.beginPath()
  ctx.moveTo(x, y - r)
  ctx.lineTo(x + r, y)
  ctx.lineTo(x, y + r)
  ctx.lineTo(x - r, y)
  ctx.closePath()
}

function drawKey(ctx: CanvasRenderingContext2D, inp: ViewDrawInput, k: DrawKey, toScreen: (uv: Vec2) => Vec2) {
  const pr = inp.cam ? project(inp.cam, k.xyz) : null
  if (!pr) return
  const [x, y] = toScreen(pr.uv)
  const here = k.frame === inp.frame
  const r = Math.min(10, (here ? 7 : 4.5) * Math.max(1, Math.sqrt(inp.zoom) * 0.8))
  const st = KEY_SOURCE_STYLE[k.source]
  const colour = SEGMENT_STYLE[k.kind].colour
  ctx.save()
  ctx.globalAlpha = here ? 1 : 0.55
  ctx.lineWidth = 1.5
  if (st.marker === "square") {
    ctx.strokeStyle = "rgba(0,0,0,0.7)"
    ctx.fillStyle = colour
    ctx.fillRect(x - r * 0.8, y - r * 0.8, r * 1.6, r * 1.6)
    ctx.strokeRect(x - r * 0.8, y - r * 0.8, r * 1.6, r * 1.6)
  } else {
    diamond(ctx, x, y, r)
    if (st.marker === "diamond-hollow") {
      halo(ctx, () => ctx.stroke())
      ctx.strokeStyle = colour
      ctx.stroke()
    } else {
      ctx.fillStyle = colour
      ctx.fill()
      ctx.strokeStyle = st.marker === "diamond-filled" ? "#ffffff" : "rgba(0,0,0,0.7)"
      ctx.stroke()
      if (st.marker === "diamond-tether") {
        ctx.beginPath()
        ctx.moveTo(x, y + r)
        ctx.lineTo(x, y + r + 6)
        ctx.strokeStyle = colour
        ctx.stroke()
      }
    }
  }
  if (st.glyph) {
    ctx.fillStyle = "#ffffff"
    ctx.font = `600 10px ${MONO}`
    ctx.fillText(st.glyph, x + r + 3, y + 3)
  }
  if (k.selected) {
    ctx.globalAlpha = 1
    ctx.strokeStyle = getCss("--info", "#60a5fa")
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.arc(x, y, r + 5, 0, Math.PI * 2)
    ctx.stroke()
    label(ctx, k.id, x + r + 7, y - r - 3)
  }
  ctx.restore()

  // Residual vector: picked pixel -> where the solver reprojects it.
  if (here && inp.showResiduals && k.pickedUv && k.projectedUv) {
    const a = toScreen(k.pickedUv)
    const b = toScreen(k.projectedUv)
    const px = Math.hypot(k.projectedUv[0] - k.pickedUv[0], k.projectedUv[1] - k.pickedUv[1])
    const col = SEVERITY_CANVAS[residualSeverity(px)]
    ctx.save()
    ctx.strokeStyle = col
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.moveTo(a[0], a[1])
    ctx.lineTo(b[0], b[1])
    ctx.stroke()
    ctx.fillStyle = col
    ctx.beginPath()
    ctx.arc(a[0], a[1], 2, 0, Math.PI * 2)
    ctx.fill()
    if (px < 2 && px > 0.01) {
      // magnified 4x ghost so a sub-2px miss stays visible
      const mx = a[0] + (b[0] - a[0]) * 4
      const my = a[1] + (b[1] - a[1]) * 4
      ctx.setLineDash([2, 2])
      ctx.beginPath()
      ctx.moveTo(a[0], a[1])
      ctx.lineTo(mx, my)
      ctx.stroke()
    }
    ctx.restore()
    label(ctx, `${px.toFixed(1)} px`, b[0] + 8, b[1] + 14, col)
  }
}

const getCss = cssVar

function label(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, colour = "#ffffff") {
  ctx.save()
  ctx.font = `500 11px ${MONO}`
  ctx.lineWidth = 3
  ctx.strokeStyle = "rgba(0,0,0,0.75)"
  ctx.strokeText(text, x, y)
  ctx.fillStyle = colour
  ctx.fillText(text, x, y)
  ctx.restore()
}

function plus(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, colour: string, ring = false) {
  ctx.save()
  ctx.lineWidth = 1.5
  const draw = () => {
    ctx.beginPath()
    ctx.moveTo(x - r, y)
    ctx.lineTo(x + r, y)
    ctx.moveTo(x, y - r)
    ctx.lineTo(x, y + r)
    ctx.stroke()
  }
  halo(ctx, draw)
  ctx.strokeStyle = colour
  draw()
  if (ring) {
    ctx.beginPath()
    ctx.arc(x, y, r + 3, 0, Math.PI * 2)
    ctx.stroke()
  }
  ctx.restore()
}

function drawEpipolar(ctx: CanvasRenderingContext2D, inp: ViewDrawInput, e: DrawEpipolar, toScreen: (uv: Vec2) => Vec2) {
  const colour = viewColour(e.fromView)
  ctx.save()
  ctx.lineCap = "round"
  ctx.lineJoin = "round"
  for (const run of e.epi.runs) {
    if (run.length < 2) continue
    ctx.beginPath()
    run.forEach((p, i) => {
      const s = toScreen(p.uv)
      if (i === 0) ctx.moveTo(s[0], s[1])
      else ctx.lineTo(s[0], s[1])
    })
    ctx.lineWidth = 6
    ctx.strokeStyle = colour
    ctx.globalAlpha = 0.18
    ctx.stroke()
    ctx.lineWidth = 1.5
    ctx.globalAlpha = 1
    ctx.stroke()
  }
  for (const t of e.epi.ticks) {
    const s = toScreen(t.uv)
    if (s[0] < -20 || s[0] > inp.width + 20 || s[1] < -20 || s[1] > inp.height + 20) continue
    ctx.fillStyle = colour
    ctx.beginPath()
    ctx.arc(s[0], s[1], 3, 0, Math.PI * 2)
    ctx.fill()
    ctx.strokeStyle = "rgba(0,0,0,0.7)"
    ctx.lineWidth = 1
    ctx.stroke()
    label(ctx, t.label, s[0] + 6, s[1] - 5, colour)
  }
  ctx.restore()
}

/** Everything for one view, in order: ghost, track, epipolar, keys, observations, picks. */
export function drawView(canvas: HTMLCanvasElement, inp: ViewDrawInput): void {
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  const { toScreen } = setup(ctx, inp)

  drawPipeline(ctx, inp, toScreen)
  if (inp.showPipeline) {
    for (const uv of inp.pipelineAnchors) {
      const [x, y] = toScreen(uv)
      ctx.save()
      ctx.strokeStyle = PIPELINE_GHOST.colour
      ctx.globalAlpha = PIPELINE_GHOST.alpha
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.arc(x, y, 6, 0, Math.PI * 2)
      ctx.stroke()
      ctx.restore()
    }
  }
  drawTrack(ctx, inp, toScreen)
  if (inp.showRays) for (const e of inp.epipolar) drawEpipolar(ctx, inp, e, toScreen)

  for (const k of inp.keys) drawKey(ctx, inp, k, toScreen)

  if (inp.nowUv) {
    const [x, y] = toScreen(inp.nowUv)
    ctx.save()
    ctx.lineWidth = 1.5
    halo(ctx, () => {
      ctx.beginPath()
      ctx.arc(x, y, 7, 0, Math.PI * 2)
      ctx.stroke()
    })
    ctx.strokeStyle = "#ffffff"
    ctx.beginPath()
    ctx.arc(x, y, 7, 0, Math.PI * 2)
    ctx.stroke()
    ctx.restore()
  }

  const colour = viewColour(inp.viewIndex)
  for (const o of inp.soft) {
    const [x, y] = toScreen(o.uv)
    plus(ctx, x, y, 5, colour, o.selected)
    if (inp.showResiduals && o.projectedUv) {
      const [px, py] = toScreen(o.projectedUv)
      ctx.save()
      ctx.strokeStyle = SEVERITY_CANVAS[residualSeverity(Math.hypot(o.uv[0] - o.projectedUv[0], o.uv[1] - o.projectedUv[1]))]
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(x, y)
      ctx.lineTo(px, py)
      ctx.stroke()
      ctx.restore()
    }
  }

  // Ghost: where the pending pick / hovered constraint lands in THIS view.
  if (inp.ghost && inp.cam) {
    const pr = project(inp.cam, inp.ghost)
    if (pr) {
      const [x, y] = toScreen(pr.uv)
      ctx.save()
      ctx.strokeStyle = "#ffffff"
      ctx.lineWidth = 1.5
      ctx.setLineDash([3, 3])
      halo(ctx, () => {
        ctx.beginPath()
        ctx.arc(x, y, 9, 0, Math.PI * 2)
        ctx.stroke()
      })
      ctx.strokeStyle = "#ffffff"
      ctx.beginPath()
      ctx.arc(x, y, 9, 0, Math.PI * 2)
      ctx.stroke()
      ctx.restore()
    }
  }

  if (inp.pending) {
    const [x, y] = toScreen(inp.pending)
    ctx.save()
    ctx.lineWidth = 2
    halo(ctx, () => {
      ctx.beginPath()
      ctx.arc(x, y, 8, 0, Math.PI * 2)
      ctx.stroke()
    })
    ctx.strokeStyle = colour
    ctx.beginPath()
    ctx.arc(x, y, 8, 0, Math.PI * 2)
    ctx.stroke()
    ctx.restore()
    plus(ctx, x, y, 3, colour)
  }

  if (inp.hover) {
    const [x, y] = toScreen(inp.hover.uv)
    ctx.save()
    ctx.strokeStyle = inp.hover.snapped ? colour : "rgba(255,255,255,0.7)"
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(x - 10, y)
    ctx.lineTo(x - 3, y)
    ctx.moveTo(x + 3, y)
    ctx.lineTo(x + 10, y)
    ctx.moveTo(x, y - 10)
    ctx.lineTo(x, y - 3)
    ctx.moveTo(x, y + 3)
    ctx.lineTo(x, y + 10)
    ctx.stroke()
    if (inp.hover.snapped) {
      ctx.fillStyle = colour
      ctx.beginPath()
      ctx.arc(x, y, 2.5, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.restore()
  }
}
