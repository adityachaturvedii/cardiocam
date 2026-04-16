// Extract the mean R, G, B intensities from two cheek patches.
//
// The signal is the three color-channel means over ROI skin pixels, which is
// rotation-invariant — cropping an axis-aligned rectangle straight from the
// source frame gives a skin patch that the downstream BVP methods (GREEN,
// POS, CHROM) can all consume identically.
//
// We use polygon fills over MediaPipe Face Mesh cheek landmarks rather than
// bounding rectangles to avoid catching non-skin pixels (eyebrows, lip edge)
// at extreme head angles.

import type { NormalizedLandmark } from '@mediapipe/tasks-vision'

// Cheek-apex polygons on the MediaPipe 478-point Face Mesh. These bound the
// zygomatic region — high, well clear of the nasolabial fold, the lips, and
// the nose — so the channel means sample clean cheek skin with no hair or
// lip edge contamination.
//
// Subject-left cheek (viewer's-right side of the image).
const LEFT_CHEEK_RING = [345, 352, 411, 425, 266, 371, 355]
// Subject-right cheek (viewer's-left side of the image).
const RIGHT_CHEEK_RING = [116, 123, 187, 205, 36, 142, 126]

export interface CheekSample {
  /** Mean red channel over both cheek polygons, 0-255. */
  r: number
  /** Mean green channel over both cheek polygons, 0-255. */
  g: number
  /** Mean blue channel over both cheek polygons, 0-255. */
  b: number
  /** Alias for g — keeps the old green-only pipeline trivially backwards compatible. */
  greenMean: number
  leftArea: number
  rightArea: number
}

/**
 * Sample the two cheek ROIs from the current video frame and return the
 * per-channel means. Returns null if either ROI ends up with no pixels
 * (face partly off-frame).
 */
export function sampleCheekRgb(
  landmarks: NormalizedLandmark[],
  ctx: CanvasRenderingContext2D,
  frameWidth: number,
  frameHeight: number
): CheekSample | null {
  const leftPoly = LEFT_CHEEK_RING.map((i) => ({
    x: landmarks[i].x * frameWidth,
    y: landmarks[i].y * frameHeight,
  }))
  const rightPoly = RIGHT_CHEEK_RING.map((i) => ({
    x: landmarks[i].x * frameWidth,
    y: landmarks[i].y * frameHeight,
  }))

  const left = meanRgbInPolygon(ctx, leftPoly, frameWidth, frameHeight)
  const right = meanRgbInPolygon(ctx, rightPoly, frameWidth, frameHeight)
  if (left === null || right === null) return null

  // Equal-weight average across both cheeks.
  const r = (left.r + right.r) / 2
  const g = (left.g + right.g) / 2
  const b = (left.b + right.b) / 2

  return {
    r,
    g,
    b,
    greenMean: g,
    leftArea: left.area,
    rightArea: right.area,
  }
}

/** Old name kept as a thin alias for any external callers. */
export const sampleCheekGreen = sampleCheekRgb

function meanRgbInPolygon(
  ctx: CanvasRenderingContext2D,
  poly: { x: number; y: number }[],
  frameWidth: number,
  frameHeight: number
): { r: number; g: number; b: number; area: number } | null {
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity
  for (const p of poly) {
    if (p.x < minX) minX = p.x
    if (p.x > maxX) maxX = p.x
    if (p.y < minY) minY = p.y
    if (p.y > maxY) maxY = p.y
  }
  minX = Math.max(0, Math.floor(minX))
  minY = Math.max(0, Math.floor(minY))
  maxX = Math.min(frameWidth - 1, Math.ceil(maxX))
  maxY = Math.min(frameHeight - 1, Math.ceil(maxY))
  const w = maxX - minX + 1
  const h = maxY - minY + 1
  if (w <= 0 || h <= 0) return null

  let imageData: ImageData
  try {
    imageData = ctx.getImageData(minX, minY, w, h)
  } catch {
    return null
  }
  const buf = imageData.data

  let rSum = 0
  let gSum = 0
  let bSum = 0
  let count = 0
  for (let py = 0; py < h; py++) {
    const worldY = minY + py + 0.5
    for (let px = 0; px < w; px++) {
      const worldX = minX + px + 0.5
      if (pointInPolygon(worldX, worldY, poly)) {
        const i = (py * w + px) * 4
        rSum += buf[i]
        gSum += buf[i + 1]
        bSum += buf[i + 2]
        count++
      }
    }
  }
  if (count === 0) return null
  return { r: rSum / count, g: gSum / count, b: bSum / count, area: count }
}

function pointInPolygon(
  x: number,
  y: number,
  poly: { x: number; y: number }[]
): boolean {
  let inside = false
  const n = poly.length
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = poly[i].x
    const yi = poly[i].y
    const xj = poly[j].x
    const yj = poly[j].y
    const intersect =
      yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi
    if (intersect) inside = !inside
  }
  return inside
}

export { LEFT_CHEEK_RING, RIGHT_CHEEK_RING }
