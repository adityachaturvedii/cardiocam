// Extract the mean green-channel intensity from two cheek patches.
//
// Unlike the Python desktop app, we don't rotate-and-crop the face first.
// The signal is a scalar (mean green over ROI skin pixels) and is
// rotation-invariant — cropping an axis-aligned rectangle straight from
// the source frame gives a slightly different but equally valid skin
// patch. The aligned-face step in Python was architectural, not
// accuracy-critical.
//
// We use polygon fills over MediaPipe Face Mesh cheek landmarks instead
// of bounding rectangles to avoid catching non-skin pixels (eyebrows,
// lip edge) at extreme head angles.

import type { NormalizedLandmark } from '@mediapipe/tasks-vision'

// Cheek-apex polygons on the MediaPipe 478-point Face Mesh. These bound the
// zygomatic region — high, well clear of the nasolabial fold, the lips, and
// the nose — so the green-channel mean samples clean cheek skin with no
// hair or lip edge contamination.
//
// Subject-left cheek (viewer's-right side of the image).
const LEFT_CHEEK_RING = [345, 352, 411, 425, 266, 371, 355]
// Subject-right cheek (viewer's-left side of the image).
const RIGHT_CHEEK_RING = [116, 123, 187, 205, 36, 142, 126]

export interface CheekSample {
  greenMean: number
  leftArea: number
  rightArea: number
}

/**
 * Sample the two cheek ROIs from the current video frame and return the
 * mean green-channel intensity. Returns null if either ROI ends up with
 * no pixels (face partly off-frame).
 */
export function sampleCheekGreen(
  landmarks: NormalizedLandmark[],
  ctx: CanvasRenderingContext2D,
  frameWidth: number,
  frameHeight: number
): CheekSample | null {
  // Pull out the two polygons in pixel coordinates.
  const leftPoly = LEFT_CHEEK_RING.map((i) => ({
    x: landmarks[i].x * frameWidth,
    y: landmarks[i].y * frameHeight,
  }))
  const rightPoly = RIGHT_CHEEK_RING.map((i) => ({
    x: landmarks[i].x * frameWidth,
    y: landmarks[i].y * frameHeight,
  }))

  const left = meanGreenInPolygon(ctx, leftPoly, frameWidth, frameHeight)
  const right = meanGreenInPolygon(ctx, rightPoly, frameWidth, frameHeight)
  if (left === null || right === null) return null

  return {
    greenMean: (left.mean + right.mean) / 2,
    leftArea: left.area,
    rightArea: right.area,
  }
}

function meanGreenInPolygon(
  ctx: CanvasRenderingContext2D,
  poly: { x: number; y: number }[],
  frameWidth: number,
  frameHeight: number
): { mean: number; area: number } | null {
  // Compute AABB of the polygon, clamped to frame bounds.
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

  // getImageData for just this region, then ray-cast each pixel against
  // the polygon to decide inclusion. 478-point landmarks give us polygons
  // ~20-40px across, so the pixel count is bounded and this runs in <1ms.
  let imageData: ImageData
  try {
    imageData = ctx.getImageData(minX, minY, w, h)
  } catch {
    return null
  }
  const buf = imageData.data

  let greenSum = 0
  let count = 0
  for (let py = 0; py < h; py++) {
    const worldY = minY + py + 0.5
    for (let px = 0; px < w; px++) {
      const worldX = minX + px + 0.5
      if (pointInPolygon(worldX, worldY, poly)) {
        const i = (py * w + px) * 4
        greenSum += buf[i + 1]
        count++
      }
    }
  }
  if (count === 0) return null
  return { mean: greenSum / count, area: count }
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
