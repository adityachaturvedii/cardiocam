import FFT from 'fft.js'

// Port of the signal-processing pipeline that runs in Python
// process.py + signal_processing.py. Kept as pure functions — no state,
// no React — so they're trivially unit-testable against synthetic signals.

/** Remove a best-fit linear trend from the data (scipy.signal.detrend default). */
export function detrend(data: Float64Array): Float64Array {
  const n = data.length
  if (n < 2) return new Float64Array(data)
  // Linear least-squares: y = a + b*x where x = 0..n-1
  let sx = 0
  let sy = 0
  let sxx = 0
  let sxy = 0
  for (let i = 0; i < n; i++) {
    const x = i
    const y = data[i]
    sx += x
    sy += y
    sxx += x * x
    sxy += x * y
  }
  const denom = n * sxx - sx * sx
  const b = denom === 0 ? 0 : (n * sxy - sx * sy) / denom
  const a = (sy - b * sx) / n
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    out[i] = data[i] - (a + b * i)
  }
  return out
}

/** Return a symmetric Hamming window of length n (numpy.hamming default). */
export function hammingWindow(n: number): Float64Array {
  const w = new Float64Array(n)
  if (n === 1) {
    w[0] = 1
    return w
  }
  const twoPi = 2 * Math.PI
  for (let i = 0; i < n; i++) {
    w[i] = 0.54 - 0.46 * Math.cos((twoPi * i) / (n - 1))
  }
  return w
}

/** L2 norm of a vector. */
export function l2Norm(v: Float64Array): number {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i] * v[i]
  return Math.sqrt(s)
}

export interface BpmEstimate {
  /** Peak-BPM candidate in the 50–180 band; always set. */
  bpm: number
  /** Peak power divided by in-band median power. */
  snr: number
  /** Full FFT power spectrum over the in-band freqs (same length as freqsBpm). */
  fft: Float64Array
  /** Frequency axis in BPM, same length as fft. */
  freqsBpm: Float64Array
  /** Estimated sample rate used for the frequency axis. */
  fps: number
}

/**
 * Compute a BPM estimate from a single full-buffer run.
 *
 * Mirrors Python process.py line-for-line: detrend → Hamming window →
 * L2 normalize → FFT zero-padded to nFft → power spectrum → restrict
 * to 50–180 BPM → argmax. SNR = peak / median.
 */
export function estimateBpm(
  buffer: Float64Array,
  times: Float64Array,
  nFft: number,
  bpmMin = 50,
  bpmMax = 180
): BpmEstimate {
  const L = buffer.length
  // L samples captured in times[L-1] - times[0] seconds span L-1 intervals,
  // so sample rate = (L-1) / duration. The Python code had this wrong as
  // L/duration for years; we fixed that in Phase 2 and keep the correction.
  const duration = times[L - 1] - times[0]
  const fps = duration > 0 ? (L - 1) / duration : 30

  // Resample the raw buffer onto an even time grid (Python uses linear
  // interpolation between observed times so jitter in frame delivery does
  // not distort the spectrum).
  const evenTimes = new Float64Array(L)
  for (let i = 0; i < L; i++) {
    evenTimes[i] = times[0] + (i * duration) / (L - 1)
  }
  const detrended = detrend(buffer)
  const interpolated = new Float64Array(L)
  interpolate(evenTimes, times, detrended, interpolated)

  // Window + L2 normalize.
  const win = hammingWindow(L)
  for (let i = 0; i < L; i++) interpolated[i] *= win[i]
  const norm = l2Norm(interpolated)
  if (norm > 0) {
    for (let i = 0; i < L; i++) interpolated[i] /= norm
  }
  // Python multiplies by 30 before the FFT. Does not affect the peak
  // position but preserves numerical scale parity with the reference.
  for (let i = 0; i < L; i++) interpolated[i] *= 30

  // Zero-padded real FFT via fft.js (requires power-of-two length).
  const fft = new FFT(nFft)
  const input = fft.createComplexArray()
  const output = fft.createComplexArray()
  // fft.js complex layout: [re0, im0, re1, im1, ...]
  for (let i = 0; i < input.length; i++) input[i] = 0
  for (let i = 0; i < L; i++) input[2 * i] = interpolated[i]
  fft.transform(output, input)

  // Build power spectrum for the non-negative freqs only (first nFft/2 + 1).
  const halfBins = nFft / 2 + 1
  const power = new Float64Array(halfBins)
  for (let i = 0; i < halfBins; i++) {
    const re = output[2 * i]
    const im = output[2 * i + 1]
    power[i] = re * re + im * im
  }

  // Restrict to 50–180 BPM band.
  const binBpm = (60 * fps) / nFft
  const loIdx = Math.ceil(bpmMin / binBpm)
  const hiIdx = Math.min(halfBins - 1, Math.floor(bpmMax / binBpm))
  const bandLen = Math.max(0, hiIdx - loIdx + 1)
  const bandFft = new Float64Array(bandLen)
  const bandFreqs = new Float64Array(bandLen)
  for (let i = 0; i < bandLen; i++) {
    bandFft[i] = power[loIdx + i]
    bandFreqs[i] = (loIdx + i) * binBpm
  }

  let peakIdx = 0
  let peakVal = 0
  for (let i = 0; i < bandLen; i++) {
    if (bandFft[i] > peakVal) {
      peakVal = bandFft[i]
      peakIdx = i
    }
  }
  const bpm = bandLen > 0 ? bandFreqs[peakIdx] : 0
  const median = bandLen > 0 ? medianOf(bandFft) : 0
  const snr = median > 0 ? peakVal / median : 0

  return { bpm, snr, fft: bandFft, freqsBpm: bandFreqs, fps }
}

/** numpy.interp: monotonically increasing xp assumed. */
function interpolate(
  x: Float64Array,
  xp: Float64Array,
  fp: Float64Array,
  out: Float64Array
) {
  const n = x.length
  const m = xp.length
  let j = 0
  for (let i = 0; i < n; i++) {
    const xi = x[i]
    while (j < m - 2 && xp[j + 1] < xi) j++
    if (xi <= xp[0]) out[i] = fp[0]
    else if (xi >= xp[m - 1]) out[i] = fp[m - 1]
    else {
      const t = (xi - xp[j]) / (xp[j + 1] - xp[j])
      out[i] = fp[j] + t * (fp[j + 1] - fp[j])
    }
  }
}

/** Median of a Float64Array via a copied sort. OK for 100-ish elements. */
function medianOf(a: Float64Array): number {
  const s = Array.from(a).sort((x, y) => x - y)
  const n = s.length
  if (n === 0) return 0
  return n % 2 === 0 ? (s[n / 2 - 1] + s[n / 2]) / 2 : s[(n - 1) / 2]
}
