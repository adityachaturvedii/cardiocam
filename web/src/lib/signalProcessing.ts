import FFT from 'fft.js'

// Port of the signal-processing pipeline that runs in Python
// process.py + signal_processing.py. Kept as pure functions — no state,
// no React — so they're trivially unit-testable against synthetic signals.

/**
 * POS (plane-orthogonal-to-skin) BVP extractor — Wang, den Brinker, Stuijk,
 * de Haan, "Algorithmic principles of remote PPG", IEEE TBME 2017. Replaces
 * the naive green-channel-only signal with a chrominance-style projection
 * that cancels specular reflections and lighting drift.
 *
 * Input: three channel-mean time series (R, G, B), same length, same fps.
 * Output: a single BVP time series of the same length, ready for the
 * existing detrend → Hamming → FFT pipeline.
 *
 * Implementation mirrors the pyVHR reference (MIT license). For a window
 * of length w = round(1.6 * fps): temporal-normalize, project through
 * P = [[0, 1, -1], [-2, 1, 1]], tune the two projected rows with an
 * std-ratio alpha, overlap-add into H.
 */
export function posTransform(
  r: Float64Array,
  g: Float64Array,
  b: Float64Array,
  fps: number
): Float64Array {
  const T = r.length
  if (g.length !== T || b.length !== T) {
    throw new Error('posTransform: r, g, b must all be the same length')
  }
  const H = new Float64Array(T)
  const w = Math.max(2, Math.round(1.6 * fps))
  if (T < w) return H // not enough history yet — all zeros

  const eps = 1e-9

  for (let n = w - 1; n < T; n++) {
    const m = n - w + 1

    // Temporal mean over the window, per channel.
    let mR = 0
    let mG = 0
    let mB = 0
    for (let k = m; k <= n; k++) {
      mR += r[k]
      mG += g[k]
      mB += b[k]
    }
    mR /= w
    mG /= w
    mB /= w
    if (mR < eps || mG < eps || mB < eps) {
      // Degenerate window (a fully black patch). Skip.
      continue
    }

    // Normalize and project onto two axes: S1 = G/meanG - B/meanB,
    // S2 = -2 R/meanR + G/meanG + B/meanB (Wang's P matrix applied to the
    // temporally-normalized RGB window).
    const S1 = new Float64Array(w)
    const S2 = new Float64Array(w)
    for (let i = 0; i < w; i++) {
      const k = m + i
      const rn = r[k] / mR
      const gn = g[k] / mG
      const bn = b[k] / mB
      S1[i] = gn - bn
      S2[i] = -2 * rn + gn + bn
    }

    // alpha = std(S1) / std(S2) — the "tuning" step.
    const sd1 = stdZeroMean(S1)
    const sd2 = stdZeroMean(S2)
    const alpha = sd1 / (sd2 + eps)

    // Hn = (S1 + alpha * S2) - mean — contribute to overlap-add.
    let sumHn = 0
    for (let i = 0; i < w; i++) sumHn += S1[i] + alpha * S2[i]
    const mHn = sumHn / w

    for (let i = 0; i < w; i++) {
      H[m + i] += S1[i] + alpha * S2[i] - mHn
    }
  }

  return H
}

/** Standard deviation assuming the series mean is already removed, matching
 *  the pyVHR POS reference which std's the detrended projected rows. */
function stdZeroMean(v: Float64Array): number {
  let mean = 0
  for (let i = 0; i < v.length; i++) mean += v[i]
  mean /= v.length
  let s = 0
  for (let i = 0; i < v.length; i++) {
    const d = v[i] - mean
    s += d * d
  }
  return Math.sqrt(s / v.length)
}

/**
 * OMIT (Orthogonal Matrix Image Transformation) BVP extractor —
 * Álvarez Casado & Bordallo López, "Face2PPG: An unsupervised pipeline
 * for blood volume pulse extraction from faces", arXiv:2202.04101 (2022).
 *
 * Algorithm, verbatim from the pyVHR reference (MIT license):
 *   Q, R = qr(X)              where X is the 3×T RGB sample matrix
 *   S    = Q[:, 0]            first column of Q — unit vector along
 *                             the dominant color direction
 *   P    = I - S Sᵀ            projector orthogonal to S
 *   Y    = P · X              project each frame out of the S direction
 *   BVP  = Y[1, :]             second row (green-like post-projection)
 *
 * Geometric note: for a 3×T matrix X (T ≫ 3) np.linalg.qr in reduced
 * mode gives Q ∈ ℝ^{3×3} and R ∈ ℝ^{3×T}. The first column of Q is
 * simply X[:, 0] normalized to unit length (with an optional sign
 * flip) — a direct consequence of X = Q·R with R upper-triangular.
 * We reproduce that specific behavior rather than a full orthonormal
 * basis of the column space, because matching the reference is the
 * point (the published benchmark numbers assume this form).
 *
 * Robust-to-compression claim (from the paper): the QR projection
 * handles small RGB quantization errors more gracefully than POS's
 * fixed projection matrix, particularly for H.264-compressed mobile
 * video streams.
 */
export function omitTransform(
  r: Float64Array,
  g: Float64Array,
  b: Float64Array
): Float64Array {
  const T = r.length
  if (g.length !== T || b.length !== T) {
    throw new Error('omitTransform: r, g, b must all be the same length')
  }
  if (T === 0) return new Float64Array(0)

  // S = normalized first column of X = [r[0], g[0], b[0]] / ‖·‖.
  const r0 = r[0]
  const g0 = g[0]
  const b0 = b[0]
  const norm = Math.sqrt(r0 * r0 + g0 * g0 + b0 * b0)
  if (norm < 1e-9) return new Float64Array(T)
  const s0 = r0 / norm
  const s1 = g0 / norm
  const s2 = b0 / norm

  // Y = (I - S Sᵀ) X, extract row 1 (green-like).
  // Per-frame projection: y_i = X_i - (S · X_i) * S, we want y_i[1].
  const out = new Float64Array(T)
  for (let i = 0; i < T; i++) {
    const dot = s0 * r[i] + s1 * g[i] + s2 * b[i]
    out[i] = g[i] - dot * s1
  }
  return out
}

/**
 * CHROM BVP extractor — de Haan & Jeanne, "Robust pulse rate from
 * chrominance-based rPPG", IEEE TBME 2013. Older than POS, simpler math:
 * one pass over the full buffer, no sliding window.
 *
 *   Xcomp = 3*R - 2*G
 *   Ycomp = 1.5*R + G - 1.5*B
 *   alpha = std(Xcomp) / std(Ycomp)
 *   BVP   = Xcomp - alpha * Ycomp
 *
 * Hardcoded RGB coefficients are calibrated for Caucasian skin in the
 * original paper; CHROM generally trails POS slightly on diverse skin
 * tones but in the Liu 2023 benchmark table it was within 0.5 BPM of
 * POS on UBFC-rPPG and UBFC-Phys. Shipping both gives users an easy A/B.
 */
export function chromTransform(
  r: Float64Array,
  g: Float64Array,
  b: Float64Array
): Float64Array {
  const T = r.length
  if (g.length !== T || b.length !== T) {
    throw new Error('chromTransform: r, g, b must all be the same length')
  }
  const Xcomp = new Float64Array(T)
  const Ycomp = new Float64Array(T)
  for (let i = 0; i < T; i++) {
    Xcomp[i] = 3 * r[i] - 2 * g[i]
    Ycomp[i] = 1.5 * r[i] + g[i] - 1.5 * b[i]
  }
  const sX = stdZeroMean(Xcomp)
  const sY = stdZeroMean(Ycomp)
  const alpha = sY > 1e-9 ? sX / sY : 0
  const bvp = new Float64Array(T)
  for (let i = 0; i < T; i++) {
    bvp[i] = Xcomp[i] - alpha * Ycomp[i]
  }
  return bvp
}

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
  /** Peak-BPM candidate inside the configured band; always set. */
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
 * Pipeline: detrend → Hamming window → L2 normalize → zero-padded FFT →
 * power spectrum → restrict to [bpmMin, bpmMax] → argmax. SNR = peak /
 * in-band median power.
 *
 * Optional arguments:
 *   priorBpm      if set, restricts the argmax search to the window
 *                 [priorBpm - maxDeltaBpm, priorBpm + maxDeltaBpm] ∩
 *                 [bpmMin, bpmMax]. Useful once we have a stable prior:
 *                 FFT noise can crown a spurious peak far from the true
 *                 HR, and instantaneous HR can't change by tens of BPM
 *                 between frames, so searching only a physiologically
 *                 plausible window suppresses spurious jumps. SNR is
 *                 still computed against the full-band median so the
 *                 validity gate behaves the same.
 *   maxDeltaBpm   half-width of the prior window. Default 20 BPM.
 *   priorSigmaBpm if set along with priorBpm, applies a soft Gaussian
 *                 bias to the peak-picking score inside the prior window:
 *                 score[i] = fft[i] * exp(-0.5 * ((bpm[i] - prior) / sigma)^2).
 *                 Disambiguates cases where two peaks are comparable in
 *                 amplitude but only one is physiologically consistent
 *                 with the recent rolling median (e.g. cardiac at 75 vs
 *                 respiration harmonic at 90 in the same spectrum). The
 *                 raw FFT returned for plotting is unaffected — only the
 *                 selected BPM changes.
 */
export function estimateBpm(
  buffer: Float64Array,
  times: Float64Array,
  nFft: number,
  bpmMin = 45,
  bpmMax = 150,
  priorBpm: number | null = null,
  maxDeltaBpm = 20,
  priorSigmaBpm: number | null = null
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

  const { bpm, snr } = pickPeak(
    bandFft,
    bandFreqs,
    priorBpm,
    maxDeltaBpm,
    priorSigmaBpm
  )
  return { bpm, snr, fft: bandFft, freqsBpm: bandFreqs, fps }
}

/**
 * Pick the argmax BPM from an in-band power spectrum, with optional
 * prior-based window clamp and Gaussian bias. Returns the selected BPM
 * plus the SNR (peak / in-band median), computed on the RAW spectrum
 * values so the validity gate reflects genuine spectral strength
 * regardless of any bias weighting.
 *
 * Exposed for callers that maintain their own accumulated spectrum
 * (e.g. HeartRateEstimator's time-weighted spectrum accumulator) and
 * want to run peak-picking without re-running the whole FFT.
 */
export function pickPeak(
  bandFft: Float64Array,
  bandFreqs: Float64Array,
  priorBpm: number | null,
  maxDeltaBpm: number,
  priorSigmaBpm: number | null
): { bpm: number; snr: number; peakIdx: number } {
  const bandLen = bandFft.length
  let searchLo = 0
  let searchHi = bandLen - 1
  if (priorBpm !== null && bandLen > 0) {
    const lo = priorBpm - maxDeltaBpm
    const hi = priorBpm + maxDeltaBpm
    let newLo = -1
    let newHi = -1
    for (let i = 0; i < bandLen; i++) {
      if (bandFreqs[i] >= lo && newLo < 0) newLo = i
      if (bandFreqs[i] <= hi) newHi = i
    }
    if (newLo >= 0 && newHi >= newLo) {
      searchLo = newLo
      searchHi = newHi
    }
  }

  let peakIdx = searchLo
  let peakScore = -Infinity
  const useBias =
    priorBpm !== null && priorSigmaBpm !== null && priorSigmaBpm > 0
  const sigma = priorSigmaBpm ?? 1
  const sigma2 = 2 * sigma * sigma
  for (let i = searchLo; i <= searchHi; i++) {
    let score = bandFft[i]
    if (useBias) {
      const d = bandFreqs[i] - (priorBpm as number)
      score *= Math.exp(-(d * d) / sigma2)
    }
    if (score > peakScore) {
      peakScore = score
      peakIdx = i
    }
  }
  const bpm = bandLen > 0 ? bandFreqs[peakIdx] : 0
  const peakVal = bandLen > 0 ? bandFft[peakIdx] : 0
  const median = bandLen > 0 ? medianOf(bandFft) : 0
  const snr = median > 0 && peakVal > 0 ? peakVal / median : 0
  return { bpm, snr, peakIdx }
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
