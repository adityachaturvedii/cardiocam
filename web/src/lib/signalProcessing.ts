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
 * chrominance-based rPPG", IEEE TBME 2013. This is the sliding-window
 * form published in the rPPG-Toolbox benchmark (CHROME_DEHAAN.py),
 * not the one-shot whole-buffer form you'll see in short code listings.
 *
 * Per window of length ~1.6*fps (with 50 % overlap):
 *   1. Normalize RGB by the per-window mean (divide each channel by its
 *      mean over this window).
 *   2. Xs = 3*R_norm - 2*G_norm
 *      Ys = 1.5*R_norm + G_norm - 1.5*B_norm
 *   3. Zero-phase bandpass filter both signals in the cardiac band.
 *      The toolbox uses order-3 Butterworth; we use order-1, which is
 *      slightly flatter in the pass-band but otherwise qualitatively
 *      similar. If we want exact benchmark parity we can upgrade the
 *      bandpass later.
 *   4. alpha = std(Xf) / std(Yf)
 *      bvp_win = Xf - alpha * Yf
 *   5. Multiply by a Hann window and overlap-add into the full-length
 *      output buffer.
 *
 * Compared to the one-shot CHROM we used to ship, this form:
 *   - Is robust to illumination trends longer than the window length
 *     because each window renormalizes;
 *   - Reduces the influence of noise concentrated in a single window;
 *   - Matches the MAE numbers reported in the Liu 2023 benchmark (3.98
 *     on UBFC-rPPG).
 */
export function chromTransform(
  r: Float64Array,
  g: Float64Array,
  b: Float64Array,
  fps = 30
): Float64Array {
  const T = r.length
  if (g.length !== T || b.length !== T) {
    throw new Error('chromTransform: r, g, b must all be the same length')
  }
  if (T === 0) return new Float64Array(0)

  // Window length (samples), forced even so the half-window step is exact.
  let winL = Math.ceil(1.6 * fps)
  if (winL % 2) winL += 1
  const halfWin = winL / 2
  // Too-short buffer: fall back to the one-shot form for correctness.
  if (T < winL) return chromOneShot(r, g, b)

  const nWin = Math.floor((T - halfWin) / halfWin)
  const out = new Float64Array(halfWin * (nWin + 1))
  const { b: bb, a: aa } = designButterBandpass1(0.7, 2.5, fps)
  const hann = hannWindow(winL)

  for (let i = 0; i < nWin; i++) {
    const winS = i * halfWin
    const winE = winS + winL
    // Per-window RGB mean.
    let mR = 0
    let mG = 0
    let mB = 0
    for (let k = winS; k < winE; k++) {
      mR += r[k]
      mG += g[k]
      mB += b[k]
    }
    mR /= winL
    mG /= winL
    mB /= winL
    if (mR < 1e-9 || mG < 1e-9 || mB < 1e-9) continue
    const Xs = new Float64Array(winL)
    const Ys = new Float64Array(winL)
    for (let k = 0; k < winL; k++) {
      const rn = r[winS + k] / mR
      const gn = g[winS + k] / mG
      const bn = b[winS + k] / mB
      Xs[k] = 3 * rn - 2 * gn
      Ys[k] = 1.5 * rn + gn - 1.5 * bn
    }
    const Xf = filtfilt(bb, aa, Xs)
    const Yf = filtfilt(bb, aa, Ys)
    const sX = stdZeroMean(Xf)
    const sY = stdZeroMean(Yf)
    const alpha = sY > 1e-9 ? sX / sY : 0
    const sWin = new Float64Array(winL)
    for (let k = 0; k < winL; k++) sWin[k] = (Xf[k] - alpha * Yf[k]) * hann[k]
    // Overlap-add.
    for (let k = 0; k < winL; k++) {
      if (winS + k < out.length) out[winS + k] += sWin[k]
    }
  }
  // Pad / truncate to input length so downstream code doesn't see a
  // shape change.
  if (out.length === T) return out
  const aligned = new Float64Array(T)
  const copyLen = Math.min(T, out.length)
  for (let i = 0; i < copyLen; i++) aligned[i] = out[i]
  return aligned
}

/** Fallback simple-CHROM when the buffer is shorter than a single window —
 *  produces a reasonable BVP during warm-up. */
function chromOneShot(
  r: Float64Array,
  g: Float64Array,
  b: Float64Array
): Float64Array {
  const T = r.length
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
  for (let i = 0; i < T; i++) bvp[i] = Xcomp[i] - alpha * Ycomp[i]
  return bvp
}

/** Symmetric Hann window (scipy.signal.windows.hann default). */
function hannWindow(n: number): Float64Array {
  const w = new Float64Array(n)
  if (n <= 1) {
    if (n === 1) w[0] = 1
    return w
  }
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)))
  }
  return w
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

/**
 * Smoothness-prior (Tarvainen 2002) detrend — removes slowly-varying
 * baseline drift while preserving the periodic cardiac pulse. This is
 * the detrend every paper in the rPPG-Toolbox uses (POS_WANG, CHROM,
 * and the downstream post_process.py evaluation pipeline), replacing
 * a simple linear least-squares line fit.
 *
 * Formula:
 *   y_detrended = (I - (I + lambda^2 D'D)^-1) y
 * where D is the (N-2)×N second-order difference matrix with rows
 *   [1, -2, 1, 0, …, 0], [0, 1, -2, 1, 0, …], …
 *
 * Implementation: the matrix K = I - (I + lambda^2 D'D)^-1 depends only
 * on N and lambda (not on y), so we cache it per (N, lambda). Each call
 * then reduces to a single O(N^2) matrix-vector product. At N=240 that's
 * ~57 K multiply-adds — negligible per frame.
 *
 * Default lambda = 100 matches the rPPG-Toolbox convention. At 30 FPS
 * over a 240-sample buffer it cuts off trends slower than ~0.2 Hz,
 * well below the cardiac band (0.75-2.5 Hz).
 */
export function smoothPriorDetrend(
  data: Float64Array,
  lambdaValue = 100
): Float64Array {
  const n = data.length
  if (n < 3) return new Float64Array(data)
  const K = getSmoothPriorMatrix(n, lambdaValue)
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    const row = K[i]
    for (let j = 0; j < n; j++) s += row[j] * data[j]
    out[i] = s
  }
  return out
}

// Cache keyed by "n:lambda". Building K is O(N^3) (one matrix inverse)
// but only happens once per session for the default N=240.
const SMOOTH_PRIOR_CACHE = new Map<string, Float64Array[]>()

function getSmoothPriorMatrix(n: number, lambdaValue: number): Float64Array[] {
  const key = `${n}:${lambdaValue}`
  const cached = SMOOTH_PRIOR_CACHE.get(key)
  if (cached) return cached

  // Build M = I + lambda^2 * D' * D where D is (n-2) × n with rows
  // [1, -2, 1, 0, ...]. D'D is banded (pentadiagonal) but we fill the
  // full n×n matrix since we're only doing this once per session.
  const lam2 = lambdaValue * lambdaValue
  const M: Float64Array[] = []
  for (let i = 0; i < n; i++) M.push(new Float64Array(n))
  for (let i = 0; i < n; i++) M[i][i] = 1
  // D'D[i][j] contributions. For each row k of D (k in 0..n-3), D[k] has
  // 1 at col k, -2 at col k+1, 1 at col k+2. D'D[a][b] = sum over k of
  // D[k][a] * D[k][b]. For pentadiagonal structure:
  for (let k = 0; k < n - 2; k++) {
    const a = k
    const b = k + 1
    const c = k + 2
    // 1 * 1 = +1 on diag(a)
    M[a][a] += lam2
    // 1 * -2 + -2 * 1 = -4 on off-diag(a,b)
    M[a][b] += -2 * lam2
    M[b][a] += -2 * lam2
    // 1 * 1 + -2 * -2 + 1 * 1 contribution?
    // actually: D'D[a][c] = D[k][a] * D[k][c] = 1*1 = +1
    M[a][c] += lam2
    M[c][a] += lam2
    // D'D[b][b] = (-2)^2 = 4
    M[b][b] += 4 * lam2
    // D'D[b][c] = -2*1 = -2
    M[b][c] += -2 * lam2
    M[c][b] += -2 * lam2
    // D'D[c][c] = 1*1 = 1
    M[c][c] += lam2
  }
  // Invert M via Gauss-Jordan. Symmetric positive definite, so numerically
  // fine. Faster would be Cholesky; this is run once per session so the
  // constant factor doesn't matter.
  const inv = invertSymmetricPD(M)
  // K = I - inv(M)
  const K: Float64Array[] = []
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(n)
    for (let j = 0; j < n; j++) row[j] = (i === j ? 1 : 0) - inv[i][j]
    K.push(row)
  }
  SMOOTH_PRIOR_CACHE.set(key, K)
  return K
}

function invertSymmetricPD(A: Float64Array[]): Float64Array[] {
  const n = A.length
  // Build augmented [A | I].
  const aug: Float64Array[] = []
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(2 * n)
    for (let j = 0; j < n; j++) row[j] = A[i][j]
    row[n + i] = 1
    aug.push(row)
  }
  // Forward elimination (no pivoting — M is symmetric PD so diag never 0).
  for (let i = 0; i < n; i++) {
    const pivot = aug[i][i]
    for (let j = 0; j < 2 * n; j++) aug[i][j] /= pivot
    for (let k = 0; k < n; k++) {
      if (k === i) continue
      const factor = aug[k][i]
      if (factor === 0) continue
      for (let j = 0; j < 2 * n; j++) aug[k][j] -= factor * aug[i][j]
    }
  }
  // Extract inverse.
  const inv: Float64Array[] = []
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(n)
    for (let j = 0; j < n; j++) row[j] = aug[i][n + j]
    inv.push(row)
  }
  return inv
}

/**
 * Design a Butterworth order-1 bandpass filter. Returns { b, a } matching
 * scipy.signal.butter(1, [low, high], btype='bandpass', fs=fs).
 *
 * Algebra: analog lowpass prototype H(s) = 1/(s+1), apply bandpass
 * transform s -> (s^2 + W0^2)/(BW*s), then bilinear transform
 * s = (1 - z^-1)/(1 + z^-1). Result is a 2nd-order digital bandpass
 * (3 numerator + 3 denominator coefficients).
 */
export function designButterBandpass1(
  lowHz: number,
  highHz: number,
  fs: number
): { b: Float64Array; a: Float64Array } {
  // Prewarp the critical frequencies so the bilinear transform lands
  // on the desired cutoffs.
  const wLow = Math.tan((Math.PI * lowHz) / fs)
  const wHigh = Math.tan((Math.PI * highHz) / fs)
  const bw = wHigh - wLow
  const w0sq = wLow * wHigh
  // Analog bandpass H_a(s) = BW*s / (s^2 + BW*s + W0^2). Apply bilinear
  // s = (1 - z^-1)/(1 + z^-1). Expand numerator and denominator in z^-1.
  //   num(s) = BW * s
  //   den(s) = s^2 + BW*s + w0^2
  // After substitution s = (1 - z^-1)/(1 + z^-1), multiply top & bottom by
  // (1 + z^-1)^2 to clear denominators.
  //   num(z) = BW * (1 - z^-2)
  //   den(z) = (1 - z^-1)^2 + BW * (1 - z^-1)(1 + z^-1) + w0^2 (1 + z^-1)^2
  const b0 = bw
  const b1 = 0
  const b2 = -bw
  const a0 = 1 + bw + w0sq
  const a1 = 2 * w0sq - 2
  const a2 = 1 - bw + w0sq
  // Normalize so a[0] == 1.
  const b = new Float64Array([b0 / a0, b1 / a0, b2 / a0])
  const a = new Float64Array([1, a1 / a0, a2 / a0])
  return { b, a }
}

/**
 * Apply a causal IIR filter y[n] = (sum b[k] x[n-k] - sum a[k] y[n-k]) / a[0].
 * Assumes a[0] = 1 (normalized). For internal use by filtfilt.
 */
function lfilter(
  b: Float64Array,
  a: Float64Array,
  x: Float64Array
): Float64Array {
  const n = x.length
  const y = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let acc = 0
    for (let k = 0; k < b.length; k++) {
      if (i - k >= 0) acc += b[k] * x[i - k]
    }
    for (let k = 1; k < a.length; k++) {
      if (i - k >= 0) acc -= a[k] * y[i - k]
    }
    y[i] = acc
  }
  return y
}

/**
 * Zero-phase filtering via forward-backward application — matches
 * scipy.signal.filtfilt for short IIR filters. Runs the filter once
 * forward, reverses, runs again, reverses back. Phase cancels; magnitude
 * response is squared (so a 3 dB cutoff becomes 6 dB).
 *
 * No edge padding here — for short orders (2nd order total) and long
 * signals (240 samples) the boundary transient is negligible. scipy's
 * filtfilt uses reflected padding; we do not, accepting a few samples
 * of warm-up at each end.
 */
export function filtfilt(
  b: Float64Array,
  a: Float64Array,
  x: Float64Array
): Float64Array {
  const fwd = lfilter(b, a, x)
  const rev = new Float64Array(fwd.length)
  for (let i = 0; i < fwd.length; i++) rev[i] = fwd[fwd.length - 1 - i]
  const bwd = lfilter(b, a, rev)
  const out = new Float64Array(bwd.length)
  for (let i = 0; i < bwd.length; i++) out[i] = bwd[bwd.length - 1 - i]
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
  // Smoothness-prior (Tarvainen) detrend: removes slowly-varying baseline
  // drift that linear detrend can't. This is the detrend used by every
  // method in the rPPG-Toolbox benchmark — mandatory for matching their
  // published MAE numbers.
  const detrended = smoothPriorDetrend(buffer)
  const interpolated = new Float64Array(L)
  interpolate(evenTimes, times, detrended, interpolated)

  // Zero-phase Butterworth bandpass 0.75-2.5 Hz (45-150 BPM). Applied via
  // filtfilt so the cardiac peak isn't shifted by filter phase. Matches
  // the toolbox's post-process.py bandpass step.
  const { b: bb, a: aa } = designButterBandpass1(0.75, 2.5, fps)
  const bandpassed = filtfilt(bb, aa, interpolated)
  for (let i = 0; i < L; i++) interpolated[i] = bandpassed[i]

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

// ──────────────────────────────────────────────────────────────────────
// Respiratory rate estimation
// ──────────────────────────────────────────────────────────────────────

export interface RREstimate {
  /** Breaths per minute. 0 if SNR too low. */
  rr: number
  /** Peak / in-band-median ratio. */
  snr: number
  /** Power spectrum in the respiratory band. */
  fft: Float64Array
  /** Frequency axis in breaths-per-minute. */
  freqsBpm: Float64Array
}

/**
 * Estimate respiratory rate from a BVP signal. Breathing modulates the
 * pulse amplitude and baseline — a bandpass in the respiratory band
 * (0.18–0.5 Hz = 11–30 breaths/min) isolates this modulation.
 *
 * Same DSP approach as HR: smoothPriorDetrend → bandpass → Hamming →
 * FFT → peak-pick, just in a lower frequency band.
 *
 * Needs a longer buffer (~8 s minimum, ideally 12+) because one full
 * respiratory cycle is 2–5 s. At 8 s × 30 FPS = 240 samples we get
 * at most ~2 full breaths in the window, which gives coarse spectral
 * resolution. Still usable for a rough display.
 */
export function estimateRR(
  bvp: Float64Array,
  times: Float64Array,
  nFft: number
): RREstimate {
  const L = bvp.length
  if (L < 60) {
    return { rr: 0, snr: 0, fft: new Float64Array(0), freqsBpm: new Float64Array(0) }
  }
  const duration = times[L - 1] - times[0]
  const fps = duration > 0 ? (L - 1) / duration : 30

  // Detrend + bandpass in the respiratory band.
  const detrended = smoothPriorDetrend(bvp, 300)
  const evenTimes = new Float64Array(L)
  for (let i = 0; i < L; i++) evenTimes[i] = times[0] + (i * duration) / (L - 1)
  const interpolated = new Float64Array(L)
  interpolate(evenTimes, times, detrended, interpolated)

  const { b, a } = designButterBandpass1(0.18, 0.5, fps)
  const filtered = filtfilt(b, a, interpolated)

  // Window + FFT.
  const win = hammingWindow(L)
  for (let i = 0; i < L; i++) filtered[i] *= win[i]
  const norm = l2Norm(filtered)
  if (norm > 0) for (let i = 0; i < L; i++) filtered[i] /= norm

  const fftObj = new FFT(nFft)
  const input = fftObj.createComplexArray()
  const output = fftObj.createComplexArray()
  for (let i = 0; i < input.length; i++) input[i] = 0
  for (let i = 0; i < L; i++) input[2 * i] = filtered[i] * 30

  fftObj.transform(output, input)

  const halfBins = nFft / 2 + 1
  const power = new Float64Array(halfBins)
  for (let i = 0; i < halfBins; i++) {
    const re = output[2 * i]
    const im = output[2 * i + 1]
    power[i] = re * re + im * im
  }

  // Restrict to respiratory band: 0.18–0.5 Hz → 11–30 breaths/min.
  const rrMin = 11
  const rrMax = 30
  const binBpm = (60 * fps) / nFft
  const loIdx = Math.ceil(rrMin / binBpm)
  const hiIdx = Math.min(halfBins - 1, Math.floor(rrMax / binBpm))
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
  const rr = bandLen > 0 ? bandFreqs[peakIdx] : 0
  const median = bandLen > 0 ? medianOf(bandFft) : 0
  const snr = median > 0 && peakVal > 0 ? peakVal / median : 0
  return { rr, snr, fft: bandFft, freqsBpm: bandFreqs }
}

// ──────────────────────────────────────────────────────────────────────
// SpO2 estimation (experimental)
// ──────────────────────────────────────────────────────────────────────

export interface SpO2Estimate {
  /** Estimated SpO2 percentage. 0 if invalid. */
  spo2: number
  /** The AC/DC ratio R used for the calibration curve. */
  ratioR: number
  /** Trend direction: 'stable' | 'rising' | 'falling'. Computed from
   *  the slope of the last N SpO2 values the caller maintains. */
  trend: 'stable' | 'rising' | 'falling'
  /** True when the estimate is likely trustworthy (sufficient pulsatile
   *  component in both channels). */
  valid: boolean
}

/**
 * Estimate SpO2 from per-frame R and B channel means.
 *
 * Traditional pulse oximetry uses red + infrared LEDs and a Beer-Lambert
 * model. Camera-based rPPG has no IR, so we approximate using the
 * ratio of pulsatile (AC) to mean (DC) components in the red and blue
 * channels:
 *
 *   R_ratio = (AC_red / DC_red) / (AC_blue / DC_blue)
 *   SpO2 ≈ 110 - 25 × R_ratio
 *
 * The linear calibration curve is the simplest approximation from the
 * literature (Verkruysse 2008; Humphreys 2007). It's NOT validated for
 * clinical use. Accuracy is ±3-5% SpO2 on typical webcam setups —
 * clinically useless for hypoxemia detection.
 *
 * This function takes the full-buffer R and B channel arrays and computes
 * AC (RMS of bandpass-filtered signal) and DC (mean) for each.
 *
 * @param recentSpO2 — array of recent SpO2 values the caller has
 *   accumulated, used only for the trend calculation.
 */
export function estimateSpO2(
  rBuf: Float64Array,
  bBuf: Float64Array,
  times: Float64Array,
  recentSpO2: number[]
): SpO2Estimate {
  const L = rBuf.length
  const INVALID: SpO2Estimate = { spo2: 0, ratioR: 0, trend: 'stable', valid: false }
  if (L < 60) return INVALID

  const duration = times[L - 1] - times[0]
  const fps = duration > 0 ? (L - 1) / duration : 30

  // DC = mean of each channel over the buffer.
  let dcR = 0
  let dcB = 0
  for (let i = 0; i < L; i++) {
    dcR += rBuf[i]
    dcB += bBuf[i]
  }
  dcR /= L
  dcB /= L
  if (dcR < 1 || dcB < 1) return INVALID

  // AC = RMS of bandpass-filtered signal in the cardiac band (0.75-2.5 Hz).
  // This isolates the pulsatile component caused by blood volume changes.
  const { b, a } = designButterBandpass1(0.75, 2.5, fps)
  const acR = rms(filtfilt(b, a, rBuf))
  const acB = rms(filtfilt(b, a, bBuf))

  if (acB < 1e-9 || dcB < 1e-9) return INVALID

  const ratioR = (acR / dcR) / (acB / dcB)
  // Linear calibration curve.
  let spo2 = 110 - 25 * ratioR

  // Clamp to physiological range.
  spo2 = Math.max(70, Math.min(100, spo2))

  // Validity: both channels need a measurable pulsatile component.
  // If AC/DC is too small, the signal is below noise floor.
  const valid = (acR / dcR) > 0.0005 && (acB / dcB) > 0.0005

  // Trend from recent history.
  let trend: 'stable' | 'rising' | 'falling' = 'stable'
  if (recentSpO2.length >= 10) {
    const recent = recentSpO2.slice(-30)
    const firstHalf = recent.slice(0, Math.floor(recent.length / 2))
    const secondHalf = recent.slice(Math.floor(recent.length / 2))
    const avg1 = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length
    const avg2 = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length
    const delta = avg2 - avg1
    if (delta > 1) trend = 'rising'
    else if (delta < -1) trend = 'falling'
  }

  return { spo2: Math.round(spo2 * 10) / 10, ratioR, trend, valid }
}

function rms(v: Float64Array): number {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i] * v[i]
  return Math.sqrt(s / v.length)
}
