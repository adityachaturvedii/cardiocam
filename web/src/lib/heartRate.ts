import { estimateBpm } from './signalProcessing'

export interface HeartRateState {
  /** Valid instantaneous BPM (FFT peak in band). 0 until the SNR gate opens. */
  bpm: number
  /** Peak / in-band-median ratio. Below snrThreshold the reading is gated. */
  snr: number
  /** True when SNR + warmup both pass and the BPM should be trusted. */
  valid: boolean
  /** Current estimated capture sample rate in Hz. */
  fps: number
  /** 0-1 progress while the initial 100-sample buffer is filling. */
  warmup: number
  /** Most recent detrended signal waveform, for plotting. */
  signal: Float64Array
  /** Most recent in-band FFT power, for plotting. */
  fft: Float64Array
  /** Frequency axis (BPM) aligned with fft. */
  freqsBpm: Float64Array
  /** Smoothed BPM across a trailing window of valid estimates. */
  stableBpm: number
}

export class HeartRateEstimator {
  private readonly bufferSize = 100
  private readonly nFft = 1024
  private readonly snrThreshold = 4
  private readonly warmupFrames = 10

  // Rolling buffer of green-channel means and their wall-clock timestamps (s).
  private samples: number[] = []
  private timestamps: number[] = []

  private framesSinceFull = 0
  private validBpms: number[] = []

  private state: HeartRateState = {
    bpm: 0,
    snr: 0,
    valid: false,
    fps: 0,
    warmup: 0,
    signal: new Float64Array(0),
    fft: new Float64Array(0),
    freqsBpm: new Float64Array(0),
    stableBpm: 0,
  }

  /** Returns a read-only snapshot of the latest estimator state. */
  getState(): HeartRateState {
    return this.state
  }

  reset() {
    this.samples = []
    this.timestamps = []
    this.framesSinceFull = 0
    this.validBpms = []
    this.state = {
      bpm: 0,
      snr: 0,
      valid: false,
      fps: 0,
      warmup: 0,
      signal: new Float64Array(0),
      fft: new Float64Array(0),
      freqsBpm: new Float64Array(0),
      stableBpm: 0,
    }
  }

  /**
   * Feed one frame's green-channel mean into the estimator. Returns the
   * updated state for convenience. Timestamp is seconds-since-anything
   * monotonic (e.g. performance.now() / 1000).
   */
  pushSample(green: number, tSeconds: number): HeartRateState {
    if (!Number.isFinite(green)) return this.state
    // Outlier clamp — matches Python process.py guard against sudden jumps
    // (e.g. when a finger brushes the ROI). Once the buffer is full and a
    // new sample differs from the running mean by more than 10, fall back
    // to the previous sample so the buffer doesn't get poisoned.
    const L = this.samples.length
    if (L > 99) {
      let mean = 0
      for (let i = 0; i < L; i++) mean += this.samples[i]
      mean /= L
      if (Math.abs(green - mean) > 10) {
        green = this.samples[L - 1]
      }
    }
    this.samples.push(green)
    this.timestamps.push(tSeconds)

    // Keep the buffer bounded.
    if (this.samples.length > this.bufferSize) {
      this.samples.shift()
      this.timestamps.shift()
    }

    const len = this.samples.length
    const warmup = Math.min(1, len / this.bufferSize)

    if (len < this.bufferSize) {
      this.state = {
        ...this.state,
        warmup,
        valid: false,
        bpm: 0,
        snr: 0,
      }
      return this.state
    }

    // Full buffer: run the DSP pipeline.
    const buf = new Float64Array(this.samples)
    const ts = new Float64Array(this.timestamps)
    const est = estimateBpm(buf, ts, this.nFft)

    this.framesSinceFull++
    const pastWarmup = this.framesSinceFull >= this.warmupFrames
    const valid = pastWarmup && est.snr >= this.snrThreshold

    if (valid) {
      this.validBpms.push(est.bpm)
      if (this.validBpms.length > this.bufferSize / 2) this.validBpms.shift()
    }

    // Stable BPM: median of last N valid samples. More robust to one
    // outlier bin than the raw mean used in the Python GUI.
    let stable = 0
    if (this.validBpms.length >= 25) {
      const sorted = [...this.validBpms].sort((a, b) => a - b)
      stable = sorted[Math.floor(sorted.length / 2)]
    }

    this.state = {
      bpm: est.bpm,
      snr: est.snr,
      valid,
      fps: est.fps,
      warmup: 1,
      signal: buf,
      fft: est.fft,
      freqsBpm: est.freqsBpm,
      stableBpm: stable,
    }
    return this.state
  }
}
