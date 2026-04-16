import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useCamera } from '../hooks/useCamera'
import { useFaceLandmarker } from '../hooks/useFaceLandmarker'
import {
  HeartRateEstimator,
  type BvpMethod,
  type HeartRateState,
} from '../lib/heartRate'
import {
  FOREHEAD_RING,
  LEFT_CHEEK_RING,
  RIGHT_CHEEK_RING,
  sampleCheekRgb,
} from '../lib/roi'
import Footer from '../components/Footer'
import HeartIcon from '../components/HeartIcon'
import SignalPlot from '../components/SignalPlot'
import FFTPlot from '../components/FFTPlot'

const INITIAL_HR_STATE: HeartRateState = {
  bpm: 0,
  snr: 0,
  valid: false,
  fps: 0,
  warmup: 0,
  signal: new Float64Array(0),
  fft: new Float64Array(0),
  freqsBpm: new Float64Array(0),
  stableBpm: 0,
  stableReadable: false,
  framesSinceValid: Infinity,
  method: 'POS',
}

export default function Reader() {
  const camera = useCamera()
  const landmarker = useFaceLandmarker()
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const samplingCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const rafRef = useRef<number | null>(null)
  const estimatorRef = useRef<HeartRateEstimator>(new HeartRateEstimator())

  const [faceDetected, setFaceDetected] = useState(false)
  const [fps, setFps] = useState(0)
  const [hr, setHr] = useState<HeartRateState>(INITIAL_HR_STATE)
  // BVP extractor choice. POS by default; GREEN exists so a user can A/B
  // against the pre-Apr-2026 behavior on their own face. Swapping resets
  // the estimator because the two methods' signal shapes differ.
  const [method, setMethodState] = useState<BvpMethod>('POS')

  const runLoop = useCallback(() => {
    const video = camera.videoRef.current
    const overlay = overlayCanvasRef.current
    const sampling = samplingCanvasRef.current
    if (!video || !overlay || !sampling || landmarker.status !== 'ready') return

    const overlayCtx = overlay.getContext('2d')
    const samplingCtx = sampling.getContext('2d', { willReadFrequently: true })
    if (!overlayCtx || !samplingCtx) return

    let frameCount = 0
    let lastFpsTick = performance.now()

    const tick = () => {
      if (video.readyState < 2 || video.videoWidth === 0) {
        rafRef.current = requestAnimationFrame(tick)
        return
      }
      const vw = video.videoWidth
      const vh = video.videoHeight
      if (overlay.width !== vw) overlay.width = vw
      if (overlay.height !== vh) overlay.height = vh
      if (sampling.width !== vw) sampling.width = vw
      if (sampling.height !== vh) sampling.height = vh

      const now = performance.now()
      const result = landmarker.landmarker.detectForVideo(video, now)
      samplingCtx.drawImage(video, 0, 0, vw, vh)

      overlayCtx.clearRect(0, 0, vw, vh)
      const faces = result.faceLandmarks
      if (faces && faces.length > 0) {
        setFaceDetected(true)
        const lm = faces[0]
        drawPolygon(overlayCtx, LEFT_CHEEK_RING, lm, vw, vh)
        drawPolygon(overlayCtx, RIGHT_CHEEK_RING, lm, vw, vh)
        drawPolygon(overlayCtx, FOREHEAD_RING, lm, vw, vh)

        const sample = sampleCheekRgb(lm, samplingCtx, vw, vh)
        if (sample) {
          const state = estimatorRef.current.pushRgb(
            sample.r,
            sample.g,
            sample.b,
            now / 1000
          )
          setHr(state)
        }
      } else {
        setFaceDetected(false)
      }

      frameCount++
      if (now - lastFpsTick > 500) {
        setFps((frameCount * 1000) / (now - lastFpsTick))
        frameCount = 0
        lastFpsTick = now
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
  }, [camera.videoRef, landmarker])

  useEffect(() => {
    if (camera.state.status === 'running' && landmarker.status === 'ready') {
      runLoop()
    }
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
    }
  }, [camera.state.status, landmarker.status, runLoop])

  const onStart = () => {
    estimatorRef.current.reset()
    setHr(INITIAL_HR_STATE)
    camera.start()
  }
  const onStop = () => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    camera.stop()
    estimatorRef.current.reset()
    setFaceDetected(false)
    setFps(0)
    setHr(INITIAL_HR_STATE)
  }

  // Swap camera mid-session: stop the current stream, then ask for the new
  // one. Reset the estimator so the buffer doesn't mix signals from two
  // different sensors.
  const onChangeCamera = (id: string) => {
    camera.setDeviceId(id)
    if (camera.state.status === 'running') {
      estimatorRef.current.reset()
      setHr(INITIAL_HR_STATE)
      camera.start(id)
    }
  }

  // Swap BVP extractor. The two methods produce different signal shapes,
  // so we wipe the running buffer and re-fill it under the new method.
  const onChangeMethod = (next: BvpMethod) => {
    setMethodState(next)
    estimatorRef.current.setMethod(next)
    setHr(INITIAL_HR_STATE)
  }

  // Show the stable BPM whenever it's readable — even during brief SNR
  // dips. It only disappears after ~5 seconds without a valid sample.
  const canShowNumber = hr.stableReadable || hr.valid
  const displayBpm = hr.stableReadable
    ? hr.stableBpm
    : hr.valid
      ? hr.bpm
      : 0
  const bpmText = canShowNumber ? displayBpm.toFixed(0) : '--'
  const acquiring = camera.state.status === 'running' && !canShowNumber
  // True when we're showing a held reading but the instantaneous window
  // has dropped out — useful for a softer status note without flapping.
  const holding = hr.stableReadable && !hr.valid

  const statusLine = (() => {
    if (landmarker.status === 'loading') return 'Loading face model…'
    if (landmarker.status === 'error')
      return `Model error: ${landmarker.error}`
    if (camera.state.status === 'error')
      return `Camera error: ${camera.state.error}`
    if (camera.state.status !== 'running') return 'Ready — press Start'
    if (!faceDetected) return 'Looking for your face…'
    if (acquiring)
      return `Acquiring · SNR ${hr.snr.toFixed(1)} · buffer ${(hr.warmup * 100).toFixed(0)}%`
    if (holding)
      return `Holding reading · reacquiring (SNR ${hr.snr.toFixed(1)})`
    return `Instantaneous ${hr.bpm.toFixed(1)} · SNR ${hr.snr.toFixed(1)} · ${hr.fps.toFixed(1)} fps`
  })()

  const isRunning = camera.state.status === 'running'

  return (
    <div className="min-h-full flex flex-col">
      <header className="flex items-center justify-between px-4 md:px-8 py-4 border-b border-ink/5">
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-ink2 hover:text-ink text-sm"
        >
          ← <span className="hidden sm:inline">back</span>
        </Link>
        <div className="inline-flex items-center gap-2 text-heart font-medium">
          <HeartIcon bpm={hr.valid ? displayBpm : 0} size={18} />
          cardiocam
        </div>
        <div className="text-xs text-ink2 tabular-nums">
          {fps.toFixed(0)} fps
        </div>
      </header>

      <main className="flex-1 px-4 md:px-8 py-4 md:py-8">
        {/* On mobile: stacked. On desktop: video on left, metrics+plots on right. */}
        <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_380px] gap-6 max-w-6xl mx-auto">
          {/* Video column */}
          <div>
            {/* No fixed aspect ratio — iPhone front cam returns a portrait
                720x1280 stream, desktop webcams return landscape 1280x720.
                Let the video element's intrinsic size drive the container
                height; the overlay canvas is absolute-positioned on top so
                its coordinates always match the rendered video. */}
            {/* Mobile: 4:3 box capped at 40vh so the BPM card is visible
                without scrolling. object-cover center-crops the feed to
                fill the box — the face stays centered (MediaPipe landmarks
                are in source coords, and the overlay canvas matches via
                the same object-cover + aspect). Desktop: uncapped. */}
            <div className="relative w-full rounded-2xl overflow-hidden bg-black shadow-lg aspect-[4/3] max-h-[40vh] lg:max-h-none lg:aspect-video">
              <video
                ref={camera.videoRef}
                playsInline
                muted
                className="absolute inset-0 w-full h-full object-cover scale-x-[-1]"
              />
              <canvas
                ref={overlayCanvasRef}
                className="absolute inset-0 w-full h-full object-cover scale-x-[-1] pointer-events-none"
              />
              <canvas ref={samplingCanvasRef} className="hidden" />
              {!isRunning && (
                <div className="absolute inset-0 flex items-center justify-center text-white/80 text-center px-6 min-h-[12rem]">
                  {camera.state.status === 'error'
                    ? `Camera error: ${camera.state.error}`
                    : 'Camera stopped'}
                </div>
              )}
              {acquiring && faceDetected && (
                <div className="absolute top-3 left-3 rounded-full bg-black/60 text-white text-xs px-3 py-1 backdrop-blur">
                  Acquiring · sit still
                </div>
              )}
            </div>

            {/* Controls */}
            <div className="mt-4 flex flex-wrap items-center gap-3">
              {isRunning ? (
                <button
                  onClick={onStop}
                  className="rounded-full bg-ink px-6 py-2 text-white font-medium text-sm hover:bg-ink/85 transition"
                >
                  Stop
                </button>
              ) : (
                <button
                  onClick={onStart}
                  disabled={landmarker.status !== 'ready'}
                  className="rounded-full bg-heart px-6 py-2 text-white font-medium text-sm hover:bg-heart2 transition disabled:opacity-50"
                >
                  {landmarker.status === 'ready' ? 'Start' : 'Preparing…'}
                </button>
              )}

              {camera.devices.length > 1 && camera.deviceId && (
                <select
                  value={camera.deviceId}
                  onChange={(e) => onChangeCamera(e.target.value)}
                  className="rounded-full border border-ink/15 bg-white px-4 py-2 text-sm text-ink hover:border-ink/30 transition"
                >
                  {camera.devices.map((d) => (
                    <option key={d.deviceId} value={d.deviceId}>
                      {d.label || `Camera ${d.deviceId.slice(0, 6)}`}
                    </option>
                  ))}
                </select>
              )}

              {/* BVP method A/B toggle. POS (default) is the chrominance-based
                  extractor from Wang 2017; GREEN is the pre-Apr-2026 baseline
                  kept as an escape hatch. Swapping wipes the signal buffer. */}
              <label className="inline-flex items-center gap-2 text-xs text-ink2">
                Method:
                <select
                  value={method}
                  onChange={(e) => onChangeMethod(e.target.value as BvpMethod)}
                  className="rounded-full border border-ink/15 bg-white px-3 py-1 text-xs text-ink hover:border-ink/30 transition"
                  title="Pure: POS / CHROM / GREEN. Hybrids average z-scored BVPs."
                >
                  <option value="POS">POS</option>
                  <option value="CHROM">CHROM</option>
                  <option value="GREEN">GREEN</option>
                  <option value="POS+CHROM">POS + CHROM</option>
                  <option value="POS+GREEN">POS + GREEN</option>
                  <option value="CHROM+GREEN">CHROM + GREEN</option>
                  <option value="POS+CHROM+GREEN">POS + CHROM + GREEN</option>
                </select>
              </label>
            </div>
          </div>

          {/* Metrics + plots column */}
          <aside className="flex flex-col gap-6">
            {/* BPM card */}
            <div className="rounded-2xl bg-white shadow-sm border border-ink/5 p-6 md:p-8 text-center">
              <div
                className={`inline-flex items-baseline gap-3 ${hr.valid ? 'text-heart' : holding ? 'text-heart/60' : 'text-ink2/40'}`}
              >
                <HeartIcon
                  bpm={canShowNumber ? displayBpm : 0}
                  size={32}
                  className="self-center"
                />
                <div className="text-7xl md:text-8xl font-semibold tabular-nums leading-none">
                  {bpmText}
                </div>
              </div>
              <div className="mt-2 text-sm text-ink2 uppercase tracking-wider">
                beats per minute
              </div>
              <div className="mt-4 text-xs text-ink2 tabular-nums min-h-[1em]">
                {statusLine}
              </div>
            </div>

            {/* Plots */}
            <div className="rounded-2xl bg-white shadow-sm border border-ink/5 p-5">
              <div className="text-xs font-medium text-ink2 uppercase tracking-wider mb-2">
                Pulse signal
              </div>
              <SignalPlot signal={hr.signal} />
            </div>
            <div className="rounded-2xl bg-white shadow-sm border border-ink/5 p-5">
              <div className="text-xs font-medium text-ink2 uppercase tracking-wider mb-2">
                Spectrum
              </div>
              <FFTPlot
                fft={hr.fft}
                freqsBpm={hr.freqsBpm}
                peakBpm={hr.valid ? hr.bpm : 0}
              />
            </div>
          </aside>
        </div>
      </main>
      <Footer />
    </div>
  )
}

function drawPolygon(
  ctx: CanvasRenderingContext2D,
  indices: number[],
  lm: { x: number; y: number }[],
  w: number,
  h: number
) {
  ctx.beginPath()
  for (let i = 0; i < indices.length; i++) {
    const p = lm[indices[i]]
    const x = p.x * w
    const y = p.y * h
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.closePath()
  ctx.fillStyle = 'rgba(20, 184, 166, 0.22)'
  ctx.fill()
  ctx.strokeStyle = 'rgba(20, 184, 166, 0.95)'
  ctx.lineWidth = 2
  ctx.stroke()
}
