import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useCamera } from '../hooks/useCamera'
import { useFaceLandmarker } from '../hooks/useFaceLandmarker'
import { HeartRateEstimator, type HeartRateState } from '../lib/heartRate'
import {
  LEFT_CHEEK_RING,
  RIGHT_CHEEK_RING,
  sampleCheekGreen,
} from '../lib/roi'
import Footer from '../components/Footer'

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

      // Pull the raw frame into the sampling canvas so we can read pixel
      // data — you can't getImageData off a <video> element directly.
      samplingCtx.drawImage(video, 0, 0, vw, vh)

      overlayCtx.clearRect(0, 0, vw, vh)
      const faces = result.faceLandmarks
      if (faces && faces.length > 0) {
        setFaceDetected(true)
        const lm = faces[0]

        // BBox.
        let minX = 1,
          maxX = 0,
          minY = 1,
          maxY = 0
        for (const p of lm) {
          if (p.x < minX) minX = p.x
          if (p.x > maxX) maxX = p.x
          if (p.y < minY) minY = p.y
          if (p.y > maxY) maxY = p.y
        }
        overlayCtx.strokeStyle = 'rgba(244, 63, 94, 0.7)'
        overlayCtx.lineWidth = 2
        overlayCtx.strokeRect(
          minX * vw,
          minY * vh,
          (maxX - minX) * vw,
          (maxY - minY) * vh
        )

        // Cheek ROIs — draw on the overlay and sample from the sampling canvas.
        drawPolygon(overlayCtx, LEFT_CHEEK_RING, lm, vw, vh)
        drawPolygon(overlayCtx, RIGHT_CHEEK_RING, lm, vw, vh)

        const sample = sampleCheekGreen(lm, samplingCtx, vw, vh)
        if (sample) {
          const state = estimatorRef.current.pushSample(
            sample.greenMean,
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

  const bpmDisplay = hr.valid
    ? hr.stableBpm > 0
      ? hr.stableBpm.toFixed(0)
      : hr.bpm.toFixed(0)
    : '--'
  const acquiring = camera.state.status === 'running' && !hr.valid

  return (
    <div className="min-h-full flex flex-col px-4 py-6 md:px-10 md:py-10">
      <header className="flex items-center justify-between mb-6">
        <Link to="/" className="text-ink2 hover:text-ink text-sm">
          ← back
        </Link>
        <div className="text-sm text-ink2 tabular-nums">
          FPS {fps.toFixed(1)}
        </div>
      </header>

      <section className="flex-1 flex flex-col items-center">
        <div className="relative w-full max-w-3xl rounded-2xl overflow-hidden bg-black shadow">
          <video
            ref={camera.videoRef}
            playsInline
            muted
            className="block w-full h-auto object-contain scale-x-[-1]"
          />
          <canvas
            ref={overlayCanvasRef}
            className="absolute inset-0 w-full h-full scale-x-[-1] pointer-events-none"
          />
          <canvas ref={samplingCanvasRef} className="hidden" />
          {camera.state.status !== 'running' && (
            <div className="absolute inset-0 flex items-center justify-center text-white/80">
              {camera.state.status === 'error'
                ? `Camera error: ${camera.state.error}`
                : 'Camera stopped'}
            </div>
          )}
        </div>

        <div className="mt-8 flex flex-col items-center gap-2">
          <div
            className={`text-7xl font-semibold tabular-nums ${
              hr.valid ? 'text-heart' : 'text-ink2/50'
            }`}
          >
            {bpmDisplay}
            <span className="text-2xl text-ink2 font-normal ml-2">BPM</span>
          </div>
          <div className="text-sm text-ink2 tabular-nums">
            {landmarker.status === 'loading' && 'Loading face model…'}
            {landmarker.status === 'error' && `Model error: ${landmarker.error}`}
            {landmarker.status === 'ready' &&
              camera.state.status !== 'running' &&
              'Ready — press Start'}
            {landmarker.status === 'ready' &&
              camera.state.status === 'running' &&
              !faceDetected &&
              'Looking for your face…'}
            {acquiring &&
              faceDetected &&
              `Acquiring signal · SNR ${hr.snr.toFixed(1)} · buffer ${(
                hr.warmup * 100
              ).toFixed(0)}%`}
            {hr.valid &&
              `Instantaneous ${hr.bpm.toFixed(1)} BPM · SNR ${hr.snr.toFixed(
                1
              )} · ${hr.fps.toFixed(1)} fps`}
          </div>
        </div>

        <div className="mt-8">
          {camera.state.status === 'running' ? (
            <button
              onClick={onStop}
              className="rounded-full bg-ink px-8 py-3 text-white font-medium"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={onStart}
              disabled={landmarker.status !== 'ready'}
              className="rounded-full bg-heart px-8 py-3 text-white font-medium disabled:opacity-50"
            >
              {landmarker.status === 'ready' ? 'Start' : 'Preparing…'}
            </button>
          )}
        </div>
      </section>
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
  ctx.fillStyle = 'rgba(20, 184, 166, 0.25)'
  ctx.fill()
  ctx.strokeStyle = 'rgba(20, 184, 166, 1)'
  ctx.lineWidth = 2
  ctx.stroke()
}
