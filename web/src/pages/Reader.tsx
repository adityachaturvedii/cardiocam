import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useCamera } from '../hooks/useCamera'
import { useFaceLandmarker } from '../hooks/useFaceLandmarker'

export default function Reader() {
  const camera = useCamera()
  const landmarker = useFaceLandmarker()
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const rafRef = useRef<number | null>(null)
  const [faceDetected, setFaceDetected] = useState(false)
  const [fps, setFps] = useState(0)

  // Main detection loop — runs once camera + landmarker are both ready.
  // We use rAF so the browser paces us to display refresh and we don't
  // burn cycles when the tab is backgrounded.
  const runLoop = useCallback(() => {
    const video = camera.videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || landmarker.status !== 'ready') return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let lastTick = performance.now()
    let frameCount = 0
    let lastFpsTick = lastTick

    const tick = () => {
      if (video.readyState < 2 || video.videoWidth === 0) {
        rafRef.current = requestAnimationFrame(tick)
        return
      }
      // Sync canvas to video dimensions so overlay aligns pixel-for-pixel.
      if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth
      if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight

      const now = performance.now()
      const result = landmarker.landmarker.detectForVideo(video, now)
      lastTick = now

      ctx.clearRect(0, 0, canvas.width, canvas.height)
      const faces = result.faceLandmarks
      if (faces && faces.length > 0) {
        setFaceDetected(true)
        // Rough bbox from normalized landmarks — session 2 will upgrade
        // this with the same 5-point + cheek-ROI logic as the Python app.
        const lm = faces[0]
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
        const x = minX * canvas.width
        const y = minY * canvas.height
        const w = (maxX - minX) * canvas.width
        const h = (maxY - minY) * canvas.height
        ctx.strokeStyle = '#F43F5E'
        ctx.lineWidth = 3
        ctx.strokeRect(x, y, w, h)
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

  const onStart = () => camera.start()
  const onStop = () => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    camera.stop()
    setFaceDetected(false)
    setFps(0)
  }

  return (
    <main className="min-h-full flex flex-col px-4 py-6 md:px-10 md:py-10">
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
            // object-contain keeps the whole frame visible so the canvas
            // overlay's coordinates map 1:1 to what the user sees. A scale
            // -1 makes the selfie view mirror-natural; the canvas is NOT
            // flipped because detection runs on the unflipped source — the
            // overlay CSS flips would otherwise shift bbox horizontally.
            className="block w-full h-auto object-contain scale-x-[-1]"
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full scale-x-[-1] pointer-events-none"
          />
          {camera.state.status !== 'running' && (
            <div className="absolute inset-0 flex items-center justify-center text-white/80">
              {camera.state.status === 'error'
                ? `Camera error: ${camera.state.error}`
                : 'Camera stopped'}
            </div>
          )}
        </div>

        <div className="mt-6 flex flex-col items-center gap-4">
          <div className="text-lg">
            {landmarker.status === 'loading' && (
              <span className="text-ink2">Loading face model…</span>
            )}
            {landmarker.status === 'error' && (
              <span className="text-heart">Model error: {landmarker.error}</span>
            )}
            {landmarker.status === 'ready' && (
              <span
                className={faceDetected ? 'text-pulse' : 'text-ink2'}
              >
                Face detected: {faceDetected ? 'yes' : 'no'}
              </span>
            )}
          </div>

          {camera.state.status === 'running' ? (
            <button
              onClick={onStop}
              className="rounded-full bg-ink px-6 py-2 text-white font-medium"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={onStart}
              disabled={landmarker.status !== 'ready'}
              className="rounded-full bg-heart px-6 py-2 text-white font-medium disabled:opacity-50"
            >
              {landmarker.status === 'ready' ? 'Start' : 'Preparing…'}
            </button>
          )}
        </div>
      </section>
    </main>
  )
}
