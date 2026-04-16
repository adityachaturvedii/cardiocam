import { Link } from 'react-router-dom'
import Footer from '../components/Footer'
import HeartIcon from '../components/HeartIcon'

export default function Landing() {
  return (
    <div className="min-h-full flex flex-col">
      <main className="flex-1 flex flex-col items-center px-6 py-14 md:py-20">
        {/* Hero */}
        <div className="max-w-2xl text-center">
          <div className="inline-flex items-center gap-2 rounded-full bg-heart/10 text-heart px-4 py-1 text-sm font-medium mb-6">
            <HeartIcon bpm={72} size={14} />
            Remote photoplethysmography
          </div>
          <h1 className="text-4xl md:text-6xl font-semibold tracking-tight text-ink">
            Your heart rate,
            <br />
            from your camera.
          </h1>
          <p className="mt-6 text-lg text-ink2 max-w-xl mx-auto">
            cardiocam measures your pulse from subtle color changes in your
            face — no contact, no wearable. Everything runs in your browser;
            your video never leaves your device.
          </p>
          <Link
            to="/measure"
            className="mt-10 inline-flex items-center justify-center rounded-full bg-heart px-8 py-3.5 text-white font-medium shadow-lg shadow-heart/25 hover:bg-heart2 transition"
          >
            Start measurement →
          </Link>
          <p className="mt-3 text-xs text-ink2/70">
            Works on modern Chrome, Safari, Firefox, and Edge.
          </p>
        </div>

        {/* How it works */}
        <section className="mt-20 md:mt-28 w-full max-w-5xl">
          <h2 className="text-center text-sm font-medium text-ink2 uppercase tracking-wider">
            How it works
          </h2>
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-5">
            <Card
              step="1"
              title="Face detection"
              body="MediaPipe Face Mesh tracks 478 landmarks per frame, finding your cheeks in real time."
            />
            <Card
              step="2"
              title="Sample the skin"
              body="Each frame we average the green channel over two clean cheek patches. Blood flow shows up as a tiny periodic wiggle."
            />
            <Card
              step="3"
              title="Find the peak"
              body="Detrend, window, FFT the last few seconds. The biggest peak between 45–150 BPM is your heart rate — smoothed over time for stability."
            />
          </div>
        </section>

        {/* Privacy */}
        <section className="mt-16 md:mt-24 max-w-2xl text-center">
          <h2 className="text-sm font-medium text-ink2 uppercase tracking-wider">
            Privacy
          </h2>
          <p className="mt-3 text-ink">
            Every frame is processed on your device. No video, no landmarks,
            no heart-rate readings leave the browser. There is no server,
            account, or analytics.
          </p>
        </section>

        {/* Research */}
        <section className="mt-16 md:mt-24 max-w-2xl text-sm text-ink2 text-center">
          <h2 className="font-medium text-ink mb-2">Based on research</h2>
          <p>
            Implements the rPPG method described in Poh, McDuff, and Picard's
            2010 paper{' '}
            <a
              href="https://doi.org/10.1364/OE.18.010762"
              target="_blank"
              rel="noopener noreferrer"
              className="underline hover:text-ink"
            >
              "Non-contact, automated cardiac pulse measurements using video
              imaging and blind source separation"
            </a>{' '}
            (<em>Optics Express</em>, 18(10), 10762–10774).
          </p>
        </section>

        {/* Disclaimer */}
        <p className="mt-10 max-w-lg text-center text-xs text-ink2/70">
          Not a medical device. cardiocam is an educational demo — do not use
          its readings for diagnosis or treatment decisions.
        </p>
      </main>
      <Footer />
    </div>
  )
}

function Card({
  step,
  title,
  body,
}: {
  step: string
  title: string
  body: string
}) {
  return (
    <div className="rounded-2xl border border-ink/5 bg-white/60 p-6 shadow-sm">
      <div className="text-xs font-mono text-heart mb-3">STEP {step}</div>
      <h3 className="font-semibold text-ink">{title}</h3>
      <p className="mt-2 text-sm text-ink2 leading-relaxed">{body}</p>
    </div>
  )
}
