import { Link } from 'react-router-dom'
import Footer from '../components/Footer'

export default function Landing() {
  return (
    <div className="min-h-full flex flex-col">
      <main className="flex-1 flex flex-col items-center justify-center px-6 py-16 text-center">
        <h1 className="text-4xl md:text-5xl font-semibold tracking-tight text-ink">
          cardiocam
        </h1>
        <p className="mt-4 max-w-xl text-ink2">
          Measure your heart rate using just your camera. No app install, no
          upload — your video never leaves your device.
        </p>
        <Link
          to="/measure"
          className="mt-10 inline-flex items-center justify-center rounded-full bg-heart px-8 py-3 text-white font-medium shadow-md hover:bg-heart2 transition"
        >
          Start measurement
        </Link>

        <section className="mt-16 max-w-md text-sm text-ink2">
          <h2 className="font-medium text-ink mb-2">Based on research</h2>
          <p>
            Implements remote photoplethysmography (rPPG) as described in Poh,
            McDuff, and Picard's 2010 paper "Non-contact, automated cardiac
            pulse measurements using video imaging and blind source
            separation" (<em>Optics Express</em>, 18(10), 10762–10774).
          </p>
        </section>
      </main>
      <Footer />
    </div>
  )
}
