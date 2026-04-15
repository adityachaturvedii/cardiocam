import { Link } from 'react-router-dom'

export default function Landing() {
  return (
    <main className="min-h-full flex flex-col items-center justify-center px-6 py-16 text-center">
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
      <p className="mt-8 text-xs text-ink2/70">
        Session 1 scaffold — landing copy and layout polish come in session 3.
      </p>
    </main>
  )
}
