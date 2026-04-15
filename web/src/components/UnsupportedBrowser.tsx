import Footer from './Footer'

interface Props {
  missing: string[]
}

export default function UnsupportedBrowser({ missing }: Props) {
  return (
    <div className="min-h-full flex flex-col">
      <main className="flex-1 flex flex-col items-center justify-center px-6 py-16 text-center">
        <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-ink">
          Browser not supported
        </h1>
        <p className="mt-4 max-w-xl text-ink2">
          cardiocam needs a modern browser with camera access and GPU
          compute. Your current browser is missing:
        </p>
        <ul className="mt-4 text-ink font-medium">
          {missing.map((m) => (
            <li key={m}>· {m}</li>
          ))}
        </ul>
        <p className="mt-8 max-w-md text-sm text-ink2">
          Try a recent version of Chrome, Safari, Firefox, or Edge on a
          laptop or phone with a front-facing camera.
        </p>
      </main>
      <Footer />
    </div>
  )
}
