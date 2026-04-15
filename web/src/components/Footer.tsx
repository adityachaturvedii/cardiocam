export default function Footer() {
  return (
    <footer className="mt-auto pt-8 pb-4 text-center text-xs text-ink2/70">
      Built by{' '}
      <a
        href="https://github.com/adityachaturvedii"
        target="_blank"
        rel="noopener noreferrer"
        className="underline hover:text-ink"
      >
        Aditya Chaturvedi
      </a>{' '}
      · Based on{' '}
      <a
        href="https://doi.org/10.1364/OE.18.010762"
        target="_blank"
        rel="noopener noreferrer"
        className="underline hover:text-ink"
        title="Poh, McDuff, Picard — Non-contact automated cardiac pulse measurements, Optics Express 2010"
      >
        Poh et al. (2010)
      </a>
    </footer>
  )
}
