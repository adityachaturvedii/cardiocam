/**
 * STEP 2: Evaluate cardiocam's ACTUAL TypeScript DSP pipeline on
 * pre-extracted RGB CSVs from UBFC-rPPG v2.
 *
 * Reads CSVs produced by extract_rgb.py (frame, r, g, b, gt_hr, fps).
 * Runs our real signalProcessing.ts functions (posTransform, chromTransform,
 * smoothPriorDetrend, estimateBpm) on sliding windows. Compares predicted
 * BPM against ground-truth HR.
 *
 * Usage:
 *   cd web
 *   npx tsx ../benchmark/evaluate_ts.ts
 */
import * as fs from 'fs'
import * as path from 'path'
import {
  estimateBpm,
  posTransform,
  chromTransform,
  omitTransform,
} from '../web/src/lib/signalProcessing'

const CSV_DIR = path.join(__dirname, 'data', 'ubfc_rgb')
const OUT_DIR = path.join(__dirname, 'data')
const BUFFER_SIZE = 240
const NFFT = 2048
const WINDOW_STEP = 30
const BPM_MIN = 45
const BPM_MAX = 150
const SNR_THRESHOLD = 2
const METHODS = ['POS', 'CHROM', 'OMIT', 'GREEN'] as const

// ─── Helpers ─────────────────────────────────────────────────────────

interface CsvRow {
  frame: number
  r: number
  g: number
  b: number
  gt_hr: number
  fps: number
}

function loadCsv(csvPath: string): CsvRow[] {
  const text = fs.readFileSync(csvPath, 'utf8')
  const lines = text.trim().split('\n')
  const header = lines[0].split(',')
  return lines.slice(1).map(line => {
    const vals = line.split(',')
    const row: Record<string, number> = {}
    header.forEach((h, i) => {
      row[h] = vals[i] === '' ? NaN : parseFloat(vals[i])
    })
    return row as unknown as CsvRow
  })
}

function pearsonR(a: number[], b: number[]): number {
  const n = a.length
  if (n < 3) return 0
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0
  for (let i = 0; i < n; i++) {
    sx += a[i]; sy += b[i]
    sxx += a[i] * a[i]; syy += b[i] * b[i]
    sxy += a[i] * b[i]
  }
  const num = n * sxy - sx * sy
  const den = Math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
  return den > 0 ? num / den : 0
}

function computeBvp(
  method: string,
  r: Float64Array,
  g: Float64Array,
  b: Float64Array,
  fps: number
): Float64Array {
  switch (method) {
    case 'POS': return posTransform(r, g, b, fps)
    case 'CHROM': return chromTransform(r, g, b, fps)
    case 'OMIT': return omitTransform(r, g, b)
    case 'GREEN': return g
    default: return g
  }
}

// ─── Main ────────────────────────────────────────────────────────────

function main() {
  if (!fs.existsSync(CSV_DIR)) {
    console.error(`No CSV directory at ${CSV_DIR}`)
    console.error('Run extract_rgb.py first.')
    process.exit(1)
  }

  const csvFiles = fs.readdirSync(CSV_DIR)
    .filter(f => f.endsWith('.csv'))
    .sort()

  console.log(`Found ${csvFiles.length} subject CSVs in ${CSV_DIR}`)

  type Row = {
    subject: string; method: string; mae: number; rmse: number
    pearson_r: number; n_windows: number; gt_hr_mean: number
  }
  const results: Row[] = []

  for (const csvFile of csvFiles) {
    const subject = csvFile.replace('.csv', '')
    const rows = loadCsv(path.join(CSV_DIR, csvFile))
    if (rows.length < BUFFER_SIZE) {
      console.log(`  ${subject}: too short (${rows.length} frames), skip`)
      continue
    }
    const fps = rows[0].fps

    process.stdout.write(`  ${subject}...`)
    const t0 = Date.now()

    for (const method of METHODS) {
      const preds: number[] = []
      const gts: number[] = []

      for (let start = 0; start + BUFFER_SIZE <= rows.length; start += WINDOW_STEP) {
        const window = rows.slice(start, start + BUFFER_SIZE)
        // Skip windows with missing face detections
        if (window.some(w => isNaN(w.r) || isNaN(w.g) || isNaN(w.b))) continue

        const rArr = new Float64Array(window.map(w => w.r))
        const gArr = new Float64Array(window.map(w => w.g))
        const bArr = new Float64Array(window.map(w => w.b))
        const ts = new Float64Array(BUFFER_SIZE)
        for (let i = 0; i < BUFFER_SIZE; i++) ts[i] = i / fps

        const bvp = computeBvp(method, rArr, gArr, bArr, fps)
        const est = estimateBpm(bvp, ts, NFFT, BPM_MIN, BPM_MAX)

        if (est.snr < SNR_THRESHOLD) continue

        const gtMean = window.reduce((s, w) => s + w.gt_hr, 0) / BUFFER_SIZE
        preds.push(est.bpm)
        gts.push(gtMean)
      }

      if (preds.length < 3) {
        process.stdout.write(` ${method}:skip`)
        continue
      }

      const errors = preds.map((p, i) => Math.abs(p - gts[i]))
      const mae = errors.reduce((a, b) => a + b) / errors.length
      const rmse = Math.sqrt(errors.map(e => e * e).reduce((a, b) => a + b) / errors.length)
      const r = pearsonR(preds, gts)
      const gtMean = gts.reduce((a, b) => a + b) / gts.length

      results.push({
        subject,
        method,
        mae: +mae.toFixed(2),
        rmse: +rmse.toFixed(2),
        pearson_r: +r.toFixed(3),
        n_windows: preds.length,
        gt_hr_mean: +gtMean.toFixed(1),
      })
    }
    console.log(` ${((Date.now() - t0) / 1000).toFixed(1)}s`)
  }

  if (results.length === 0) { console.log('No results!'); return }

  // Write CSV
  fs.mkdirSync(OUT_DIR, { recursive: true })
  const csvPath = path.join(OUT_DIR, 'ubfc_results.csv')
  const header = Object.keys(results[0]).join(',')
  const csvRows = results.map(r => Object.values(r).join(','))
  fs.writeFileSync(csvPath, [header, ...csvRows].join('\n'))
  console.log(`\nResults saved to ${csvPath}\n`)

  // Aggregate table
  console.log('='.repeat(65))
  console.log('AGGREGATE RESULTS — cardiocam actual TS pipeline on UBFC-rPPG v2')
  console.log('='.repeat(65))
  for (const method of METHODS) {
    const mr = results.filter(r => r.method === method)
    if (mr.length === 0) continue
    const mae = mr.reduce((s, r) => s + r.mae, 0) / mr.length
    const rmse = mr.reduce((s, r) => s + r.rmse, 0) / mr.length
    const pr = mr.reduce((s, r) => s + r.pearson_r, 0) / mr.length
    console.log(
      `  ${method.padEnd(8)} MAE: ${mae.toFixed(2).padStart(6)}  ` +
      `RMSE: ${rmse.toFixed(2).padStart(6)}  ` +
      `r: ${pr.toFixed(3).padStart(6)}  (${mr.length} subjects)`
    )
  }

  // Published reference (Liu 2023 rPPG-Toolbox Table 2, UBFC-rPPG)
  console.log('\n--- Published reference (Liu 2023, full 42 subjects) ---')
  console.log('  GREEN   MAE: 19.81')
  console.log('  POS     MAE:  4.00')
  console.log('  CHROM   MAE:  3.98')

  // Per-subject breakdown, worst POS first
  console.log('\nPer-subject MAE (POS), worst first:')
  const posResults = results.filter(r => r.method === 'POS').sort((a, b) => b.mae - a.mae)
  for (const r of posResults) {
    const bar = '█'.repeat(Math.min(40, Math.round(r.mae)))
    console.log(
      `  ${r.subject.padEnd(12)} MAE: ${r.mae.toFixed(2).padStart(6)}  ` +
      `r: ${r.pearson_r.toFixed(3).padStart(6)}  GT: ${r.gt_hr_mean.toString().padStart(5)}  ${bar}`
    )
  }
}

main()
