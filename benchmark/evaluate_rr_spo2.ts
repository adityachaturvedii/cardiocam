/**
 * Evaluate cardiocam's respiratory rate + SpO2 on UBFC-rPPG datasets.
 *
 * SpO2: Uses DATASET_1 CSVs (have GT SpO2 from pulse oximeter).
 * RR:   Uses DATASET_2 CSVs (derives reference RR from contact PPG GT
 *       by bandpassing the PPG waveform at 0.18-0.5 Hz and peak-picking).
 *
 * Usage:
 *   cd web
 *   npx tsx ../benchmark/evaluate_rr_spo2.ts
 */
import * as fs from 'fs'
import * as path from 'path'
import {
  estimateBpm,
  estimateRR,
  estimateSpO2,
  posTransform,
  smoothPriorDetrend,
  designButterBandpass1,
  filtfilt,
} from '../web/src/lib/signalProcessing'

const DS1_DIR = path.join(__dirname, 'data', 'ubfc_ds1_rgb')
const DS2_DIR = path.join(__dirname, 'data', 'ubfc_rgb')

const BUFFER_SIZE = 240
const NFFT = 2048
const WINDOW_STEP = 30

// ─── Helpers ─────────────────────────────────────────────────────────

function loadCsv(csvPath: string): Record<string, number>[] {
  const text = fs.readFileSync(csvPath, 'utf8')
  const lines = text.trim().split('\n')
  const header = lines[0].split(',')
  return lines.slice(1).map(line => {
    const vals = line.split(',')
    const row: Record<string, number> = {}
    header.forEach((h, i) => {
      row[h] = vals[i] === '' ? NaN : parseFloat(vals[i])
    })
    return row
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

// ─── SpO2 benchmark (DATASET_1) ─────────────────────────────────────

function benchmarkSpO2() {
  if (!fs.existsSync(DS1_DIR)) {
    console.log('No DATASET_1 CSVs found. Run extract_rgb_ds1.py first.')
    return
  }
  const files = fs.readdirSync(DS1_DIR).filter(f => f.endsWith('.csv')).sort()
  console.log(`\n${'='.repeat(60)}`)
  console.log('SpO2 BENCHMARK (DATASET_1, GT from pulse oximeter)')
  console.log('='.repeat(60))

  const allPreds: number[] = []
  const allGts: number[] = []

  for (const f of files) {
    const subject = f.replace('.csv', '')
    const rows = loadCsv(path.join(DS1_DIR, f))
    if (rows.length < BUFFER_SIZE) continue
    const fps = rows[0].fps

    const valid = rows.filter(r => !isNaN(r.r) && !isNaN(r.g))
    if (valid.length < BUFFER_SIZE) continue

    const preds: number[] = []
    const gts: number[] = []

    for (let start = 0; start + BUFFER_SIZE <= valid.length; start += WINDOW_STEP) {
      const win = valid.slice(start, start + BUFFER_SIZE)
      if (win.some(w => isNaN(w.r))) continue

      const rArr = new Float64Array(win.map(w => w.r))
      const gArr = new Float64Array(win.map(w => w.g))
      const bArr = new Float64Array(win.map(w => w.b))
      const ts = new Float64Array(BUFFER_SIZE)
      for (let i = 0; i < BUFFER_SIZE; i++) ts[i] = i / fps

      const est = estimateSpO2(rArr, gArr, bArr, ts, [])
      if (!est.valid) continue

      const gtSpo2 = win.reduce((s, w) => s + (w.gt_spo2 || 0), 0) / BUFFER_SIZE
      if (gtSpo2 < 80 || gtSpo2 > 100) continue

      preds.push(est.spo2)
      gts.push(gtSpo2)
    }

    if (preds.length < 3) {
      console.log(`  ${subject}: insufficient valid windows`)
      continue
    }

    const errors = preds.map((p, i) => Math.abs(p - gts[i]))
    const mae = errors.reduce((a, b) => a + b) / errors.length
    const r = pearsonR(preds, gts)
    const gtMean = gts.reduce((a, b) => a + b) / gts.length
    const predMean = preds.reduce((a, b) => a + b) / preds.length
    console.log(
      `  ${subject.padEnd(16)} MAE: ${mae.toFixed(2)}%  ` +
      `GT: ${gtMean.toFixed(1)}%  Pred: ${predMean.toFixed(1)}%  r: ${r.toFixed(3)}  (${preds.length} windows)`
    )
    allPreds.push(...preds)
    allGts.push(...gts)
  }

  if (allPreds.length > 0) {
    const totalMae = allPreds.map((p, i) => Math.abs(p - allGts[i])).reduce((a, b) => a + b) / allPreds.length
    const r = pearsonR(allPreds, allGts)
    console.log(`\n  AGGREGATE  MAE: ${totalMae.toFixed(2)}%  r: ${r.toFixed(3)}  (${allPreds.length} windows)`)
    console.log('  Note: camera SpO2 is ±3-5% in literature. Shown for transparency.')
  }
}

// ─── RR benchmark (DATASET_2) ───────────────────────────────────────

function benchmarkRR() {
  if (!fs.existsSync(DS2_DIR)) {
    console.log('No DATASET_2 CSVs found. Run extract_rgb.py first.')
    return
  }
  const files = fs.readdirSync(DS2_DIR).filter(f => f.endsWith('.csv')).sort()
  console.log(`\n${'='.repeat(60)}`)
  console.log('RESPIRATORY RATE BENCHMARK (DATASET_2, derived GT from contact PPG)')
  console.log('='.repeat(60))

  // DATASET_2 ground_truth.txt line 1 is the contact PPG signal.
  // We derive reference RR by bandpassing at 0.18-0.5 Hz + FFT peak-pick.
  const DS2_ROOT = path.join(process.env.HOME!, 'UBFC_DATASET', 'DATASET_2')

  const allPreds: number[] = []
  const allGts: number[] = []

  for (const f of files) {
    const subject = f.replace('.csv', '')
    const rows = loadCsv(path.join(DS2_DIR, f))
    if (rows.length < BUFFER_SIZE) continue
    const fps = rows[0].fps

    // Load contact PPG from ground_truth.txt line 1
    const gtPath = path.join(DS2_ROOT, subject, 'ground_truth.txt')
    if (!fs.existsSync(gtPath)) continue
    const gtLines = fs.readFileSync(gtPath, 'utf8').trim().split('\n')
    const contactPpg = gtLines[0].trim().split(/\s+/).map(Number)

    const valid = rows.filter(r => !isNaN(r.r) && !isNaN(r.g))
    if (valid.length < BUFFER_SIZE) continue

    // Run full-video POS BVP
    const rFull = new Float64Array(valid.map(r => r.r))
    const gFull = new Float64Array(valid.map(r => r.g))
    const bFull = new Float64Array(valid.map(r => r.b))
    const bvpFull = posTransform(rFull, gFull, bFull, fps)

    const preds: number[] = []
    const gts: number[] = []

    for (let start = 0; start + BUFFER_SIZE <= valid.length; start += WINDOW_STEP) {
      // Camera RR
      const bvpWin = bvpFull.slice(start, start + BUFFER_SIZE)
      const ts = new Float64Array(BUFFER_SIZE)
      for (let i = 0; i < BUFFER_SIZE; i++) ts[i] = i / fps

      const rrEst = estimateRR(bvpWin, ts, NFFT)
      if (rrEst.snr < 1.5 || rrEst.rr < 8 || rrEst.rr > 35) continue

      // Reference RR from contact PPG (same window, same bandpass logic)
      if (start + BUFFER_SIZE > contactPpg.length) continue
      const ppgWin = new Float64Array(contactPpg.slice(start, start + BUFFER_SIZE))
      const refRR = estimateRR(ppgWin, ts, NFFT)
      if (refRR.snr < 1.5 || refRR.rr < 8 || refRR.rr > 35) continue

      preds.push(rrEst.rr)
      gts.push(refRR.rr)
    }

    if (preds.length < 3) {
      console.log(`  ${subject}: insufficient valid windows`)
      continue
    }

    const errors = preds.map((p, i) => Math.abs(p - gts[i]))
    const mae = errors.reduce((a, b) => a + b) / errors.length
    const r = pearsonR(preds, gts)
    console.log(
      `  ${subject.padEnd(12)} MAE: ${mae.toFixed(2)} brpm  r: ${r.toFixed(3)}  (${preds.length} windows)`
    )
    allPreds.push(...preds)
    allGts.push(...gts)
  }

  if (allPreds.length > 0) {
    const totalMae = allPreds.map((p, i) => Math.abs(p - allGts[i])).reduce((a, b) => a + b) / allPreds.length
    const r = pearsonR(allPreds, allGts)
    console.log(`\n  AGGREGATE  MAE: ${totalMae.toFixed(2)} brpm  r: ${r.toFixed(3)}  (${allPreds.length} windows)`)
    console.log('  Note: derived GT (contact PPG bandpassed) — not a direct respiration belt.')
    console.log('  Literature reports camera-RR MAE ~2-4 brpm on cooperative subjects.')
  }
}

// ─── Main ────────────────────────────────────────────────────────────

benchmarkSpO2()
benchmarkRR()
