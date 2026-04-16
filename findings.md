# cardiocam accuracy — findings

_Running synthesis of what we've learned and shipped. Read top-to-bottom
for the current state. Appended rather than rewritten, so earlier
recommendations are preserved as history._

## Where we are (as of the long-term-accumulation branch)

Currently deployed on master **plus** the long-term-accumulation branch
pending merge:

- **7 BVP methods**: GREEN, POS, CHROM, plus 4 hybrids (POS+CHROM,
  POS+GREEN, CHROM+GREEN, POS+CHROM+GREEN). POS is default.
- **FFT buffer: 240 samples (~8s)** at 30 FPS, 2048-point zero-padded FFT
  (~0.88 BPM bin width).
- **Accumulated power spectrum** with exponential decay (half-life ~8s).
  Peak-picking uses the accumulated, not instantaneous, spectrum.
- **Rate limiter**: ±20 BPM search window around rolling median anchor.
- **Prior-biased peak selection**: Gaussian weighting (sigma 12 BPM)
  around the anchor.
- **Hysteresis SNR gate**: enter at 4, exit at 2.5, 1-second exit
  debounce.
- **stableReadable display gate** holds the BPM through brief signal
  dropouts (~5s) so the UI doesn't flicker.
- **EWMA on the emitted stableBpm** with 3-second half-life.

## Shipped fixes and expected vs measured gains

| Change | Shipped | Measured impact (user's face) |
|---|---|---|
| GREEN → POS | Yes | Big — user confirmed ~75 BPM readings align with finger-pulse reference |
| Rate limiter + prior bias | Yes | Big — killed the 50s → 170s jumps |
| Hysteresis + debounce | Yes | Big — killed the valid/acquiring flap |
| 8s buffer + accumulated spectrum + EWMA | Yes, pending commit on branch | Not yet field-tested |

## What the Finnish paper (Álvarez Casado 2025) adds

Álvarez Casado's group at Oulu, same team that published Face2PPG, shipped a
real-time rPPG system paper in Aug 2025. Their benchmark table on four
public datasets:

| Config | LGI-PPGI MAE | COHFACE MAE | UBFC1 MAE | UBFC2 MAE |
|---|---:|---:|---:|---:|
| Server-Normalized (single ROI, POS) | 8.7 | 9.4 | 1.2 | 1.4 |
| **Server-Multiregion** (three ROIs, POS) | **4.5** | 8.0 | 0.9 | 0.9 |
| RT Config 1 (Lab + FFT) | 6.4 | 10.8 | 1.4 | 4.7 |
| RT Config 2 (Lab + Welch) | 5.9 | 11.3 | 1.5 | 6.7 |

Key takeaways for cardiocam:

1. **Multi-region ROI is the single biggest gain.** Going from single-
   region to three regions halves MAE on LGI-PPGI (8.7 → 4.5). This is
   bigger than any algorithm change they tried.
2. **Lab color space a-channel** is a lightweight alternative to POS
   for the BVP extraction step. Works without the sliding-window
   temporal dependency POS needs.
3. **Welch's method** (overlapping FFT segments averaged) is what they
   use for spectral analysis in their server configs. We recently
   shipped an exponential-decay-accumulator equivalent in spirit.
4. **Alpha-Beta filter on landmarks** stabilizes the per-frame
   landmark points to reduce ROI jitter. We currently use raw
   MediaPipe landmarks.

## What the Kolosov paper (Sensors 2023) adds

Very little. It's a hardware-benchmarking paper comparing EVM and
GREEN-rPPG across Raspberry Pi / Jetson platforms. Uses
**forehead + both cheeks** as three ROIs — same multi-region idea — but
doesn't quantify the ROI-count benefit. EVM is presented as a
visualization method repurposed for HR; modern rPPG literature has
moved past it.

## EVM as a technique — honest assessment

Eulerian Video Magnification is great for *visualizing* the pulse on a
face. It's *not* a meaningful improvement over POS/CHROM for BPM
estimation. No modern rPPG benchmark paper (Liu 2023, Álvarez Casado
2022/2025) includes EVM-derived BPM in its tables. Implementing EVM in
cardiocam would be a large engineering effort (Laplacian pyramid,
temporal bandpass per pyramid scale, up-sampling) for an unclear
accuracy gain. **Skip.**

## Revised ranked techniques

After this latest round of research:

| Rank | Technique | Effort | Expected gain on cardiocam |
|------|-----------|--------|----------------------------|
| 1 | **Add forehead as a third ROI**, average across cheeks+forehead z-scored BVPs | Small (~30 LOC in roi.ts, heartRate.ts) | Large — Finnish paper shows single → multi-region ≈ halves MAE on hard datasets |
| 2 | **Per-region SNR weighting** (extend #1): compute the BVP quality per ROI and weight higher-SNR regions more | Medium (~60 LOC) | Moderate incremental on top of #1 |
| 3 | **Alpha-Beta filter on landmark positions** before sampling | Small (~30 LOC) | Moderate on phones (mobile has more motion) |
| 4 | **CIE-Lab a-channel as an 8th BVP method** | Small (~20 LOC) | Marginal; a faster alternative to POS, not inherently better |
| 5 | Welch's method as a separate spectral analyzer (we already have the exp-decay accumulator which is similar) | Medium | Marginal over what we have |

## What I recommend next

**#1 (three-ROI average)** — highest-impact change available, small
diff, well-supported in literature. The same multi-region framing is
endorsed by Kolosov 2023, Face2PPG 2022/2023, and Álvarez Casado 2025.

After #1 ships and is user-verified, consider #2 (per-region SNR
weighting) or #3 (Alpha-Beta landmark filter) based on whether
accuracy on mobile is still the weak link.

## Constraints and lessons (carried forward)

- Zenodo search via WebFetch is unreliable; most relevant rPPG datasets
  are gate-kept by license forms anyway. Literature > datasets.
- pyVHR is MIT-licensed and has clean reference implementations.
- PDFs accessed via WebFetch save to disk as binary; `pdftotext -layout`
  (poppler) turns them into searchable text. Essential for thesis-style
  content that isn't in HTML form.
- The Finnish paper's reported MAE numbers come from cooperative
  subjects in reasonable lighting. Expect cardiocam on phones in
  arbitrary lighting to land at 2-3× worse MAE than their reported
  best. That's fine for an educational demo.
