# cardiocam accuracy — findings so far

_Not a paper. Running synthesis of what the literature survey tells us about
improving cardiocam's BPM accuracy. Read from top to bottom for the current
state of understanding._

## Current understanding

cardiocam ships a **green-channel-only** rPPG pipeline. That's the
**worst** classical method in every published cross-dataset benchmark.
The rPPG-Toolbox paper (Liu et al., NeurIPS 2023, Table 2) gives direct
numbers: MAE 19.81 BPM on UBFC-rPPG for GREEN vs 4.00 BPM for POS — a 5×
gap on the most standard benchmark, and similar gaps on PURE, UBFC-Phys,
and MMPD.

We don't need a dataset to know we should change this. The expected
accuracy gain from GREEN → POS is the largest single lever available.

## Ranked techniques, effort and expected gain

| Rank | Technique | Effort | Expected gain on cardiocam (my honest prior) |
|------|-----------|--------|----------------------------------------------|
| 1 | **POS** (Wang 2017) | Small — ~30 lines of TS in signalProcessing.ts, plus collecting R/G/B means instead of just green | Large. 2-5× MAE reduction. This is the dominant lever. |
| 2 | **CHROM** (de Haan 2013) | Small — ~10 lines | Modest incremental over POS. Worth shipping as an option. |
| 3 | **Rigid mesh normalization** (Face2PPG 2023) | Medium — affine-warp each frame to a canonical pose using MediaPipe landmarks before sampling ROIs | Moderate on mobile/motion cases. Overkill if user sits still. |
| 4 | **Dynamic ROI selection** (Face2PPG 2023) | Medium — evaluate multiple cheek patches per frame, weight by SNR | Small but consistent. |
| 5 | **Adaptive bandpass** (post-processing) | Small | Marginal. Nice polish but not worth doing before (1). |
| 6 | **Deep learning methods** (PhysNet, EfficientPhys, etc.) | Very large — ONNX Runtime Web, model download, browser memory budget | Unclear gain over POS on simple desktop/seated cases; maybe large on arbitrary mobile. Defer. |

## What I recommend

Port **POS** first, as a single focused PR on a new branch. The diff is
small, the payoff is well-established in literature, and it doesn't
change any public API of cardiocam. Everything downstream of the
green-channel sample (detrend, Hamming, FFT, SNR gate, median smoothing)
stays identical.

After POS ships and is verified against a manual finger-pulse reference
on your own face — the cheapest honest test we have — we re-evaluate.
If accuracy is good we stop. If we want to push further, rigid mesh
normalization is the next logical step.

## What I'm NOT recommending

- **Spinning up UBFC-rPPG evaluation infrastructure first.** The
  published MAE delta between GREEN and POS is so large (5×) that a
  validation dataset isn't needed to justify the change. We'd be
  spending days of engineering to reconfirm a result that's already
  in every rPPG paper's table. Defer that infra until we want to do
  something the literature hasn't already answered.
- **Chasing DL methods.** The gap between POS and the best supervised
  methods on UBFC-rPPG is ~2 BPM. The gap between GREEN (us, today) and
  POS is ~16 BPM. We're in the wrong part of the Pareto frontier to
  care about DL yet.

## Lessons and constraints

- Zenodo search via WebFetch is unreliable; most relevant rPPG datasets
  are gate-kept by license forms anyway. Literature > datasets for our
  first moves.
- pyVHR is MIT-licensed and has clean reference implementations we can
  transcribe for POS and CHROM.
- The rPPG-Toolbox's README stores its headline result as a PNG. The
  Read tool can parse the PNG directly, which is how we got the numbers.

## Open questions

- How much does POS help on MediaPipe's ROI placement specifically?
  (All the literature benchmarks use a particular face detector; POS's
  relative gain should transfer, but the absolute MAE won't.)
- On a phone in low light, what's the floor? MMPD MAE for POS is 12.36 —
  substantially worse than desktop. Post-POS work should likely focus
  on the mobile case.
