---
title: Design, Implementation and Evaluation of a Real-Time Remote Photoplethysmography (rPPG) Acquisition System for Non-Invasive Vital Sign Monitoring
authors: Álvarez Casado, Sharifipour, Lage Cañellas, Nhi Nguyen, Le Nguyen, Bordallo López
year: 2025
venue: arXiv:2508.18787v1 (26 Aug 2025)
group: Center for Machine Vision and Signal Analysis (CMVS), University of Oulu + VTT, Finland
relevance: ★★★★★ — same group as Face2PPG; most directly applicable modern reference we've found
---

# Why this paper is unusually useful

It's a **product paper** — describes a working real-time rPPG system
with published performance numbers on four public datasets, four
different configurations, and explicit algorithmic choices. Every
decision is justified by tradeoffs rather than hand-waved. The benchmark
table is concrete and published.

# System architecture highlights

- **Face detection + alignment**: Ensemble of Regression Trees (ERT, Dlib)
  optimized to ~530 FPS on CPU. Small non-DL model. For cardiocam we
  use MediaPipe which already does this well — no change needed.
- **Landmark stabilization: Alpha-Beta filter** on the per-frame
  landmark points. Removes frame-to-frame jitter without killing
  responsiveness to real head movement. We do nothing today.
- **Skin segmentation**: geometric triangulation of facial landmarks
  onto a reference shape, creating a spatio-temporal "normalized face"
  matrix. Landmark-based is preferred over deep segmentation because
  annotated skin-mask datasets are small.
- **ROI selection**: **three regions — both cheeks + forehead**.
  Multi-region with per-region quality assessment beats single-region
  in their benchmark.
- **RGB-to-BVP**: **CIE-Lab color space** in their real-time configs.
  POS in their server configs. They argue Lab's a-channel is a wider
  dynamic range than the green channel approach and has no temporal
  dependency (unlike POS's sliding window), making it cheaper to
  compute frame-by-frame.
- **Buffering**: 12-second rolling windows (360 samples at 30 FPS)
  after 8-second stability check. We currently use 8 seconds.
- **Filtering**: 61-tap FIR band-pass with linear phase. No phase
  distortion. HR band 0.75–4.00 Hz (45–240 BPM); narrower display
  band 0.8–2.0 Hz (48–120 BPM).
- **Spectral analysis**: FFT or **Welch's method** depending on
  configuration. Welch's averages multiple overlapping FFT segments
  and is lower-variance than a single FFT — a true accumulated-
  spectrum approach, conceptually similar to what we just built in
  our `long-term-accumulation` branch.

# Benchmark table (their Table 2)

MAE ± SD in BPM; PCC = Pearson correlation coefficient. Lower MAE /
higher PCC better.

| Configuration | LGI-PPGI MAE | COHFACE MAE | UBFC1 MAE | UBFC2 MAE |
|---|---:|---:|---:|---:|
| Face2PPG-Server Normalized  | 8.7 ± 8.4 | 9.4 ± 4.8  | 1.2 ± 0.4 | 1.4 ± 1.5 |
| Face2PPG-Server Multiregion | 4.5 ± 3.3 | 8.0 ± 4.4  | 0.9 ± 0.4 | 0.9 ± 0.9 |
| Face2PPG-RT Config. 1 (Lab+FFT)     | 6.4 ± 6.8 | 10.8 ± 5.5 | 1.4 ± 0.5 | 4.7 ± 4.6 |
| Face2PPG-RT Config. 2 (Lab+Welch)   | 5.9 ± 8.0 | 11.3 ± 7.3 | 1.5 ± 1.2 | 6.7 ± 6.1 |

Observations:
- **Multi-region server config wins on every dataset.** Most
  consistent improvement comes from three ROIs instead of one.
- **Welch's method slightly beats single FFT** on LGI-PPGI (the
  only dataset where RT configs can be meaningfully ranked by
  accuracy); underperforms on UBFC2.
- **Server Multiregion gets 0.9 BPM MAE on UBFC** — state-of-the-art
  for classical rPPG; rivals the best supervised methods.
- COHFACE results (8–11 MAE, PCC near zero) show the method strains
  on heavily compressed video. All methods do.

# Actionable techniques for cardiocam, ranked

1. **Multi-region ROI: forehead + both cheeks, per-region SNR
   weighting** — the single biggest win in their benchmark
   (Server-Normalized 8.7 → Multiregion 4.5 on LGI-PPGI).
2. **Alpha-Beta filter on landmarks** — absorbs frame jitter before
   ROI sampling; cheap to implement in JS.
3. **CIE-Lab a-channel as an alternative BVP extractor** — another
   method to add to our dropdown; lightweight compared to POS.
4. **Welch's-method spectral averaging** — overlap multiple FFT
   segments and average the power spectra. Similar in spirit to our
   exponential-decay accumulator but with mathematical pedigree.
5. **61-tap linear-phase FIR instead of our current implicit filter
   chain** — phase preservation matters less for BPM estimation than
   for BVP waveform shape; low priority.

# Caveats

- This paper focuses on a **server-streaming product** (they ship video
  to an HTTP endpoint + REST API). Our constraint is different (pure
  browser). Some optimization choices they made for embedded Linux
  don't translate.
- Their **skin segmentation** uses triangulation to a reference mesh.
  Implementing this in TypeScript is nontrivial; we'd be adding ~200
  lines for a marginal gain over our landmark-polygon sampling.
- The dataset MAE numbers come from recordings with cooperative
  subjects in reasonable lighting. Our users are on phones, in
  arbitrary lighting, with arbitrary face angles. Expect our MAE to be
  2-3× worse than their reported numbers even after adopting their
  techniques.
