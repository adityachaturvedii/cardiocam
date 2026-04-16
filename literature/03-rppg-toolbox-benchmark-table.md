---
title: rPPG-Toolbox benchmark Table 2
source: Liu et al. NeurIPS 2023, extracted from the repo's figures/results.png
url: https://github.com/ubicomplab/rPPG-Toolbox
relevance: ★★★★★ — the definitive cross-dataset cheat sheet for what to port
---

# Benchmark results (MAE in BPM, MAPE in %)

Test sets span four datasets: PURE, UBFC-rPPG, UBFC-Phys, MMPD. Lower is better.

## Unsupervised / classical methods (no training required)

| Method   | PURE MAE | UBFC-rPPG MAE | UBFC-Phys MAE | MMPD MAE |
|----------|---------:|--------------:|--------------:|---------:|
| **GREEN** (current cardiocam) | 10.09 | 19.81 | 13.55 | 21.68 |
| ICA      |  4.77   |  14.70 |  10.03 |  18.60 |
| CHROM    |  5.77   |   3.98 |   4.49 |  13.66 |
| LGI      |  4.61   |  15.80 |   6.27 |  17.08 |
| PBV      |  3.91   |  15.17 |  12.34 |  17.95 |
| **POS**  |  3.67   |   4.00 |   4.51 |  12.36 |

## Supervised / deep-learning methods (cross-dataset eval)

Best DL numbers land near POS on PURE and UBFC-rPPG, but DL methods cost
training + inference latency that doesn't fit in-browser without major
engineering. Not our primary target.

# Key takeaways for cardiocam

1. **GREEN — what we ship today — is the worst classical method on every
   dataset.** MAE 19.81 BPM on UBFC-rPPG. This quantifies the gap.
2. **POS is the clear winner** across all four datasets. On UBFC-rPPG
   POS drops MAE from ~20 to ~4 — a **5× improvement**.
3. **CHROM is a very close second** on UBFC-rPPG and UBFC-Phys, and beats
   POS slightly on UBFC-Phys. Running both and picking the better one per
   session could be a small extra win.
4. **MMPD (mobile phone videos)** is the hardest dataset across the board.
   Our web app runs on phones, so this is the most realistic condition.
   Even POS sits at 12.36 MAE there — accurate rPPG on arbitrary phones
   is genuinely hard.
5. **LGI and PBV are not worth porting first** — they're sometimes worse
   than CHROM/POS. Defer.

# Implications for cardiocam roadmap

- **First port:** POS → expect MAE on your personal finger-pulse
  reference to drop substantially (5× is generous; 2-3× is realistic).
- **Second port:** CHROM as a side-by-side comparison option.
- **Do NOT port** ICA, LGI, PBV first — they don't buy enough.
- **DL methods:** out of scope until we have a browser-compatible path
  (ONNX Runtime Web, TFJS) and a tiny enough model.
