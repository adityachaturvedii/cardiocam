---
title: Robust Pulse Rate from Chrominance-based rPPG (CHROM)
authors: de Haan, Jeanne
year: 2013
venue: IEEE Transactions on Biomedical Engineering 60(10): 2878-2886
doi: 10.1109/TBME.2013.2266196
relevance: ★★★★ — second-best classical candidate; pairs naturally with POS in an A/B
---

# Core idea

Combine two chrominance signals derived from RGB so that skin-color-
invariant hemoglobin pulse information survives while specular motion
cancels. First popular chrominance rPPG method, predates POS.

# Algorithm (one-shot, per window)

Given RGB signal `X[3, T]`:

1. `Xcomp = 3*R - 2*G`
2. `Ycomp = 1.5*R + G - 1.5*B`
3. `alpha = std(Xcomp) / std(Ycomp)`
4. `bvp = Xcomp - alpha * Ycomp`

Constants are the paper's calibrated values for Caucasian skin. Works
reasonably across skin tones but not optimal for all.

# Verbatim reference implementation (pyVHR, MIT license)

```python
def cpu_CHROM(signal):
    X = signal
    Xcomp = 3*X[:, 0] - 2*X[:, 1]
    Ycomp = 1.5*X[:, 0] + X[:, 1] - 1.5*X[:, 2]
    sX = np.std(Xcomp, axis=1)
    sY = np.std(Ycomp, axis=1)
    alpha = (sX / sY).reshape(-1, 1)
    alpha = np.repeat(alpha, Xcomp.shape[1], 1)
    bvp = Xcomp - alpha * Ycomp
    return bvp
```

# Why it's interesting despite being older than POS

- Even simpler than POS (no sliding window; runs on the whole buffer at once).
- In benchmarks CHROM and POS are usually within 0.5 BPM MAE of each other.
- Ship both in cardiocam and let the user A/B with a toggle.

# Caveat

Fixed RGB coefficients are skin-tone dependent. POS adapts to the frame's
skin mean, which is why it generally wins on diverse datasets.
