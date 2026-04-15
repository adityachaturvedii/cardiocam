---
title: Algorithmic Principles of Remote PPG (POS)
authors: Wang, den Brinker, Stuijk, de Haan
year: 2016 (IEEE TBME), often cited as 2017
venue: IEEE Transactions on Biomedical Engineering 64(7): 1479-1491
doi: 10.1109/TBME.2016.2609282
relevance: ★★★★★ — top candidate for porting to cardiocam
---

# Core idea

Green-channel rPPG treats the pulse as a variation in one color dimension.
**POS projects the per-frame RGB color into a 2D subspace orthogonal to the
mean skin tone**, then combines the two projected traces with an alpha term
tuned so the output matches the blood-volume pulse direction.

Rationale: specular reflections and diffuse illumination changes move
color along the mean-skin direction. Projecting orthogonal to that axis
cancels those confounds and leaves the pulse-induced chromatic shift
(different in rgb because hemoglobin absorbs green more than red/blue).

# Algorithm (sliding window, length w = 1.6 * fps frames)

For each window of RGB samples `C[c=3, w]`:

1. **Temporal normalize:** `Cn = C / mean(C, axis=time)`
2. **Project:** `S = P · Cn` where `P = [[0, 1, -1], [-2, 1, 1]]` (2×3)
3. **Tune:** `alpha = std(S[0]) / std(S[1])`, `H = S[0] + alpha * S[1]`
4. **Overlap-add** into the running BVP trace.

That's the whole algorithm. Parameter-free except for the window length.

# Verbatim reference implementation (pyVHR, MIT license)

```python
def cpu_POS(signal, fps):
    eps = 1e-9
    X = signal                       # shape [e, 3, T]
    e, c, f = X.shape
    w = int(1.6 * fps)
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)
    H = np.zeros((e, f))
    for n in np.arange(w, f):
        m = n - w + 1
        Cn = X[:, :, m:(n+1)]
        M = 1.0 / (np.mean(Cn, axis=2) + eps)
        Cn = M[:, :, None] * Cn
        S = np.dot(Q, Cn)[0]         # project, drop singleton
        S = np.swapaxes(S, 0, 1)
        S1, S2 = S[:, 0, :], S[:, 1, :]
        alpha = (np.std(S1, axis=1) / (np.std(S2, axis=1) + eps))[:, None]
        Hn = S1 + alpha * S2
        Hn = Hn - np.mean(Hn, axis=1, keepdims=True)
        H[:, m:(n+1)] += Hn
    return H
```

# Why this port is cheap for cardiocam

- Replaces `greenMean` input with `[red, green, blue]` means per frame (trivial
  change in [roi.ts](../web/src/lib/roi.ts) — compute three channel means,
  not just green).
- POS runs inside `signalProcessing.ts` as a preprocessing step before our
  existing detrend → Hamming → FFT. Adds one sliding-window pass; O(T·w).
- Entire algorithm is ~20 lines of JS, fully deterministic, no training.

# Expected improvement (literature)

Reported gains over green-channel on UBFC-rPPG in the rPPG-Toolbox paper
(Liu 2023, table 3): MAE **~4–7 BPM for GREEN** vs **~2–4 BPM for POS**
depending on preprocessing. POS is consistently within 0.5 BPM of CHROM
and slightly better than CHROM on datasets with illumination changes.

# Caveat

POS needs at least ~1.6 seconds of RGB history to produce one output
sample. Our current buffer is 3.3 s which is fine. The first ~50 frames
after Start will be zero-filled from the overlap-add.
