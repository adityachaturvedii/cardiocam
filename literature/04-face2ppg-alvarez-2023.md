---
title: Face2PPG — An unsupervised pipeline for blood volume pulse extraction from faces
authors: Constantino Álvarez Casado, Miguel Bordallo López
year: 2022/2023 (final revision)
venue: arXiv:2202.04101
relevance: ★★★★ — best modern reference for the pieces around POS we are still missing
---

# Why this paper matters for cardiocam

It makes three concrete claims, each a possible follow-up PR after POS
lands:

1. **Rigid mesh normalization** — stabilize the detected face in time
   using a canonical mesh, not the per-frame bounding box. This kills
   motion artifacts that fool temporal signals. **We don't do this
   today.** MediaPipe gives us a mesh every frame; we could warp each
   frame to a fixed canonical pose and sample ROIs from the warped
   frame.

2. **Dynamic facial region selection** — pick the cheek regions with the
   best signal quality at runtime instead of always using a fixed
   polygon. Each region is evaluated by a proxy (e.g. SNR or spectral
   flatness), the best N contribute to the BVP. **We currently ALWAYS
   use the same two cheek polygons.** Worth considering once POS is in.

3. **OMIT (Orthogonal Matrix Image Transformation)** — a new RGB→rPPG
   transformation using QR decomposition. Robust to compression
   artifacts (H.264, phone video). The paper claims it's close to
   learned methods' accuracy. In the rPPG-Toolbox table OMIT wasn't
   separately reported; treat as exploratory.

# Priority ordering

- POS first (already decided)
- Then rigid mesh normalization — biggest expected gain on mobile
  because phone users move their heads
- Then dynamic region selection
- OMIT is a tier below POS/CHROM for our case

# Caveat

The paper's full numerical comparisons aren't in the arXiv abstract.
If we want the exact MAE deltas for contributions 1-3 individually,
we'd need the PDF body. Deferred unless a specific port needs
validation.
