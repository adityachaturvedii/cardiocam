# rPPG literature survey — cardiocam accuracy project

## Scope

Techniques published after Poh, McDuff, Picard (2010) that could improve
the green-channel + FFT baseline cardiocam currently uses. Focus on
methods that are:

- Classical (no training), **or** trained but small and shippable to the browser
- Robust to real-world conditions (motion, illumination, skin tone, phone cameras)
- Straightforward enough to implement in TS/Python without a GPU

One file per key paper lives alongside this survey; this document is
the index and running narrative.

## Papers reviewed

(filled in below as the survey proceeds)

## Current cardiocam pipeline, for comparison

- **Signal source:** spatial mean of the green channel over two cheek
  polygon ROIs extracted from MediaPipe Face Mesh.
- **Window:** 100-sample rolling buffer (~3.3 s at 30 FPS).
- **Preprocessing:** linear detrend → Hamming window → L2 normalize.
- **Spectral estimate:** zero-padded FFT to 1024 bins → peak in 50–180 BPM.
- **Gating:** SNR = peak / in-band median, threshold 4; warmup 10 frames
  after buffer first fills.
- **Smoothing:** median of the most recent 50 valid BPM estimates.

## What I'm looking for

1. Better chrominance-based signals (POS, CHROM, PBV) that cancel specular
   reflections and motion-induced lighting changes.
2. Motion-robust ROI tracking beyond static cheek polygons.
3. Illumination normalization that doesn't need a calibration target.
4. Simple post-processing (filter banks, harmonic templates, Kalman) that
   lowers BPM variance without retraining.
5. Cross-device / skin-tone fairness findings.
