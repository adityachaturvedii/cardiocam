---
title: Contactless Camera-Based Heart Rate and Respiratory Rate Monitoring Using AI on Hardware
authors: Kolosov, Kelefouras, Kourtessis, Mporas
year: 2023
venue: Sensors 23(9), 4550
doi: 10.3390/s23094550
relevance: ★★ — hardware benchmarking paper; limited algorithm content for our case
---

# What this paper is

A hardware-oriented comparison of two classical contactless vital-sign
pipelines — **Eulerian Video Magnification** (EVM) and **rPPG with the
GREEN method** — deployed across five platforms (PC, two Raspberry Pi
variants, Jetson Nano, Jetson Xavier NX). The focus is **FPS,
efficiency (FPS/Watt), value (FPS/cost), latency**, not BPM accuracy.

# What's new (algorithmically)

Nothing. The pipelines are textbook implementations:
- **EVM**: Laplacian pyramid → temporal bandpass → FFT → peak. ROI resized to 180×180.
- **rPPG**: Extract mean of green channel over ROI → detrend → interpolate → Hamming window → L2 normalize → FFT → peak-in-band.

Both pipelines match, step-for-step, what cardiocam's Python reference
app did before the POS migration. They pick GREEN specifically *because*
it's the cheapest on edge hardware — an explicit accuracy trade-off.

# What's relevant to cardiocam

1. **ROI selection: forehead + both cheeks (three patches), not just
   two.** The paper uses FaceMesh landmarks to extract all three. They
   don't show numbers on three-vs-two, but **three ROIs is standard in
   the rPPG literature** (Poh 2010; Face2PPG) and would give cardiocam
   another signal source to average.
2. **Buffer size 180 frames** at 30 FPS = 6 seconds. Sits between our
   old 100 (3.3s) and new 240 (8s). No new information beyond what we
   already chose.
3. **Band 0.83–3.0 Hz = 50–180 BPM** (standard). We went 45–150 for
   better noise rejection; their range is wider.

# What is NOT relevant

- **EVM as a pulse-rate primary method** — it's a *visualization*
  technique that got repurposed for HR by finding peaks in the amplified
  color signal. Their reported accuracies (93–98%) are from small-n
  studies with different ground-truth definitions and are not directly
  comparable to modern rPPG MAE. The Face2PPG and rPPG-Toolbox papers
  did not include EVM in their recent benchmarks for a reason.
- **Edge-device hardware comparison** — we run in a browser; the FPS
  numbers on a Jetson Nano don't tell us anything about our platform.

# One-line takeaway

The only actionable bit for cardiocam is **add the forehead as a third
ROI** alongside the two cheeks. Everything else is either already done,
not transferable, or not an accuracy lever.
