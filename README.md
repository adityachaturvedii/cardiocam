# cardiocam

Measure your heart rate from a webcam. Runs entirely in your browser — no
install, no upload, no server.

**Live app:** https://adityachaturvedii.github.io/cardiocam/

cardiocam implements remote photoplethysmography (rPPG): subtle
pulse-driven color changes in the skin of your face are extracted from
video and turned into a BPM estimate in real time. The algorithm pipeline
follows well-established methods from the rPPG literature; all
computation happens on-device via MediaPipe (GPU-accelerated where
available) and a small TypeScript DSP core.

> Not a medical device. cardiocam is a research and demo project — do not
> use its readings for diagnosis or treatment decisions.

## How it works

1. **Face tracking** — MediaPipe Face Mesh gives 478 landmarks per frame.
   Two cheek-apex polygons are derived from the zygomatic region and
   sampled each frame for R, G, B means.
2. **BVP extraction** — the RGB time series is transformed into a blood
   volume pulse signal. Multiple methods are available:
   - **POS** (Wang et al., 2017) — plane-orthogonal-to-skin projection,
     default. Cancels specular reflections and common-mode illumination
     drift.
   - **CHROM** (de Haan & Jeanne, 2013) — chrominance subtraction. Older
     and simpler, still competitive on stable scenes.
   - **GREEN** (Verkruysse et al., 2008) — green-channel mean. Baseline.
   - Plus four hybrid modes that z-score and average the pure methods.
3. **BPM estimation** — detrend → Hamming window → zero-padded FFT (2048
   bins over ~8 s) → argmax inside the 45–150 BPM band.
4. **Stability** — the power spectrum is exponentially accumulated across
   frames (half-life ~8 s) so the persistent cardiac peak stands out
   from transient noise. The peak search is clamped to `±20 BPM` of the
   rolling median and softly biased toward it. An SNR gate with
   hysteresis (enter at 4, exit at 2.5, ~1 s debounce) decides when to
   trust the reading; the displayed BPM is an EWMA of the robust median
   of recent valid estimates.

## Running the web app locally

```bash
cd web
npm install
npm run dev
```

Then open http://localhost:5173/. Grant camera access when prompted. The
`Method` dropdown lets you A/B the seven BVP modes.

### Tests

```bash
cd web
npm test
```

16 unit tests cover the DSP pipeline: detrend, Hamming window, FFT peak
detection with and without prior-biased search, POS and CHROM synthetic
recovery, and SNR on noise vs signal.

### Production build

```bash
cd web
npm run build
```

The output under `web/dist/` is a ~400 KB static bundle (+ a 3.6 MB
MediaPipe model loaded lazily from `/models/`). GitHub Pages builds and
deploys `web/**` on every push to `master` via `.github/workflows/deploy-web.yml`.

## Python reference implementation

A PyQt5 desktop app lives at the repo root. It predates the web port and
is retained as a reference — same signal pipeline, slightly different
stability defaults. Not the recommended way to use cardiocam.

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python GUI.py
```

Requires a Python 3.10 virtualenv because of MediaPipe's build wheels.

## Project structure

```
.
├── web/                     Browser app (primary artifact)
│   ├── src/lib/             DSP core — BVP methods, FFT, estimator
│   ├── src/pages/           Landing + Reader
│   ├── src/components/      Plots, heart icon, footer
│   └── public/models/       Vendored MediaPipe face_landmarker.task
├── literature/              Survey notes + rPPG benchmark references
├── findings.md              Running notes on accuracy decisions
├── GUI.py                   Python reference implementation
└── mediapipe_face.py        Python MediaPipe wrapper
```

## References

- Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). "Remote
  plethysmographic imaging using ambient light." *Optics Express*,
  16(26), 21434–21445.
- Poh, M.-Z., McDuff, D. J., & Picard, R. W. (2010). "Non-contact,
  automated cardiac pulse measurements using video imaging and blind
  source separation." *Optics Express*, 18(10), 10762–10774.
  [doi:10.1364/OE.18.010762](https://doi.org/10.1364/OE.18.010762)
- de Haan, G., & Jeanne, V. (2013). "Robust pulse rate from
  chrominance-based rPPG." *IEEE TBME*, 60(10), 2878–2886.
  [doi:10.1109/TBME.2013.2266196](https://doi.org/10.1109/TBME.2013.2266196)
- Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
  "Algorithmic principles of remote PPG." *IEEE TBME*, 64(7), 1479–1491.
  [doi:10.1109/TBME.2016.2609282](https://doi.org/10.1109/TBME.2016.2609282)
- Liu, X. et al. (2023). "rPPG-Toolbox: Deep Remote PPG Toolbox."
  *NeurIPS 2023* ([arXiv:2210.00716](https://arxiv.org/abs/2210.00716)).
  Source of the cross-dataset MAE figures that informed the method
  choice — see `literature/03-rppg-toolbox-benchmark-table.md`.

## License

MIT.
