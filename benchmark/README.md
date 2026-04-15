# cardiocam accuracy benchmark

Offline accuracy-engineering workspace. Kept deliberately separate from
`web/` (shipped product) and the top-level Python app (reference implementation
for the web port).

Goal: measure the current rPPG algorithm against a public dataset with
ground-truth BPM, then propose + evaluate improvements. Winners get
ported to `web/src/lib/` as a separate commit after the accuracy gain
is demonstrated.

## Directory map

```
benchmark/
  datasets/       Symlinks or download scripts for public datasets
                  (actual video data lives outside the repo — too big)
  loaders/        Per-dataset loaders: video frames + GT BPM aligned in time
  algorithms/     rPPG algorithm variants (green-only, POS, CHROM, ...)
                  Each exposes estimate_bpm(frames, fps) -> per-window BPM
  evaluation/     MAE / RMSE / Pearson r / Bland-Altman; per-subject and
                  aggregate tables
  experiments/    Per-hypothesis runs: config + predictions.csv + metrics.json
  data/           Small derived files checked in: summary CSVs, plots
  plots/          Figure outputs
```

## Discipline

- Lock each experiment's intent in a commit BEFORE running it:
  `research(protocol): H{n} {what}` — makes the git history a
  lightweight pre-registration.
- Record the outcome in `experiments/H{n}-{slug}/metrics.json` and
  commit `research(results): H{n} — {outcome}`.
- Never delete a run's results, even if the hypothesis was refuted.
  Negative results are progress.

## Not a shipped artifact

Nothing in this directory is loaded by the web app or by the Python
desktop app. Safe to iterate on without touching user-facing code.
