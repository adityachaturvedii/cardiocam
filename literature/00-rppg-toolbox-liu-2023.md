---
title: rPPG-Toolbox — Deep Remote PPG Toolbox
authors: Liu, Narayanswamy, Paruchuri, Zhang, Tang, Zhang, Wang, Sengupta, Patel, McDuff
year: 2023
venue: NeurIPS 2023 (arXiv:2210.00716)
url: https://github.com/ubicomplab/rPPG-Toolbox
relevance: ★★★★★ — authoritative benchmark suite; names and implementations of every major rPPG method we'd evaluate
---

# Summary

Open-source PyTorch benchmark suite from McDuff's group (UW Ubicomp Lab),
published at NeurIPS 2023. Implements every major classical and deep
learning rPPG method with a unified preprocessing and evaluation pipeline
over UBFC-rPPG, PURE, SCAMPS, UBFC-Phys, BP4D+, MMPD.

# Methods catalogued (our search space)

## Classical / unsupervised (no training)

- **GREEN** (Verkruysse 2008) — simple green-channel mean. Our current baseline.
- **ICA** (Poh 2011) — independent component analysis on RGB channels; picks the BVP-like source.
- **CHROM** (de Haan & Jeanne 2013) — project RGB to a chrominance space invariant
  to diffuse skin reflection; combines two hues with alpha-tuning.
- **LGI** (Pilz 2018) — local group invariance; projects through skin color model.
- **PBV** (de Haan 2014) — uses the empirical blood-volume pulse signature vector
  to weight RGB channels.
- **POS** (Wang 2016/2017) — plane-orthogonal-to-skin: projects into the 2D
  subspace orthogonal to the mean skin color to cancel specular reflections.
- **OMIT** (Álvarez 2023) — unsupervised pipeline with orthonormal matrix,
  recent.

## Deep learning (require training; not shippable to browser in v1)

DeepPhys, PhysNet, TS-CAN, EfficientPhys, BigSmall, PhysFormer, iBVPNet,
PhysMamba, RhythmFormer, FactorizePhys.

# Why this matters for cardiocam

- The toolbox reports **MAE / MAPE** for every method on standard datasets.
  POS and CHROM are typically within 1-2 BPM of each other and substantially
  better than GREEN on non-trivial conditions (motion, varied illumination).
- All classical methods are stateless per-window computations — direct drop-in
  replacements for our current green-channel mean.
- DL methods need training + model deployment; defer.

# Next references to chase

- Wang 2016/2017 POS — the specific algorithm we'd port first
- de Haan & Jeanne 2013 CHROM — second priority
- A recent cross-dataset benchmark (pick one with numbers we can cite)
