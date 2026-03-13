# FusionKick

**A Temporal Fusion Transformer Framework for Multi-Modal Wearable Sports Analytics**

> *Ayaan Ahmed Khan   
> Paper under review | [Preprint coming soon]

---

## What this is

FusionKick is a Temporal Fusion Transformer (TFT) adapted for real-time performance state prediction in football, using simultaneous input from four wearable sensor modalities: IMU (100 Hz), GPS (10 Hz), PPG heart rate / HRV (1 Hz), and EDA skin conductance (4 Hz).

The core finding: **HRV in the 0.04–0.15 Hz band dominates GPS kinematics as a predictor of pre-fatigue state transitions** — by 0.19 normalized importance units in the model's attention weights. An architecture that only watches kinematics is watching the downstream consequence of fatigue. FusionKick watches the process.

---

## This repository

Direct access to club-authorized wearable data is restricted. This repo provides the **full simulation pipeline** used to generate the evaluation dataset — every generative equation parameterized from published sports science literature, every noise model sourced from device validation studies. The goal is that any researcher can reproduce the dataset, benchmark a different architecture against ours, or extend the simulation to new modalities.

Model architecture code will be released upon paper acceptance. The simulation is the primary reproducibility contribution.

```
fusionkick/
├── simulation/
│   ├── config.py              ← All parameters, fully sourced
│   ├── generate_dataset.py    ← Main script: runs Equations 1–10
│   ├── noise_models.py        ← Per-modality noise injection
│   └── label_generator.py     ← Probabilistic labeling (Eq. 10)
├── model/
│   └── README.md              ← Architecture details + release timeline
├── requirements.txt
└── CITATION.md
```

---

## Quickstart

```bash
git clone https://github.com/[yourusername]/fusionkick.git
cd fusionkick
pip install -r requirements.txt
cd simulation
python generate_dataset.py --output ../data --seed 42
```

Expected output:

```
Generating FusionKick simulation dataset (seed=42)
Players: 68, Matches: 38
──────────────────────────────────────────────────
  Players simulated: 10/68
  Players simulated: 20/68
  ...
Total windows: ~61,342
  peak:         11.3%
  sustained:    34.7%
  transitional: 21.4%
  pre_fatigue:  19.8%
  fatigued:     12.8%

Splitting dataset (time-stratified)...
  Train: ~36,800 windows
  Val:   ~12,270 windows
  Test:  ~12,272 windows
```

Runtime: approximately 8–12 minutes on a standard laptop CPU.

---

## Simulation design

The simulation is not a shortcut around real data — it is a structured hypothesis about what real data would look like, built from peer-reviewed physiology.

### What is explicitly parameterized

| Signal | Distribution | Source |
|--------|-------------|--------|
| HR | `HR_t = HR_rest + (HR_max - HR_rest)·[α·W_t + β·F_t] + ε` | Stolen et al. (2005) |
| Fatigue | `F_t = F_{t-1}·exp(-λ·Δt) + (1-exp(-λ·Δt))·W_t` | Banister et al. (1975) |
| HRV RMSSD | `RMSSD_t = RMSSD_rest·exp(-γ·[W_t + 1.4·F_t])` | Plews et al. (2013) |
| IMU | Mixture model: baseline + sprint events | Wundersitz et al. (2015) |
| GPS velocity | `v_t ~ LogNormal(μ_v(W_t), σ_v = 0.31)` | Scott et al. (2016) |
| EDA | Linear arousal model + motion artifact | Benedek & Kaernbach (2010) |
| Cross-modal covariance | Cholesky decomposition, ρ from literature | Casamichana & Castellano (2010) |

### What makes evaluation non-trivial

Labels are generated via a probabilistic softmax scheme (τ = 0.35 default, τ = 0.65 at state transitions). A logistic regression trained directly on the generative variables achieves F1 ≈ 0.51 — the labels are not recoverable by thresholding.

---

## Results

| Model | Weighted F1 | Latency |
|-------|------------|---------|
| Logistic Regression | 0.598 | < 1 ms |
| Random Forest | 0.781 | 12 ms |
| XGBoost | 0.819 | 8 ms |
| LSTM (GPS only) | 0.843 | 94 ms |
| Ensemble GBM | 0.874 | 310 ms |
| **FusionKick** | **0.917** | **180 ms** |

Edge deployment on NVIDIA Jetson Orin Nano: 180 ± 23 ms end-to-end latency.

---

## Known limitations

- Dataset is synthetic. Real-data validation is the necessary next step.
- EDA wristbands are not standard kit in professional football; removing EDA costs 0.031 F1 (see ablation in paper).
- The counterfactual substitution result (23% fatigue reduction) is a simulation outcome, not an empirical claim.

---

## License

MIT License. See `LICENSE`.

---

## Contact

Ayaan Ahmed Khan — ayaan0703rohila@gmail.com  
