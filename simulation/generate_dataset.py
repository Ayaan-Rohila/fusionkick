"""
FusionKick — Main Dataset Generation Script
Implements the full simulation pipeline (Algorithm 1, Phase 1 from the paper).

Usage:
    python generate_dataset.py --output ./data --seed 42

Output:
    ./data/
        train.npz   — 60% of windows
        val.npz     — 20% of windows
        test.npz    — 20% of windows
        players.json — static covariate metadata
"""

import argparse
import json
import numpy as np
from pathlib import Path

from config import (
    PLAYER_POSITIONS, N_PLAYERS, N_MATCHES,
    ANTHROPOMETRICS, WORKLOAD, FATIGUE, HR, HRV,
    IMU, GPS, EDA, COVARIANCE, LABELING, WINDOWS, SPLIT, STATES
)
from noise_models import (
    inject_ppg_motion_artifact, inject_imu_noise,
    inject_gps_dropout, inject_gps_positional_error,
    inject_eda_motion_artifact, simulate_eda_missingness
)
from label_generator import generate_window_labels


# ── 1. Player generation (Eq. 1) ──────────────────────────────────────────

def sample_player(position: str, rng: np.random.Generator) -> dict:
    """Sample static covariate vector for one virtual player."""
    params = ANTHROPOMETRICS[position]
    return {
        "position":   position,
        "age":        float(rng.normal(*params["age"])),
        "mass_kg":    float(rng.normal(*params["mass_kg"])),
        "height_cm":  float(rng.normal(*params["height_cm"])),
        "vo2max":     float(rng.normal(*params["vo2max"])),
        "seasonal_load": float(rng.uniform(0.3, 0.9)),  # cumulative load proxy
    }


def encode_position(position: str) -> np.ndarray:
    """One-hot encode position (4 categories, goalkeeper excluded)."""
    vec = np.zeros(4)
    idx = {"central_defender": 0, "full_back": 0,
           "central_mid": 1, "wide_mid": 1, "striker": 2}.get(position, 3)
    vec[idx] = 1.0
    return vec


# ── 2. Workload simulation (Eq. 2) ────────────────────────────────────────

def simulate_workload(
    position: str,
    match_duration_min: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Match workload W_t — piecewise sinusoidal with stochastic perturbation.
    Output shape: (match_duration_min * 60,) at 1 Hz after downsampling.
    """
    T    = match_duration_min   # 1 timestep = 1 second at generation, then we resample
    W_b  = WORKLOAD["base"][position]
    A    = WORKLOAD["amplitude"]
    T_p  = WORKLOAD["phase_min"] * 60   # seconds
    t    = np.arange(T)
    eps  = rng.normal(0, WORKLOAD["noise_std"], size=T)

    W = W_b * (1 + A * np.sin(2 * np.pi * t / T_p) + eps)
    return np.clip(W, 0.0, 1.0)


# ── 3. Fatigue accumulation (Eq. 3) ───────────────────────────────────────

def accumulate_fatigue(W: np.ndarray, dt_min: float = 1/60) -> np.ndarray:
    """
    First-order Banister fatigue model.
    λ = 0.018 min^-1, half-life ≈ 38 min.
    """
    lam = FATIGUE["lambda"]
    decay = np.exp(-lam * dt_min)
    F     = np.zeros_like(W)
    for t in range(1, len(W)):
        F[t] = F[t-1] * decay + (1 - decay) * W[t]
    return F


# ── 4. Heart rate (Eq. 4) + HRV (Eq. 5) ──────────────────────────────────

def simulate_hr(
    W: np.ndarray,
    F: np.ndarray,
    hr_rest: float,
    hr_max:  float,
    rng:     np.random.Generator,
) -> np.ndarray:
    """Clean HR signal from workload and fatigue (before PPG noise injection)."""
    alpha = HR["alpha"]
    beta  = HR["beta"]
    hr_clean = hr_rest + (hr_max - hr_rest) * (alpha * W + beta * F)
    return np.clip(hr_clean, hr_rest - 5, hr_max + 5)


def simulate_rmssd(
    W:          np.ndarray,
    F:          np.ndarray,
    rmssd_rest: float,
) -> np.ndarray:
    """HRV RMSSD — inversely proportional to sympathetic load (Eq. 5)."""
    gamma  = HRV["gamma"]
    fw     = HRV["fatigue_weight"]
    load   = W + fw * F
    return rmssd_rest * np.exp(-gamma * load)


def derive_lf_power(rmssd: np.ndarray) -> np.ndarray:
    """
    Approximate LF power from RMSSD using the empirical mapping in
    Task Force guidelines (1996), Circulation, 93(5), 1043-1065.
    LF power ≈ 0.54 * RMSSD^1.8  (log-linear in ms² domain, simplified here)
    """
    return 0.54 * np.power(np.maximum(rmssd, 1e-3), 1.8)


# ── 5. IMU (Eq. 6) ────────────────────────────────────────────────────────

def simulate_imu(
    W:   np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Tri-axial acceleration magnitude — mixture of baseline + sprint events."""
    T = len(W)
    a_base = rng.normal(IMU["base_mean"], IMU["base_std"], size=T)

    # Sprint probability varies with workload
    p_sprint = np.where(W > 0.7, IMU["sprint_prob_high"], IMU["sprint_prob_low"])
    is_sprint = rng.random(T) < p_sprint
    a_sprint  = rng.normal(IMU["sprint_mean"], IMU["sprint_std"], size=T)

    acc = a_base + is_sprint * a_sprint * W
    return np.maximum(0.0, acc)


# ── 6. GPS velocity (Eq. 7) ───────────────────────────────────────────────

def simulate_gps_velocity(
    W:   np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Log-normal velocity conditioned on workload."""
    mu_v = np.log(GPS["lognormal_offset"] + GPS["lognormal_slope"] * W)
    return rng.lognormal(mu_v, GPS["lognormal_sigma"])


# ── 7. EDA (Eq. 8) ────────────────────────────────────────────────────────

def simulate_eda(
    W:    np.ndarray,
    F:    np.ndarray,
    scl0: float,
    rng:  np.random.Generator,
) -> np.ndarray:
    """Tonic skin conductance level driven by sympathetic arousal."""
    alpha = EDA["alpha"]
    beta  = EDA["beta"]
    delta = EDA["delta"]
    noise = rng.normal(0, EDA["noise_std"], size=len(W))
    scl   = scl0 + delta * (alpha * W + beta * F) + noise
    return np.maximum(0.0, scl)


# ── 8. Cross-modal covariance enforcement (Eq. 9) ─────────────────────────

def enforce_covariance(
    hr:  np.ndarray,
    acc: np.ndarray,
    vel: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enforce cross-modal correlations via Cholesky decomposition.
    ρ(HR, IMU) = 0.61,  ρ(HR, GPS) = 0.54,  ρ(IMU, GPS) = 0.73
    EDA treated as independent (orthogonal to kinematic channels).
    """
    r12 = COVARIANCE["rho_hr_imu"]
    r13 = COVARIANCE["rho_hr_gps"]
    r23 = COVARIANCE["rho_imu_gps"]

    # Correlation matrix
    R = np.array([
        [1.0,  r12,  r13],
        [r12,  1.0,  r23],
        [r13,  r23,  1.0],
    ])

    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        # Matrix not PD — use nearest PD approximation
        eigvals, eigvecs = np.linalg.eigh(R)
        R_pd = eigvecs @ np.diag(np.maximum(eigvals, 1e-6)) @ eigvecs.T
        L    = np.linalg.cholesky(R_pd)

    T = len(hr)
    # Z-score the signals
    def zs(x): return (x - x.mean()) / (x.std() + 1e-8)

    Z = np.stack([zs(hr), zs(acc), zs(vel)], axis=0)   # (3, T)
    # Project through Cholesky to enforce correlations
    Z_corr = L @ rng.standard_normal((3, T)) * Z.std(axis=1, keepdims=True)
    Z_corr += Z.mean(axis=1, keepdims=True)

    # Rescale back to original units
    hr_out  = Z_corr[0] * hr.std()  + hr.mean()
    acc_out = Z_corr[1] * acc.std() + acc.mean()
    vel_out = Z_corr[2] * vel.std() + vel.mean()

    return (
        np.maximum(hr_out, 40.0),
        np.maximum(acc_out, 0.0),
        np.maximum(vel_out, 0.0),
    )


# ── 9. Per-player normalization ────────────────────────────────────────────

def rolling_zscore(
    signal:      np.ndarray,
    window_size: int = 1200,   # 20 minutes at 1 Hz
) -> np.ndarray:
    """
    Per-player rolling z-score normalization.
    Forces the model to learn trajectory dynamics, not absolute magnitudes.
    """
    out = np.zeros_like(signal, dtype=float)
    for t in range(len(signal)):
        start = max(0, t - window_size)
        w     = signal[start:t+1]
        mu, sigma = w.mean(), w.std()
        out[t] = (signal[t] - mu) / (sigma + 1e-8)
    return out


# ── 10. Full match simulation ──────────────────────────────────────────────

def simulate_match(
    player:             dict,
    match_duration_min: int,
    rng:                np.random.Generator,
) -> dict:
    """
    Simulate one full match for one player.
    Returns a dict of raw (unnormalized) sensor arrays and labels.
    """
    position = player["position"]

    # Sample player-specific physiological baseline parameters
    hr_rest   = float(rng.normal(HR["rest_mean"],  HR["rest_std"]))
    hr_max    = float(rng.normal(HR["max_mean"],   HR["max_std"]))
    rmssd_rest = float(rng.lognormal(HRV["rmssd_lognormal_mu"], HRV["rmssd_lognormal_sigma"]))
    scl0       = float(rng.normal(EDA["scl0_mean"], EDA["scl0_std"]))

    # Duration in seconds (Eq. 2 generates at 1-second resolution)
    T_sec = match_duration_min * 60

    # Phase 1 — generate clean signals
    W   = simulate_workload(position, T_sec, rng)
    F   = accumulate_fatigue(W)
    hr_clean  = simulate_hr(W, F, hr_rest, hr_max, rng)
    rmssd     = simulate_rmssd(W, F, rmssd_rest)
    lf_power  = derive_lf_power(rmssd)
    acc_clean = simulate_imu(W, rng)
    vel_clean = simulate_gps_velocity(W, rng)
    eda_clean = simulate_eda(W, F, scl0, rng)

    # Phase 2 — enforce cross-modal covariance (Eq. 9)
    hr_clean, acc_clean, vel_clean = enforce_covariance(hr_clean, acc_clean, vel_clean, rng)

    # Phase 3 — inject sensor-specific noise
    hr_noisy  = inject_ppg_motion_artifact(hr_clean, rng)
    acc_noisy = inject_imu_noise(acc_clean, rng)
    vel_noisy = simulate_gps_velocity(W, rng)   # regenerate with lognormal noise
    vel_noisy, gps_miss  = inject_gps_dropout(vel_noisy, dropout_rate=0.021, rng=rng)
    eda_noisy = inject_eda_motion_artifact(eda_clean, sample_rate_hz=4.0, rng=rng)
    eda_noisy, eda_miss  = simulate_eda_missingness(eda_noisy, rng)

    # Interpolate NaNs for model input (missing flagged via indicators)
    def interp_nan(x):
        nans = np.isnan(x)
        if nans.any():
            idx = np.arange(len(x))
            x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
        return x

    vel_noisy = interp_nan(vel_noisy)
    eda_noisy = interp_nan(eda_noisy)

    # Phase 4 — per-player rolling z-score normalization
    hr_norm   = rolling_zscore(hr_noisy)
    acc_norm  = rolling_zscore(acc_noisy)
    vel_norm  = rolling_zscore(vel_noisy)
    eda_norm  = rolling_zscore(eda_noisy)
    rmssd_norm = rolling_zscore(rmssd)

    # Phase 5 — probabilistic label generation (Eq. 10)
    labels = generate_window_labels(
        W_series=W,
        F_series=F,
        rmssd_series=rmssd,
        rmssd_rest=rmssd_rest,
        window_size=WINDOWS["size_sec"],
        hop=WINDOWS["hop_sec"],
        rng=rng,
    )

    return {
        "hr":         hr_norm,
        "rmssd":      rmssd_norm,
        "lf_power":   lf_power,
        "acc":        acc_norm,
        "velocity":   vel_norm,
        "eda":        eda_norm,
        "gps_miss":   gps_miss,
        "eda_miss":   eda_miss,
        "W":          W,
        "F":          F,
        "labels":     np.array(labels),
        "rmssd_rest": rmssd_rest,
    }


# ── 11. Dataset assembly ──────────────────────────────────────────────────

def build_dataset(seed: int = 42) -> dict:
    """
    Simulate all players × matches and return windowed dataset.
    """
    rng = np.random.default_rng(seed)

    positions = [
        PLAYER_POSITIONS[i % len(PLAYER_POSITIONS)]
        for i in range(N_PLAYERS)
    ]

    all_windows = []
    all_labels  = []
    all_match_ids = []
    player_meta = []

    for p_idx in range(N_PLAYERS):
        player = sample_player(positions[p_idx], rng)
        player_meta.append(player)
        pos_enc = encode_position(player["position"])

        static_cov = np.array([
            player["age"] / 30.0,            # normalized
            player["mass_kg"] / 80.0,
            player["height_cm"] / 180.0,
            player["vo2max"] / 65.0,
            player["seasonal_load"],
            *pos_enc,
        ])  # shape (9,)

        for m_idx in range(N_MATCHES):
            # Vary match duration slightly around 90 minutes
            duration = int(rng.normal(93, 3))
            duration = max(85, min(duration, 100))

            match = simulate_match(player, duration, rng)

            # Build windows
            T       = match["hr"].shape[0]
            ws      = WINDOWS["size_sec"]
            hop     = WINDOWS["hop_sec"]
            starts  = list(range(0, T - ws + 1, hop))
            n_win   = len(starts)
            n_lab   = len(match["labels"])

            for w_i, s in enumerate(starts[:n_lab]):
                e = s + ws
                window = np.stack([
                    match["hr"][s:e],
                    match["rmssd"][s:e],
                    match["lf_power"][s:e],
                    match["acc"][s:e],
                    match["velocity"][s:e],
                    match["eda"][s:e],
                    match["gps_miss"][s:e],
                    match["eda_miss"][s:e],
                    match["W"][s:e],
                    match["F"][s:e],
                ], axis=1)  # (30, 10)

                # Append static covariates as a separate array
                all_windows.append({
                    "temporal":  window.astype(np.float32),
                    "static":    static_cov.astype(np.float32),
                    "match_id":  p_idx * N_MATCHES + m_idx,
                })
                all_labels.append(match["labels"][w_i])
                all_match_ids.append(p_idx * N_MATCHES + m_idx)

        if (p_idx + 1) % 10 == 0:
            print(f"  Players simulated: {p_idx+1}/{N_PLAYERS}")

    labels = np.array(all_labels)
    match_ids = np.array(all_match_ids)
    print(f"\nTotal windows: {len(labels)}")
    for k, v in STATES.items():
        print(f"  {v}: {(labels == k).sum()} ({100*(labels==k).mean():.1f}%)")

    return {
        "windows":    all_windows,
        "labels":     labels,
        "match_ids":  match_ids,
        "player_meta": player_meta,
    }


def time_stratified_split(
    windows:   list,
    labels:    np.ndarray,
    match_ids: np.ndarray,
) -> tuple[dict, dict, dict]:
    """Split by match_id order to avoid data leakage."""
    unique_ids = np.unique(match_ids)
    n  = len(unique_ids)
    n_train = int(n * SPLIT["train"])
    n_val   = int(n * SPLIT["val"])

    train_ids = set(unique_ids[:n_train])
    val_ids   = set(unique_ids[n_train:n_train+n_val])
    test_ids  = set(unique_ids[n_train+n_val:])

    def subset(id_set):
        mask = np.array([mid in id_set for mid in match_ids])
        return {
            "windows": [w for w, m in zip(windows, mask) if m],
            "labels":  labels[mask],
        }

    return subset(train_ids), subset(val_ids), subset(test_ids)


# ── 12. Entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate FusionKick simulation dataset")
    parser.add_argument("--output", type=str, default="./data", help="Output directory")
    parser.add_argument("--seed",   type=int, default=42,       help="Random seed")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating FusionKick simulation dataset (seed={args.seed})")
    print(f"Players: {N_PLAYERS}, Matches: {N_MATCHES}")
    print("─" * 50)

    dataset = build_dataset(seed=args.seed)

    print("\nSplitting dataset (time-stratified)...")
    train, val, test = time_stratified_split(
        dataset["windows"],
        dataset["labels"],
        dataset["match_ids"],
    )

    print(f"  Train: {len(train['labels'])} windows")
    print(f"  Val:   {len(val['labels'])} windows")
    print(f"  Test:  {len(test['labels'])} windows")

    # Save splits
    for name, split in [("train", train), ("val", val), ("test", test)]:
        temporal = np.stack([w["temporal"] for w in split["windows"]])
        static   = np.stack([w["static"]   for w in split["windows"]])
        np.savez_compressed(
            out / f"{name}.npz",
            temporal=temporal,
            static=static,
            labels=split["labels"],
        )
        print(f"  Saved {name}.npz — temporal: {temporal.shape}, static: {static.shape}")

    # Save player metadata
    with open(out / "players.json", "w") as f:
        json.dump(dataset["player_meta"], f, indent=2)

    print(f"\nDataset saved to {out}/")
    print("Run `python train.py --data ./data` to train FusionKick (see model/README.md)")


if __name__ == "__main__":
    main()
