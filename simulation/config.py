"""
FusionKick Simulation Configuration
All parameters drawn from peer-reviewed sports science literature.
Sources cited inline — see paper for full references.
"""

# ── Player population ──────────────────────────────────────────────────────
# Carling et al. (2008), Sports Medicine
PLAYER_POSITIONS = ["central_defender", "full_back", "central_mid", "wide_mid", "striker"]
N_PLAYERS        = 68
N_MATCHES        = 38

ANTHROPOMETRICS = {
    # (mean, std) for each field
    "central_defender": {"age": (26.4, 3.1), "mass_kg": (82.3, 4.8), "height_cm": (185.2, 5.1), "vo2max": (58.1, 4.2)},
    "full_back":        {"age": (25.1, 2.8), "mass_kg": (76.6, 3.9), "height_cm": (179.4, 4.3), "vo2max": (62.4, 3.8)},
    "central_mid":      {"age": (25.8, 3.4), "mass_kg": (77.1, 4.1), "height_cm": (180.1, 4.9), "vo2max": (63.7, 4.1)},
    "wide_mid":         {"age": (24.6, 2.9), "mass_kg": (74.3, 3.6), "height_cm": (176.8, 4.7), "vo2max": (65.2, 3.5)},
    "striker":          {"age": (25.3, 3.2), "mass_kg": (78.9, 4.3), "height_cm": (181.6, 5.3), "vo2max": (61.8, 4.0)},
}

# ── Workload dynamics (Eq. 2) ─────────────────────────────────────────────
# Bradley et al. (2009), Journal of Sports Sciences
WORKLOAD = {
    "base": {
        "central_defender": 0.65,
        "full_back":         0.72,
        "central_mid":       0.78,
        "wide_mid":          0.75,
        "striker":           0.71,
    },
    "amplitude":    0.22,       # sinusoidal intensity swing
    "phase_min":    8.0,        # T_phase — match ebb-and-flow period (minutes)
    "noise_std":    0.04,       # ε_w white noise std
}

# ── Fatigue accumulation (Eq. 3) ──────────────────────────────────────────
# Banister et al. (1975), Australian Journal of Sports Medicine
FATIGUE = {
    "lambda":       0.018,      # decay constant (min^-1), half-life ~38 min
}

# ── Heart rate model (Eq. 4) ─────────────────────────────────────────────
# Stolen et al. (2005), Sports Medicine
HR = {
    "rest_mean":    58.0,       # bpm
    "rest_std":     6.0,
    "max_mean":     194.0,      # bpm
    "max_std":      8.0,
    "alpha":        0.71,       # workload coefficient
    "beta":         0.29,       # fatigue coefficient
    "noise_std":    3.2,        # PPG motion artifact — Gillinov et al. (2017)
}

# ── HRV model (Eq. 5) ─────────────────────────────────────────────────────
# Plews et al. (2013), Sports Medicine
HRV = {
    "rmssd_lognormal_mu":    3.6,   # log-space mean
    "rmssd_lognormal_sigma": 0.4,   # log-space std
    "gamma":                 1.8,   # sympathetic suppression coefficient
    "fatigue_weight":        1.4,   # fatigue contribution multiplier
}

# ── IMU acceleration model (Eq. 6) ────────────────────────────────────────
# Wundersitz et al. (2015), European Journal of Sport Science
IMU = {
    "base_mean":        1.05,   # g (resting gravitational loading)
    "base_std":         0.08,
    "sprint_mean":      2.8,    # g (peak sprint acceleration)
    "sprint_std":       0.4,
    "sprint_prob_high": 0.09,   # P(sprint) in high-intensity phase
    "sprint_prob_low":  0.02,   # P(sprint) at rest
    "noise_std":        0.12,   # g (manufacturer RMS noise floor)
}

# ── GPS velocity model (Eq. 7) ────────────────────────────────────────────
# Scott et al. (2016), Journal of Strength and Conditioning Research
GPS = {
    "lognormal_offset":  2.1,   # minimum velocity component
    "lognormal_slope":   4.8,   # workload scaling
    "lognormal_sigma":   0.31,  # shape parameter
    "ar1_rho":           0.7,   # positional error autocorrelation
    "ar1_sigma_m":       0.8,   # positional error std (metres)
}

# ── EDA model (Eq. 8) ─────────────────────────────────────────────────────
# Benedek & Kaernbach (2010), Journal of Neuroscience Methods
EDA = {
    "scl0_mean":        2.1,    # µS
    "scl0_std":         0.6,
    "delta":            1.4,    # µS per unit arousal
    "alpha":            0.6,    # workload coefficient
    "beta":             0.4,    # fatigue coefficient
    "noise_std":        0.04,   # µS
    "missing_rate":     0.083,  # wristband removal probability
    "hf_artifact_hz":   1.5,    # motion artifact cutoff
}

# ── Cross-modal covariance (Eq. 9) ────────────────────────────────────────
# rho_1: HR-IMU  — Casamichana & Castellano (2010)
# rho_2: HR-GPS  — Coutts & Duffield (2010)
# rho_3: IMU-GPS — Barrett et al. (2014)
COVARIANCE = {
    "rho_hr_imu":   0.61,
    "rho_hr_gps":   0.54,
    "rho_imu_gps":  0.73,
}

# ── Probabilistic label generation (Eq. 10) ───────────────────────────────
LABELING = {
    "tau_default":       0.35,  # softmax temperature (sharp)
    "tau_boundary":      0.65,  # temperature at state transitions (soft)
    "boundary_delta_f":  0.08,  # fatigue change threshold to trigger boundary mode
    "adjacent_noise":    0.22,  # probability of adjacent-state label at boundary
}

# ── Window / dataset settings ─────────────────────────────────────────────
WINDOWS = {
    "size_sec":     30,
    "hop_sec":      15,
    "hz":           1,          # unified temporal grid after downsampling
}

SPLIT = {
    "train": 0.60,
    "val":   0.20,
    "test":  0.20,
    "strategy": "time_stratified",
}

# ── Performance states ────────────────────────────────────────────────────
STATES = {
    0: "peak",
    1: "sustained",
    2: "transitional",
    3: "pre_fatigue",
    4: "fatigued",
}
