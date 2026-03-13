"""
FusionKick — Sensor Noise Models
Implements per-modality noise injection as described in Section 3.4 of the paper.
Each noise model is parameterized from published device validation studies.
"""

import numpy as np
from config import HR, IMU, GPS, EDA


def inject_ppg_motion_artifact(hr_signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Adds PPG motion artifact noise to a clean heart rate signal.
    
    Noise std = 3.2 bpm, calibrated from:
        Gillinov et al. (2017). Variable accuracy of wearable heart rate monitors
        during aerobic exercise. Med Sci Sports Exerc, 49(8), 1697-1703.

    Args:
        hr_signal: Clean HR array (bpm), shape (T,)
        rng:       NumPy random generator for reproducibility

    Returns:
        Noisy HR array, same shape
    """
    noise = rng.normal(0, HR["noise_std"], size=hr_signal.shape)
    return hr_signal + noise


def inject_imu_noise(acc_signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Adds accelerometer sensor noise.

    RMS noise floor = 0.12 g, from STATSports Apex Pro manufacturer specification,
    consistent with Wundersitz et al. (2015). Validity of a trunk-mounted accelerometer
    to assess peak accelerations. Eur J Sport Sci, 15(5), 382-390.

    Args:
        acc_signal: Clean acceleration array (g), shape (T,)
        rng:        NumPy random generator

    Returns:
        Noisy acceleration array
    """
    noise = rng.normal(0, IMU["noise_std"], size=acc_signal.shape)
    return np.maximum(0.0, acc_signal + noise)   # acceleration magnitude >= 0


def inject_gps_positional_error(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adds AR(1) positional error to GPS coordinates.

    Error model: ρ = 0.7, σ = 0.8 m
    Reflects stadium multipath interference documented in:
        Scott et al. (2016). The validity and reliability of global positioning
        systems in team sport. J Strength Cond Res, 30(5), 1470-1490.

    Args:
        x, y: Clean coordinate arrays (metres), shape (T,)
        rng:  NumPy random generator

    Returns:
        (x_noisy, y_noisy) tuple
    """
    T = len(x)
    rho   = GPS["ar1_rho"]
    sigma = GPS["ar1_sigma_m"]

    err_x = np.zeros(T)
    err_y = np.zeros(T)
    innovation_std = sigma * np.sqrt(1 - rho**2)

    for t in range(1, T):
        err_x[t] = rho * err_x[t-1] + rng.normal(0, innovation_std)
        err_y[t] = rho * err_y[t-1] + rng.normal(0, innovation_std)

    return x + err_x, y + err_y


def inject_gps_dropout(
    velocity: np.ndarray,
    dropout_rate: float,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates GPS satellite dropout (e.g., stadium roof occlusion).
    Returns the signal with NaN at dropout timesteps, plus a binary
    missingness indicator that can be fed as an auxiliary model input.

    Args:
        velocity:     GPS velocity array, shape (T,)
        dropout_rate: Fraction of timesteps to drop (paper: 0.021)
        rng:          NumPy random generator

    Returns:
        (velocity_with_nans, missingness_indicator) both shape (T,)
    """
    mask    = rng.random(velocity.shape) < dropout_rate
    v_clean = velocity.copy().astype(float)
    v_clean[mask] = np.nan
    return v_clean, mask.astype(float)


def inject_eda_motion_artifact(
    scl: np.ndarray,
    sample_rate_hz: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Adds high-frequency motion artifact to EDA signal.

    Artifact model from:
        Benedek & Kaernbach (2010). Decomposition of skin conductance data.
        J Neurosci Methods, 190(1), 80-91.

    Injects bandlimited noise above 1.5 Hz — the physiological EDA signal
    is almost entirely below 1 Hz; contamination above 1.5 Hz is pure artifact.

    Args:
        scl:            Tonic skin conductance level (µS), shape (T,)
        sample_rate_hz: Sampling rate of the signal
        rng:            NumPy random generator

    Returns:
        Artifact-contaminated EDA signal
    """
    T = len(scl)
    white = rng.normal(0, EDA["noise_std"], size=T)

    # High-pass filter above 1.5 Hz to isolate artifact band
    cutoff_ratio = EDA["hf_artifact_hz"] / (sample_rate_hz / 2)
    cutoff_ratio = min(cutoff_ratio, 0.99)  # must be < 1

    from scipy.signal import butter, filtfilt
    b, a   = butter(2, cutoff_ratio, btype="high")
    artifact = filtfilt(b, a, white)

    return scl + artifact


def simulate_eda_missingness(
    scl: np.ndarray,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates wristband removal events (EDA missing rate: 8.3%).
    Returns signal with NaN at missing timesteps and a missingness mask.

    Args:
        scl: EDA signal, shape (T,)
        rng: NumPy random generator

    Returns:
        (scl_with_nans, missingness_mask) both shape (T,)
    """
    mask     = rng.random(scl.shape) < EDA["missing_rate"]
    scl_out  = scl.copy().astype(float)
    scl_out[mask] = np.nan
    return scl_out, mask.astype(float)
