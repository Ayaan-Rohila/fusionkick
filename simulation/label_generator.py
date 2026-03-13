"""
FusionKick — Probabilistic Label Generator
Implements Equation 10 from the paper.

Key design decision: labels are NOT a hard threshold of the generative variables.
A logistic regression on the raw variables (W_t, F_t, HRV_t) achieves F1 ≈ 0.51.
The probabilistic scheme is what makes the evaluation non-trivial.
"""

import numpy as np
from config import LABELING, STATES


# ── State scoring functions f_k ───────────────────────────────────────────
# Derived from:
#   Borg (1982). Psychophysical bases of perceived exertion.
#   Med Sci Sports Exerc, 14(5), 377-381.
# and the Banister fatigue-fitness model.
#
# Each function returns a raw score (higher = more likely to be in that state).
# Scores are NOT probabilities yet — softmax in label_from_state_vars() handles that.

def score_peak(W: float, F: float, hrv_norm: float) -> float:
    """High workload, low fatigue, good cardiac autonomic status."""
    return 3.0 * W - 2.5 * F + 1.5 * hrv_norm - 0.5


def score_sustained(W: float, F: float, hrv_norm: float) -> float:
    """Moderate-high workload, moderate fatigue, stable HRV."""
    return 2.0 * W - 1.2 * F + 0.8 * hrv_norm


def score_transitional(W: float, F: float, hrv_norm: float) -> float:
    """Workload declining, fatigue building, HRV starting to drop."""
    return -abs(W - 0.55) * 2.0 - 1.0 * F + 0.3 * hrv_norm + 0.8


def score_pre_fatigue(W: float, F: float, hrv_norm: float) -> float:
    """High fatigue, HRV clearly depressed, workload maintained but costly."""
    return -1.5 * W + 3.0 * F - 2.0 * hrv_norm + 0.5


def score_fatigued(W: float, F: float, hrv_norm: float) -> float:
    """Very high fatigue, low HRV, workload collapsing."""
    return -3.0 * W + 4.0 * F - 3.0 * hrv_norm + 1.0


SCORE_FUNCTIONS = [
    score_peak,
    score_sustained,
    score_transitional,
    score_pre_fatigue,
    score_fatigued,
]


def softmax(scores: np.ndarray, tau: float) -> np.ndarray:
    """Temperature-scaled softmax. Lower tau = sharper distribution."""
    scaled = scores / tau
    scaled -= scaled.max()   # numerical stability
    exp    = np.exp(scaled)
    return exp / exp.sum()


def label_from_state_vars(
    W: float,
    F: float,
    rmssd: float,
    rmssd_rest: float,
    delta_F: float,
    rng: np.random.Generator,
) -> int:
    """
    Assign a hard label to a single 30-second window via probabilistic sampling.

    Implements Eq. 10:
        P(state=k | W_w, F_w, HRV_w) ∝ exp(f_k(W_w, F_w, HRV_w) / τ)

    Temperature τ is raised at state transition boundaries (|ΔF| > threshold),
    producing deliberately ambiguous labels that model annotation uncertainty.

    Args:
        W:          Mean workload for the window (0-1 scale)
        F:          Mean fatigue for the window (0-1 scale)
        rmssd:      Window-mean RMSSD (ms)
        rmssd_rest: Player-specific resting RMSSD (ms)
        delta_F:    |F_t - F_{t-1}| — fatigue change magnitude
        rng:        NumPy random generator

    Returns:
        Integer label 0–4 (see config.STATES)
    """
    # Normalize HRV relative to player's resting baseline
    hrv_norm = np.clip(rmssd / (rmssd_rest + 1e-6), 0.0, 2.0) - 1.0

    # Compute raw scores for each state
    scores = np.array([f(W, F, hrv_norm) for f in SCORE_FUNCTIONS])

    # Select temperature based on whether we are at a state transition boundary
    at_boundary = delta_F > LABELING["boundary_delta_f"]
    tau = LABELING["tau_boundary"] if at_boundary else LABELING["tau_default"]

    # Convert to probability distribution
    probs = softmax(scores, tau)

    # Sample hard label
    return int(rng.choice(len(STATES), p=probs))


def generate_window_labels(
    W_series:     np.ndarray,
    F_series:     np.ndarray,
    rmssd_series: np.ndarray,
    rmssd_rest:   float,
    window_size:  int,
    hop:          int,
    rng:          np.random.Generator,
) -> list[int]:
    """
    Generate labels for all windows in a match.

    Args:
        W_series:     Workload time series, shape (T,)
        F_series:     Fatigue time series, shape (T,)
        rmssd_series: RMSSD time series, shape (T,)
        rmssd_rest:   Player resting RMSSD baseline (ms)
        window_size:  Window length in timesteps (paper: 30)
        hop:          Hop size in timesteps (paper: 15, giving 50% overlap)
        rng:          NumPy random generator

    Returns:
        List of integer labels, one per window
    """
    T      = len(W_series)
    labels = []

    starts = range(0, T - window_size + 1, hop)
    for s in starts:
        e = s + window_size

        W_mean    = float(np.nanmean(W_series[s:e]))
        F_mean    = float(np.nanmean(F_series[s:e]))
        rmssd_mean = float(np.nanmean(rmssd_series[s:e]))
        delta_F   = float(abs(F_series[e-1] - F_series[s]))

        label = label_from_state_vars(
            W=W_mean,
            F=F_mean,
            rmssd=rmssd_mean,
            rmssd_rest=rmssd_rest,
            delta_F=delta_F,
            rng=rng,
        )
        labels.append(label)

    return labels


def label_difficulty_check(
    W_series:     np.ndarray,
    F_series:     np.ndarray,
    rmssd_series: np.ndarray,
    rmssd_rest:   float,
    window_size:  int = 30,
    hop:          int = 15,
    n_trials:     int = 5,
    seed:         int = 0,
) -> float:
    """
    Sanity check: estimate how consistent labels are across multiple sampling runs.
    High consistency (>0.95) suggests the temperature is too low — labels are
    effectively deterministic and the evaluation will be too easy.
    Target: agreement ~0.82-0.88 on non-boundary windows.

    Returns:
        Mean label agreement across trials
    """
    rng   = np.random.default_rng(seed)
    all_labels = []
    for _ in range(n_trials):
        lbls = generate_window_labels(
            W_series, F_series, rmssd_series, rmssd_rest,
            window_size, hop, rng
        )
        all_labels.append(lbls)

    agreements = []
    for i in range(len(all_labels[0])):
        votes = [all_labels[t][i] for t in range(n_trials)]
        mode_count = max(votes.count(v) for v in set(votes))
        agreements.append(mode_count / n_trials)

    return float(np.mean(agreements))
