from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logsumexp(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return -math.inf
    m = max(vals)
    if m == -math.inf:
        return -math.inf
    return float(m + math.log(sum(math.exp(v - m) for v in vals)))


def beta_logpdf(x: float, a: float, b: float) -> float:
    x2 = float(np.clip(x, 1e-12, 1.0 - 1e-12))
    a2 = float(max(a, 1e-9))
    b2 = float(max(b, 1e-9))
    log_b = math.lgamma(a2) + math.lgamma(b2) - math.lgamma(a2 + b2)
    return (a2 - 1.0) * math.log(x2) + (b2 - 1.0) * math.log(1.0 - x2) - log_b


def entropy(p: np.ndarray) -> float:
    p2 = np.asarray(p, dtype=np.float64)
    p2 = p2[p2 > 0.0]
    if p2.size == 0:
        return 0.0
    return float(-np.sum(p2 * np.log(p2)))


def now_unix_ms() -> int:
    import time

    return int(time.time() * 1000)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p2 = np.clip(p, eps, 1.0)
    q2 = np.clip(q, eps, 1.0)
    return float(np.sum(p2 * (np.log(p2) - np.log(q2))))


def security_health_score(
    session_risks: Iterable[float],
    critical_sessions: int,
    total_sessions: int,
    attack_velocity: float,
    entropy_avg: float,
    entropy_max: float,
    drift: float,
) -> int:
    """Compute a 0-100 Security Health score per specification.

    Returns an integer health score (rounded).
    """
    # Session risk
    sess_list = list(session_risks) if session_risks is not None else []
    if sess_list:
        session_risk = float(sum(float(x) for x in sess_list) / max(1, len(sess_list)))
    else:
        session_risk = 0.0
    session_risk_score = float(session_risk * 100.0)

    # Critical density
    total = int(max(1, int(total_sessions or 0)))
    critical_density = float(critical_sessions or 0) / float(total)
    critical_score = min(critical_density * 500.0, 100.0)

    # Attack velocity
    velocity_score = min(float(attack_velocity or 0.0) * 2.0, 100.0)

    # Uncertainty
    uncertainty_score = 0.0
    try:
        if float(entropy_max or 0.0) > 0.0:
            uncertainty_score = (float(entropy_avg or 0.0) / float(entropy_max)) * 100.0
    except Exception:
        uncertainty_score = 0.0

    # Drift (assumed to be in 0..1 range or small float); convert to 0-100
    drift_score = min(float(drift or 0.0) * 100.0, 100.0)

    # Weighted risk index
    risk_index = (
        0.35 * session_risk_score
        + 0.30 * critical_score
        + 0.15 * velocity_score
        + 0.10 * uncertainty_score
        + 0.10 * drift_score
    )

    health = max(0.0, min(100.0, 100.0 - float(risk_index)))
    return int(round(health))
