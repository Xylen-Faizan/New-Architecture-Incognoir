"""Health score computation for enterprise dashboards and compliance.

Health score combines:
- Model confidence: normalized mean belief confidence
- Alert posture: alert volume normalized against baseline
- Recovery trajectory: false positive feedback reducing beliefs
- Operational stability: inference latency and success rates
"""

import time
from typing import Dict, Any, Optional
import numpy as np


class HealthScoreCompute:
    def __init__(self, baseline_rps: float = 10.0, baseline_p95_ms: float = 50.0):
        self.baseline_rps = float(baseline_rps)
        self.baseline_p95_ms = float(baseline_p95_ms)
        self.started_at_unix_s = time.time()

    def compute(
        self,
        mean_confidence: float,
        alert_count_24h: int,
        baseline_alerts_24h: int = 5,
        feedback_reduction_factor: float = 1.0,
        current_rps: float = 10.0,
        p95_latency_ms: float = 50.0,
    ) -> Dict[str, Any]:
        """Compute health score and component breakdown.
        
        Returns:
            {
                'score': 0-100,
                'components': {
                    'model_confidence': 0-100,
                    'alert_posture': 0-100,
                    'recovery': 0-100,
                    'stability': 0-100,
                },
                'uptime_days': float,
            }
        """
        # Component 1: Model Confidence (0-100)
        conf_score = float(np.clip(mean_confidence * 100.0, 0.0, 100.0))

        # Component 2: Alert Posture (0-100, lower is better)
        # Normalize alert count; too few or too many both penalize
        baseline = max(1, baseline_alerts_24h)
        alert_ratio = alert_count_24h / float(baseline)
        # Target is 1.0x baseline; ratio > 2 or < 0.1 penalizes
        if alert_ratio > 2.0:
            alert_posture = max(0.0, 100.0 - (alert_ratio - 2.0) * 20.0)
        elif alert_ratio < 0.1:
            alert_posture = max(0.0, 100.0 - (0.1 - alert_ratio) * 100.0)
        else:
            alert_posture = 100.0 - abs(alert_ratio - 1.0) * 20.0
        alert_posture = float(np.clip(alert_posture, 0.0, 100.0))

        # Component 3: Recovery Trajectory (0-100)
        # feedback_reduction_factor > 1 means model is recovering from false positives
        recovery_score = float(np.clip((1.0 / feedback_reduction_factor) * 100.0, 0.0, 100.0))

        # Component 4: Operational Stability (0-100)
        # Combine RPS and latency: both deviations penalize
        rps_ratio = current_rps / max(0.1, self.baseline_rps)
        latency_ratio = p95_latency_ms / max(1.0, self.baseline_p95_ms)
        # RPS too low (< 0.5x) or too high (> 2x) penalizes
        if rps_ratio < 0.5:
            rps_score = max(0.0, 100.0 - (0.5 - rps_ratio) * 100.0)
        elif rps_ratio > 2.0:
            rps_score = max(0.0, 100.0 - (rps_ratio - 2.0) * 30.0)
        else:
            rps_score = 100.0 - abs(rps_ratio - 1.0) * 30.0
        rps_score = float(np.clip(rps_score, 0.0, 100.0))

        # Latency: > 3x baseline penalizes significantly
        if latency_ratio > 3.0:
            latency_score = max(0.0, 100.0 - (latency_ratio - 3.0) * 15.0)
        else:
            latency_score = 100.0 - max(0.0, (latency_ratio - 1.0) * 25.0)
        latency_score = float(np.clip(latency_score, 0.0, 100.0))

        stability_score = (rps_score + latency_score) / 2.0

        # Composite health score: weighted average
        weights = {
            "confidence": 0.35,
            "alert_posture": 0.25,
            "recovery": 0.20,
            "stability": 0.20,
        }

        overall = (
            conf_score * weights["confidence"]
            + alert_posture * weights["alert_posture"]
            + recovery_score * weights["recovery"]
            + stability_score * weights["stability"]
        )

        uptime_days = (time.time() - self.started_at_unix_s) / 86400.0

        return {
            "score": float(np.clip(overall, 0.0, 100.0)),
            "components": {
                "model_confidence": conf_score,
                "alert_posture": alert_posture,
                "recovery": recovery_score,
                "stability": stability_score,
            },
            "uptime_days": float(uptime_days),
        }
