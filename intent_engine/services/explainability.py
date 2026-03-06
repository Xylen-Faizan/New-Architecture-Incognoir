from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ExplainabilityEngine:
    risk_band_normal_max: float
    risk_band_suspicious_max: float
    risk_band_high_max: float

    def risk_band(self, score: float) -> str:
        s = float(score)
        if s < float(self.risk_band_normal_max):
            return "Normal"
        if s < float(self.risk_band_suspicious_max):
            return "Suspicious"
        if s < float(self.risk_band_high_max):
            return "High"
        return "Critical"

    def build_response(
        self,
        session_id: str,
        state_id: int,
        belief: List[float],
        risk_score: float,
        top_intent_state: int,
        confidence: float,
        lstm_prob: float,
        micro_prob: float,
        model_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "session_id": str(session_id),
            "state_id": int(state_id),
            "risk_score": float(risk_score),
            "risk_band": self.risk_band(float(risk_score)),
            "top_intent_state": int(top_intent_state),
            "confidence": float(confidence),
            "belief": [float(x) for x in belief],
            "evidence": {
                "lstm_next_token_probability": float(lstm_prob),
                "micro_probability": float(micro_prob),
            },
            "model": dict(model_snapshot),
        }

