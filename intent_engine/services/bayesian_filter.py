from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from typing import Sequence

EPS = 1e-12

from .lstm_emission import LSTMEmission
from .markov import MarkovArtifacts


@dataclass(frozen=True)
class FilterState:
    belief: List[float]
    history: List[int]


@dataclass(frozen=True)
class FilterResult:
    belief: List[float]
    history: List[int]
    risk_score: float
    top_intent_state: int
    confidence: float
    lstm_prob: float
    micro_prob: float


class BayesianFilter:
    def __init__(
        self,
        artifacts: MarkovArtifacts,
        lstm: LSTMEmission,
        max_history_len: int,
        alpha: float = 0.4,
        decay_lambda: float = 0.03,
        temperature: float = 1.0,
        calibration_temperature: float = 1.5,
        entropy_floor: float = 0.02,
        logit_clamp_min: float = -5.0,
        logit_clamp_max: float = 5.0,
    ) -> None:
        self._artifacts = artifacts
        self._lstm = lstm
        self._max_history_len = int(max_history_len)

        self._a = artifacts.transition_model.transition_matrix()
        self._h = int(self._a.shape[0])

        self._alpha = float(alpha)
        self._decay_lambda = float(decay_lambda)
        self._temperature = float(max(1e-6, temperature))
        self._calibration_temperature = float(max(1e-6, calibration_temperature))
        self._entropy_floor = float(max(0.0, entropy_floor))
        self._logit_clamp_min = float(logit_clamp_min)
        self._logit_clamp_max = float(logit_clamp_max)

    def _compute_evidence_score(
        self,
        state_token: int,
        history: List[int],
        lstm_prob: float,
        micro_features: Dict[str, Any],
        micro_prob: float,
        prev_risk: float,
    ) -> float:
        """Composite evidence score from multiple signals.
        
        Combines:
        - Rare state transition score: based on transition probability
        - LSTM surprise: -log(P(state | history))
        - Microbehavior anomaly: deviation from typical micro-features
        - Response latency anomaly: deviation from baseline response time
        """
        scores = []
        
        # 1. Rare transition score: how unusual is this seq transition?
        if len(history) > 0:
            prev_state = history[-1]
            trans_prob = float(self._a[prev_state, state_token]) if prev_state < self._a.shape[0] and state_token < self._a.shape[1] else 1.0 / max(1, self._a.shape[1])
            trans_prob = float(np.clip(trans_prob, EPS, 1.0))
            # rare transitions have low prob → high score
            rare_transition_score = 1.0 - trans_prob  # range [0, 1]
            scores.append(("rare_transition", rare_transition_score))
        
        # 2. LSTM surprise: -log(P(current_state | history))
        lstm_prob_clipped = float(np.clip(lstm_prob, EPS, 1.0))
        lstm_surprise = -np.log(lstm_prob_clipped) / 10.0  # normalize to rough [0, 1] range
        lstm_surprise = float(np.clip(lstm_surprise, 0.0, 1.0))
        scores.append(("lstm_surprise", lstm_surprise))
        
        # 3. Microbehavior anomaly: how deviant are micro-features?
        micro_prob_clipped = float(np.clip(micro_prob, EPS, 1.0))
        micro_anomaly = 1.0 - micro_prob_clipped  # deviation from typical features
        scores.append(("micro_anomaly", micro_anomaly))
        
        # 4. Response latency anomaly (if resp_time_ms available)
        resp_latency_anomaly = 0.0
        if "resp_time_ms" in micro_features:
            try:
                resp_ms = float(micro_features.get("resp_time_ms", 0))
                # simple heuristic: flag if > 1000ms or < 10ms
                if resp_ms > 1000 or resp_ms < 10:
                    resp_latency_anomaly = min(1.0, abs(resp_ms - 100.0) / 1000.0)
                scores.append(("resp_latency_anomaly", resp_latency_anomaly))
            except (ValueError, TypeError):
                pass
        
        # Composite: weighted average
        weights = {"rare_transition": 0.3, "lstm_surprise": 0.3, "micro_anomaly": 0.25, "resp_latency_anomaly": 0.15}
        total_weight = 0.0
        weighted_sum = 0.0
        for name, score in scores:
            w = weights.get(name, 0.0)
            weighted_sum += w * score
            total_weight += w
        
        evidence = weighted_sum / total_weight if total_weight > 0.0 else 0.0
        return float(np.clip(evidence, 0.0, 1.0))

    @property
    def num_hidden_states(self) -> int:
        return self._h

    def init_state(self) -> FilterState:
        belief = (np.ones((self._h,), dtype=np.float64) / float(self._h)).tolist()
        return FilterState(belief=belief, history=[])

    def update(
        self,
        state_token: int,
        micro_features: Dict[str, Any],
        prev: Optional[FilterState],
    ) -> FilterResult:
        if prev is None:
            prev = self.init_state()

        belief = np.asarray(prev.belief, dtype=np.float64)
        if belief.size != self._h:
            belief = np.ones((self._h,), dtype=np.float64) / float(self._h)
        else:
            s = float(belief.sum())
            if s > 0.0:
                belief = belief / s
            else:
                belief = np.ones((self._h,), dtype=np.float64) / float(self._h)

        history = list(prev.history)
        history = history[-int(self._max_history_len) :]

        lstm_prob = float(self._lstm.next_token_probability(history, int(state_token)))
        micro_prob = float(self._artifacts.micro_model.probability(micro_features))

        predicted = self._a.T @ belief
        ps = float(predicted.sum())
        if ps > 0.0:
            predicted = predicted / ps
        else:
            predicted = np.ones((self._h,), dtype=np.float64) / float(self._h)

        emission_vec = np.asarray(
            self._artifacts.emission_config.emission_vector(
                p_seq=lstm_prob,
                micro_p=micro_prob,
                num_hidden_states=self._h,
            ),
            dtype=np.float64,
        )

        # Temperature scaling for emission: soften or sharpen evidence
        # Convert to log-space safely and apply temperature
        ev = np.clip(emission_vec, EPS, None)
        logev = np.log(ev)
        scaled = np.exp(logev / float(self._temperature))
        if scaled.sum() > 0:
            scaled = scaled / float(scaled.sum())
        else:
            scaled = np.ones((self._h,), dtype=np.float64) / float(self._h)

        updated = predicted * scaled
        us = float(updated.sum())
        if us > 0.0:
            belief2 = updated / us
        else:
            belief2 = np.ones((self._h,), dtype=np.float64) / float(self._h)

        history2 = (history + [int(state_token)])[-int(self._max_history_len) :]

        idx: Sequence[int] = tuple(getattr(self._artifacts, "malicious_states", ()))
        # raw risk from belief
        mass_mal = float(np.sum(belief2[list(idx)])) if idx else 0.0
        mass_mal = float(np.clip(mass_mal, 0.0, 1.0))

        # smoothing in logit space with decay and clamping
        prev_risk = float(np.sum(np.asarray(belief, dtype=np.float64)[list(idx)])) if idx else 0.0
        prev_risk = float(np.clip(prev_risk, EPS, 1.0 - EPS))
        raw_risk = float(np.clip(mass_mal, EPS, 1.0 - EPS))

        # evidence accumulation with temporal decay
        # L_t = (1 - lambda) * L_{t-1} + alpha * evidence_score
        # Compute composite evidence score from multiple signals
        anomaly_score = self._compute_evidence_score(
            int(state_token), history, lstm_prob, micro_features, micro_prob, float(prev_risk)
        )
        logit_prev = np.log(prev_risk) - np.log(1.0 - prev_risk)
        # apply decay: reduce accumulated evidence over time
        logit_prev_decayed = (1.0 - float(self._decay_lambda)) * logit_prev
        logit_new = logit_prev_decayed + float(self._alpha) * float(anomaly_score)
        # clamp logit to prevent explosion
        logit_new = float(np.clip(logit_new, self._logit_clamp_min, self._logit_clamp_max))
        # convert back to probability
        p_raw = 1.0 / (1.0 + float(np.exp(-logit_new)))
        # apply calibration temperature for final scaling
        logit_calibrated = float(np.log(np.clip(p_raw, EPS, 1.0 - EPS)) - np.log(1.0 - np.clip(p_raw, EPS, 1.0 - EPS)))
        logit_calibrated = logit_calibrated / float(self._calibration_temperature)
        new_risk = 1.0 / (1.0 + float(np.exp(-logit_calibrated)))
        new_risk = float(np.clip(new_risk, EPS, 1.0 - EPS))

        # enforce entropy floor by mixing towards uniform if needed
        def entropy(p: np.ndarray) -> float:
            p = np.clip(p, EPS, 1.0)
            return float(-np.sum(p * np.log(p)))

        # adjust belief2 masses to match new_risk while preserving relative shapes
        h = int(belief2.size)
        if idx:
            mal_idx = list(idx)
            non_idx = [i for i in range(h) if i not in mal_idx]
            mass_mal = float(np.sum(belief2[mal_idx]))
            mass_non = 1.0 - mass_mal
            target_mal = float(np.clip(new_risk, 0.0, 1.0))
            target_non = 1.0 - target_mal

            if mass_mal > 0:
                belief2[mal_idx] = belief2[mal_idx] * (target_mal / mass_mal)
            if mass_non > 0 and non_idx:
                belief2[non_idx] = belief2[non_idx] * (target_non / mass_non)

        # final normalization and entropy floor enforcement
        s2 = float(belief2.sum())
        if s2 > 0:
            belief2 = belief2 / s2
        else:
            belief2 = np.ones((self._h,), dtype=np.float64) / float(self._h)

        # apply entropy floor by mixing with uniform if below threshold
        cur_ent = entropy(belief2)
        if cur_ent < float(self._entropy_floor):
            uniform = np.ones((self._h,), dtype=np.float64) / float(self._h)
            # simple incremental mixing to reach entropy floor
            beta = 0.01
            for _ in range(40):
                mixed = (1.0 - beta) * belief2 + beta * uniform
                if entropy(mixed) >= float(self._entropy_floor) or beta >= 0.99:
                    belief2 = mixed
                    break
                beta = min(0.99, beta * 1.6)

        top_state = int(np.argmax(belief2)) if belief2.size else 0
        conf = float(np.max(belief2)) if belief2.size else 0.0

        return FilterResult(
            belief=belief2.astype(np.float64).tolist(),
            history=history2,
            risk_score=float(np.clip(new_risk, 0.0, 1.0)),
            top_intent_state=top_state,
            confidence=conf,
            lstm_prob=lstm_prob,
            micro_prob=micro_prob,
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "num_hidden_states": int(self._h),
            "malicious_states": list(self._artifacts.malicious_states),
            "lstm_max_len": int(self._lstm.max_len),
        }

