from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class TransitionModel:
    a_counts: np.ndarray

    @property
    def num_hidden_states(self) -> int:
        return int(self.a_counts.shape[0])

    def transition_matrix(self) -> np.ndarray:
        counts = np.asarray(self.a_counts, dtype=np.float64)
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
        return counts / row_sums


@dataclass(frozen=True)
class EmissionConfig:
    beta_params: Tuple[Tuple[float, float], ...]
    per_step_epsilon: float

    def emission_vector(self, p_seq: float, micro_p: float, num_hidden_states: int) -> np.ndarray:
        eps = float(self.per_step_epsilon)
        p = float(np.clip(p_seq, eps, 1.0 - eps))
        m = float(np.clip(micro_p, eps, 1.0 - eps))

        params = self.beta_params
        if len(params) != num_hidden_states:
            params = tuple(list(params)[:num_hidden_states] + [(2.0, 2.0)] * max(num_hidden_states - len(params), 0))

        from .math_utils import beta_logpdf

        log_emissions = np.zeros((num_hidden_states,), dtype=np.float64)
        for z in range(num_hidden_states):
            a, b = params[z]
            log_emissions[z] = beta_logpdf(p, float(a), float(b)) + beta_logpdf(m, float(a), float(b))

        log_emissions = log_emissions - float(np.max(log_emissions))
        emissions = np.exp(log_emissions)
        emissions = np.clip(emissions, eps, np.inf)
        return emissions


@dataclass(frozen=True)
class MicroModel:
    feature_means: dict
    feature_stds: dict
    ll_mean: float
    ll_std: float

    def log_likelihood(self, micro_features: dict) -> float:
        ll = 0.0
        for col, mean in self.feature_means.items():
            if col not in micro_features:
                continue
            std = float(self.feature_stds.get(col, 0.0))
            if std <= 0.0:
                continue
            try:
                val = float(micro_features[col])
            except Exception:
                continue
            ll += -0.5 * np.log(2.0 * np.pi * (std * std)) - ((val - float(mean)) ** 2) / (2.0 * (std * std))
        return float(ll)

    def probability(self, micro_features: dict) -> float:
        from .math_utils import sigmoid

        ll = self.log_likelihood(micro_features)
        denom = float(self.ll_std if self.ll_std > 1e-9 else 1.0)
        z = (ll - float(self.ll_mean)) / denom
        return float(sigmoid(float(z)))


@dataclass(frozen=True)
class MarkovArtifacts:
    transition_model: TransitionModel
    emission_config: EmissionConfig
    micro_model: MicroModel
    malicious_states: List[int]
    lstm_max_len: int


def load_markov_artifacts(path: Path) -> MarkovArtifacts:
    with path.open("rb") as f:
        obj = pickle.load(f)

    a_counts = np.asarray(obj["hmm_a_counts"], dtype=np.float64)
    beta_params = tuple((float(a), float(b)) for a, b in obj["hmm_emission_beta_params"])
    per_step_eps = float(obj.get("hmm_per_step_epsilon", 1e-12))
    malicious_states = [int(x) for x in obj.get("hmm_malicious_states", [3, 4, 5])]

    micro_feature_means = {str(k): float(v) for k, v in obj["micro_feature_means"].items()}
    micro_feature_stds = {str(k): float(v) for k, v in obj["micro_feature_stds"].items()}
    micro_ll_mean = float(obj.get("micro_ll_mean", 0.0))
    micro_ll_std = float(obj.get("micro_ll_std", 1.0))

    lstm_max_len = int(obj.get("lstm_max_len", 64))

    return MarkovArtifacts(
        transition_model=TransitionModel(a_counts=a_counts),
        emission_config=EmissionConfig(beta_params=beta_params, per_step_epsilon=per_step_eps),
        micro_model=MicroModel(
            feature_means=micro_feature_means,
            feature_stds=micro_feature_stds,
            ll_mean=micro_ll_mean,
            ll_std=micro_ll_std,
        ),
        malicious_states=malicious_states,
        lstm_max_len=lstm_max_len,
    )
