from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bayesian_filter import BayesianFilter
from .explainability import ExplainabilityEngine
from .lstm_emission import LSTMEmission, load_lstm
from .markov import MarkovArtifacts, load_markov_artifacts
from .state_encoder import StateEncoder, load_encoder


@dataclass(frozen=True)
class ModelRegistry:
    encoder: StateEncoder
    markov: MarkovArtifacts
    lstm: LSTMEmission
    bayes_filter: BayesianFilter
    explainability: ExplainabilityEngine


def load_registry(model_dir: Path, max_history_len: int, risk_band_thresholds: tuple[float, float, float], **kwargs) -> ModelRegistry:
    from ..config import settings
    encoder = load_encoder(model_dir / "encoder.pkl")
    markov = load_markov_artifacts(model_dir / "markov.pkl")
    lstm = load_lstm(model_dir / "lstm.h5", max_len=int(markov.lstm_max_len))

    bayes_filter = BayesianFilter(
        artifacts=markov,
        lstm=lstm,
        max_history_len=int(max_history_len),
        alpha=float(settings.risk_update_alpha),
        decay_lambda=float(settings.risk_decay_lambda),
        temperature=float(settings.emission_temperature),
        calibration_temperature=float(settings.calibration_temperature),
        entropy_floor=float(settings.entropy_floor),
        logit_clamp_min=float(settings.logit_clamp_min),
        logit_clamp_max=float(settings.logit_clamp_max),
    )
    explainability = ExplainabilityEngine(
        risk_band_normal_max=float(risk_band_thresholds[0]),
        risk_band_suspicious_max=float(risk_band_thresholds[1]),
        risk_band_high_max=float(risk_band_thresholds[2]),
    )

    return ModelRegistry(
        encoder=encoder,
        markov=markov,
        lstm=lstm,
        bayes_filter=bayes_filter,
        explainability=explainability,
    )

