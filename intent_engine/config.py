from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path(os.environ.get("INTENT_ENGINE_MODEL_DIR", "intent_engine/models"))

    dev_mode: bool = os.environ.get("INTENT_ENGINE_DEV_MODE", "1").strip() in ("1", "true", "True")
    api_key: str = os.environ.get("INTENT_ENGINE_API_KEY", "")

    redis_url: str = os.environ.get("INTENT_ENGINE_REDIS_URL", "")
    session_ttl_seconds: int = int(os.environ.get("INTENT_ENGINE_SESSION_TTL_SECONDS", "1800"))
    max_session_events: int = int(os.environ.get("INTENT_ENGINE_MAX_SESSION_EVENTS", "512"))

    kafka_bootstrap_servers: str = os.environ.get("INTENT_ENGINE_KAFKA_BOOTSTRAP_SERVERS", "")
    kafka_topic_risk_events: str = os.environ.get("INTENT_ENGINE_KAFKA_TOPIC_RISK_EVENTS", "risk_events")
    cors_origins: Tuple[str, ...] = tuple(
        x.strip()
        for x in os.environ.get("INTENT_ENGINE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
        if x.strip()
    )
    demo_heuristics_enabled: bool = os.environ.get("INTENT_ENGINE_DEMO_HEURISTICS", "0").strip() in ("1", "true", "True")

    pseudonymization_key_id: str = os.environ.get("INTENT_ENGINE_PSEUDONYMIZATION_KEY_ID", "v1")
    pseudonymization_secret: str = os.environ.get("INTENT_ENGINE_PSEUDONYMIZATION_SECRET", "")

    hmm_malicious_states: Tuple[int, ...] = tuple(
        int(x) for x in os.environ.get("INTENT_ENGINE_HMM_MALICIOUS_STATES", "3,4,5").split(",") if x.strip()
    )

    risk_band_normal_max: float = float(os.environ.get("INTENT_ENGINE_RISK_BAND_NORMAL_MAX", "0.3"))
    risk_band_suspicious_max: float = float(os.environ.get("INTENT_ENGINE_RISK_BAND_SUSPICIOUS_MAX", "0.6"))
    risk_band_high_max: float = float(os.environ.get("INTENT_ENGINE_RISK_BAND_HIGH_MAX", "0.8"))

    max_history_len: int = int(os.environ.get("INTENT_ENGINE_MAX_HISTORY_LEN", "64"))
    # Calibration / smoothing parameters
    risk_update_alpha: float = float(os.environ.get("INTENT_ENGINE_RISK_UPDATE_ALPHA", "0.4"))
    risk_decay_lambda: float = float(os.environ.get("INTENT_ENGINE_RISK_DECAY_LAMBDA", "0.03"))
    emission_temperature: float = float(os.environ.get("INTENT_ENGINE_EMISSION_TEMPERATURE", "1.0"))
    calibration_temperature: float = float(os.environ.get("INTENT_ENGINE_CALIBRATION_TEMPERATURE", "1.5"))
    entropy_floor: float = float(os.environ.get("INTENT_ENGINE_ENTROPY_FLOOR", "0.02"))
    logit_clamp_min: float = float(os.environ.get("INTENT_ENGINE_LOGIT_CLAMP_MIN", "-5.0"))
    logit_clamp_max: float = float(os.environ.get("INTENT_ENGINE_LOGIT_CLAMP_MAX", "5.0"))
    risk_decay_factor: float = float(os.environ.get("INTENT_ENGINE_RISK_DECAY_FACTOR", "0.95"))
    demo_heuristics_weight: float = float(os.environ.get("INTENT_ENGINE_DEMO_HEURISTICS_WEIGHT", "0.25"))


settings = Settings()
