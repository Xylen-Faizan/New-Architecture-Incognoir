from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from behavioral_intelligence_core import BehavioralCoreConfig, BehavioralIntelligenceCore


def build_models(csv_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    core = BehavioralIntelligenceCore(config=BehavioralCoreConfig())
    df = core.load_and_normalize(str(csv_path))
    df = core.encode_states(df)
    core.fit(df)

    encoder_obj = {
        "state_to_id": dict(core._state_to_id),
        "num_states": int(core.num_states),
    }
    with (out_dir / "encoder.pkl").open("wb") as f:
        pickle.dump(encoder_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    markov_obj = {
        "hmm_a_counts": core._hmm_a_counts,
        "hmm_a_baseline": core._hmm_a_baseline,
        "hmm_emission_beta_params": list(core.config.hmm_emission_beta_params),
        "hmm_per_step_epsilon": float(core.config.hmm_per_step_epsilon),
        "hmm_malicious_states": list(core.config.hmm_malicious_states),
        "micro_feature_means": dict(core._micro_means),
        "micro_feature_stds": dict(core._micro_stds),
        "micro_ll_mean": float(core._micro_ll_mean),
        "micro_ll_std": float(core._micro_ll_std),
        "lstm_max_len": int(core._lstm_max_len),
    }
    with (out_dir / "markov.pkl").open("wb") as f:
        pickle.dump(markov_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    if core._lstm_model is None:
        raise RuntimeError("LSTM model was not trained; cannot export lstm.h5")
    core._lstm_model.save(str(out_dir / "lstm.h5"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="user_sessions_final.csv")
    parser.add_argument("--out", default="intent_engine/models")
    args = parser.parse_args()

    build_models(csv_path=Path(args.csv), out_dir=Path(args.out))


if __name__ == "__main__":
    main()

