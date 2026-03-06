from __future__ import annotations

import importlib.util
import math
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


MICRO_COLS_DEFAULT: Tuple[str, ...] = (
    "cursor_speed_wsession",
    "scroll_speed_wsession",
    "typing_speed_wsession",
    "dwell_time_wsession",
    "hesitation_time_wsession",
    "click_accuracy_wsession",
    "movement_pattern_score_wsession",
)


@dataclass(frozen=True)
class BehavioralCoreConfig:
    dirichlet_alpha: float = 0.5
    prior_malicious: float = 0.1
    malicious_evidence_shift_std: float = 2.0
    hybrid_prior_malicious: float = 0.1
    hybrid_malicious_mode: str = "uniform"
    hybrid_like_malicious: float = 1e-6
    hybrid_weight_markov: float = 0.4
    hybrid_weight_lstm: float = 0.4
    hybrid_weight_entropy: float = 0.1
    hybrid_weight_micro: float = 0.1

    lstm_enabled: bool = True
    lstm_embedding_dim: int = 32
    lstm_hidden_dim: int = 64
    lstm_dense_dim: int = 64
    lstm_epochs: int = 5
    lstm_batch_size: int = 256
    lstm_max_len: int = 64
    lstm_perplexity_drift_threshold_z: float = 3.0
    rare_transition_threshold: float = 0.01
    drift_recent_window_days: int = 1
    drift_threshold: float = 0.25
    online_update_max_posterior: float = 0.2
    online_update_require_label_normal: bool = True
    online_update_min_transitions: int = 3
    confidence_bootstrap_samples: int = 30
    confidence_ci_alpha: float = 0.05
    evidence_loglr_clip: float = 12.0
    hmm_num_hidden_states: int = 6
    hmm_dirichlet_alpha: float = 0.5
    hmm_malicious_states: Tuple[int, ...] = (3, 4, 5)
    hmm_emission_beta_params: Tuple[Tuple[float, float], ...] = (
        (6.0, 1.5),
        (5.0, 2.0),
        (4.5, 2.2),
        (1.8, 5.0),
        (1.6, 5.5),
        (1.4, 6.0),
    )
    hmm_offline_update_eta: float = 1.0
    hmm_online_update_eta: float = 0.05
    hmm_online_max_posterior: float = 0.2
    hmm_min_transitions: int = 3
    hmm_per_step_epsilon: float = 1e-12
    hmm_belief_entropy_drift_threshold_z: float = 3.0
    micro_cols: Tuple[str, ...] = MICRO_COLS_DEFAULT


class BehavioralIntelligenceCore:
    def __init__(self, config: Optional[BehavioralCoreConfig] = None) -> None:
        self.config = config or BehavioralCoreConfig()

        self._state_to_id: Dict[str, int] = {}
        self._id_to_state: List[str] = []

        self._transition_counts: Dict[Tuple[int, int], int] = {}
        self._from_counts: Dict[int, int] = {}

        self._entropy_mean: float = 0.0
        self._entropy_std: float = 0.0
        self._micro_means: Dict[str, float] = {}
        self._micro_stds: Dict[str, float] = {}

        self._log_evidence_mean: float = 0.0
        self._log_evidence_std: float = 1.0

        self._entropy_n: int = 0
        self._entropy_sum: float = 0.0
        self._entropy_sumsq: float = 0.0

        self._micro_n: Dict[str, int] = {}
        self._micro_sum: Dict[str, float] = {}
        self._micro_sumsq: Dict[str, float] = {}

        self._evidence_n: int = 0
        self._evidence_sum: float = 0.0
        self._evidence_sumsq: float = 0.0

        self._transition_probs_dense: Optional[np.ndarray] = None

        self._lstm_model: Any = None
        self._lstm_max_len: int = 0
        self._lstm_vocab_size: int = 0
        self._lstm_perplexity_mean: float = 0.0
        self._lstm_perplexity_std: float = 0.0

        self._micro_ll_mean: float = 0.0
        self._micro_ll_std: float = 1.0

        self._hmm_a_counts: Optional[np.ndarray] = None
        self._hmm_a_baseline: Optional[np.ndarray] = None
        self._hmm_belief_entropy_mean: float = 0.0
        self._hmm_belief_entropy_std: float = 1.0
        self._fitted: bool = False

    def load_and_normalize(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df["req_timestamp"] = pd.to_datetime(df["req_timestamp"], errors="coerce", utc=True)
        df = df.sort_values(["session_id", "req_timestamp"], kind="mergesort").reset_index(drop=True)
        return df

    def encode_states(self, df: pd.DataFrame) -> pd.DataFrame:
        state_raw = (
            df["req_path"].astype(str)
            + "|"
            + df["req_method"].astype(str)
            + "|"
            + df["resp_code"].astype(str)
        )

        codes, uniques = pd.factorize(state_raw, sort=False)
        df = df.copy()
        df["state_raw"] = state_raw
        df["state"] = codes.astype(np.int32)

        self._state_to_id = {str(s): int(i) for i, s in enumerate(uniques)}
        self._id_to_state = [str(s) for s in uniques]
        return df

    @property
    def num_states(self) -> int:
        return len(self._id_to_state)

    @property
    def entropy_baseline(self) -> Dict[str, float]:
        return {"mean": float(self._entropy_mean), "std": float(self._entropy_std)}

    @property
    def evidence_baseline(self) -> Dict[str, float]:
        return {"mean": float(self._log_evidence_mean), "std": float(self._log_evidence_std)}

    def fit(self, df: pd.DataFrame) -> None:
        if "state" not in df.columns:
            df = self.encode_states(df)

        normal_df = df[df["label"] == 0]
        self._build_transition_baseline(normal_df)
        self._transition_probs_dense = self._build_transition_matrix_dense()
        self._fit_entropy_baseline(normal_df)
        self._fit_micro_baseline(normal_df)
        self._fit_evidence_baseline(normal_df)
        self._fit_lstm_baseline(normal_df)
        self._fit_micro_ll_baseline(normal_df)
        self._fitted = True
        self._fit_hmm_baseline(normal_df)

    def transition_probability(self, prev_state: int, next_state: int) -> float:
        k = self.num_states
        alpha = float(self.config.dirichlet_alpha)

        denom = float(self._from_counts.get(int(prev_state), 0))
        num = float(self._transition_counts.get((int(prev_state), int(next_state)), 0))
        return float((num + alpha) / (denom + alpha * k)) if k > 0 else 0.0

    def session_log_likelihood(self, states: Sequence[int]) -> float:
        if len(states) < 2:
            return 0.0
        log_likelihood = 0.0
        for prev_state, next_state in zip(states[:-1], states[1:]):
            prob = self.transition_probability(int(prev_state), int(next_state))
            log_likelihood += math.log(max(prob, 1e-300))
        return float(log_likelihood)

    def session_entropy(self, states: Sequence[int]) -> float:
        if len(states) < 2:
            return 0.0

        pairs = np.empty((len(states) - 1, 2), dtype=np.int32)
        pairs[:, 0] = np.asarray(states[:-1], dtype=np.int32)
        pairs[:, 1] = np.asarray(states[1:], dtype=np.int32)

        view = pairs.view(dtype=[("a", np.int32), ("b", np.int32)])
        _, counts = np.unique(view, return_counts=True)
        probs = counts.astype(np.float64) / float(counts.sum())
        return float(-np.sum(probs * np.log(probs)))

    def entropy_log_likelihood(self, entropy_value: float) -> float:
        return float(_gaussian_logpdf(entropy_value, self._entropy_mean, self._entropy_std))

    def micro_log_likelihood(self, session_df: pd.DataFrame) -> float:
        log_like = 0.0
        for col in self.config.micro_cols:
            if col not in session_df.columns:
                continue

            mean = self._micro_means.get(col, 0.0)
            std = self._micro_stds.get(col, 0.0)
            val = float(pd.to_numeric(session_df[col], errors="coerce").mean())
            if np.isnan(val):
                continue
            log_like += _gaussian_logpdf(val, mean, std)
        return float(log_like)

    def hmm_filter_session(
        self,
        session_df: pd.DataFrame,
        update_transitions: bool = False,
        eta: Optional[float] = None,
    ) -> Dict[str, Any]:
        self._require_fitted()
        self._require_hmm_ready()

        states = session_df["state"].to_numpy(dtype=np.int32)
        if len(states) < 1:
            belief = np.ones((self.config.hmm_num_hidden_states,), dtype=np.float64) / float(
                self.config.hmm_num_hidden_states
            )
            return {"belief": belief, "belief_entropy": 0.0, "belief_history": []}

        h = int(self.config.hmm_num_hidden_states)
        belief = np.ones((h,), dtype=np.float64) / float(h)

        token_seq = (states + 1).tolist()
        p_seq = self._lstm_next_token_probabilities(token_seq)

        micro_ll = float(self.micro_log_likelihood(session_df))
        micro_p = self._micro_ll_to_probability(micro_ll)

        belief_history: List[List[float]] = []
        entropy_history: List[float] = []

        prev_belief = belief.copy()

        for t in range(len(token_seq)):
            a = self._hmm_transition_matrix()
            predicted = a.T @ belief
            predicted_sum = float(predicted.sum())
            if predicted_sum > 0.0:
                predicted = predicted / predicted_sum
            else:
                predicted = np.ones((h,), dtype=np.float64) / float(h)

            if t == 0:
                emission_vec = np.ones((h,), dtype=np.float64)
            else:
                p_t = float(p_seq[t])
                emission_vec = self._hmm_emission_vector(p_t, micro_p)

            updated = predicted * emission_vec
            updated_sum = float(updated.sum())
            if updated_sum > 0.0:
                belief = updated / updated_sum
            else:
                belief = np.ones((h,), dtype=np.float64) / float(h)

            ent = float(_entropy(belief))
            belief_history.append(belief.astype(np.float64).tolist())
            entropy_history.append(ent)

            if update_transitions and t > 0:
                step_eta = float(self.config.hmm_online_update_eta if eta is None else eta)
                xi = self._hmm_expected_transition(prev_belief, emission_vec)
                self._hmm_a_counts = self._hmm_a_counts + step_eta * xi

            prev_belief = belief.copy()

        return {
            "belief": belief,
            "belief_entropy": float(entropy_history[-1]) if entropy_history else 0.0,
            "belief_history": belief_history,
            "entropy_history": entropy_history,
            "micro_ll": micro_ll,
            "micro_p": float(micro_p),
        }

    def hmm_posterior_malicious_probability(self, session_df: pd.DataFrame) -> float:
        res = self.hmm_filter_session(session_df, update_transitions=False)
        belief = np.asarray(res["belief"], dtype=np.float64)
        idx = list(self.config.hmm_malicious_states)
        return float(np.sum(belief[idx])) if idx else 0.0

    def lstm_log_likelihood(self, states: Sequence[int]) -> float:
        self._require_fitted()
        if not self.config.lstm_enabled or self._lstm_model is None or self._lstm_max_len <= 2:
            return 0.0

        seq = (np.asarray(states, dtype=np.int32) + 1).tolist()
        if len(seq) < 2:
            return 0.0

        tf = self._require_tensorflow()
        pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

        padded = pad_sequences([seq], maxlen=self._lstm_max_len, padding="pre", truncating="pre", value=0)
        x = padded[:, :-1]
        y = padded[:, 1:]
        mask = (y != 0).astype(np.float32)

        pred = self._lstm_model.predict(x, verbose=0)
        probs = np.take_along_axis(pred, y[..., None], axis=-1).squeeze(-1)
        probs = np.clip(probs, 1e-12, 1.0)

        ll = float(np.sum(np.log(probs) * mask))
        return ll

    def _lstm_log_likelihoods_for_sequences(self, sequences: Sequence[Sequence[int]]) -> np.ndarray:
        self._require_fitted()
        if not self.config.lstm_enabled or self._lstm_model is None or self._lstm_max_len <= 2:
            return np.zeros((len(sequences),), dtype=np.float64)

        if not sequences:
            return np.zeros((0,), dtype=np.float64)

        seqs: List[List[int]] = []
        for seq in sequences:
            s = list(seq)
            if len(s) < 2:
                s = [0, 0]
            seqs.append(s)

        tf = self._require_tensorflow()
        pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

        padded = pad_sequences(seqs, maxlen=self._lstm_max_len, padding="pre", truncating="pre", value=0)
        x = padded[:, :-1]
        y = padded[:, 1:]
        mask = (y != 0).astype(np.float64)

        pred = self._lstm_model.predict(x, batch_size=int(self.config.lstm_batch_size), verbose=0)
        probs = np.take_along_axis(pred, y[..., None], axis=-1).squeeze(-1)
        probs = np.clip(probs, 1e-12, 1.0)

        ll = np.sum(np.log(probs) * mask, axis=1)
        ll = np.asarray(ll, dtype=np.float64)
        ll[np.isnan(ll)] = 0.0
        return ll

    def _lstm_next_token_probabilities(self, token_seq: Sequence[int]) -> np.ndarray:
        self._require_fitted()

        probs_out = np.ones((len(token_seq),), dtype=np.float64)
        if not self.config.lstm_enabled or self._lstm_model is None or self._lstm_max_len <= 2:
            return probs_out
        if len(token_seq) < 2:
            return probs_out

        tf = self._require_tensorflow()
        pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

        padded = pad_sequences([list(token_seq)], maxlen=self._lstm_max_len, padding="pre", truncating="pre", value=0)
        x = padded[:, :-1]
        y = padded[:, 1:]

        pred = self._lstm_model.predict(x, verbose=0)[0]
        y_row = y[0]
        token_probs = np.take_along_axis(pred, y_row[..., None], axis=-1).squeeze(-1)
        token_probs = np.clip(token_probs.astype(np.float64), 1e-12, 1.0)

        eff_len = int(min(len(token_seq), self._lstm_max_len))
        base = int(self._lstm_max_len - eff_len)
        start = int(len(token_seq) - eff_len)

        for i in range(base, self._lstm_max_len - 1):
            tok = int(y_row[i])
            if tok == 0:
                continue
            t_idx = start + (i - base) + 1
            if 0 <= t_idx < len(token_seq):
                probs_out[t_idx] = float(token_probs[i])

        return probs_out

    def _micro_ll_to_probability(self, micro_ll: float) -> float:
        denom = float(self._micro_ll_std if self._micro_ll_std > 1e-9 else 1.0)
        z = (float(micro_ll) - float(self._micro_ll_mean)) / denom
        return float(_sigmoid(z))

    def _hmm_transition_matrix(self) -> np.ndarray:
        self._require_hmm_ready()
        counts = np.asarray(self._hmm_a_counts, dtype=np.float64)
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
        return counts / row_sums

    def _hmm_emission_vector(self, p_seq: float, micro_p: float) -> np.ndarray:
        h = int(self.config.hmm_num_hidden_states)
        eps = float(self.config.hmm_per_step_epsilon)
        p = float(np.clip(p_seq, eps, 1.0 - eps))
        m = float(np.clip(micro_p, eps, 1.0 - eps))

        params = self.config.hmm_emission_beta_params
        if len(params) != h:
            params = tuple(list(params)[:h] + [(2.0, 2.0)] * max(h - len(params), 0))

        log_emissions = np.zeros((h,), dtype=np.float64)
        for z in range(h):
            a, b = params[z]
            log_emissions[z] = _beta_logpdf(p, float(a), float(b)) + _beta_logpdf(m, float(a), float(b))

        log_emissions = log_emissions - float(np.max(log_emissions))
        emissions = np.exp(log_emissions)
        emissions = np.clip(emissions, eps, np.inf)
        return emissions

    def _hmm_expected_transition(self, prev_belief: np.ndarray, emission_vec: np.ndarray) -> np.ndarray:
        a = self._hmm_transition_matrix()
        numer = prev_belief[:, None] * a * emission_vec[None, :]
        s = float(numer.sum())
        if s <= 0.0:
            return np.zeros_like(a)
        return numer / s

    def lstm_perplexity(self, states: Sequence[int]) -> float:
        if len(states) < 2:
            return 1.0
        ll = self.lstm_log_likelihood(states)
        n = max(len(states) - 1, 1)
        return float(np.exp(-ll / float(n)))

    def hybrid_log_likelihood(self, session_df: pd.DataFrame) -> Dict[str, float]:
        self._require_fitted()

        states = session_df["state"].to_numpy(dtype=np.int32)
        markov_ll = float(self.session_log_likelihood(states))
        lstm_ll = float(self.lstm_log_likelihood(states))

        entropy_val = float(self.session_entropy(states))
        log_entropy = float(self.entropy_log_likelihood(entropy_val))
        log_micro = float(self.micro_log_likelihood(session_df))

        total = (
            self.config.hybrid_weight_markov * markov_ll
            + self.config.hybrid_weight_lstm * lstm_ll
            + self.config.hybrid_weight_entropy * log_entropy
            + self.config.hybrid_weight_micro * log_micro
        )

        return {
            "markov_log_likelihood": float(markov_ll),
            "lstm_log_likelihood": float(lstm_ll),
            "entropy": float(entropy_val),
            "entropy_log_likelihood": float(log_entropy),
            "micro_log_likelihood": float(log_micro),
            "hybrid_log_likelihood_normal": float(total),
        }

    def hybrid_posterior_malicious_probability(
        self,
        session_df: pd.DataFrame,
        prior_malicious: Optional[float] = None,
        like_malicious: Optional[float] = None,
    ) -> float:
        self._require_fitted()

        prior = float(self.config.hybrid_prior_malicious if prior_malicious is None else prior_malicious)
        prior = min(max(prior, 1e-9), 1.0 - 1e-9)

        evidence = self.hybrid_log_likelihood(session_df)
        log_like_normal = float(evidence["hybrid_log_likelihood_normal"])

        states = session_df["state"].to_numpy(dtype=np.int32)
        n_transitions = max(int(len(states) - 1), 1)

        if like_malicious is not None:
            lm = float(max(like_malicious, 1e-300))
            log_like_malicious = math.log(lm)
        elif self.config.hybrid_malicious_mode == "uniform":
            uniform_prob = 1.0 / float(max(self.num_states, 1))
            log_uniform = math.log(max(uniform_prob, 1e-300))
            log_like_malicious = (
                self.config.hybrid_weight_markov * n_transitions * log_uniform
                + self.config.hybrid_weight_lstm * n_transitions * log_uniform
            )
        else:
            lm = float(max(self.config.hybrid_like_malicious, 1e-300))
            log_like_malicious = math.log(lm)

        log_num = log_like_malicious + math.log(prior)
        log_denom = _logsumexp([log_num, log_like_normal + math.log(1.0 - prior)])
        posterior = math.exp(log_num - log_denom)
        return float(min(max(posterior, 0.0), 1.0))

    def compute_hybrid_bayesian_risk(
        self,
        df: pd.DataFrame,
        session_id: str,
        prior_malicious: Optional[float] = None,
        like_malicious: Optional[float] = None,
    ) -> float:
        session = df[df["session_id"] == session_id]
        return self.hybrid_posterior_malicious_probability(
            session, prior_malicious=prior_malicious, like_malicious=like_malicious
        )

    def log_evidence_under_normal(self, session_df: pd.DataFrame) -> Dict[str, float]:
        states = session_df["state"].to_numpy(dtype=np.int32)
        log_trans = self.session_log_likelihood(states)
        entropy_val = self.session_entropy(states)
        log_entropy = self.entropy_log_likelihood(entropy_val)
        log_micro = self.micro_log_likelihood(session_df)
        log_evidence = log_trans + log_entropy + log_micro

        return {
            "log_transition": float(log_trans),
            "entropy": float(entropy_val),
            "log_entropy": float(log_entropy),
            "log_micro": float(log_micro),
            "log_evidence_normal": float(log_evidence),
        }

    def posterior_malicious_probability(
        self,
        session_df: pd.DataFrame,
        prior_malicious: Optional[float] = None,
    ) -> float:
        self._require_fitted()

        prior = float(self.config.prior_malicious if prior_malicious is None else prior_malicious)
        prior = min(max(prior, 1e-9), 1.0 - 1e-9)

        evidence = self.log_evidence_under_normal(session_df)
        log_e = float(evidence["log_evidence_normal"])

        sigma = float(self._log_evidence_std if self._log_evidence_std > 0 else 1.0)
        mu_n = float(self._log_evidence_mean)
        mu_m = float(mu_n - self.config.malicious_evidence_shift_std * sigma)

        log_p_n = float(_gaussian_logpdf(log_e, mu_n, sigma))
        log_p_m = float(_gaussian_logpdf(log_e, mu_m, sigma))
        log_lr = float(log_p_m - log_p_n)
        log_lr = float(np.clip(log_lr, -self.config.evidence_loglr_clip, self.config.evidence_loglr_clip))

        logit_prior = math.log(prior) - math.log(1.0 - prior)
        posterior = _sigmoid(logit_prior + log_lr)
        return float(min(max(posterior, 0.0), 1.0))

    def compute_bayesian_risk(self, df: pd.DataFrame, session_id: str, prior_malicious: Optional[float] = None) -> float:
        session = df[df["session_id"] == session_id]
        return self.posterior_malicious_probability(session, prior_malicious=prior_malicious)

    def explain_session(self, df: pd.DataFrame, session_id: str) -> str:
        self._require_fitted()

        session = df[df["session_id"] == session_id]
        states = session["state"].to_numpy(dtype=np.int32)

        rare_transitions: List[Dict[str, float]] = []
        for prev_state, next_state in zip(states[:-1], states[1:]):
            prob = self.transition_probability(int(prev_state), int(next_state))
            if prob < self.config.rare_transition_threshold:
                rare_transitions.append(
                    {
                        "from": int(prev_state),
                        "to": int(next_state),
                        "probability": float(prob),
                    }
                )

        evidence = self.log_evidence_under_normal(session)
        posterior = self.posterior_malicious_probability(session)
        hybrid_evidence = self.hybrid_log_likelihood(session)
        hybrid_posterior = self.hybrid_posterior_malicious_probability(session)
        lstm_perplexity = self.lstm_perplexity(states)
        hmm_res = self.hmm_filter_session(session, update_transitions=False)
        hmm_belief = np.asarray(hmm_res["belief"], dtype=np.float64)
        hmm_posterior = float(np.sum(hmm_belief[list(self.config.hmm_malicious_states)]))
        hmm_top_state = int(np.argmax(hmm_belief)) if hmm_belief.size else 0
        conf = self.bootstrap_confidence(df, session_id, n=self.config.confidence_bootstrap_samples)
        drift = self.compute_hybrid_drift(df, window_days=self.config.drift_recent_window_days)
        hmm_drift = self.compute_hmm_drift(df, window_days=self.config.drift_recent_window_days)

        explanation: Dict[str, Any] = {
            "session_id": session_id,
            "posterior_malicious_probability": float(posterior),
            "prior_malicious": float(self.config.prior_malicious),
            "evidence": evidence,
            "hybrid_posterior_malicious_probability": float(hybrid_posterior),
            "hybrid_prior_malicious": float(self.config.hybrid_prior_malicious),
            "hybrid_evidence": hybrid_evidence,
            "lstm_perplexity": float(lstm_perplexity),
            "hmm_posterior_malicious_probability": float(hmm_posterior),
            "hmm_belief": hmm_belief.astype(np.float64).tolist(),
            "hmm_top_state": int(hmm_top_state),
            "baseline": {
                "dirichlet_alpha": float(self.config.dirichlet_alpha),
                "malicious_evidence_shift_std": float(self.config.malicious_evidence_shift_std),
                "entropy_mean": float(self._entropy_mean),
                "entropy_std": float(self._entropy_std),
                "log_evidence_mean": float(self._log_evidence_mean),
                "log_evidence_std": float(self._log_evidence_std),
                "evidence_loglr_clip": float(self.config.evidence_loglr_clip),
                "lstm_max_len": int(self._lstm_max_len),
                "lstm_perplexity_mean": float(self._lstm_perplexity_mean),
                "lstm_perplexity_std": float(self._lstm_perplexity_std),
                "micro_ll_mean": float(self._micro_ll_mean),
                "micro_ll_std": float(self._micro_ll_std),
                "hmm_num_hidden_states": int(self.config.hmm_num_hidden_states),
                "hmm_dirichlet_alpha": float(self.config.hmm_dirichlet_alpha),
                "hmm_malicious_states": list(self.config.hmm_malicious_states),
                "hybrid_weights": {
                    "markov": float(self.config.hybrid_weight_markov),
                    "lstm": float(self.config.hybrid_weight_lstm),
                    "entropy": float(self.config.hybrid_weight_entropy),
                    "micro": float(self.config.hybrid_weight_micro),
                },
                "hybrid_malicious_mode": str(self.config.hybrid_malicious_mode),
                "hybrid_like_malicious": float(self.config.hybrid_like_malicious),
            },
            "rare_transitions": rare_transitions,
            "confidence": conf,
            "drift": drift,
            "hmm_drift": hmm_drift,
        }
        return json.dumps(explanation, indent=2)

    def score_all_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()

        rows = []
        for session_id, session in df.groupby("session_id"):
            states = session["state"].to_numpy(dtype=np.int32)
            log_trans = self.session_log_likelihood(states)
            entropy_val = self.session_entropy(states)
            log_entropy = self.entropy_log_likelihood(entropy_val)
            log_micro = self.micro_log_likelihood(session)
            log_evidence = log_trans + log_entropy + log_micro
            posterior = self.posterior_malicious_probability(session)
            rows.append(
                {
                    "session_id": session_id,
                    "posterior_malicious_probability": float(posterior),
                    "log_transition": float(log_trans),
                    "entropy": float(entropy_val),
                    "log_entropy": float(log_entropy),
                    "log_micro": float(log_micro),
                    "log_evidence_normal": float(log_evidence),
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values("posterior_malicious_probability", ascending=False)
            .reset_index(drop=True)
        )

    def score_all_sessions_hybrid(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()

        session_ids: List[str] = []
        markov_lls: List[float] = []
        n_transitions_list: List[int] = []
        entropies: List[float] = []
        log_entropy_lls: List[float] = []
        micro_lls: List[float] = []

        sequences: List[List[int]] = []

        for session_id, session in df.groupby("session_id"):
            session_ids.append(str(session_id))
            states = session["state"].to_numpy(dtype=np.int32)
            n_transitions_list.append(max(int(len(states) - 1), 1))

            markov_ll = float(self.session_log_likelihood(states))
            entropy_val = float(self.session_entropy(states))
            log_entropy = float(self.entropy_log_likelihood(entropy_val))
            micro_ll = float(self.micro_log_likelihood(session))

            markov_lls.append(markov_ll)
            entropies.append(entropy_val)
            log_entropy_lls.append(log_entropy)
            micro_lls.append(micro_ll)

            sequences.append((states + 1).tolist())

        lstm_lls = self._lstm_log_likelihoods_for_sequences(sequences)

        w_m = float(self.config.hybrid_weight_markov)
        w_l = float(self.config.hybrid_weight_lstm)
        w_h = float(self.config.hybrid_weight_entropy)
        w_u = float(self.config.hybrid_weight_micro)

        log_like_normal = (
            w_m * np.asarray(markov_lls, dtype=np.float64)
            + w_l * np.asarray(lstm_lls, dtype=np.float64)
            + w_h * np.asarray(log_entropy_lls, dtype=np.float64)
            + w_u * np.asarray(micro_lls, dtype=np.float64)
        )

        prior = float(self.config.hybrid_prior_malicious)
        prior = min(max(prior, 1e-9), 1.0 - 1e-9)

        if self.config.hybrid_malicious_mode == "uniform":
            uniform_prob = 1.0 / float(max(self.num_states, 1))
            log_uniform = math.log(max(uniform_prob, 1e-300))
            n_arr = np.asarray(n_transitions_list, dtype=np.float64)
            log_like_malicious = (w_m * n_arr * log_uniform) + (w_l * n_arr * log_uniform)
        else:
            lm = float(max(self.config.hybrid_like_malicious, 1e-300))
            log_like_malicious = float(math.log(lm))

        log_num = log_like_malicious + math.log(prior)
        log_denom = np.logaddexp(log_num, log_like_normal + math.log(1.0 - prior))
        posteriors = np.exp(log_num - log_denom)

        out = pd.DataFrame(
            {
                "session_id": session_ids,
                "hybrid_posterior_malicious_probability": posteriors.astype(np.float64),
                "markov_log_likelihood": np.asarray(markov_lls, dtype=np.float64),
                "lstm_log_likelihood": np.asarray(lstm_lls, dtype=np.float64),
                "entropy": np.asarray(entropies, dtype=np.float64),
                "entropy_log_likelihood": np.asarray(log_entropy_lls, dtype=np.float64),
                "micro_log_likelihood": np.asarray(micro_lls, dtype=np.float64),
                "hybrid_log_likelihood_normal": log_like_normal.astype(np.float64),
            }
        )

        return out.sort_values("hybrid_posterior_malicious_probability", ascending=False).reset_index(drop=True)

    def compute_drift_score(self, df: pd.DataFrame, window_days: int = 1) -> Dict[str, float]:
        self._require_fitted()
        if self._transition_probs_dense is None:
            return {"drift_score": 0.0}

        if "req_timestamp" not in df.columns:
            return {"drift_score": 0.0}

        max_ts = df["req_timestamp"].max()
        if pd.isna(max_ts):
            return {"drift_score": 0.0}

        recent_df = df[df["req_timestamp"] > max_ts - pd.Timedelta(days=int(window_days))]
        if recent_df.empty:
            return {"drift_score": 0.0}

        k = self.num_states
        if k == 0:
            return {"drift_score": 0.0}

        alpha = float(self.config.dirichlet_alpha)
        counts = np.zeros((k, k), dtype=np.float64)
        for _, group in recent_df.groupby("session_id"):
            states = group["state"].to_numpy(dtype=np.int32)
            if len(states) < 2:
                continue
            for prev_state, next_state in zip(states[:-1], states[1:]):
                counts[int(prev_state), int(next_state)] += 1.0

        row_sums = counts.sum(axis=1)
        mask = row_sums > 0.0
        if not np.any(mask):
            return {"drift_score": 0.0}

        denom = row_sums[mask][:, None] + alpha * k
        p = (counts[mask] + alpha) / denom
        q = self._transition_probs_dense[mask]

        eps = 1e-12
        kl_rows = np.sum(p * (np.log(np.clip(p, eps, 1.0)) - np.log(np.clip(q, eps, 1.0))), axis=1)
        weights = row_sums[mask] / float(row_sums[mask].sum())
        drift_score = float(np.sum(weights * kl_rows))
        return {"drift_score": float(drift_score)}

    def compute_hybrid_drift(self, df: pd.DataFrame, window_days: int = 1) -> Dict[str, float]:
        self._require_fitted()

        drift_markov = self.compute_drift_score(df, window_days=window_days)
        drift_score = float(drift_markov.get("drift_score", 0.0))

        out: Dict[str, float] = {"markov_kl": drift_score}

        if "req_timestamp" in df.columns:
            max_ts = df["req_timestamp"].max()
            if not pd.isna(max_ts):
                recent_df = df[df["req_timestamp"] > max_ts - pd.Timedelta(days=int(window_days))]
            else:
                recent_df = df.iloc[0:0]
        else:
            recent_df = df.iloc[0:0]

        if self.config.lstm_enabled and self._lstm_model is not None and not recent_df.empty:
            sequences: List[List[int]] = []
            for _, group in recent_df.groupby("session_id"):
                seq = (group["state"].to_numpy(dtype=np.int32) + 1).tolist()
                if len(seq) >= 2:
                    sequences.append(seq)

            if sequences:
                tf = self._require_tensorflow()
                pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
                padded = pad_sequences(
                    sequences,
                    maxlen=self._lstm_max_len,
                    padding="pre",
                    truncating="pre",
                    value=0,
                )
                x = padded[:, :-1]
                y = padded[:, 1:]
                mask = (y != 0).astype(np.float64)

                pred = self._lstm_model.predict(x, batch_size=int(self.config.lstm_batch_size), verbose=0)
                probs = np.take_along_axis(pred, y[..., None], axis=-1).squeeze(-1)
                probs = np.clip(probs, 1e-12, 1.0)
                ll = np.sum(np.log(probs) * mask, axis=1)
                n_tokens = np.maximum(mask.sum(axis=1), 1.0)
                perplexities = np.exp(-ll / n_tokens)

                recent_mean = float(np.mean(perplexities))
                out["lstm_perplexity_mean_recent"] = recent_mean
                if self._lstm_perplexity_std > 1e-9:
                    out["lstm_perplexity_z"] = float(
                        (recent_mean - self._lstm_perplexity_mean) / self._lstm_perplexity_std
                    )
                else:
                    out["lstm_perplexity_z"] = 0.0

        if not recent_df.empty:
            entropies: List[float] = []
            for _, group in recent_df.groupby("session_id"):
                entropies.append(float(self.session_entropy(group["state"].to_numpy(dtype=np.int32))))
            if entropies:
                recent_entropy_mean = float(np.mean(entropies))
                out["entropy_mean_recent"] = recent_entropy_mean
                if self._entropy_std > 1e-9:
                    out["entropy_mean_z"] = float((recent_entropy_mean - self._entropy_mean) / self._entropy_std)
                else:
                    out["entropy_mean_z"] = 0.0

        return out

    def compute_hmm_drift(self, df: pd.DataFrame, window_days: int = 1) -> Dict[str, float]:
        self._require_fitted()
        self._require_hmm_ready()

        if "req_timestamp" in df.columns:
            max_ts = df["req_timestamp"].max()
            if not pd.isna(max_ts):
                recent_df = df[df["req_timestamp"] > max_ts - pd.Timedelta(days=int(window_days))]
            else:
                recent_df = df.iloc[0:0]
        else:
            recent_df = df.iloc[0:0]

        if recent_df.empty:
            return {"hmm_transition_kl": 0.0, "hmm_belief_entropy_z": 0.0}

        h = int(self.config.hmm_num_hidden_states)
        counts = np.ones((h, h), dtype=np.float64) * float(self.config.hmm_dirichlet_alpha)
        final_entropies: List[float] = []

        a_ref = self._hmm_a_baseline if self._hmm_a_baseline is not None else self._hmm_transition_matrix()

        for _, group in recent_df.groupby("session_id"):
            states = group["state"].to_numpy(dtype=np.int32)
            if len(states) < 2:
                continue

            token_seq = (states + 1).tolist()
            p_seq = self._lstm_next_token_probabilities(token_seq)
            micro_p = self._micro_ll_to_probability(float(self.micro_log_likelihood(group)))

            belief = np.ones((h,), dtype=np.float64) / float(h)
            prev_belief = belief.copy()

            for t in range(len(token_seq)):
                predicted = a_ref.T @ belief
                s = float(predicted.sum())
                if s > 0.0:
                    predicted = predicted / s
                else:
                    predicted = np.ones((h,), dtype=np.float64) / float(h)

                if t == 0:
                    emission_vec = np.ones((h,), dtype=np.float64)
                else:
                    emission_vec = self._hmm_emission_vector(float(p_seq[t]), float(micro_p))

                updated = predicted * emission_vec
                us = float(updated.sum())
                if us > 0.0:
                    belief = updated / us
                else:
                    belief = np.ones((h,), dtype=np.float64) / float(h)

                if t > 0:
                    numer = prev_belief[:, None] * a_ref * emission_vec[None, :]
                    ns = float(numer.sum())
                    if ns > 0.0:
                        counts += numer / ns

                prev_belief = belief.copy()

            final_entropies.append(float(_entropy(belief)))

        a_recent = counts / np.where(counts.sum(axis=1, keepdims=True) > 0.0, counts.sum(axis=1, keepdims=True), 1.0)
        hmm_transition_kl = float(_kl_divergence(a_recent, a_ref))

        if final_entropies:
            recent_mean = float(np.mean(final_entropies))
        else:
            recent_mean = 0.0

        denom = float(self._hmm_belief_entropy_std if self._hmm_belief_entropy_std > 1e-9 else 1.0)
        z = float((recent_mean - float(self._hmm_belief_entropy_mean)) / denom)

        return {
            "hmm_transition_kl": float(hmm_transition_kl),
            "hmm_belief_entropy_mean_recent": float(recent_mean),
            "hmm_belief_entropy_z": float(z),
        }

    def safe_online_update(self, df_new: pd.DataFrame) -> Dict[str, Any]:
        self._require_fitted()

        drift = self.compute_hybrid_drift(df_new, window_days=self.config.drift_recent_window_days)
        hmm_drift = self.compute_hmm_drift(df_new, window_days=self.config.drift_recent_window_days)
        drift_score = float(drift.get("markov_kl", 0.0))
        if drift_score > self.config.drift_threshold:
            return {"updated": False, "drift": drift, "hmm_drift": hmm_drift, "accepted_sessions": 0}

        perplexity_z = float(drift.get("lstm_perplexity_z", 0.0))
        if abs(perplexity_z) > self.config.lstm_perplexity_drift_threshold_z:
            return {"updated": False, "drift": drift, "hmm_drift": hmm_drift, "accepted_sessions": 0}

        hmm_kl = float(hmm_drift.get("hmm_transition_kl", 0.0))
        if hmm_kl > self.config.drift_threshold:
            return {"updated": False, "drift": drift, "hmm_drift": hmm_drift, "accepted_sessions": 0}

        hmm_entropy_z = float(hmm_drift.get("hmm_belief_entropy_z", 0.0))
        if abs(hmm_entropy_z) > self.config.hmm_belief_entropy_drift_threshold_z:
            return {"updated": False, "drift": drift, "hmm_drift": hmm_drift, "accepted_sessions": 0}

        accepted = 0
        accepted_hmm = 0
        for session_id, session in df_new.groupby("session_id"):
            states = session["state"].to_numpy(dtype=np.int32) if "state" in session.columns else None
            if states is None or len(states) - 1 < self.config.online_update_min_transitions:
                continue

            if self.config.online_update_require_label_normal and "label" in session.columns:
                if int(session["label"].iloc[0]) != 0:
                    continue

            posterior = self.posterior_malicious_probability(session)
            if posterior > self.config.online_update_max_posterior:
                continue

            evidence = self.log_evidence_under_normal(session)
            entropy_val = float(evidence["entropy"])
            self._update_entropy_moments(entropy_val)
            self._update_micro_moments(session)
            self._update_evidence_moments(float(evidence["log_evidence_normal"]))

            for prev_state, next_state in zip(states[:-1], states[1:]):
                key = (int(prev_state), int(next_state))
                self._transition_counts[key] = self._transition_counts.get(key, 0) + 1
                self._from_counts[int(prev_state)] = self._from_counts.get(int(prev_state), 0) + 1

            if len(states) - 1 >= int(self.config.hmm_min_transitions):
                hmm_p = self.hmm_posterior_malicious_probability(session)
                if hmm_p <= float(self.config.hmm_online_max_posterior):
                    self.hmm_filter_session(session, update_transitions=True, eta=float(self.config.hmm_online_update_eta))
                    accepted_hmm += 1

            accepted += 1

        self._refresh_baselines_from_moments()
        self._transition_probs_dense = self._build_transition_matrix_dense()

        return {
            "updated": accepted > 0,
            "drift": drift,
            "hmm_drift": hmm_drift,
            "accepted_sessions": int(accepted),
            "accepted_hmm_sessions": int(accepted_hmm),
        }

    def bootstrap_confidence(self, df: pd.DataFrame, session_id: str, n: int = 20, seed: int = 7) -> Dict[str, float]:
        self._require_fitted()

        session = df[df["session_id"] == session_id]
        states = session["state"].to_numpy(dtype=np.int32)
        if len(states) < 2:
            p = self.posterior_malicious_probability(session)
            return {"std": 0.0, "ci_low": float(p), "ci_high": float(p)}

        rng = np.random.default_rng(int(seed))
        posteriors = np.empty(int(n), dtype=np.float64)
        transitions = np.stack([states[:-1], states[1:]], axis=1)

        for i in range(int(n)):
            idx = rng.integers(0, len(transitions), size=len(transitions))
            sampled_pairs = transitions[idx]

            log_trans = 0.0
            for prev_state, next_state in sampled_pairs:
                prob = self.transition_probability(int(prev_state), int(next_state))
                log_trans += math.log(max(prob, 1e-300))

            entropy_val = _entropy_from_pairs(sampled_pairs)
            log_entropy = self.entropy_log_likelihood(float(entropy_val))

            row_idx = rng.integers(0, len(session), size=len(session))
            sampled_rows = session.iloc[row_idx]
            log_micro = self.micro_log_likelihood(sampled_rows)

            log_evidence = float(log_trans + log_entropy + log_micro)
            sigma = float(self._log_evidence_std if self._log_evidence_std > 0 else 1.0)
            mu_n = float(self._log_evidence_mean)
            mu_m = float(mu_n - self.config.malicious_evidence_shift_std * sigma)

            log_p_n = float(_gaussian_logpdf(log_evidence, mu_n, sigma))
            log_p_m = float(_gaussian_logpdf(log_evidence, mu_m, sigma))
            log_lr = float(log_p_m - log_p_n)
            log_lr = float(np.clip(log_lr, -self.config.evidence_loglr_clip, self.config.evidence_loglr_clip))

            prior = float(self.config.prior_malicious)
            prior = min(max(prior, 1e-9), 1.0 - 1e-9)
            logit_prior = math.log(prior) - math.log(1.0 - prior)
            posteriors[i] = _sigmoid(logit_prior + log_lr)

        std = float(np.std(posteriors))
        alpha = float(self.config.confidence_ci_alpha)
        ci_low = float(np.quantile(posteriors, alpha / 2.0))
        ci_high = float(np.quantile(posteriors, 1.0 - alpha / 2.0))
        return {"std": std, "ci_low": ci_low, "ci_high": ci_high}

    def _build_transition_baseline(self, normal_df: pd.DataFrame) -> None:
        self._transition_counts.clear()
        self._from_counts.clear()

        for _, group in normal_df.groupby("session_id"):
            states = group["state"].to_numpy(dtype=np.int32)
            if len(states) < 2:
                continue
            for prev_state, next_state in zip(states[:-1], states[1:]):
                key = (int(prev_state), int(next_state))
                self._transition_counts[key] = self._transition_counts.get(key, 0) + 1
                self._from_counts[int(prev_state)] = self._from_counts.get(int(prev_state), 0) + 1

    def _fit_entropy_baseline(self, normal_df: pd.DataFrame) -> None:
        entropies: List[float] = []
        self._entropy_n = 0
        self._entropy_sum = 0.0
        self._entropy_sumsq = 0.0
        for _, group in normal_df.groupby("session_id"):
            h = float(self.session_entropy(group["state"].to_numpy(dtype=np.int32)))
            entropies.append(h)
            self._update_entropy_moments(h)

        if not entropies:
            self._entropy_mean = 0.0
            self._entropy_std = 0.0
            return

        self._entropy_mean = float(np.mean(entropies))
        self._entropy_std = float(np.std(entropies, ddof=0))

    def _fit_micro_baseline(self, normal_df: pd.DataFrame) -> None:
        self._micro_means.clear()
        self._micro_stds.clear()
        self._micro_n.clear()
        self._micro_sum.clear()
        self._micro_sumsq.clear()

        per_session_means: Dict[str, List[float]] = {c: [] for c in self.config.micro_cols}
        for _, group in normal_df.groupby("session_id"):
            for col in self.config.micro_cols:
                if col not in group.columns:
                    continue
                val = float(pd.to_numeric(group[col], errors="coerce").mean())
                if np.isnan(val):
                    continue
                per_session_means[col].append(val)
                self._update_micro_moment(col, val)

        for col in self.config.micro_cols:
            values = per_session_means.get(col, [])
            if not values:
                self._micro_means[col] = 0.0
                self._micro_stds[col] = 0.0
                continue

            arr = np.asarray(values, dtype=np.float64)
            self._micro_means[col] = float(np.mean(arr))
            self._micro_stds[col] = float(np.std(arr, ddof=0))

    def _fit_evidence_baseline(self, normal_df: pd.DataFrame) -> None:
        self._evidence_n = 0
        self._evidence_sum = 0.0
        self._evidence_sumsq = 0.0

        values: List[float] = []
        for _, group in normal_df.groupby("session_id"):
            evidence = self.log_evidence_under_normal(group)
            v = float(evidence["log_evidence_normal"])
            values.append(v)
            self._update_evidence_moments(v)

        if not values:
            self._log_evidence_mean = 0.0
            self._log_evidence_std = 1.0
            return

        arr = np.asarray(values, dtype=np.float64)
        self._log_evidence_mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        self._log_evidence_std = std if std > 1e-9 else 1.0

    def _fit_lstm_baseline(self, normal_df: pd.DataFrame) -> None:
        self._lstm_model = None
        self._lstm_max_len = 0
        self._lstm_vocab_size = 0
        self._lstm_perplexity_mean = 0.0
        self._lstm_perplexity_std = 0.0

        if not self.config.lstm_enabled:
            return
        if self.num_states <= 0:
            return
        if importlib.util.find_spec("tensorflow") is None:
            return

        tf = self._require_tensorflow()
        pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

        sequences: List[List[int]] = []
        for _, group in normal_df.groupby("session_id"):
            seq = (group["state"].to_numpy(dtype=np.int32) + 1).tolist()
            if len(seq) >= 2:
                sequences.append(seq)

        if not sequences:
            return

        max_len_observed = max(len(s) for s in sequences)
        max_len = int(min(max_len_observed, int(self.config.lstm_max_len)))
        if max_len < 3:
            return

        vocab_size = int(self.num_states + 1)

        padded = pad_sequences(sequences, maxlen=max_len, padding="pre", truncating="pre", value=0)
        x = padded[:, :-1]
        y = padded[:, 1:]
        sample_weight = (y != 0).astype(np.float32)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    input_dim=vocab_size,
                    output_dim=int(self.config.lstm_embedding_dim),
                    mask_zero=True,
                ),
                tf.keras.layers.LSTM(int(self.config.lstm_hidden_dim), return_sequences=True),
                tf.keras.layers.Dense(int(self.config.lstm_dense_dim), activation="relu"),
                tf.keras.layers.Dense(vocab_size, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[],
        )

        model.fit(
            x,
            y,
            sample_weight=sample_weight,
            epochs=int(self.config.lstm_epochs),
            batch_size=int(self.config.lstm_batch_size),
            verbose=0,
        )

        pred = model.predict(x, batch_size=int(self.config.lstm_batch_size), verbose=0)
        probs = np.take_along_axis(pred, y[..., None], axis=-1).squeeze(-1)
        probs = np.clip(probs, 1e-12, 1.0)
        mask = sample_weight.astype(np.float64)
        ll = np.sum(np.log(probs) * mask, axis=1)
        n_tokens = np.maximum(mask.sum(axis=1), 1.0)
        perplexities = np.exp(-ll / n_tokens)

        self._lstm_model = model
        self._lstm_max_len = max_len
        self._lstm_vocab_size = vocab_size
        self._lstm_perplexity_mean = float(np.mean(perplexities))
        self._lstm_perplexity_std = float(np.std(perplexities, ddof=0))

    def _fit_micro_ll_baseline(self, normal_df: pd.DataFrame) -> None:
        values: List[float] = []
        for _, group in normal_df.groupby("session_id"):
            values.append(float(self.micro_log_likelihood(group)))

        if not values:
            self._micro_ll_mean = 0.0
            self._micro_ll_std = 1.0
            return

        arr = np.asarray(values, dtype=np.float64)
        self._micro_ll_mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        self._micro_ll_std = std if std > 1e-9 else 1.0

    def _fit_hmm_baseline(self, normal_df: pd.DataFrame) -> None:
        h = int(self.config.hmm_num_hidden_states)
        alpha = float(self.config.hmm_dirichlet_alpha)
        self._hmm_a_counts = np.ones((h, h), dtype=np.float64) * alpha

        entropies: List[float] = []
        eta = float(self.config.hmm_offline_update_eta)

        if not self.config.lstm_enabled or self._lstm_model is None or self._lstm_max_len <= 2:
            self._hmm_a_baseline = self._hmm_transition_matrix().copy()
            self._hmm_belief_entropy_mean = 0.0
            self._hmm_belief_entropy_std = 1.0
            return

        tf = self._require_tensorflow()
        pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

        seqs: List[List[int]] = []
        micros: List[float] = []

        for _, group in normal_df.groupby("session_id"):
            states = group["state"].to_numpy(dtype=np.int32)
            if len(states) - 1 < int(self.config.hmm_min_transitions):
                continue

            token_seq = (states + 1).tolist()
            eff_len = int(min(len(token_seq), int(self._lstm_max_len)))
            token_seq = token_seq[-eff_len:]
            if len(token_seq) < 2:
                continue

            seqs.append(token_seq)
            micros.append(self._micro_ll_to_probability(float(self.micro_log_likelihood(group))))

        if not seqs:
            self._hmm_a_baseline = self._hmm_transition_matrix().copy()
            self._hmm_belief_entropy_mean = 0.0
            self._hmm_belief_entropy_std = 1.0
            return

        batch_sessions = int(max(128, min(4096, int(self.config.lstm_batch_size) * 4)))
        for batch_start in range(0, len(seqs), batch_sessions):
            batch_end = min(batch_start + batch_sessions, len(seqs))
            seqs_b = seqs[batch_start:batch_end]
            micros_b = micros[batch_start:batch_end]

            padded = pad_sequences(seqs_b, maxlen=self._lstm_max_len, padding="pre", truncating="pre", value=0)
            x = padded[:, :-1]
            y = padded[:, 1:]

            pred = self._lstm_model.predict(x, batch_size=int(self.config.lstm_batch_size), verbose=0)
            probs_selected = np.take_along_axis(pred, y[..., None], axis=-1).squeeze(-1)
            probs_selected = np.clip(probs_selected.astype(np.float64), 1e-12, 1.0)

            for j, token_seq in enumerate(seqs_b):
                micro_p = float(micros_b[j])
                eff_len = int(min(len(token_seq), int(self._lstm_max_len)))
                base = int(self._lstm_max_len - eff_len)

                p_seq = np.ones((eff_len,), dtype=np.float64)
                p_seq[1:] = probs_selected[j, base : base + eff_len - 1]

                belief = np.ones((h,), dtype=np.float64) / float(h)
                prev_belief = belief.copy()

                for t in range(eff_len):
                    a = self._hmm_transition_matrix()
                    predicted = a.T @ belief
                    s = float(predicted.sum())
                    if s > 0.0:
                        predicted = predicted / s
                    else:
                        predicted = np.ones((h,), dtype=np.float64) / float(h)

                    if t == 0:
                        emission_vec = np.ones((h,), dtype=np.float64)
                    else:
                        emission_vec = self._hmm_emission_vector(float(p_seq[t]), micro_p)

                    updated = predicted * emission_vec
                    us = float(updated.sum())
                    if us > 0.0:
                        belief = updated / us
                    else:
                        belief = np.ones((h,), dtype=np.float64) / float(h)

                    if t > 0:
                        xi = self._hmm_expected_transition(prev_belief, emission_vec)
                        self._hmm_a_counts = self._hmm_a_counts + eta * xi

                    prev_belief = belief.copy()

                entropies.append(float(_entropy(belief)))

        self._hmm_a_baseline = self._hmm_transition_matrix().copy()

        if not entropies:
            self._hmm_belief_entropy_mean = 0.0
            self._hmm_belief_entropy_std = 1.0
            return

        arr = np.asarray(entropies, dtype=np.float64)
        self._hmm_belief_entropy_mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        self._hmm_belief_entropy_std = std if std > 1e-9 else 1.0

    def _build_transition_matrix_dense(self) -> np.ndarray:
        k = self.num_states
        alpha = float(self.config.dirichlet_alpha)
        if k == 0:
            return np.zeros((0, 0), dtype=np.float64)

        counts = np.zeros((k, k), dtype=np.float64)
        for (i, j), c in self._transition_counts.items():
            if 0 <= i < k and 0 <= j < k:
                counts[i, j] = float(c)

        row_sums = counts.sum(axis=1, keepdims=True)
        probs = (counts + alpha) / (row_sums + alpha * k)
        return probs

    def _compute_transition_matrix_dense(self, data: pd.DataFrame) -> np.ndarray:
        k = self.num_states
        alpha = float(self.config.dirichlet_alpha)
        if k == 0:
            return np.zeros((0, 0), dtype=np.float64)

        counts = np.zeros((k, k), dtype=np.float64)
        for _, group in data.groupby("session_id"):
            states = group["state"].to_numpy(dtype=np.int32)
            if len(states) < 2:
                continue
            for prev_state, next_state in zip(states[:-1], states[1:]):
                counts[int(prev_state), int(next_state)] += 1.0

        row_sums = counts.sum(axis=1, keepdims=True)
        probs = (counts + alpha) / (row_sums + alpha * k)
        return probs

    def _update_entropy_moments(self, value: float) -> None:
        self._entropy_n += 1
        self._entropy_sum += float(value)
        self._entropy_sumsq += float(value) * float(value)

    def _update_micro_moment(self, col: str, value: float) -> None:
        self._micro_n[col] = int(self._micro_n.get(col, 0) + 1)
        self._micro_sum[col] = float(self._micro_sum.get(col, 0.0) + float(value))
        self._micro_sumsq[col] = float(self._micro_sumsq.get(col, 0.0) + float(value) * float(value))

    def _update_micro_moments(self, session_df: pd.DataFrame) -> None:
        for col in self.config.micro_cols:
            if col not in session_df.columns:
                continue
            val = float(pd.to_numeric(session_df[col], errors="coerce").mean())
            if np.isnan(val):
                continue
            self._update_micro_moment(col, val)

    def _update_evidence_moments(self, value: float) -> None:
        self._evidence_n += 1
        self._evidence_sum += float(value)
        self._evidence_sumsq += float(value) * float(value)

    def _refresh_baselines_from_moments(self) -> None:
        if self._entropy_n > 0:
            mean = self._entropy_sum / float(self._entropy_n)
            var = max((self._entropy_sumsq / float(self._entropy_n)) - mean * mean, 0.0)
            self._entropy_mean = float(mean)
            self._entropy_std = float(math.sqrt(var))

        for col in self.config.micro_cols:
            n = int(self._micro_n.get(col, 0))
            if n <= 0:
                continue
            mean = float(self._micro_sum.get(col, 0.0)) / float(n)
            var = max((float(self._micro_sumsq.get(col, 0.0)) / float(n)) - mean * mean, 0.0)
            self._micro_means[col] = float(mean)
            self._micro_stds[col] = float(math.sqrt(var))

        if self._evidence_n > 0:
            mean = self._evidence_sum / float(self._evidence_n)
            var = max((self._evidence_sumsq / float(self._evidence_n)) - mean * mean, 0.0)
            self._log_evidence_mean = float(mean)
            std = float(math.sqrt(var))
            self._log_evidence_std = std if std > 1e-9 else 1.0

    def _require_tensorflow(self) -> Any:
        if importlib.util.find_spec("tensorflow") is None:
            raise RuntimeError("TensorFlow is not available in this environment.")
        import tensorflow as tf  # type: ignore

        return tf

    def _require_hmm_ready(self) -> None:
        h = int(self.config.hmm_num_hidden_states)
        if self._hmm_a_counts is None or self._hmm_a_counts.shape != (h, h):
            raise RuntimeError("HMM intent model is not initialized.")

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("BehavioralIntelligenceCore is not fitted yet. Call fit(df) first.")


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logsumexp(log_values: Sequence[float]) -> float:
    if not log_values:
        return -math.inf
    m = max(log_values)
    if m == -math.inf:
        return -math.inf
    return float(m + math.log(sum(math.exp(v - m) for v in log_values)))


def _beta_logpdf(x: float, a: float, b: float) -> float:
    x2 = float(np.clip(x, 1e-12, 1.0 - 1e-12))
    a2 = float(max(a, 1e-9))
    b2 = float(max(b, 1e-9))
    log_b = math.lgamma(a2) + math.lgamma(b2) - math.lgamma(a2 + b2)
    return (a2 - 1.0) * math.log(x2) + (b2 - 1.0) * math.log(1.0 - x2) - log_b


def _gaussian_logpdf(x: float, mean: float, std: float) -> float:
    if std <= 0.0 or math.isnan(std):
        return 0.0
    var = std * std
    return -0.5 * math.log(2.0 * math.pi * var) - ((x - mean) * (x - mean)) / (2.0 * var)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p2 = np.clip(p, eps, 1.0)
    q2 = np.clip(q, eps, 1.0)
    return float(np.sum(p2 * (np.log(p2) - np.log(q2))))


def _entropy_from_pairs(pairs: np.ndarray) -> float:
    if pairs.size == 0:
        return 0.0
    view = pairs.astype(np.int32, copy=False).view(dtype=[("a", np.int32), ("b", np.int32)])
    _, counts = np.unique(view, return_counts=True)
    probs = counts.astype(np.float64) / float(counts.sum())
    return float(-np.sum(probs * np.log(probs)))


def _entropy(p: np.ndarray) -> float:
    p2 = np.asarray(p, dtype=np.float64)
    p2 = p2[p2 > 0.0]
    if p2.size == 0:
        return 0.0
    return float(-np.sum(p2 * np.log(p2)))


def main() -> None:
    core = BehavioralIntelligenceCore()
    df = core.load_and_normalize("user_sessions_final.csv")
    df = core.encode_states(df)

    print("Total events:", len(df))
    print("Unique sessions:", df["session_id"].nunique())
    print("Total unique states:", df["state"].nunique())

    core.fit(df)
    print("Entropy baseline:", core.entropy_baseline)
    print("Evidence baseline:", core.evidence_baseline)
    print(
        "LSTM baseline:",
        {
            "enabled": bool(core.config.lstm_enabled),
            "max_len": int(core._lstm_max_len),
            "perplexity_mean": float(core._lstm_perplexity_mean),
            "perplexity_std": float(core._lstm_perplexity_std),
        },
    )

    scored = core.score_all_sessions_hybrid(df)
    print(scored.head(10).to_string(index=False))

    top_session = str(scored.iloc[0]["session_id"])
    print(core.explain_session(df, top_session))


if __name__ == "__main__":
    main()
