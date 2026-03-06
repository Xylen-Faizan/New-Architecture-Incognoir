import math
import numpy as np

from intent_engine.services.bayesian_filter import BayesianFilter, FilterState


class DummyTransitionModel:
    def __init__(self, mat):
        self._mat = np.asarray(mat, dtype=np.float64)

    def transition_matrix(self):
        return self._mat


class DummyMicroModel:
    def probability(self, micro_features):
        return float(micro_features.get("score", 0.5))


class DummyEmissionConfig:
    def emission_vector(self, p_seq, micro_p, num_hidden_states):
        # simple emission: favor state 1 if micro_p>0.6
        if num_hidden_states == 2:
            if micro_p > 0.6 or p_seq > 0.6:
                return [0.1, 0.9]
            return [0.6, 0.4]
        return [1.0 / num_hidden_states] * num_hidden_states


class DummyArtifacts:
    def __init__(self):
        self.transition_model = DummyTransitionModel([[0.9, 0.1], [0.1, 0.9]])
        self.micro_model = DummyMicroModel()
        self.emission_config = DummyEmissionConfig()
        self.malicious_states = (1,)


class DummyLSTM:
    def __init__(self):
        self.max_len = 8

    def next_token_probability(self, history, token):
        return 0.5


def test_bayesian_filter_entropy_floor_and_smoothing():
    artifacts = DummyArtifacts()
    lstm = DummyLSTM()
    bf = BayesianFilter(artifacts=artifacts, lstm=lstm, max_history_len=8, alpha=0.1, temperature=0.9, entropy_floor=0.02)

    # start with uniform belief
    prev = FilterState(belief=[0.5, 0.5], history=[])

    # event that strongly indicates malicious
    res = bf.update(state_token=2, micro_features={"score": 0.95}, prev=prev)
    assert isinstance(res.risk_score, float)
    # should not saturate to 1.0 immediately
    assert res.risk_score < 0.99
    # entropy should be at least the floor
    # compute entropy
    p = np.asarray(res.belief, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    ent = float(-np.sum(p * np.log(p)))
    assert ent >= 0.02

    # apply another similar event and ensure risk increases incrementally
    res2 = bf.update(state_token=2, micro_features={"score": 0.95}, prev=FilterState(belief=res.belief, history=res.history))
    assert res2.risk_score >= res.risk_score
    assert res2.risk_score < 1.0
