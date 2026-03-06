def test_evidence_score_composition():
    """Test that evidence score incorporates multiple signals."""
    from intent_engine.services.bayesian_filter import BayesianFilter
    from intent_engine.services.markov import load_markov_artifacts
    from intent_engine.services.lstm_emission import load_lstm
    import os
    from pathlib import Path
    
    model_dir = Path(os.path.join(os.path.dirname(__file__), "..", "models"))
    
    artifacts = load_markov_artifacts(model_dir / "markov.pkl")
    lstm = load_lstm(model_dir / "lstm.h5", max_len=int(artifacts.lstm_max_len))
    
    bayes = BayesianFilter(
        artifacts, lstm, max_history_len=50,
        alpha=0.4, decay_lambda=0.03, temperature=1.0, calibration_temperature=1.5
    )
    
    # Test 1: rare transition should score higher
    history = [0, 1, 2]
    state = 10  # some state
    
    # Low LSTM prob + rare features should yield higher evidence
    micro_features = {"resp_time_ms": 2000.0}  # anomalous latency
    
    score = bayes._compute_evidence_score(state, history, 0.05, micro_features, 0.1, 0.1)
    assert 0.0 <= score <= 1.0, f"Evidence score out of range: {score}"
    assert score > 0.2, f"Expected significant evidence score, got {score}"
    
    # Test 2: typical sequence should score lower
    typical_features = {"resp_time_ms": 100.0}
    score2 = bayes._compute_evidence_score(state, history, 0.7, typical_features, 0.8, 0.1)
    assert 0.0 <= score2 <= 1.0
    # This should be lower than the anomalous case
    assert score2 < score, f"Typical sequence {score2} should score lower than anomalous {score}"
    
    print("Evidence composition: OK (anomalous: {:.3f}, typical: {:.3f})".format(score, score2))


if __name__ == "__main__":
    test_evidence_score_composition()
