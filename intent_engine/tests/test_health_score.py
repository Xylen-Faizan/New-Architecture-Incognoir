def test_health_score_computation():
    """Test health score combines components correctly."""
    from intent_engine.services.health_score import HealthScoreCompute
    
    hc = HealthScoreCompute(baseline_rps=10.0, baseline_p95_ms=50.0)
    
    # Test 1: All good (high confidence, normal alerts, stable)
    result = hc.compute(
        mean_confidence=0.85,
        alert_count_24h=5,
        baseline_alerts_24h=5,
        feedback_reduction_factor=1.0,
        current_rps=10.0,
        p95_latency_ms=50.0,
    )
    assert 0.0 <= result["score"] <= 100.0
    assert result["score"] > 80.0, f"Expected high score for healthy system, got {result['score']}"
    
    # Test 2: Degraded (low confidence, many alerts, high latency)
    result2 = hc.compute(
        mean_confidence=0.5,
        alert_count_24h=20,
        baseline_alerts_24h=5,
        feedback_reduction_factor=0.8,
        current_rps=5.0,
        p95_latency_ms=150.0,
    )
    assert 0.0 <= result2["score"] <= 100.0
    assert result2["score"] < result["score"], f"Degraded system score {result2['score']} should be lower than healthy {result['score']}"
    
    # Verify components are in range
    for comp_name, comp_value in result["components"].items():
        assert 0.0 <= comp_value <= 100.0, f"{comp_name} out of range: {comp_value}"
    
    print(f"Health score test OK (healthy: {result['score']:.1f}, degraded: {result2['score']:.1f})")


if __name__ == "__main__":
    test_health_score_computation()
