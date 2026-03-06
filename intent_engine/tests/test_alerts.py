import numpy as np

from intent_engine.services.alert_layer import AlertDecisionLayer, AlertQueue, AlertPayload, now_iso


def test_alert_decision_layer():
    """Test alert threshold logic."""
    layer = AlertDecisionLayer(suspicious_threshold=0.5, high_threshold=0.75, critical_threshold=0.85)

    # No alert below threshold
    assert layer.get_alert_level(0.4) is None
    assert not layer.should_alert(0.4)

    # Alert at suspicious
    assert layer.get_alert_level(0.55) == "suspicious"
    assert layer.should_alert(0.55)  # first crossing

    # Alert at high
    assert layer.get_alert_level(0.78) == "high"

    # Alert at critical
    assert layer.get_alert_level(0.87) == "critical"

    # Delta detection: no delta
    assert not layer.should_alert(0.52, prev_risk_score=0.51, time_delta_ms=5000)

    # Delta detection: large delta within window
    assert layer.should_alert(0.65, prev_risk_score=0.50, time_delta_ms=10000)

    print("Alert decision layer: OK")


def test_alert_queue():
    """Test alert queue storage."""
    q = AlertQueue(maxlen=100)
    assert len(q.get_recent(10)) == 0

    # Enqueue alerts
    a1 = AlertPayload(
        timestamp=now_iso(),
        environment_id="prod",
        session_id="sess-1",
        risk_score=0.87,
        risk_band="critical",
        alert_level="critical",
        top_intent_state=4,
        confidence=0.91,
        entropy=0.15,
        rare_transitions_count=2,
        microbehavior_z_max=3.2,
        reason="Test alert",
    )
    q.enqueue(a1)
    assert len(q.get_recent(10)) == 1
    assert q.get_by_session("sess-1") == a1

    # Serialize
    data = q.to_dict()
    assert data["total_queued"] == 1
    assert len(data["alerts"]) == 1

    print("Alert queue: OK")


if __name__ == "__main__":
    test_alert_decision_layer()
    test_alert_queue()
    print("ALERT TESTS OK")
