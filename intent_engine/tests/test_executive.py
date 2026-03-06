from intent_engine import app as app_mod
from intent_engine.services.math_utils import security_health_score


def test_metrics_executive_json():
    data = app_mod.metrics_executive(x_api_key=None, x_user_role="ROLE_EXECUTIVE")
    assert isinstance(data, dict)
    assert "security_health_score" in data
    assert "security_health_components" in data


def test_metrics_executive_export():
    resp = app_mod.metrics_executive(export="board", x_api_key=None, x_user_role="ROLE_EXECUTIVE")
    # should be a Response object with pdf media type
    from fastapi.responses import Response
    assert isinstance(resp, Response)
    assert resp.media_type == "application/pdf"


def test_security_health_score_good_health():
    """Test when all signals are favorable (low risk)."""
    score = security_health_score(
        session_risks=[0.1, 0.15, 0.08],  # low session risk
        critical_sessions=0,
        total_sessions=100,
        attack_velocity=1.0,  # low velocity
        entropy_avg=0.5,
        entropy_max=5.0,
        drift=0.1,  # low drift
    )
    # Should be high (80+)
    assert 75 <= score <= 100, f"expected ~85, got {score}"


def test_security_health_score_elevated_risk():
    """Test when some signals are concerning."""
    score = security_health_score(
        session_risks=[0.35, 0.4, 0.32],  # moderate session risk
        critical_sessions=3,
        total_sessions=100,
        attack_velocity=5.0,  # moderate velocity
        entropy_avg=2.0,
        entropy_max=5.0,
        drift=0.15,
    )
    # Should be moderate (60-75)
    assert 60 <= score <= 80, f"expected ~70, got {score}"


def test_security_health_score_critical_risk():
    """Test when many signals are concerning."""
    score = security_health_score(
        session_risks=[0.7, 0.75, 0.65],  # high session risk
        critical_sessions=10,
        total_sessions=50,
        attack_velocity=15.0,  # high velocity
        entropy_avg=4.5,
        entropy_max=5.0,
        drift=0.35,
    )
    # Should be low (<60)
    assert 0 <= score < 60, f"expected ~40, got {score}"


def test_security_health_score_bounds():
    """Test that score is always between 0 and 100."""
    # Extreme case
    score = security_health_score(
        session_risks=[1.0, 1.0],
        critical_sessions=1000,
        total_sessions=10,
        attack_velocity=100.0,
        entropy_avg=100.0,
        entropy_max=1.0,
        drift=10.0,
    )
    assert 0 <= score <= 100, f"score out of bounds: {score}"

    # Empty case
    score = security_health_score(
        session_risks=[],
        critical_sessions=0,
        total_sessions=0,
        attack_velocity=0.0,
        entropy_avg=0.0,
        entropy_max=1.0,
        drift=0.0,
    )
    assert 0 <= score <= 100, f"score out of bounds: {score}"
