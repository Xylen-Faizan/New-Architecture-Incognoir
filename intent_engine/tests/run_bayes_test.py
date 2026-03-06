from intent_engine.tests.test_bayesian_filter import test_bayesian_filter_entropy_floor_and_smoothing
from intent_engine.tests.test_feedback import test_false_positive_feedback_applies
from intent_engine.tests.test_alerts import test_alert_decision_layer, test_alert_queue
from intent_engine.tests.test_rbac import test_apikey_store_add_get_delete
from intent_engine.tests.test_evidence_composition import test_evidence_score_composition
from intent_engine.tests.test_health_score import test_health_score_computation
from intent_engine.tests.test_governance import test_governance_status_allows_admin, test_governance_status_forbidden_without_role


if __name__ == "__main__":
    try:
        test_bayesian_filter_entropy_floor_and_smoothing()
        test_false_positive_feedback_applies()
        test_alert_decision_layer()
        test_alert_queue()
        test_apikey_store_add_get_delete()
        test_evidence_score_composition()
        test_health_score_computation()
        test_governance_status_allows_admin()
        test_governance_status_forbidden_without_role()
        print("TESTS OK")
    except AssertionError as e:
        print("TEST FAILED", e)
        raise