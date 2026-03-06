import numpy as np

from intent_engine import app as app_mod
from intent_engine.app import _session_key, session_feedback
from intent_engine.config import settings


def test_false_positive_feedback_applies():
    env = "default"
    sid = "sess-fb-1"
    key = _session_key(env, sid)
    # create a session with strong belief in state 1
    data = {
        "environment_id": env,
        "subject": {},
        "belief": [0.1, 0.9],
        "history": [2],
        "events": [],
        "updated_at_unix_ms": 0,
    }
    app_mod.cache.set_json(key, data, ttl_seconds=3600)

    fb = app_mod.FeedbackRequest(action="false_positive", note="1")
    resp = session_feedback(sid, fb, environment_id=env, x_api_key=None)
    assert resp.get("ok") is True

    new = app_mod.cache.get_json(key)
    assert new is not None
    b = np.asarray(new.get("belief", []), dtype=np.float64)
    # state 1 should have been reduced
    assert b[1] < 0.9
    assert abs(b.sum() - 1.0) < 1e-6
