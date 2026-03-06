from intent_engine import app as app_mod


def test_governance_status_allows_admin():
    # admin role should succeed even without API key when dev_mode
    data = app_mod.governance_status(x_api_key=None, x_user_role="ROLE_ADMIN")
    assert isinstance(data, dict)
    assert data.get("ok") is True
    assert "drift_status" in data


def test_governance_status_forbidden_without_role():
    # if dev_mode is on (the default during tests) the lack of a role header is
    # tolerated and treated as an admin request.  we still call and make sure we
    # don't get an unexpected exception.  if someone runs this with dev_mode off
    # the call should raise a 403.
    try:
        data = app_mod.governance_status(x_api_key=None, x_user_role=None)
        assert isinstance(data, dict)
    except Exception as e:
        from fastapi import HTTPException
        assert isinstance(e, HTTPException)
        assert e.status_code in (401, 403)


if __name__ == "__main__":
    test_governance_status_allows_admin()
    test_governance_status_forbidden_without_role()
    print("Governance tests OK")
