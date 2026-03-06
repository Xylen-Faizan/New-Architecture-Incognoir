import os
import tempfile

from intent_engine.services.rbac import APIKeyStore


def test_apikey_store_add_get_delete():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "keys.json")
        store = APIKeyStore(path)
        # initially empty
        assert store.list_keys() == {}

        # add key
        k = store.add_key("ROLE_EXECUTIVE")
        assert isinstance(k, str) and len(k) > 0

        # get role
        role = store.get_role(k)
        assert role == "ROLE_EXECUTIVE"

        # list contains mapping
        l = store.list_keys()
        assert k in l and l[k] == "ROLE_EXECUTIVE"

        # delete
        ok = store.delete_key(k)
        assert ok is True
        assert store.get_role(k) is None
