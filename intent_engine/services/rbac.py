import json
import os
import threading
import uuid
from typing import Optional, Dict


class APIKeyStore:
    """Simple file-backed API key -> role store for demo/RBAC.

    Usage: instantiate with optional `path`. Methods are thread-safe.
    """

    def __init__(self, path: Optional[str] = None):
        base = os.path.dirname(__file__)
        data_dir = os.path.join(base, "..", "data")
        data_dir = os.path.abspath(data_dir)
        os.makedirs(data_dir, exist_ok=True)
        self._path = path or os.path.join(data_dir, "api_keys.json")
        self._lock = threading.RLock()
        self._keys: Dict[str, str] = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self._path):
                with open(self._path, "r", encoding="utf-8") as f:
                    self._keys = json.load(f)
            else:
                self._keys = {}
        except Exception:
            self._keys = {}

    def _save(self):
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._keys, f)

    def add_key(self, role: str) -> str:
        with self._lock:
            k = uuid.uuid4().hex
            self._keys[k] = role
            self._save()
            return k

    def get_role(self, key: str) -> Optional[str]:
        if not key:
            return None
        with self._lock:
            return self._keys.get(str(key))

    def delete_key(self, key: str) -> bool:
        with self._lock:
            if key in self._keys:
                del self._keys[key]
                self._save()
                return True
            return False

    def list_keys(self):
        with self._lock:
            return dict(self._keys)
