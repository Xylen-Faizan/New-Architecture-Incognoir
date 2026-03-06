from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class StateEncoder:
    state_to_id: Dict[str, int]
    num_states: int

    def encode(self, req_path: str, req_method: str, resp_code: int) -> int:
        raw = f"{req_path}|{req_method}|{resp_code}"
        val = self.state_to_id.get(raw)
        if val is None:
            return 0
        return int(val)


def load_encoder(path: Path) -> StateEncoder:
    with path.open("rb") as f:
        obj = pickle.load(f)

    state_to_id = {str(k): int(v) for k, v in obj["state_to_id"].items()}
    num_states = int(obj["num_states"])
    return StateEncoder(state_to_id=state_to_id, num_states=num_states)

