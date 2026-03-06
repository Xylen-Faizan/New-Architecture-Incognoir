from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request


def _post_json(url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()


def _normal_step() -> tuple[str, str, int]:
    paths = ["/login", "/profile", "/dashboard", "/api/data", "/settings", "/checkout"]
    methods = ["GET", "GET", "GET", "POST"]
    codes = [200, 200, 200, 200, 204]
    return random.choice(paths), random.choice(methods), random.choice(codes)


def _attack_step() -> tuple[str, str, int]:
    paths = ["/admin", "/admin/users", "/export", "/internal/config", "/api/data", "/login"]
    methods = ["GET", "POST", "DELETE", "PATCH"]
    codes = [401, 403, 429, 500, 200]
    return random.choice(paths), random.choice(methods), random.choice(codes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--environment", default="demo")
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--rate", type=float, default=8.0)
    parser.add_argument("--minutes", type=float, default=2.0)
    parser.add_argument("--attack_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(int(args.seed))
    infer_url = f"{args.api.rstrip('/')}/infer"

    total_seconds = float(args.minutes) * 60.0
    period = 1.0 / float(max(args.rate, 0.1))
    started = time.time()

    session_ids = [f"demo_session_{i+1}" for i in range(int(max(args.sessions, 1)))]

    i = 0
    while True:
        now = time.time()
        if now - started >= total_seconds:
            break

        sid = random.choice(session_ids)
        is_attack = random.random() < float(args.attack_ratio)
        if is_attack:
            req_path, req_method, resp_code = _attack_step()
            typing_speed = random.uniform(2.2, 3.2)
            cursor_speed = random.uniform(2.5, 4.0)
        else:
            req_path, req_method, resp_code = _normal_step()
            typing_speed = random.uniform(0.4, 1.6)
            cursor_speed = random.uniform(0.5, 2.0)

        payload = {
            "session_id": sid,
            "environment_id": str(args.environment),
            "req_path": req_path,
            "req_method": req_method,
            "resp_code": int(resp_code),
            "micro_features": {
                "typing_speed_wsession": float(typing_speed),
                "cursor_speed_wsession": float(cursor_speed),
            },
            "timestamp_unix_ms": int(time.time() * 1000),
        }

        _post_json(infer_url, payload)
        i += 1
        time.sleep(period)

    print(f"sent_events={i}")


if __name__ == "__main__":
    main()
