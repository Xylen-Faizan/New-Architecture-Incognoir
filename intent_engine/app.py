from __future__ import annotations

import asyncio
import os
from collections import deque
from datetime import datetime, timezone
import hashlib
import hmac
import re
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .services.cache import SessionCache, build_cache
from .services.kafka_pub import RiskEventPublisher, build_publisher
from .services.math_utils import entropy, now_unix_ms, security_health_score
from .services.model_registry import ModelRegistry, load_registry
from .services.alert_layer import AlertDecisionLayer, AlertQueue, AlertPayload, now_iso
from .services.rbac import APIKeyStore
from .services.health_score import HealthScoreCompute
from fastapi import Depends


# Role constants
ROLE_EXECUTIVE = "ROLE_EXECUTIVE"
ROLE_CISO = "ROLE_CISO"
ROLE_SOC_ANALYST = "ROLE_SOC_ANALYST"
ROLE_ADMIN = "ROLE_ADMIN"


_api_key_store = APIKeyStore()


def get_current_role(x_role: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)) -> str:
    """Resolve current role for request.

    For demo purposes the role may be provided via the `X-Role` header. In production
    this should be mapped from authentication tokens or API keys.
    """
    # API-key based role resolution (persistent)
    try:
        if x_api_key:
            role = _api_key_store.get_role(str(x_api_key))
            if role:
                return role
    except Exception:
        pass

    # legacy static admin key from settings
    try:
        if settings.api_key and x_api_key and str(x_api_key) == str(settings.api_key):
            return ROLE_ADMIN
    except Exception:
        pass

    if x_role:
        r = str(x_role).strip()
        if r in (ROLE_EXECUTIVE, ROLE_CISO, ROLE_SOC_ANALYST, ROLE_ADMIN):
            return r

    raise HTTPException(status_code=403, detail="forbidden: role required or invalid")


class Event(BaseModel):
    session_id: str = Field(min_length=1)
    environment_id: str = "default"
    user_id: Optional[str] = None
    device_id: Optional[str] = None
    req_path: str
    req_method: str
    resp_code: int
    micro_features: Dict[str, Any] = Field(default_factory=dict)
    timestamp_unix_ms: Optional[int] = None


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1)
    environment_id: str = "default"

class AuditEvent(BaseModel):
    environment_id: str = "default"
    session_id: str = Field(min_length=1)
    action: str = Field(min_length=1)
    note: Optional[str] = None


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache: SessionCache = build_cache(settings.redis_url)
publisher: RiskEventPublisher = build_publisher(settings.kafka_bootstrap_servers, settings.kafka_topic_risk_events)
registry: ModelRegistry = load_registry(
    model_dir=settings.model_dir,
    max_history_len=settings.max_history_len,
    risk_band_thresholds=(settings.risk_band_normal_max, settings.risk_band_suspicious_max, settings.risk_band_high_max),
)


@app.on_event("startup")
def _startup_security_checks() -> None:
    if not settings.dev_mode:
        if not str(settings.api_key or "").strip():
            raise RuntimeError("INTENT_ENGINE_API_KEY is required when INTENT_ENGINE_DEV_MODE=0")
        if not str(settings.pseudonymization_secret or "").strip():
            raise RuntimeError("INTENT_ENGINE_PSEUDONYMIZATION_SECRET is required when INTENT_ENGINE_DEV_MODE=0")

_started_at_unix_ms = now_unix_ms()
_inference_count = 0
_inference_latency_ms_sum = 0.0
_recent_events: List[Dict[str, Any]] = []
_ws_clients: List[WebSocket] = []

_latency_ms_window: Deque[float] = deque(maxlen=5000)
_infer_ts_window: Deque[int] = deque(maxlen=5000)
_cache_gets = 0
_cache_hits = 0
_env_last_seen: Dict[str, int] = {}
_audit_log: Deque[Dict[str, Any]] = deque(maxlen=5000)
_allowlist_ip_hash: Dict[str, set[str]] = {}

# Alert system
_alert_decision = AlertDecisionLayer(suspicious_threshold=0.5, high_threshold=0.75, critical_threshold=0.85)
_alert_queue = AlertQueue(maxlen=10000)
_session_last_risk: Dict[str, Dict[str, Any]] = {}  # track prev risk for delta detection

# Health score tracker
_health_score = HealthScoreCompute(baseline_rps=10.0, baseline_p95_ms=50.0)

_ENV_RE = re.compile(r"^[a-zA-Z0-9_-]{1,32}$")


def _normalize_environment_id(value: str) -> str:
    v = str(value or "").strip()
    if not v:
        return "default"
    if not _ENV_RE.match(v):
        return "default"
    return v


def _session_key(environment_id: str, session_id: str) -> str:
    env = _normalize_environment_id(environment_id)
    sid = str(session_id)
    return f"session:{env}:{sid}"


def _require_api_key(x_api_key: Optional[str]) -> None:
    required = str(settings.api_key or "")
    if not required:
        return
    if str(x_api_key or "") != required:
        raise HTTPException(status_code=401, detail="unauthorized")


def _require_role(x_user_role: Optional[str], allowed: List[str]) -> None:
    """Raise 403 if role header not in allowed list.

    In development/demo mode we tolerate a missing role header and
    implicitly treat requests as coming from an administrator.  This
    makes it easier to open governance or admin pages before the client
    has persisted a role value.
    """
    if not x_user_role:
        if settings.dev_mode:
            # default to admin when no header supplied in dev/demo
            return
        raise HTTPException(status_code=403, detail="forbidden")
    if str(x_user_role) not in allowed:
        raise HTTPException(status_code=403, detail="forbidden")


def _pseudonymize(value: str, secret: str) -> str:
    v = str(value or "")
    if not v:
        return ""
    return hmac.new(secret.encode("utf-8"), v.encode("utf-8"), hashlib.sha256).hexdigest()

def _risk_band(score: float) -> str:
    s = float(score)
    if s < float(settings.risk_band_normal_max):
        return "Normal"
    if s < float(settings.risk_band_suspicious_max):
        return "Suspicious"
    if s < float(settings.risk_band_high_max):
        return "High"
    return "Critical"

def _ts_to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc).isoformat()

def _model_version_tag() -> Dict[str, Any]:
    base = str(settings.model_dir)
    paths = {
        "encoder": os.path.join(base, "encoder.pkl"),
        "markov": os.path.join(base, "markov.pkl"),
        "lstm": os.path.join(base, "lstm.h5"),
    }
    out: Dict[str, Any] = {"model_dir": base}
    for k, p in paths.items():
        try:
            st = os.stat(p)
            out[f"{k}_mtime_unix_ms"] = int(st.st_mtime * 1000)
        except Exception:
            out[f"{k}_mtime_unix_ms"] = None
    return out


async def _ws_broadcast(payload: Dict[str, Any]) -> None:
    if not _ws_clients:
        return
    data = payload
    clients = list(_ws_clients)

    async def send_one(ws: WebSocket) -> None:
        try:
            await ws.send_json(data)
        except Exception:
            try:
                _ws_clients.remove(ws)
            except ValueError:
                return

    await asyncio.gather(*(send_one(ws) for ws in clients), return_exceptions=True)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_dir": str(settings.model_dir),
        "cache": cache.__class__.__name__,
        "model": registry.bayes_filter.snapshot(),
    }


@app.post("/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    cache.delete(_session_key(req.environment_id, req.session_id))
    return {"ok": True}


class APIKeyCreate(BaseModel):
    role: str = Field(..., description="Role to assign to the API key")


@app.get("/admin/api_keys")
def admin_list_api_keys(role: str = Depends(get_current_role)) -> Dict[str, Any]:
    if role != ROLE_ADMIN:
        raise HTTPException(status_code=403, detail="forbidden: admin only")
    keys = _api_key_store.list_keys()
    # Do not disclose keys in production; for demo we return them
    return {"keys": keys}


@app.post("/admin/api_keys")
def admin_create_api_key(req: APIKeyCreate, role: str = Depends(get_current_role)) -> Dict[str, Any]:
    if role != ROLE_ADMIN:
        raise HTTPException(status_code=403, detail="forbidden: admin only")
    if req.role not in (ROLE_EXECUTIVE, ROLE_CISO, ROLE_SOC_ANALYST, ROLE_ADMIN):
        raise HTTPException(status_code=400, detail="invalid role")
    k = _api_key_store.add_key(req.role)
    return {"key": k, "role": req.role}


@app.delete("/admin/api_keys/{key}")
def admin_delete_api_key(key: str, role: str = Depends(get_current_role)) -> Dict[str, Any]:
    if role != ROLE_ADMIN:
        raise HTTPException(status_code=403, detail="forbidden: admin only")
    ok = _api_key_store.delete_key(key)
    return {"ok": ok}


@app.get("/admin/calibration")
def admin_get_calibration(role: str = Depends(get_current_role)) -> Dict[str, Any]:
    if role != ROLE_ADMIN:
        raise HTTPException(status_code=403, detail="forbidden: admin only")
    return {
        "risk_update_alpha": settings.risk_update_alpha,
        "risk_decay_lambda": settings.risk_decay_lambda,
        "calibration_temperature": settings.calibration_temperature,
        "entropy_floor": settings.entropy_floor,
        "logit_clamp_min": settings.logit_clamp_min,
        "logit_clamp_max": settings.logit_clamp_max,
    }


@app.get("/metrics/health")
def metrics_health(environment_id: str = "default") -> Dict[str, Any]:
    """Health score combining model confidence, alerts, recovery, and stability."""
    sys_metrics = {
        # note: original code mistakenly referenced `_infer_count` which doesn't exist and
        # caused a NameError resulting in a 500 response and missing CORS headers.
        "requests_per_second": float(_inference_count / max(1.0, (time.time() - _started_at_unix_ms / 1000.0) / 1.0)),
        "p95_inference_latency_ms": float(np.percentile(list(_latency_ms_window), 95)) if _latency_ms_window else 0.0,
    }
    
    now_ms = int(time.time() * 1000)
    recent_30d = [e for e in _recent_events if now_ms - e.get("ts", 0) <= 30 * 24 * 3600 * 1000]
    mean_confidence = float(np.mean([e.get("confidence", 0.8) for e in recent_30d])) if recent_30d else 0.8
    
    recent_24h = [e for e in _recent_events if now_ms - e.get("ts", 0) <= 24 * 3600 * 1000]
    active_threats = [e for e in recent_24h if e.get("risk", 0) >= 0.6]
    
    health_result = _health_score.compute(
        mean_confidence=mean_confidence,
        alert_count_24h=len(active_threats),
        baseline_alerts_24h=5,
        feedback_reduction_factor=1.0,
        current_rps=sys_metrics["requests_per_second"],
        p95_latency_ms=sys_metrics["p95_inference_latency_ms"],
    )
    
    return {
        "health_score": health_result["score"],
        "components": health_result["components"],
        "uptime_days": health_result["uptime_days"],
        "inference_count": _inference_count,
        "mean_confidence": mean_confidence,
    }


@app.get("/metrics/export")
def metrics_export(environment_id: str = "default", role: str = Depends(get_current_role)) -> Dict[str, Any]:
    """Export comprehensive metrics for compliance and analytics."""
    if role not in (ROLE_EXECUTIVE, ROLE_ADMIN, ROLE_CISO):
        raise HTTPException(status_code=403, detail="forbidden: officer-level access required")
    
    now_ms = int(time.time() * 1000)
    window_30d = 30 * 24 * 3600 * 1000
    
    recent_30d = [e for e in _recent_events if now_ms - e.get("ts", 0) <= window_30d]
    risks = [e.get("risk", 0) for e in recent_30d]
    confidences = [e.get("confidence", 0.5) for e in recent_30d if "confidence" in e]
    
    health_result = _health_score.compute(
        mean_confidence=float(np.mean(confidences)) if confidences else 0.5,
        alert_count_24h=len([e for e in _recent_events if now_ms - e.get("ts", 0) <= 24 * 3600 * 1000 and e.get("risk", 0) >= 0.6]),
        baseline_alerts_24h=5,
    )
    
    return {
        "export_timestamp": now_iso(now_ms),
        "environment_id": environment_id,
        "period_days": 30,
        "metrics": {
            "total_inferences": _inference_count,
            "mean_inference_latency_ms": float(_inference_latency_ms_sum / max(1, _inference_count)),
            "p95_latency_ms": float(np.percentile(list(_latency_ms_window), 95)) if _latency_ms_window else 0.0,
            "mean_risk": float(np.mean(risks)) if risks else 0.0,
            "max_risk": float(np.max(risks)) if risks else 0.0,
            "health_score": health_result["score"],
            "alert_count_30d": len(_alert_queue.get_recent(10000)) if hasattr(_alert_queue, 'get_recent') else 0,
            "unique_sessions": len(set(e.get("session_id") for e in recent_30d)),
        },
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    if settings.api_key:
        api_key = websocket.headers.get("x-api-key", "") or websocket.query_params.get("api_key", "")
        if str(api_key) != str(settings.api_key):
            await websocket.close(code=4401)
            return
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        try:
            _ws_clients.remove(websocket)
        except ValueError:
            return


@app.get("/session/{session_id}")
def get_session(session_id: str, environment_id: str = "default", x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_api_key(x_api_key)
    key = _session_key(environment_id, session_id)
    data = cache.get_json(key) or {}
    return {"session_id": session_id, "environment_id": _normalize_environment_id(environment_id), **data}


@app.get("/session/{session_id}/entropy")
def get_session_entropy(session_id: str, environment_id: str = "default", x_api_key: Optional[str] = Header(None)) -> List[Dict[str, Any]]:
    _require_api_key(x_api_key)
    key = _session_key(environment_id, session_id)
    data = cache.get_json(key) or {}
    events = data.get("events") if isinstance(data.get("events"), list) else []
    out = []
    for e in events:
        if not isinstance(e, dict):
            continue
        out.append(
            {
                "timestamp_unix_ms": int(e.get("timestamp_unix_ms") or 0),
                "entropy": float(e.get("entropy") or 0.0),
                "risk_score": float(e.get("risk_score") or 0.0),
            }
        )
    return out


@app.get("/session/{session_id}/transitions")
def get_session_transitions(
    session_id: str, environment_id: str = "default", x_api_key: Optional[str] = Header(None)
) -> List[Dict[str, Any]]:
    _require_api_key(x_api_key)
    key = _session_key(environment_id, session_id)
    data = cache.get_json(key) or {}
    events = data.get("events") if isinstance(data.get("events"), list) else []
    out = []
    prev_state = None
    for e in events:
        if not isinstance(e, dict):
            continue
        cur = e.get("state_id")
        if cur is None:
            continue
        cur_int = int(cur)
        if prev_state is not None:
            out.append(
                {
                    "from_state_id": int(prev_state),
                    "to_state_id": cur_int,
                    "timestamp_unix_ms": int(e.get("timestamp_unix_ms") or 0),
                    "risk_score": float(e.get("risk_score") or 0.0),
                }
            )
        prev_state = cur_int
    return out


@app.get("/transitions/aggregate")
def transitions_aggregate(environment_id: Optional[str] = None, window_minutes: int = 60, x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_api_key(x_api_key)
    now_ms = now_unix_ms()
    cutoff = now_ms - int(window_minutes) * 60_000
    if environment_id:
        env = _normalize_environment_id(environment_id)
        events = [e for e in _recent_events if str(e.get("environment_id")) == env and int(e.get("timestamp_unix_ms") or 0) >= cutoff]
    else:
        events = [e for e in _recent_events if int(e.get("timestamp_unix_ms") or 0) >= cutoff]

    edges = {}
    nodes = set()
    by_session = {}
    for e in events:
        sid = str(e.get("session_id") or "")
        if not sid:
            continue
        ts = int(e.get("timestamp_unix_ms") or 0)
        by_session.setdefault(sid, []).append((ts, e))

    for sid, evs in by_session.items():
        evs.sort(key=lambda x: x[0])
        prev = None
        for ts, e in evs:
            cur = e.get("state_id")
            try:
                cur = int(cur)
            except Exception:
                continue
            nodes.add(cur)
            if prev is not None:
                k = (prev, cur)
                edges[k] = edges.get(k, 0) + 1
            prev = cur

    edge_list = []
    for (a, b), c in edges.items():
        edge_list.append({"from_state_id": int(a), "to_state_id": int(b), "count": int(c)})

    return {"ok": True, "window_minutes": int(window_minutes), "nodes": sorted(list(nodes)), "edges": edge_list}


@app.get("/metrics/system")
def metrics_system(x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_api_key(x_api_key)
    avg_latency = float(_inference_latency_ms_sum / _inference_count) if _inference_count > 0 else 0.0
    lat = np.asarray(list(_latency_ms_window), dtype=np.float64)
    p50 = float(np.percentile(lat, 50)) if lat.size else 0.0
    p95 = float(np.percentile(lat, 95)) if lat.size else 0.0

    now_ms = now_unix_ms()
    cutoff_ms = now_ms - 60_000
    recent_ts = [t for t in _infer_ts_window if t >= cutoff_ms]
    rps = float(len(recent_ts) / 60.0)

    cache_hit_rate = float(_cache_hits / _cache_gets) if _cache_gets > 0 else 0.0
    active_envs = sorted([env for env, ts in _env_last_seen.items() if ts >= now_ms - 3_600_000])
    return {
        "uptime_ms": int(now_unix_ms() - _started_at_unix_ms),
        "inference_count": int(_inference_count),
        "avg_inference_latency_ms": float(avg_latency),
        "p50_inference_latency_ms": float(p50),
        "p95_inference_latency_ms": float(p95),
        "requests_per_second": float(rps),
        "cache_hit_rate": float(cache_hit_rate),
        "cache": cache.__class__.__name__,
        "kafka_enabled": publisher.__class__.__name__ != "NoopPublisher",
        "model": registry.bayes_filter.snapshot(),
        "pseudonymization_key_id": str(settings.pseudonymization_key_id),
        "environment_isolation": True,
        "active_environments": active_envs,
    }

@app.get("/governance/status")
def governance_status(x_api_key: Optional[str] = Header(None), x_user_role: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_api_key(x_api_key)
    _require_role(x_user_role, ["ROLE_ADMIN", "ROLE_CISO"])
    drift = metrics_drift(environment_id=None, x_api_key=x_api_key)
    drift_state = "Stable"
    try:
        rstd = float(drift.get("recent", {}).get("risk_std", 0.0))
        if rstd > 0.25:
            drift_state = "Red"
        elif rstd > 0.15:
            drift_state = "Amber"
    except Exception:
        drift_state = "Unknown"

    return {
        "ok": True,
        "environment_isolation": True,
        "pseudonymization": {
            "enabled": True,
            "key_id": str(settings.pseudonymization_key_id),
            "secret_configured": bool(settings.pseudonymization_secret) or bool(settings.dev_mode),
        },
        "model": {
            "version": _model_version_tag(),
            "baseline_version": _model_version_tag(),
        },
        "drift_status": str(drift_state),
        "demo_heuristics_enabled": bool(settings.demo_heuristics_enabled),
        "api_key_required": bool(settings.api_key),
    }


@app.get("/alerts")
def get_alerts(limit: int = 100, x_api_key: Optional[str] = Header(None), x_user_role: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_api_key(x_api_key)
    _require_role(x_user_role, ["ROLE_CISO", "ROLE_SOC_ANALYST", "ROLE_ADMIN"])
    alert_data = _alert_queue.to_dict()
    alert_data["alerts"] = alert_data["alerts"][-int(limit) :]
    return {
        "ok": True,
        **alert_data,
    }


@app.get("/alerts/session/{session_id}")
def get_session_alerts(session_id: str, environment_id: str = "default", x_api_key: Optional[str] = Header(None), x_user_role: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_api_key(x_api_key)
    _require_role(x_user_role, ["ROLE_CISO", "ROLE_SOC_ANALYST", "ROLE_ADMIN"])
    alert = _alert_queue.get_by_session(session_id)
    if alert is None:
        return {"ok": True, "alert": None, "session_id": session_id}
    return {
        "ok": True,
        "session_id": session_id,
        "alert": {
            "timestamp": alert.timestamp,
            "environment_id": alert.environment_id,
            "risk_score": alert.risk_score,
            "risk_band": alert.risk_band,
            "alert_level": alert.alert_level,
            "top_intent_state": alert.top_intent_state,
            "confidence": alert.confidence,
            "reason": alert.reason,
        },
    }


@app.get("/model/micro_baseline")
def model_micro_baseline(x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_api_key(x_api_key)
    means = getattr(registry.markov.micro_model, "feature_means", {}) or {}
    stds = getattr(registry.markov.micro_model, "feature_stds", {}) or {}
    features = []
    for k in sorted(set(list(means.keys()) + list(stds.keys()))):
        features.append({"feature": str(k), "mean": float(means.get(k, 0.0)), "std": float(stds.get(k, 0.0))})
    return {"ok": True, "features": features}

@app.get("/metrics/executive")
def metrics_executive(environment_id: str = "default", export: Optional[str] = None, x_api_key: Optional[str] = Header(None), x_user_role: Optional[str] = Header(None)) -> Any:
    _require_api_key(x_api_key)
    # accessible by executives and above
    _require_role(x_user_role, ["ROLE_EXECUTIVE", "ROLE_CISO", "ROLE_ADMIN"])
    env = _normalize_environment_id(environment_id)

    # if export requested, return a simple PDF placeholder
    if export == "board":
        # in real implementation generate PDF with report pages
        content = b"PDF-REPORT-BYTES-PLACEHOLDER"
        from fastapi.responses import Response
        return Response(content, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=incognoir_board_report.pdf"})

    now_ms = now_unix_ms()
    w5 = now_ms - 5 * 60_000
    w10 = now_ms - 10 * 60_000

    events = [e for e in _recent_events if str(e.get("environment_id")) == env and int(e.get("timestamp_unix_ms") or 0) >= w10]

    latest_by_session: Dict[str, Dict[str, Any]] = {}
    first_seen_5m: set[str] = set()
    first_seen_10m: set[str] = set()

    for e in events:
        sid = str(e.get("session_id") or "")
        if not sid:
            continue
        ts = int(e.get("timestamp_unix_ms") or 0)
        if ts >= w10:
            first_seen_10m.add(sid)
        if ts >= w5:
            first_seen_5m.add(sid)
        prev = latest_by_session.get(sid)
        if prev is None or int(prev.get("timestamp_unix_ms") or 0) <= ts:
            latest_by_session[sid] = e

    latest_5m = {sid: ev for sid, ev in latest_by_session.items() if int(ev.get("timestamp_unix_ms") or 0) >= w5}
    risks = np.asarray([float(ev.get("risk_score") or 0.0) for ev in latest_5m.values()], dtype=np.float64)

    high = 0
    critical = 0
    for ev in latest_5m.values():
        band = _risk_band(float(ev.get("risk_score") or 0.0))
        if band == "Critical":
            critical += 1
        elif band == "High":
            high += 1

    avg_risk = float(np.mean(risks)) if risks.size else 0.0

    prev_window = [e for e in _recent_events if str(e.get("environment_id")) == env and w10 <= int(e.get("timestamp_unix_ms") or 0) < w5]
    prev_latest: Dict[str, Dict[str, Any]] = {}
    for e in prev_window:
        sid = str(e.get("session_id") or "")
        if not sid:
            continue
        ts = int(e.get("timestamp_unix_ms") or 0)
        prev = prev_latest.get(sid)
        if prev is None or int(prev.get("timestamp_unix_ms") or 0) <= ts:
            prev_latest[sid] = e
    prev_risks = np.asarray([float(ev.get("risk_score") or 0.0) for ev in prev_latest.values()], dtype=np.float64)
    prev_avg = float(np.mean(prev_risks)) if prev_risks.size else 0.0
    trend_pct = float(((avg_risk - prev_avg) / prev_avg) * 100.0) if prev_avg > 1e-9 else 0.0

    drift = metrics_drift(environment_id=env, x_api_key=x_api_key)
    drift_state = "Stable"
    try:
        rstd = float(drift.get("recent", {}).get("risk_std", 0.0))
        if rstd > 0.25:
            drift_state = "Red"
        elif rstd > 0.15:
            drift_state = "Amber"
    except Exception:
        drift_state = "Unknown"

    # Compute components for Security Health Score
    # SessionRiskScore (0-100)
    session_risks = [float(ev.get("risk_score") or 0.0) for ev in latest_5m.values()]
    session_risk_score = float(np.mean(session_risks)) * 100.0 if session_risks else 0.0

    # Critical density
    total_sessions = int(len(latest_5m))
    critical_sessions = int(critical)

    # Attack velocity: suspicious events per minute over last 5 minutes
    suspicious_events = [e for e in _recent_events if str(e.get("environment_id")) == env and int(e.get("timestamp_unix_ms") or 0) >= w5 and float(e.get("risk_score") or 0.0) >= float(settings.risk_band_normal_max)]
    attack_velocity = float(len(suspicious_events)) / 5.0

    # Uncertainty: use entropy mean from drift metrics if available
    entropy_avg = float(drift.get("recent", {}).get("entropy_mean", 0.0)) if isinstance(drift, dict) else 0.0
    # entropy max: approximate by log(num_states)
    try:
        import math as _math

        entropy_max = float(_math.log(max(2, int(registry.encoder.num_states))))
    except Exception:
        entropy_max = 1.0

    # Drift score: use recent risk std as proxy
    drift_value = float(drift.get("recent", {}).get("risk_std", 0.0)) if isinstance(drift, dict) else 0.0

    # Compute security health using the unified function
    try:
        sec_health = security_health_score(
            session_risks=session_risks,
            critical_sessions=critical_sessions,
            total_sessions=total_sessions,
            attack_velocity=attack_velocity,
            entropy_avg=entropy_avg,
            entropy_max=entropy_max,
            drift=drift_value,
        )
    except Exception:
        sec_health = 0

    return {
        "ok": True,
        "environment_id": env,
        "security_health_score": int(sec_health),
        "security_health_components": {
            "session_risk_score": float(session_risk_score),
            "critical_score": float(min((float(critical_sessions) / max(1, total_sessions)) * 500.0, 100.0)),
            "velocity_score": float(min(attack_velocity * 2.0, 100.0)),
            "uncertainty_score": float((entropy_avg / max(1e-9, entropy_max)) * 100.0),
            "drift_score": float(min(drift_value * 100.0, 100.0)),
        },
        "active_sessions": int(len(latest_5m)),
        "new_sessions_5m": int(len(first_seen_5m)),
        "high_risk": int(high),
        "critical_risk": int(critical),
        "avg_risk": float(avg_risk),
        "avg_risk_trend_pct": float(trend_pct),
        "drift_status": str(drift_state),
        "model_stability_pct": float(max(0.0, 100.0 - abs(float(drift.get("recent", {}).get("risk_std", 0.0))) * 100.0)),
    }

@app.get("/metrics/risk_distribution")
def metrics_risk_distribution(environment_id: str = "default", window_minutes: int = 5, x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_api_key(x_api_key)
    env = _normalize_environment_id(environment_id)
    now_ms = now_unix_ms()
    w = now_ms - int(window_minutes) * 60_000
    events = [e for e in _recent_events if str(e.get("environment_id")) == env and int(e.get("timestamp_unix_ms") or 0) >= w]
    latest: Dict[str, Dict[str, Any]] = {}
    for e in events:
        sid = str(e.get("session_id") or "")
        if not sid:
            continue
        ts = int(e.get("timestamp_unix_ms") or 0)
        prev = latest.get(sid)
        if prev is None or int(prev.get("timestamp_unix_ms") or 0) <= ts:
            latest[sid] = e
    counts = {"Normal": 0, "Suspicious": 0, "High": 0, "Critical": 0}
    for ev in latest.values():
        counts[_risk_band(float(ev.get("risk_score") or 0.0))] += 1
    return {"ok": True, "environment_id": env, "window_minutes": int(window_minutes), "counts": counts}

@app.get("/metrics/risk_timeseries")
def metrics_risk_timeseries(
    environment_id: str = "default", window_minutes: int = 30, bucket_seconds: int = 60, x_api_key: Optional[str] = Header(None)
) -> Dict[str, Any]:
    _require_api_key(x_api_key)
    env = _normalize_environment_id(environment_id)
    now_ms = now_unix_ms()
    w = now_ms - int(window_minutes) * 60_000
    bucket_ms = int(bucket_seconds) * 1000
    events = [e for e in _recent_events if str(e.get("environment_id")) == env and int(e.get("timestamp_unix_ms") or 0) >= w]
    buckets: Dict[int, List[float]] = {}
    for e in events:
        ts = int(e.get("timestamp_unix_ms") or 0)
        b = ts - (ts % bucket_ms)
        buckets.setdefault(b, []).append(float(e.get("risk_score") or 0.0))
    xs = sorted(buckets.keys())
    series = []
    for b in xs:
        arr = np.asarray(buckets[b], dtype=np.float64)
        series.append({"timestamp_unix_ms": int(b), "time": _ts_to_iso(int(b)), "avg_risk": float(np.mean(arr)) if arr.size else 0.0})
    return {"ok": True, "environment_id": env, "window_minutes": int(window_minutes), "bucket_seconds": int(bucket_seconds), "series": series}

@app.get("/audit/logs")
def audit_logs(environment_id: Optional[str] = None, limit: int = 200, x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_api_key(x_api_key)
    env = _normalize_environment_id(environment_id) if environment_id else None
    rows = list(_audit_log)
    if env:
        rows = [r for r in rows if str(r.get("environment_id")) == env]
    rows = rows[-int(max(1, min(2000, limit))) :]
    rows.reverse()
    return {"ok": True, "rows": rows}

@app.post("/audit/event")
def audit_event(evt: AuditEvent, x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_api_key(x_api_key)
    env = _normalize_environment_id(evt.environment_id)
    row = {
        "timestamp_unix_ms": now_unix_ms(),
        "environment_id": env,
        "session_id": str(evt.session_id),
        "action": str(evt.action),
        "note": str(evt.note or ""),
    }
    _audit_log.append(row)
    if evt.action == "allowlist_ip" and isinstance(evt.note, str) and evt.note:
        _allowlist_ip_hash.setdefault(env, set()).add(str(evt.note))
    return {"ok": True}



class FeedbackRequest(BaseModel):
    action: str
    note: Optional[str] = None


@app.post("/session/{session_id}/feedback")
def session_feedback(session_id: str, fb: FeedbackRequest, environment_id: str = "default", x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_api_key(x_api_key)
    env = _normalize_environment_id(environment_id)
    key = _session_key(env, session_id)
    data = cache.get_json(key) or {}
    belief = data.get("belief") if isinstance(data.get("belief"), list) else None
    if not belief:
        return {"ok": False, "reason": "no_session_or_belief"}

    # support simple false positive feedback by reducing weight on a state
    if fb.action == "false_positive":
        # note expected to contain state id or comma-separated ids
        if not fb.note:
            return {"ok": False, "reason": "missing_note"}
        try:
            ids = [int(x.strip()) for x in str(fb.note).split(",") if x.strip()]
        except Exception:
            return {"ok": False, "reason": "invalid_state_id"}

        b = np.asarray(belief, dtype=np.float64)
        for sid in ids:
            if 0 <= sid < b.size:
                b[sid] = float(b[sid] * 0.5)
        s = float(b.sum())
        if s > 0:
            b = b / s
        else:
            b = np.ones_like(b) / float(b.size)
        data["belief"] = [float(x) for x in b.tolist()]
        # write back to cache preserving events/history
        cache.set_json(key, data, ttl_seconds=settings.session_ttl_seconds)
        _audit_log.append({"timestamp_unix_ms": now_unix_ms(), "environment_id": env, "session_id": session_id, "action": "feedback.false_positive", "note": str(fb.note)})
        return {"ok": True, "applied": True}

    return {"ok": False, "reason": "unsupported_action"}


@app.get("/model/hmm")
def model_hmm(x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_api_key(x_api_key)
    a = registry.markov.transition_model.transition_matrix().astype(np.float64)
    return {
        "num_hidden_states": int(a.shape[0]),
        "malicious_states": list(registry.markov.malicious_states),
        "transition_matrix": a.tolist(),
        "emission_beta_params": [list(p) for p in registry.markov.emission_config.beta_params],
    }


@app.get("/metrics/drift")
def metrics_drift(environment_id: Optional[str] = None, x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_api_key(x_api_key)
    cached = cache.get_json("metrics:drift")
    if isinstance(cached, dict) and cached:
        return dict(cached)

    if environment_id:
        env = _normalize_environment_id(environment_id)
        events = [e for e in _recent_events if str(e.get("environment_id")) == env]
    else:
        events = list(_recent_events)
    if not events:
        return {"ok": True, "recent": {"count": 0}}

    risks = np.asarray([float(e.get("risk_score") or 0.0) for e in events], dtype=np.float64)
    ents = np.asarray([float(e.get("entropy") or 0.0) for e in events], dtype=np.float64)

    return {
        "ok": True,
        "recent": {
            "count": int(len(events)),
            "risk_mean": float(np.mean(risks)),
            "risk_std": float(np.std(risks, ddof=0)),
            "entropy_mean": float(np.mean(ents)),
            "entropy_std": float(np.std(ents, ddof=0)),
        },
    }


@app.post("/infer")
async def infer(event: Event, request: Request, x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    global _inference_count, _inference_latency_ms_sum, _cache_gets, _cache_hits
    _require_api_key(x_api_key)

    t0 = time.perf_counter()
    env = _normalize_environment_id(event.environment_id)
    _env_last_seen[env] = now_unix_ms()

    secret = str(settings.pseudonymization_secret or "")
    if not secret:
        if settings.dev_mode:
            secret = "dev-secret"
        else:
            raise HTTPException(status_code=500, detail="pseudonymization_secret_missing")

    ip_raw = request.client.host if request.client else ""
    ua_raw = request.headers.get("user-agent", "")

    subject = {
        "key_id": str(settings.pseudonymization_key_id),
        "ip_hash": _pseudonymize(ip_raw, secret),
        "user_id_hash": _pseudonymize(str(event.user_id or ""), secret),
        "device_id_hash": _pseudonymize(str(event.device_id or ""), secret),
        "ua_hash": _pseudonymize(ua_raw, secret),
    }

    state_id = registry.encoder.encode(event.req_path, event.req_method, int(event.resp_code))
    token = int(state_id + 1)

    key = _session_key(env, event.session_id)
    _cache_gets += 1
    prev = cache.get_json(key)
    if prev is not None:
        _cache_hits += 1

    if prev is None:
        prev_state = None
    else:
        belief = prev.get("belief")
        history = prev.get("history")
        if isinstance(belief, list) and isinstance(history, list):
            from .services.bayesian_filter import FilterState

            prev_state = FilterState(belief=[float(x) for x in belief], history=[int(x) for x in history])
        else:
            prev_state = None

    result = registry.bayes_filter.update(
        state_token=token,
        micro_features=dict(event.micro_features),
        prev=prev_state,
    )

    ts_ms = int(event.timestamp_unix_ms) if event.timestamp_unix_ms is not None else now_unix_ms()
    belief_entropy = float(entropy(np.asarray(result.belief, dtype=np.float64)))

    prev_events = prev.get("events") if isinstance(prev, dict) and isinstance(prev.get("events"), list) else []
    events2 = list(prev_events)

    allowlist = _allowlist_ip_hash.get(env, set())
    allowlisted = bool(subject.get("ip_hash")) and str(subject.get("ip_hash")) in allowlist

    events2.append(
        {
            "timestamp_unix_ms": ts_ms,
            "state_id": int(state_id),
            "req_path": str(event.req_path),
            "req_method": str(event.req_method),
            "resp_code": int(event.resp_code),
            "micro_features": dict(event.micro_features),
            "risk_score": float(result.risk_score),
            "belief": [float(x) for x in result.belief],
            "entropy": float(belief_entropy),
            "top_intent_state": int(result.top_intent_state),
            "confidence": float(result.confidence),
            "allowlisted": bool(allowlisted),
        }
    )
    if len(events2) > int(settings.max_session_events):
        events2 = events2[-int(settings.max_session_events) :]

    # Manage simple decay: track consecutive non-anomalous events
    sess = cache.get_json(key) or {}
    no_anom = int(sess.get("no_anomaly_count") or 0)
    # consider an event non-anomalous if raw belief malicious mass below normal threshold
    try:
        mal_idx = tuple(getattr(registry.markov, "malicious_states", ()))
        raw_bel = np.asarray(result.belief, dtype=np.float64)
        mal_mass = float(np.sum(raw_bel[list(mal_idx)])) if mal_idx else 0.0
    except Exception:
        mal_mass = 0.0

    if mal_mass < float(settings.risk_band_normal_max):
        no_anom += 1
    else:
        no_anom = 0

    # apply decay if no anomaly for N steps
    try:
        decay_n = int(getattr(settings, "risk_decay_steps", 3))
    except Exception:
        decay_n = 3
    decayed_belief = np.asarray(result.belief, dtype=np.float64).copy()
    if no_anom >= decay_n:
        # reduce malicious mass gradually
        mal_idx = list(mal_idx)
        non_idx = [i for i in range(decayed_belief.size) if i not in mal_idx]
        if mal_idx:
            decayed_belief[mal_idx] = decayed_belief[mal_idx] * float(settings.risk_decay_factor)
        sdec = float(decayed_belief.sum())
        if sdec > 0:
            decayed_belief = decayed_belief / sdec

    sess_update = {
        "environment_id": env,
        "subject": subject,
        "belief": [float(x) for x in decayed_belief.tolist()],
        "history": result.history,
        "events": events2,
        "updated_at_unix_ms": int((time.time()) * 1000),
        "no_anomaly_count": int(no_anom),
    }
    cache.set_json(key, sess_update, ttl_seconds=settings.session_ttl_seconds)

    risk_score = float(result.risk_score)
    heuristic_risk = 0.0
    if settings.demo_heuristics_enabled:
        p = str(event.req_path or "").lower()
        m = str(event.req_method or "").upper()
        code = int(event.resp_code)

        if "/admin" in p or "/export" in p or "/internal" in p:
            heuristic_risk = max(heuristic_risk, 0.55)
        if code in (401, 403):
            heuristic_risk = max(heuristic_risk, 0.45)
        if code == 429:
            heuristic_risk = max(heuristic_risk, 0.65)
        if code >= 500:
            heuristic_risk = max(heuristic_risk, 0.55)
        if m in ("PUT", "DELETE", "PATCH"):
            heuristic_risk = max(heuristic_risk, 0.35)

        try:
            ts = float(event.micro_features.get("typing_speed_wsession", 0.0))
            if ts > 2.2:
                heuristic_risk = max(heuristic_risk, 0.55)
        except Exception:
            pass

        # Apply a configurable weight to demo heuristics to avoid saturation
        hw = float(getattr(settings, "demo_heuristics_weight", 0.25))
        heuristic_risk = float(np.clip(float(heuristic_risk) * hw, 0.0, 1.0))

        # combine conservatively so heuristics cannot instantly saturate
        risk_score = 1.0 - (1.0 - float(risk_score)) * (1.0 - float(heuristic_risk))
        risk_score = float(np.clip(risk_score, 0.0, 1.0))

    if allowlisted:
        risk_score = float(min(risk_score, float(settings.risk_band_normal_max) * 0.9))

    response = registry.explainability.build_response(
        session_id=event.session_id,
        state_id=state_id,
        belief=result.belief,
        risk_score=risk_score,
        top_intent_state=result.top_intent_state,
        confidence=result.confidence,
        lstm_prob=result.lstm_prob,
        micro_prob=result.micro_prob,
        model_snapshot=registry.bayes_filter.snapshot(),
    )

    risk_event = {
        "environment_id": env,
        "session_id": str(event.session_id),
        "subject": subject,
        "state_id": int(state_id),
        "risk_score": float(risk_score),
        "risk_band": str(response["risk_band"]),
        "timestamp_unix_ms": int(ts_ms),
        "belief": [float(x) for x in result.belief],
        "entropy": float(belief_entropy),
        "top_intent_state": int(result.top_intent_state),
        "confidence": float(result.confidence),
        "evidence": dict(response.get("evidence", {})),
        "req_path": str(event.req_path),
        "req_method": str(event.req_method),
        "resp_code": int(event.resp_code),
    }
    if settings.demo_heuristics_enabled:
        risk_event["heuristic_risk"] = float(heuristic_risk)

    # Alert triggering logic
    session_key = f"{env}:{event.session_id}"
    prev_data = _session_last_risk.get(session_key)
    prev_risk = prev_data.get("risk_score") if prev_data else None
    prev_time = prev_data.get("timestamp_unix_ms") if prev_data else None
    time_delta_ms = (ts_ms - prev_time) if prev_time is not None else None

    should_alert = _alert_decision.should_alert(
        risk_score=risk_score,
        prev_risk_score=prev_risk,
        time_delta_ms=time_delta_ms,
    )

    if should_alert:
        alert_level = _alert_decision.get_alert_level(risk_score)
        alert = AlertPayload(
            timestamp=now_iso(),
            environment_id=env,
            session_id=str(event.session_id),
            risk_score=float(risk_score),
            risk_band=str(response.get("risk_band", "Unknown")),
            alert_level=str(alert_level or "suspicious"),
            top_intent_state=int(result.top_intent_state),
            confidence=float(result.confidence),
            entropy=float(belief_entropy),
            rare_transitions_count=len([e for e in events2 if float(e.get("risk_score", 0)) >= 0.6]),
            microbehavior_z_max=0.0,  # placeholder; compute from microTable in real impl
            reason=f"Risk crossed {alert_level} threshold ({risk_score:.3f})" if prev_risk is None else f"Risk moved from {prev_risk:.3f} to {risk_score:.3f}",
            subject_ip_hash=str(subject.get("ip_hash", "")),
        )
        _alert_queue.enqueue(alert)

    # Track risk for next iteration
    _session_last_risk[session_key] = {
        "risk_score": float(risk_score),
        "timestamp_unix_ms": int(ts_ms),
    }

    publisher.publish(risk_event)
    await _ws_broadcast(risk_event)

    _recent_events.append(risk_event)
    if len(_recent_events) > 2000:
        del _recent_events[:1000]

    dt_ms = (time.perf_counter() - t0) * 1000.0
    _inference_count += 1
    _inference_latency_ms_sum += float(dt_ms)
    _latency_ms_window.append(float(dt_ms))
    _infer_ts_window.append(int(ts_ms))

    return response
