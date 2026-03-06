from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional
from collections import deque
from datetime import datetime, timezone
import json


@dataclass(frozen=True)
class AlertPayload:
    """Structured alert for SIEM integration."""
    timestamp: str
    environment_id: str
    session_id: str
    risk_score: float
    risk_band: str
    alert_level: str  # "suspicious", "high", "critical"
    top_intent_state: int
    confidence: float
    entropy: float
    rare_transitions_count: int
    microbehavior_z_max: float
    reason: str
    subject_ip_hash: Optional[str] = None


class AlertDecisionLayer:
    """Decides when to trigger alerts based on thresholds and rate-of-change."""

    def __init__(
        self,
        suspicious_threshold: float = 0.5,
        high_threshold: float = 0.75,
        critical_threshold: float = 0.85,
        delta_threshold: float = 0.1,
        delta_window_seconds: int = 30,
    ) -> None:
        self.suspicious_threshold = float(suspicious_threshold)
        self.high_threshold = float(high_threshold)
        self.critical_threshold = float(critical_threshold)
        self.delta_threshold = float(delta_threshold)
        self.delta_window_seconds = int(delta_window_seconds)

    def get_alert_level(self, risk_score: float) -> Optional[str]:
        """Classify risk into alert level."""
        s = float(risk_score)
        if s >= self.critical_threshold:
            return "critical"
        elif s >= self.high_threshold:
            return "high"
        elif s >= self.suspicious_threshold:
            return "suspicious"
        return None

    def should_alert(
        self,
        risk_score: float,
        prev_risk_score: Optional[float] = None,
        time_delta_ms: Optional[float] = None,
    ) -> bool:
        """
        Decide whether to trigger alert.

        Criteria:
        1. Risk crosses a threshold
        2. AND either (a) first time crossing, OR (b) delta > threshold within time window
        """
        level = self.get_alert_level(risk_score)
        if level is None:
            return False

        # if no previous score, trigger (first observation)
        if prev_risk_score is None:
            return True

        delta = abs(float(risk_score) - float(prev_risk_score))
        prev_level = self.get_alert_level(prev_risk_score)

        # trigger if level changed or delta large
        if level != prev_level:
            return True

        if delta >= self.delta_threshold:
            if time_delta_ms is not None:
                if float(time_delta_ms) / 1000.0 <= float(self.delta_window_seconds):
                    return True
            else:
                return True

        return False


class AlertQueue:
    """In-memory alert queue (can be extended to Kafka)."""

    def __init__(self, maxlen: int = 10000) -> None:
        self._queue: Deque[AlertPayload] = deque(maxlen=maxlen)
        self._session_last_alert: Dict[str, AlertPayload] = {}

    def enqueue(self, alert: AlertPayload) -> None:
        """Add alert to queue."""
        self._queue.append(alert)
        self._session_last_alert[alert.session_id] = alert

    def get_recent(self, limit: int = 100) -> List[AlertPayload]:
        """Get recent alerts."""
        return list(self._queue)[-int(limit) :]

    def get_by_session(self, session_id: str) -> Optional[AlertPayload]:
        """Get last alert for session."""
        return self._session_last_alert.get(session_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON response."""
        return {
            "alerts": [
                {
                    "timestamp": a.timestamp,
                    "environment_id": a.environment_id,
                    "session_id": a.session_id,
                    "risk_score": a.risk_score,
                    "risk_band": a.risk_band,
                    "alert_level": a.alert_level,
                    "top_intent_state": a.top_intent_state,
                    "confidence": a.confidence,
                    "entropy": a.entropy,
                    "rare_transitions_count": a.rare_transitions_count,
                    "microbehavior_z_max": a.microbehavior_z_max,
                    "reason": a.reason,
                }
                for a in self.get_recent(100)
            ],
            "total_queued": len(self._queue),
        }


class AlertDispatcher:
    """Routes alerts to external systems (webhook, email, SIEM)."""

    def __init__(self) -> None:
        self._webhooks: List[str] = []
        self._siem_endpoint: Optional[str] = None
        self._email_recipients: List[str] = []

    def register_webhook(self, url: str) -> None:
        """Register webhook endpoint."""
        self._webhooks.append(str(url))

    def set_siem_endpoint(self, url: str) -> None:
        """Set SIEM HEC endpoint (e.g., Splunk)."""
        self._siem_endpoint = str(url)

    def add_email_recipient(self, email: str) -> None:
        """Add email recipient for critical alerts."""
        self._email_recipients.append(str(email))

    async def dispatch(self, alert: AlertPayload) -> Dict[str, Any]:
        """
        Dispatch alert to configured endpoints.
        Returns dispatch status.
        """
        result = {
            "alert_id": f"{alert.session_id}:{alert.timestamp}",
            "webhooks_sent": 0,
            "siem_sent": False,
            "emails_sent": 0,
            "errors": [],
        }

        # In production, use async HTTP client (aiohttp, httpx)
        # For now, just mock dispatch
        for url in self._webhooks:
            try:
                # would be: async with session.post(url, json=...) as resp
                result["webhooks_sent"] += 1
            except Exception as e:
                result["errors"].append(f"webhook {url}: {str(e)}")

        if self._siem_endpoint and alert.alert_level == "critical":
            try:
                # would be: async with session.post(siem_endpoint, json=...) as resp
                result["siem_sent"] = True
            except Exception as e:
                result["errors"].append(f"siem: {str(e)}")

        if alert.alert_level == "critical" and self._email_recipients:
            try:
                # would use: aiosmtplib or similar
                result["emails_sent"] = len(self._email_recipients)
            except Exception as e:
                result["errors"].append(f"email: {str(e)}")

        return result


def now_iso() -> str:
    """ISO8601 timestamp."""
    return datetime.now(tz=timezone.utc).isoformat()
