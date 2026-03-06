# CISO‑Credible Mode (Local)

This mode enforces two non‑negotiables:

- API authentication between UI and engine
- Runtime pseudonymization (HMAC) before storage and streaming

## Backend (Intent Engine)

Pick long random values for the API key and pseudonymization secret.

PowerShell example:

```powershell
cd f:\documents\New_Incognoir_Architecture

$env:INTENT_ENGINE_DEV_MODE='0'
$env:INTENT_ENGINE_API_KEY='CHANGE_ME_LONG_RANDOM'
$env:INTENT_ENGINE_PSEUDONYMIZATION_KEY_ID='v1'
$env:INTENT_ENGINE_PSEUDONYMIZATION_SECRET='CHANGE_ME_LONG_RANDOM'

python -m uvicorn intent_engine.app:app --host 0.0.0.0 --port 8001
```

If `INTENT_ENGINE_DEV_MODE=0` and either `INTENT_ENGINE_API_KEY` or `INTENT_ENGINE_PSEUDONYMIZATION_SECRET` is missing, the service fails fast on startup.

## Frontend (Dashboard)

PowerShell example:

```powershell
cd f:\documents\New_Incognoir_Architecture\Incognoir-dashboard-demo

$env:VITE_INTENT_API_BASE='http://localhost:8001'
$env:VITE_RISK_WS_URL='ws://localhost:8001/ws'
$env:VITE_INTENT_API_KEY='CHANGE_ME_LONG_RANDOM'

npx vite
```

## Verify it’s locked

- Open the console with the correct API key: works
- Remove `VITE_INTENT_API_KEY` and refresh: requests fail with 401

