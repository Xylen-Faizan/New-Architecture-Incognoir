# Admin & Executive Console Fix Guide

## Issue Summary
1. AdminConsole was showing `fetchJson is not defined` error
2. ExecutiveConsole structure appeared reverted in screenshots

## Root Causes Fixed
1. ✅ Function declaration order corrected - `loadData` now defined before `useEffect` calls it
2. ✅ All async functions wrapped with `React.useCallback` with proper dependency arrays
3. ✅ `showSuccess` callback properly memoized
4. ✅ All imports from `../../lib/intent` are correct

## How to Test (Browser Console)

### Step 1: Hard Refresh (Clear Cache)
```
Ctrl + Shift + Delete (or Cmd + Shift + Delete on Mac)
```
- Select "Cached images and files" 
- Click "Clear" 
- Then refresh the page: `Ctrl + F5`

### Step 2: Check AdminConsole Loads
1. Navigate to: `http://localhost:5173/dashboard/admin`
2. Open DevTools (F12)
3. Check Console tab - should show NO red errors
4. Verify 8 tabs visible: Platform Overview, Model Control, Environments, Access & API, Security, Privacy, Alerts, Audit Logs

### Step 3: Check ExecutiveConsole Loads  
1. Navigate to: `http://localhost:5173/dashboard/executive`
2. Open DevTools (F12)
3. Check Console tab - should show NO red errors
4. Verify you see:
   - Hero section: Security Health Score / 100 with gauge circle
   - 4 KPI cards: Active Sessions, Active Threats, Risk Trend, Model Confidence
   - Threat Activity chart (AreaChart)
   - Risk Distribution pie chart
   - Risk Impact Summary
   - Top Incidents table
   - Compliance block at bottom

### Step 4: Test AdminConsole Features
1. Click on **Model Control** tab
2. Verify you can:
   - Move sliders for Risk Update Alpha, Risk Decay Lambda, etc.
   - See Apply Live and Rollback buttons when you change values
   - See success message when clicking Apply Live

3. Click on **Access & API** tab
4. Verify you can:
   - Select a role from dropdown
   - Click "Create Key" button
   - See API key created with success message
   - Delete key with confirmation

### Step 5: Backend Health Check
```bash
# Terminal 1: Run backend (if not already running)
$env:PYTHONPATH="F:\documents\New_Incognoir_Architecture(1)"
$env:INTENT_ENGINE_DEMO_HEURISTICS=1
.\venv\Scripts\Activate.ps1
uvicorn intent_engine.app:app --reload --port 8000
# note: a bug in `/metrics/health` used a wrong variable name and caused
# a 500 error and missing CORS header.  The code has been fixed – restart
# the server after pulling the latest changes so that the admin page
# can fetch health metrics successfully.
# Terminal 2: Generate demo traffic (keeps metrics populated)
# (Already running, but you can restart if needed)
python intent_engine/tools/generate_demo_traffic.py
```

## Files Modified

### 1. AdminConsole.jsx
- ✅ Fixed function declaration order
- ✅ Wrapped async functions with React.useCallback
- ✅ Added proper dependency arrays
- ✅ Imported fetchJson from ../../lib/intent

### 2. ExecutiveConsole.jsx
- ✅ Uses AreaChart for smooth trend visualization
- ✅ Fallback data for demo mode
- ✅ Security health score computed or fallback calculated
- ✅ Full enterprise layout preserved

## Common Issues & Solutions

### Issue: "fetchJson is not defined" in AdminConsole
**Solution:** This is now fixed. The import statement is correct:
```jsx
import { fetchJson } from "../../lib/intent";
```

### Issue: AdminConsole tabs not visible
**Solution:** 
1. Do a hard refresh (Ctrl+Shift+Delete + Ctrl+F5)
2. Check browser DevTools Network tab - make sure all JS files loaded (no 404s)
3. Check Console for errors

### Issue: ExecutiveConsole shows empty charts
**Solution:**
1. Make sure backend is running: `http://localhost:8000/health`
2. Make sure demo traffic generator is still running
3. Check browser console for fetch errors
4. Verify environment_id is being passed correctly

### Issue: Model Control sliders not working
**Solution:**
1. This is demo mode - sliders update UI state only
2. To persist: Backend would need POST to `/admin/calibration` (not wired in demo)
3. Apply Live button will show success message (demo behavior)

## Expected Behavior

### AdminConsole
- **Platform Overview**: Shows 4 stat cards, engine status
- **Model Control**: Editable sliders, Apply Live/Rollback buttons  
- **Environments**: Table showing demo, staging, prod
- **Access & API**: Create/delete API keys, role assignments
- **Security**: Risk thresholds, allowlist domains
- **Privacy**: Pseudonymization config, compliance checkboxes
- **Alerts**: Channel list, routing rules
- **Audit Logs**: Event timeline

### ExecutiveConsole
- **Hero**: Security Health Score (0-100) with gauge, status badge
- **KPI Row**: 4 cards with default values if backend empty
- **Trend Chart**: Smooth AreaChart visualization (48h data)
- **Distribution**: Pie chart (Normal/Suspicious/High/Critical)
- **Impact**: Estimated prevents, financial risk, MTTD
- **Incidents**: Table with fallback demo data
- **Compliance**: Footer with privacy/pseudonymization info

## Backend Endpoints Used

```
GET  /metrics/health           → Platform status (latency, inference count)
GET  /metrics/executive         → Executive metrics (KPIs, health score)
GET  /metrics/risk_distribution → Risk band counts (Normal/Susp/High/Crit)
GET  /metrics/risk_timeseries   → Trend data points (48h hourly buckets)
GET  /admin/calibration         → Model parameters
POST /admin/api_keys            → Create API key
DELETE /admin/api_keys/{key}    → Revoke API key
GET  /admin/api_keys            → List API keys
```

## Quick Debug Commands

### Check if fetchJson helper is working:
Open browser DevTools Console and run:
```javascript
// This should NOT error
const test = await fetch('http://localhost:8000/health', {
  headers: { 'X-Role': localStorage.getItem('userRole') || 'ROLE_ADMIN' }
});
console.log(test.status);  // should be 200
```

### Check Admin Console loads without errors:
```javascript
// In browser console
console.log(typeof AdminConsole);  // should be 'function' or undefined if not exposed
```

## Final Steps

1. ✅ Hard refresh both pages (Ctrl+Shift+Delete, then Ctrl+F5)
2. ✅ Check browser console - NO red errors
3. ✅ Backend running on port 8000
4. ✅ Demo traffic generator running
5. ✅ Test one admin action (create key, adjust slider)
6. ✅ Test executive console charts load

If you still see errors after hard refresh, check:
- Browser console for JavaScript syntax errors
- Network tab to see if JS files are loading (200 status)
- Backend console for API errors (check /health endpoint)
