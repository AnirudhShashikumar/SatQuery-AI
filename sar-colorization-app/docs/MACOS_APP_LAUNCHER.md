# SatQuery AI — macOS Application Launcher

## Overview

The SatQuery AI macOS launcher lets you start the entire platform (FastAPI backend + Next.js frontend) by double-clicking a single app icon — no Terminal required.

---

## Installation

### One-time setup

```bash
cd "${HOME}/Documents/Projects/SatQuery AI/sar-colorization-app"
bash scripts/macos/install_satquery_app.sh
```

The installer will:
1. Validate Python venv, Node.js, and npm
2. Install frontend npm dependencies if needed
3. Build the frontend for production (`next build`)
4. Generate a macOS app icon from project branding
5. Create `SatQuery AI.app`
6. Install it to `~/Applications/SatQuery AI.app`
7. Store project configuration in `~/Library/Application Support/SatQueryAI/`

### Prerequisites

| Dependency | Minimum | Check |
|:-----------|:--------|:------|
| Python venv | 3.9+ | `venv/bin/python --version` |
| Node.js | 18+ | `node --version` |
| npm | 8+ | `npm --version` |
| Backend dependencies | — | `venv/bin/pip install -r requirements-backend.txt` |
| Frontend dependencies | — | `cd frontend && npm ci` |

---

## Launching

### From Finder

1. Open **Finder** → **Applications** (or `~/Applications`)
2. Double-click **SatQuery AI**
3. Wait for the notification "SatQuery AI is ready"
4. Browser opens automatically

### From Terminal

```bash
open ~/Applications/SatQuery\ AI.app
```

### What happens on launch

1. **Backend check**: The process on `127.0.0.1:8010`, its working directory, command, and SatQuery health signature must all match before it is reused
2. **Backend start**: If not running, launches `uvicorn backend:app --host 127.0.0.1 --port 8010`
3. **Frontend check**: The process working directory and command must identify this SatQuery checkout, and `/api/build-info` must match the installer-generated build ID, git SHA, and backend URL
4. **Frontend start**: If not running, launches `npm run start` (production mode) on port 3000
5. **Port safety**: A verified stale SatQuery process on 3000 is replaced by exact PID; unrelated software is left untouched and SatQuery selects 3001–3010
6. **Browser open**: Once both services are healthy, opens the actual selected port recorded in state

### Duplicate instance handling

If you double-click the app while SatQuery is already running, the launcher will:
- Serialize launches through an atomic SatQuery-controlled lock
- Verify the healthy service identity and build ID
- Skip starting duplicates
- Open the browser immediately

---

## Stopping

```bash
bash "${HOME}/Documents/Projects/SatQuery AI/sar-colorization-app/scripts/macos/stop_satquery.sh"
```

This only stops processes that the SatQuery launcher started. It will:
- Verify PID ownership before killing (won't kill unrelated processes)
- Send SIGTERM first, then SIGKILL after 10 seconds if needed
- Clean up PID files and state

---

## Logs

All logs are stored in `~/Library/Logs/SatQueryAI/`:

| File | Contents |
|:-----|:---------|
| `launcher.log` | Startup orchestration, health checks, errors |
| `backend.log` | FastAPI/Uvicorn output |
| `frontend.log` | Next.js output |

**Security**: No API keys, tokens, or secrets are logged.

---

## State & Configuration

Stored in `~/Library/Application Support/SatQueryAI/`:

| File | Purpose |
|:-----|:--------|
| `config` | Project root path (re-run installer if project moves) |
| `state.json` | Actual PIDs, ports, build identity, project root, launch timestamp, and process ownership |
| `backend.pid` | Backend process ID |
| `frontend.pid` | Frontend process ID |

The production build also writes ignored local identity files at `frontend/.satquery-build-id` and `frontend/.satquery-git-sha`. The same values are baked into `/api/build-info`; the endpoint sends `Cache-Control: no-store` and exposes no secrets.

---

## Ports

| Service | Default Port | Override |
|:--------|:-------------|:---------|
| Backend (FastAPI) | `8010` | `SATQUERY_BACKEND_PORT` env var |
| Frontend (Next.js) | `3000` | `SATQUERY_FRONTEND_PORT` env var |

Both services bind to `127.0.0.1` only (not exposed to network).

If port 3000 is busy with a non-SatQuery process, the launcher leaves that exact process untouched and tries ports 3001–3010. An alternate existing frontend is reused only when the state file names that port and its complete current-build identity verifies.

---

## Reinstalling After Project Move

If you move the project directory:

```bash
cd "/new/path/to/sar-colorization-app"
bash scripts/macos/install_satquery_app.sh
```

The installer updates the stored project root automatically.

---

## Changing the Icon

1. Place your icon as a PNG (preferably 1024×1024) at:
   ```
   frontend/public/icon-512.png
   ```
2. Re-run the installer:
   ```bash
   bash scripts/macos/install_satquery_app.sh
   ```

---

## Uninstalling

```bash
# Remove the app
rm -rf ~/Applications/SatQuery\ AI.app

# Remove state and config
rm -rf ~/Library/Application\ Support/SatQueryAI

# Remove logs
rm -rf ~/Library/Logs/SatQueryAI
```

The project code itself is not affected.

---

## Developer Mode vs. App Launcher Mode

| Feature | Developer Mode | App Launcher |
|:--------|:---------------|:-------------|
| Backend | `uvicorn backend:app --reload` | `uvicorn backend:app` (no reload) |
| Frontend | `npm run dev` (HMR) | `npm run start` (production) |
| Browser | Manual | Automatic |
| Logs | Terminal stdout | `~/Library/Logs/SatQueryAI/` |

For development, continue using Terminal directly. The app launcher is for demo/presentation use.

---

## Troubleshooting

### "SatQuery AI project not found"
The app was moved or the project directory was relocated. Re-run the installer.

### "Python virtual environment not found"
```bash
cd "/path/to/sar-colorization-app"
python3 -m venv venv
venv/bin/pip install -r requirements-backend.txt
```

### "Frontend build not found"
```bash
cd "/path/to/sar-colorization-app"
bash scripts/macos/install_satquery_app.sh
```

### "Backend startup timed out"
Check `~/Library/Logs/SatQueryAI/backend.log` for model loading errors or missing dependencies.

### "Agent unavailable" in the UI
Ensure the backend is healthy: `curl http://127.0.0.1:8010/health`

If it returns HTTP 200 with `"status": "ready"`, the issue is in the frontend build. Rebuild:
```bash
bash scripts/macos/install_satquery_app.sh
```
