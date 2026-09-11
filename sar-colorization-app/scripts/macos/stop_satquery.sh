#!/usr/bin/env bash
# SatQuery AI — stop only processes recorded as launcher-owned.
set -euo pipefail

STATE_DIR="${SATQUERY_STATE_DIR:-${HOME}/Library/Application Support/SatQueryAI}"
LOG_DIR="${SATQUERY_LOG_DIR:-${HOME}/Library/Logs/SatQueryAI}"
STATE_FILE="${STATE_DIR}/state.json"
LAUNCHER_LOG="${LOG_DIR}/launcher.log"
BACKEND_PID_FILE="${STATE_DIR}/backend.pid"
FRONTEND_PID_FILE="${STATE_DIR}/frontend.pid"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() {
    echo "[$(ts)] $*"
    [[ -d "${LOG_DIR}" ]] && echo "[$(ts)] $*" >> "${LAUNCHER_LOG}" 2>/dev/null || true
}

state_value() {
    /usr/bin/python3 -c '
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    value = data.get(sys.argv[2], "")
    if isinstance(value, bool): print("true" if value else "false")
    else: print(value)
except Exception:
    raise SystemExit(1)
' "${STATE_FILE}" "$1" 2>/dev/null
}

process_cwd() {
    lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1
}

process_command() {
    ps -p "$1" -o command= 2>/dev/null || true
}

safe_stop() {
    local service="$1" label="$2" expected_cwd="$3" expected_pattern="$4"
    local owned pid cwd command i
    owned="$(state_value "launcher_started_${service}" || true)"
    pid="$(state_value "${service}_pid" || true)"
    if [[ "${owned}" != "true" || ! "${pid}" =~ ^[0-9]+$ ]]; then
        log "${label}: not recorded as launcher-owned — leaving it untouched"
        return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
        log "${label}: recorded PID ${pid} is no longer running"
        return 0
    fi
    cwd="$(process_cwd "${pid}")"
    command="$(process_command "${pid}")"
    if [[ "${cwd}" != "${expected_cwd}" ]] || ! grep -q "${expected_pattern}" <<< "${command}"; then
        log "${label}: PID ${pid} ownership no longer matches SatQuery — leaving it untouched"
        return 0
    fi
    log "${label}: sending SIGTERM to verified launcher-owned PID ${pid}"
    kill "${pid}" 2>/dev/null || true
    for ((i=0; i<20; i++)); do
        kill -0 "${pid}" 2>/dev/null || { log "${label}: stopped"; return 0; }
        sleep 0.5
    done
    cwd="$(process_cwd "${pid}")"
    command="$(process_command "${pid}")"
    if [[ "${cwd}" == "${expected_cwd}" ]] && grep -q "${expected_pattern}" <<< "${command}"; then
        log "${label}: sending SIGKILL to still-verified PID ${pid}"
        kill -9 "${pid}" 2>/dev/null || true
    fi
}

log "════════════════════════════════════════════════════════"
log "Stopping launcher-owned SatQuery AI services"
log "════════════════════════════════════════════════════════"

if [[ ! -f "${STATE_FILE}" ]]; then
    log "No launcher state found — no processes were terminated"
    exit 0
fi

PROJECT_ROOT="$(state_value project_root || true)"
if [[ -z "${PROJECT_ROOT}" ]]; then
    log "State does not identify a project root — no processes were terminated"
    exit 1
fi

safe_stop frontend "Frontend" "${PROJECT_ROOT}/frontend" "next-server\|next start\|npm run start"
safe_stop backend "Backend" "${PROJECT_ROOT}" "uvicorn.*backend:app"

rm -f "${FRONTEND_PID_FILE}" "${BACKEND_PID_FILE}" "${STATE_FILE}"
log "SatQuery AI stop completed"
if [[ "${SATQUERY_SKIP_NOTIFICATIONS:-0}" != "1" ]]; then
    osascript -e 'display notification "SatQuery AI launcher-owned services have been stopped." with title "SatQuery AI"' 2>/dev/null || true
fi
