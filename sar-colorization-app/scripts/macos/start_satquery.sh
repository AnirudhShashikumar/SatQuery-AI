#!/usr/bin/env bash
# SatQuery AI — deterministic macOS production launcher.
set -euo pipefail

BACKEND_PORT="${SATQUERY_BACKEND_PORT:-8010}"
BASE_FRONTEND_PORT="${SATQUERY_FRONTEND_PORT:-3000}"
MAX_FRONTEND_PORT="${SATQUERY_FRONTEND_MAX_PORT:-3010}"
BACKEND_HOST="127.0.0.1"
FRONTEND_HOST="127.0.0.1"
BACKEND_TIMEOUT="${SATQUERY_BACKEND_TIMEOUT:-120}"
FRONTEND_TIMEOUT="${SATQUERY_FRONTEND_TIMEOUT:-60}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRONTEND_DIR="${PROJECT_ROOT}/frontend"

LOG_DIR="${SATQUERY_LOG_DIR:-${HOME}/Library/Logs/SatQueryAI}"
STATE_DIR="${SATQUERY_STATE_DIR:-${HOME}/Library/Application Support/SatQueryAI}"
BACKEND_LOG="${LOG_DIR}/backend.log"
FRONTEND_LOG="${LOG_DIR}/frontend.log"
LAUNCHER_LOG="${LOG_DIR}/launcher.log"
BACKEND_PID_FILE="${STATE_DIR}/backend.pid"
FRONTEND_PID_FILE="${STATE_DIR}/frontend.pid"
STATE_FILE="${STATE_DIR}/state.json"
LAUNCH_LOCK="${STATE_DIR}/launch.lock"

PYTHON="${PROJECT_ROOT}/venv/bin/python"
NPM="$(command -v npm 2>/dev/null || true)"
NODE="$(command -v node 2>/dev/null || true)"
BUILD_ID_FILE="${FRONTEND_DIR}/.satquery-build-id"
GIT_SHA_FILE="${FRONTEND_DIR}/.satquery-git-sha"
EXPECTED_API_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"
EXPECTED_FRONTEND_BUILD_ID=""
EXPECTED_GIT_SHA=""

LAUNCHER_STARTED_BACKEND=false
LAUNCHER_STARTED_FRONTEND=false
BACKEND_PID=""
FRONTEND_PID=""
FRONTEND_PORT="${BASE_FRONTEND_PORT}"
FRONTEND_ACTION=""

ts() { date "+%Y-%m-%d %H:%M:%S"; }

log() {
    local msg="[$(ts)] $*"
    echo "${msg}" >> "${LAUNCHER_LOG}" 2>/dev/null || true
    echo "${msg}"
}

notify() {
    [[ "${SATQUERY_SKIP_NOTIFICATIONS:-0}" == "1" ]] && return 0
    osascript -e "display notification \"$1\" with title \"SatQuery AI\"" 2>/dev/null || true
}

die() {
    log "FATAL: $*"
    if [[ "${SATQUERY_SKIP_NOTIFICATIONS:-0}" != "1" ]]; then
        osascript -e "display dialog \"SatQuery AI could not start.\" & return & return & \"$*\" & return & return & \"See: ${LAUNCHER_LOG}\" buttons {\"OK\"} default button \"OK\" with icon stop with title \"SatQuery AI\"" 2>/dev/null || true
    fi
    exit 1
}

port_listener_pid() {
    lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -n 1
}

process_cwd() {
    lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1
}

process_command() {
    ps -p "$1" -o command= 2>/dev/null || true
}

is_satquery_frontend_process() {
    local pid="$1" cwd command
    [[ -n "${pid}" ]] || return 1
    cwd="$(process_cwd "${pid}" || true)"
    [[ "${cwd}" == "${FRONTEND_DIR}" ]] || return 1
    command="$(process_command "${pid}")"
    case "${command}" in
        *next-server*|*"next start"*|*"npm run start"*) return 0 ;;
        *) return 1 ;;
    esac
}

is_satquery_backend_process() {
    local pid="$1" cwd command
    [[ -n "${pid}" ]] || return 1
    cwd="$(process_cwd "${pid}" || true)"
    [[ "${cwd}" == "${PROJECT_ROOT}" ]] || return 1
    command="$(process_command "${pid}")"
    case "${command}" in
        *uvicorn*backend:app*) return 0 ;;
        *) return 1 ;;
    esac
}

backend_signature_ok() {
    curl -fsS --max-time 3 "http://${BACKEND_HOST}:${BACKEND_PORT}/health" 2>/dev/null |
        "${PYTHON}" -c 'import json,sys; d=json.load(sys.stdin); m=d.get("models", {}); raise SystemExit(0 if d.get("status") == "ready" and isinstance(m, dict) and "pix2pix" in m and "sarfusionformer" in m else 1)' 2>/dev/null
}

frontend_identity() {
    local port="$1"
    curl -fsS --max-time 3 "http://${FRONTEND_HOST}:${port}/api/build-info" 2>/dev/null |
        "${PYTHON}" -c 'import json,sys; d=json.load(sys.stdin); print("\t".join(str(d.get(k, "")) for k in ("app", "build_id", "git_sha", "api_url")))' 2>/dev/null
}

frontend_is_current() {
    local port="$1" pid identity app build_id git_sha api_url
    pid="$(port_listener_pid "${port}" || true)"
    [[ -n "${pid}" ]] || return 1
    is_satquery_frontend_process "${pid}" || return 1
    identity="$(frontend_identity "${port}")" || return 1
    IFS=$'\t' read -r app build_id git_sha api_url <<< "${identity}"
    [[ "${app}" == "SatQuery AI" ]] || return 1
    [[ "${build_id}" == "${EXPECTED_FRONTEND_BUILD_ID}" ]] || return 1
    [[ "${git_sha}" == "${EXPECTED_GIT_SHA}" ]] || return 1
    [[ "${api_url}" == "${EXPECTED_API_URL}" ]] || return 1
}

state_frontend_port() {
    [[ -f "${STATE_FILE}" ]] || return 1
    "${PYTHON}" -c '
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    valid = data.get("project_root") == sys.argv[2] and data.get("frontend_build_id") == sys.argv[3]
    port = data.get("frontend_port")
    if valid and isinstance(port, int): print(port)
    else: raise SystemExit(1)
except Exception:
    raise SystemExit(1)
' "${STATE_FILE}" "${PROJECT_ROOT}" "${EXPECTED_FRONTEND_BUILD_ID}" 2>/dev/null
}

state_service_owned() {
    local service="$1" pid="$2"
    [[ -f "${STATE_FILE}" ]] || return 1
    "${PYTHON}" -c '
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    service, pid, root = sys.argv[2], int(sys.argv[3]), sys.argv[4]
    valid = data.get("project_root") == root and data.get(f"{service}_pid") == pid and data.get(f"launcher_started_{service}") is True
    raise SystemExit(0 if valid else 1)
except Exception:
    raise SystemExit(1)
' "${STATE_FILE}" "${service}" "${pid}" "${PROJECT_ROOT}" 2>/dev/null
}

terminate_exact_pid() {
    local pid="$1" label="$2" verifier="$3" i
    "${verifier}" "${pid}" || return 1
    log "${label}: sending SIGTERM to verified PID ${pid}"
    kill "${pid}" 2>/dev/null || return 1
    for ((i=0; i<20; i++)); do
        kill -0 "${pid}" 2>/dev/null || return 0
        sleep 0.5
    done
    "${verifier}" "${pid}" || return 1
    log "${label}: verified PID ${pid} did not stop; sending SIGKILL"
    kill -9 "${pid}" 2>/dev/null || true
}

choose_frontend() {
    local pid state_port port
    FRONTEND_PORT="${BASE_FRONTEND_PORT}"
    pid="$(port_listener_pid "${FRONTEND_PORT}" || true)"
    if [[ -z "${pid}" ]]; then
        FRONTEND_ACTION="start"
        return 0
    fi
    if frontend_is_current "${FRONTEND_PORT}"; then
        FRONTEND_ACTION="reuse"
        FRONTEND_PID="${pid}"
        return 0
    fi
    if is_satquery_frontend_process "${pid}"; then
        log "Stale SatQuery frontend detected on port ${FRONTEND_PORT} (PID ${pid})"
        terminate_exact_pid "${pid}" "Frontend" is_satquery_frontend_process || return 1
        FRONTEND_ACTION="start"
        return 0
    fi

    log "Port ${FRONTEND_PORT} belongs to an unrelated process (PID ${pid}); leaving it untouched"
    state_port="$(state_frontend_port 2>/dev/null || true)"
    if [[ "${state_port}" =~ ^[0-9]+$ ]] && (( state_port > BASE_FRONTEND_PORT && state_port <= MAX_FRONTEND_PORT )); then
        if frontend_is_current "${state_port}"; then
            FRONTEND_PORT="${state_port}"
            FRONTEND_PID="$(port_listener_pid "${state_port}")"
            FRONTEND_ACTION="reuse"
            return 0
        fi
    fi

    for ((port=BASE_FRONTEND_PORT + 1; port<=MAX_FRONTEND_PORT; port++)); do
        if [[ -z "$(port_listener_pid "${port}" || true)" ]]; then
            FRONTEND_PORT="${port}"
            FRONTEND_ACTION="start"
            return 0
        fi
    done
    return 1
}

acquire_launch_lock() {
    local owner i
    for ((i=0; i<180; i++)); do
        if mkdir "${LAUNCH_LOCK}" 2>/dev/null; then
            echo "$$" > "${LAUNCH_LOCK}/pid"
            trap 'rm -rf "${LAUNCH_LOCK}"' EXIT
            return 0
        fi
        owner="$(cat "${LAUNCH_LOCK}/pid" 2>/dev/null || true)"
        if [[ ! "${owner}" =~ ^[0-9]+$ ]] || ! kill -0 "${owner}" 2>/dev/null; then
            rm -rf "${LAUNCH_LOCK}"
            continue
        fi
        sleep 0.5
    done
    return 1
}

write_state() {
    local launch_timestamp
    launch_timestamp="$(date -u "+%Y-%m-%dT%H:%M:%SZ")"
    "${PYTHON}" - "${STATE_FILE}" "${BACKEND_PID}" "${BACKEND_PORT}" "${FRONTEND_PID}" "${FRONTEND_PORT}" "${EXPECTED_FRONTEND_BUILD_ID}" "${EXPECTED_GIT_SHA}" "${PROJECT_ROOT}" "${launch_timestamp}" "${LAUNCHER_STARTED_BACKEND}" "${LAUNCHER_STARTED_FRONTEND}" <<'PY'
import json, sys
path, backend_pid, backend_port, frontend_pid, frontend_port, build_id, git_sha, project_root, launched, started_backend, started_frontend = sys.argv[1:]
payload = {
    "backend_pid": int(backend_pid),
    "backend_port": int(backend_port),
    "frontend_pid": int(frontend_pid),
    "frontend_port": int(frontend_port),
    "frontend_build_id": build_id,
    "frontend_git_sha": git_sha,
    "frontend_url": f"http://127.0.0.1:{frontend_port}",
    "project_root": project_root,
    "launch_timestamp": launched,
    "launcher_started_backend": started_backend == "true",
    "launcher_started_frontend": started_frontend == "true",
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

validate_installation() {
    [[ -x "${PYTHON}" ]] || die "Python virtual environment not found at ${PYTHON}."
    [[ -n "${NPM}" && -n "${NODE}" ]] || die "Node.js or npm is not available in PATH."
    [[ -f "${FRONTEND_DIR}/package.json" ]] || die "Frontend package not found at ${FRONTEND_DIR}."
    [[ -d "${FRONTEND_DIR}/.next" && -f "${FRONTEND_DIR}/.next/BUILD_ID" ]] || die "Production frontend build is missing. Re-run scripts/macos/install_satquery_app.sh."
    [[ -s "${BUILD_ID_FILE}" && -s "${GIT_SHA_FILE}" ]] || die "SatQuery build identity is missing. Re-run scripts/macos/install_satquery_app.sh."
    IFS= read -r EXPECTED_FRONTEND_BUILD_ID < "${BUILD_ID_FILE}"
    IFS= read -r EXPECTED_GIT_SHA < "${GIT_SHA_FILE}"
    [[ "${EXPECTED_FRONTEND_BUILD_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "SatQuery build identity is invalid. Re-run the installer."
    [[ "${EXPECTED_GIT_SHA}" =~ ^[A-Za-z0-9._-]+$ ]] || die "SatQuery git identity is invalid. Re-run the installer."
}

ensure_backend() {
    local pid launch_pid i
    pid="$(port_listener_pid "${BACKEND_PORT}" || true)"
    if [[ -n "${pid}" ]] && is_satquery_backend_process "${pid}" && backend_signature_ok; then
        BACKEND_PID="${pid}"
        if state_service_owned backend "${BACKEND_PID}"; then
            LAUNCHER_STARTED_BACKEND=true
        fi
        echo "${BACKEND_PID}" > "${BACKEND_PID_FILE}"
        log "Current SatQuery backend healthy on port ${BACKEND_PORT} (PID ${BACKEND_PID}) — reusing"
        return 0
    fi
    if [[ -n "${pid}" ]]; then
        if is_satquery_backend_process "${pid}"; then
            log "Stale SatQuery backend detected on port ${BACKEND_PORT} (PID ${pid})"
            terminate_exact_pid "${pid}" "Backend" is_satquery_backend_process || die "Could not stop the verified stale SatQuery backend."
        else
            die "Port ${BACKEND_PORT} is occupied by an unrelated process (PID ${pid}); it was not terminated."
        fi
    fi

    log "Starting backend on port ${BACKEND_PORT}"
    notify "Starting backend…"
    cd "${PROJECT_ROOT}"
    nohup "${PYTHON}" -m uvicorn backend:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" >> "${BACKEND_LOG}" 2>&1 &
    launch_pid=$!
    LAUNCHER_STARTED_BACKEND=true
    for ((i=0; i<BACKEND_TIMEOUT; i++)); do
        pid="$(port_listener_pid "${BACKEND_PORT}" || true)"
        if [[ -n "${pid}" ]] && is_satquery_backend_process "${pid}" && backend_signature_ok; then
            BACKEND_PID="${pid}"
            echo "${BACKEND_PID}" > "${BACKEND_PID_FILE}"
            log "Backend healthy on port ${BACKEND_PORT} (PID ${BACKEND_PID})"
            return 0
        fi
        kill -0 "${launch_pid}" 2>/dev/null || die "Backend process exited unexpectedly. Check ${BACKEND_LOG}."
        sleep 1
    done
    die "Backend startup timed out after ${BACKEND_TIMEOUT}s. Check ${BACKEND_LOG}."
}

ensure_frontend() {
    local launch_pid i
    choose_frontend || die "No safe frontend port available in ${BASE_FRONTEND_PORT}-${MAX_FRONTEND_PORT}."
    if [[ "${FRONTEND_ACTION}" == "reuse" ]]; then
        if state_service_owned frontend "${FRONTEND_PID}"; then
            LAUNCHER_STARTED_FRONTEND=true
        fi
        echo "${FRONTEND_PID}" > "${FRONTEND_PID_FILE}"
        log "Current SatQuery build ${EXPECTED_FRONTEND_BUILD_ID} already serves port ${FRONTEND_PORT} (PID ${FRONTEND_PID}) — reusing"
        return 0
    fi

    log "Starting current frontend build ${EXPECTED_FRONTEND_BUILD_ID} on port ${FRONTEND_PORT}"
    notify "Starting interface…"
    cd "${FRONTEND_DIR}"
    NEXT_PUBLIC_API_URL="${EXPECTED_API_URL}" \
    NEXT_PUBLIC_SATQUERY_BUILD_ID="${EXPECTED_FRONTEND_BUILD_ID}" \
    NEXT_PUBLIC_SATQUERY_GIT_SHA="${EXPECTED_GIT_SHA}" \
    nohup "${NPM}" run start -- --hostname "${FRONTEND_HOST}" --port "${FRONTEND_PORT}" >> "${FRONTEND_LOG}" 2>&1 &
    launch_pid=$!
    LAUNCHER_STARTED_FRONTEND=true
    for ((i=0; i<FRONTEND_TIMEOUT; i++)); do
        if frontend_is_current "${FRONTEND_PORT}"; then
            FRONTEND_PID="$(port_listener_pid "${FRONTEND_PORT}")"
            echo "${FRONTEND_PID}" > "${FRONTEND_PID_FILE}"
            log "Frontend ready on port ${FRONTEND_PORT} (PID ${FRONTEND_PID})"
            return 0
        fi
        kill -0 "${launch_pid}" 2>/dev/null || die "Frontend process exited unexpectedly. Check ${FRONTEND_LOG}."
        sleep 1
    done
    die "Frontend startup timed out after ${FRONTEND_TIMEOUT}s or returned the wrong build identity. Check ${FRONTEND_LOG}."
}

main() {
    mkdir -p "${LOG_DIR}" "${STATE_DIR}"
    acquire_launch_lock || die "Another SatQuery launch is still in progress."
    log "════════════════════════════════════════════════════════"
    log "SatQuery AI launcher starting"
    log "Project root: ${PROJECT_ROOT}"
    log "════════════════════════════════════════════════════════"
    notify "Starting SatQuery AI…"
    validate_installation
    log "Expected frontend build: ${EXPECTED_FRONTEND_BUILD_ID} (${EXPECTED_GIT_SHA})"
    ensure_backend
    ensure_frontend
    write_state

    local frontend_url="http://${FRONTEND_HOST}:${FRONTEND_PORT}"
    log "Opening browser: ${frontend_url}"
    notify "SatQuery AI is ready"
    if [[ "${SATQUERY_SKIP_BROWSER_OPEN:-0}" != "1" ]]; then
        open "${frontend_url}"
    fi
    log "SatQuery AI launched successfully"
}

if [[ "${SATQUERY_LAUNCHER_LIB_ONLY:-0}" != "1" ]]; then
    main "$@"
fi
