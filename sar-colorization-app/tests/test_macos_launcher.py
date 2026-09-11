"""Regression coverage for the macOS launcher's ownership and build decisions."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts/macos/start_satquery.sh"
INSTALL = ROOT / "scripts/macos/install_satquery_app.sh"
BUILD_ROUTE = ROOT / "frontend/app/api/build-info/route.ts"


def run_library(shell_body: str, *, state_dir: Path | None = None) -> str:
    with tempfile.TemporaryDirectory(prefix="SatQuery AI launcher test ") as temporary:
        state = state_dir or Path(temporary) / "Application Support" / "SatQueryAI"
        logs = Path(temporary) / "Logs" / "SatQueryAI"
        state.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        environment = {
            **os.environ,
            "SATQUERY_LAUNCHER_LIB_ONLY": "1",
            "SATQUERY_STATE_DIR": str(state),
            "SATQUERY_LOG_DIR": str(logs),
            "SATQUERY_SKIP_NOTIFICATIONS": "1",
            "START_SCRIPT": str(START),
        }
        command = f'source "$START_SCRIPT"\n{shell_body}'
        completed = subprocess.run(
            ["/bin/bash", "-c", command],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return completed.stdout.strip()


def test_nothing_running_starts_current_build_on_3000() -> None:
    output = run_library(
        """
port_listener_pid() { return 1; }
choose_frontend
printf '%s:%s' "$FRONTEND_ACTION" "$FRONTEND_PORT"
"""
    )
    assert output == "start:3000"


def test_current_satquery_on_3000_is_reused() -> None:
    output = run_library(
        """
port_listener_pid() { echo 4100; }
frontend_is_current() { [[ "$1" == "3000" ]]; }
choose_frontend
printf '%s:%s:%s' "$FRONTEND_ACTION" "$FRONTEND_PORT" "$FRONTEND_PID"
"""
    )
    assert output == "reuse:3000:4100"


def test_stale_verified_satquery_on_3000_is_replaced_by_exact_pid() -> None:
    output = run_library(
        """
port_listener_pid() { echo 4200; }
frontend_is_current() { return 1; }
is_satquery_frontend_process() { return 0; }
terminate_exact_pid() { printf 'terminated=%s\n' "$1"; }
log() { :; }
choose_frontend
printf '%s:%s' "$FRONTEND_ACTION" "$FRONTEND_PORT"
"""
    )
    assert output == "terminated=4200\nstart:3000"


def test_unrelated_process_on_3000_is_left_and_3001_is_selected() -> None:
    output = run_library(
        """
port_listener_pid() { [[ "$1" == "3000" ]] && echo 4300; }
frontend_is_current() { return 1; }
is_satquery_frontend_process() { return 1; }
state_frontend_port() { return 1; }
log() { :; }
choose_frontend
printf '%s:%s' "$FRONTEND_ACTION" "$FRONTEND_PORT"
"""
    )
    assert output == "start:3001"


def test_unrecorded_stale_satquery_on_3011_is_not_reused_or_killed() -> None:
    output = run_library(
        """
MAX_FRONTEND_PORT=3011
port_listener_pid() {
  case "$1" in 3000) echo 4400 ;; 3001) echo 4401 ;; 3002) echo 4402 ;; 3003) echo 4403 ;; 3004) echo 4404 ;; 3005) echo 4405 ;; 3006) echo 4406 ;; 3007) echo 4407 ;; 3008) echo 4408 ;; 3009) echo 4409 ;; 3010) echo 4410 ;; 3011) echo 4411 ;; esac
}
frontend_is_current() { return 1; }
is_satquery_frontend_process() { return 1; }
state_frontend_port() { return 1; }
terminate_exact_pid() { echo should-not-terminate; return 1; }
log() { :; }
if choose_frontend; then echo unexpected-reuse; else echo no-safe-port; fi
"""
    )
    assert output == "no-safe-port"


def test_rebuilt_identity_mismatch_marks_verified_process_stale() -> None:
    output = run_library(
        """
EXPECTED_FRONTEND_BUILD_ID=new-build
EXPECTED_GIT_SHA=abc123
port_listener_pid() { echo 4500; }
process_cwd() { printf '%s\n' "$FRONTEND_DIR"; }
process_command() { echo 'next-server (v15.5.23)'; }
frontend_identity() { printf 'SatQuery AI\told-build\tabc123\thttp://127.0.0.1:8010\n'; }
terminate_exact_pid() { printf 'replaced=%s\n' "$1"; }
log() { :; }
choose_frontend
printf '%s:%s' "$FRONTEND_ACTION" "$FRONTEND_PORT"
"""
    )
    assert output == "replaced=4500\nstart:3000"


def test_process_ownership_handles_project_paths_with_spaces() -> None:
    output = run_library(
        """
FRONTEND_DIR='/tmp/SatQuery AI/frontend'
process_cwd() { printf '%s\n' '/tmp/SatQuery AI/frontend'; }
process_command() { echo 'next-server (v15.5.23)'; }
if is_satquery_frontend_process 4600; then echo verified; else echo failed; fi
"""
    )
    assert output == "verified"


def test_healthy_existing_backend_is_reused() -> None:
    with tempfile.TemporaryDirectory(prefix="SatQuery AI backend reuse ") as temporary:
        state = Path(temporary) / "state"
        output = run_library(
            f"""
BACKEND_PID_FILE='{state}/backend.pid'
mkdir -p '{state}'
port_listener_pid() {{ echo 4700; }}
is_satquery_backend_process() {{ return 0; }}
backend_signature_ok() {{ return 0; }}
state_service_owned() {{ return 1; }}
log() {{ :; }}
ensure_backend
printf '%s:%s' "$BACKEND_PID" "$(cat "$BACKEND_PID_FILE")"
"""
        )
    assert output == "4700:4700"


def test_duplicate_launches_serialize_on_atomic_lock() -> None:
    with tempfile.TemporaryDirectory(prefix="SatQuery AI duplicate launch ") as temporary:
        state = Path(temporary) / "Application Support" / "SatQueryAI"
        logs = Path(temporary) / "Logs"
        state.mkdir(parents=True)
        logs.mkdir(parents=True)
        environment = {
            **os.environ,
            "SATQUERY_LAUNCHER_LIB_ONLY": "1",
            "SATQUERY_STATE_DIR": str(state),
            "SATQUERY_LOG_DIR": str(logs),
            "START_SCRIPT": str(START),
        }
        first = subprocess.Popen(
            ["/bin/bash", "-c", 'source "$START_SCRIPT"; acquire_launch_lock; sleep 1'],
            env=environment,
        )
        try:
            second = subprocess.run(
                ["/bin/bash", "-c", 'source "$START_SCRIPT"; acquire_launch_lock; echo acquired'],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
                timeout=5,
            )
        finally:
            first.wait(timeout=5)
        assert second.stdout.strip() == "acquired"


def test_installer_and_build_endpoint_define_production_identity_contract() -> None:
    installer = INSTALL.read_text(encoding="utf-8")
    route = BUILD_ROUTE.read_text(encoding="utf-8")
    assert "npm run build" in installer
    assert "npm run dev" not in START.read_text(encoding="utf-8")
    assert "npm run start" in START.read_text(encoding="utf-8")
    assert "NEXT_PUBLIC_API_URL=http://127.0.0.1:8010" in installer
    assert "NEXT_PUBLIC_SATQUERY_BUILD_ID" in installer
    assert 'app: "SatQuery AI"' in route
    assert '"Cache-Control": "no-store, max-age=0"' in route
