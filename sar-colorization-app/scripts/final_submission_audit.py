#!/usr/bin/env python3
"""Read-only SIH submission checks with optional full test/build execution."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

try:
    from verify_models import verify as verify_models
except ModuleNotFoundError:
    from scripts.verify_models import verify as verify_models


REQUIRED_DOCS = (
    "README.md", "docs/SIH_FINAL_ENGINEERING_AUDIT.md", "docs/PROBLEM_STATEMENT_26167_COMPLIANCE.md",
    "docs/OPTICAL_SAR_EVALUATION_PROTOCOL.md", "docs/MODEL_DATASET_PROVENANCE.md",
    "docs/CARTOSAT_RISAT_READINESS.md", "docs/OFFLINE_DEMO_CHECKLIST.md", "docs/SIH_FINAL_SUBMISSION_AUDIT.md",
)
TEXT_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".json", ".md", ".txt", ".toml", ".yaml", ".yml"}
README_MARKERS = ("Python 3.11", "npm", "verify_models.py", "smoke_all_workflows.py", "MODEL_DATASET_PROVENANCE.md")


def scan_private_paths(root: Path) -> list[str]:
    findings = []
    excluded = {"node_modules", ".next", "venv", ".git", "artifacts", "tests"}
    pattern = re.compile(r"/(Users|home)/[A-Za-z0-9._-]+/")
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES or excluded.intersection(path.parts) or ".test." in path.name:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if pattern.search(content):
            findings.append(str(path.relative_to(root)))
    return findings


def scan_secrets(root: Path) -> list[str]:
    findings = []
    excluded = {"node_modules", ".next", "venv", ".git", "artifacts", "tests"}
    patterns = (re.compile(r"AIza[0-9A-Za-z_-]{30,}"), re.compile(r"sk-[A-Za-z0-9]{20,}"))
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES or excluded.intersection(path.parts) or ".test." in path.name:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(content) for pattern in patterns):
            findings.append(str(path.relative_to(root)))
    return findings


def scan_large_accidental_files(root: Path, threshold_bytes: int = 100 * 1024 * 1024) -> list[str]:
    """Flag large files outside declared model, dependency, build and artifact stores."""
    findings = []
    excluded = {"node_modules", ".next", "venv", ".git", "models", "artifacts"}
    for path in root.rglob("*"):
        if path.is_file() and not excluded.intersection(path.parts) and path.stat().st_size > threshold_bytes:
            findings.append(str(path.relative_to(root)))
    return findings


def readme_command_coverage(root: Path) -> dict[str, bool]:
    content = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").is_file() else ""
    return {marker: marker in content for marker in README_MARKERS}


def command(root: Path, values: list[str]) -> dict:
    completed = subprocess.run(values, cwd=root, text=True, capture_output=True)
    return {"command": " ".join(values), "passed": completed.returncode == 0, "returncode": completed.returncode, "output_tail": (completed.stdout + completed.stderr)[-4000:]}


def audit(root: Path, run_tests: bool = False) -> dict:
    checks = {
        "required_docs": {name: (root / name).is_file() for name in REQUIRED_DOCS},
        "private_absolute_paths": scan_private_paths(root),
        "apparent_secrets": scan_secrets(root),
        "large_accidental_files": scan_large_accidental_files(root),
        "readme_reproducibility_markers": readme_command_coverage(root),
        "provenance_document_present": (root / "docs/MODEL_DATASET_PROVENANCE.md").is_file(),
        "models": verify_models(root),
    }
    commands = [command(root, [str(root / "venv/bin/python"), "scripts/validate_benchmark_results.py"])]
    if run_tests:
        commands.extend([
            command(root, [str(root / "venv/bin/python"), "-m", "pytest", "tests/test_benchmark_schema.py", "tests/compliance", "-q"]),
            command(root, [str(root / "venv/bin/python"), "-m", "pytest", "-q"]),
            command(root / "frontend", ["npm", "test"]),
            command(root / "frontend", ["npx", "tsc", "--noEmit", "--incremental", "false"]),
            command(root / "frontend", ["npm", "run", "lint"]),
            command(root / "frontend", ["npm", "run", "build"]),
            command(root, [str(root / "venv/bin/python"), "scripts/smoke_all_workflows.py", "--output", "artifacts/final_readiness/workflow_smoke.json"]),
        ])
    ready = (
        all(checks["required_docs"].values())
        and not checks["private_absolute_paths"]
        and not checks["apparent_secrets"]
        and not checks["large_accidental_files"]
        and all(checks["readme_reproducibility_markers"].values())
        and checks["provenance_document_present"]
        and checks["models"]["required_ok"]
        and all(item["passed"] for item in commands)
    )
    return {"status": "passed" if ready else "action_required", "checks": checks, "commands": commands}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root.expanduser().resolve(), args.run_tests)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["status"] == "passed" else 1)
