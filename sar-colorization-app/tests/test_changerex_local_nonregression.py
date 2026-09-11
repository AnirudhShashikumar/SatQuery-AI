import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(ROOT).as_posix()
        digest.update(f"{_sha(path)}  {relative}\n".encode())
    return digest.hexdigest()


def test_router_public_models_backend_and_ttp_remain_unchanged():
    expected = json.loads((Path(__file__).with_name("changerex_protected_hashes.json")).read_text())
    for relative, digest in expected["files"].items():
        assert _sha(ROOT / relative) == digest, f"protected production file changed: {relative}"


def test_no_frontend_changes():
    expected = json.loads((Path(__file__).with_name("changerex_protected_hashes.json")).read_text())
    paths = [
        path for path in (ROOT / "frontend").rglob("*")
        if path.is_file()
        and path.suffix in {".ts", ".tsx", ".css", ".json"}
        and "node_modules" not in path.parts
        and ".next" not in path.parts
    ]
    assert _aggregate(paths) == expected["frontend_digest"]


def test_ttp_service_tree_unchanged():
    expected = json.loads((Path(__file__).with_name("changerex_protected_hashes.json")).read_text())
    assert _aggregate((ROOT / "ttp_service").rglob("*.py")) == expected["ttp_service_digest"]


def test_standalone_package_has_no_forbidden_runtime_imports():
    forbidden = ("satquery_agent", "mmcv", "mmseg", "mmdet", "mmpretrain", "opencd", "fastapi")
    source = "\n".join(path.read_text() for path in (ROOT / "changerex_local").glob("*.py"))
    for name in forbidden:
        assert f"import {name}" not in source
        assert f"from {name}" not in source
