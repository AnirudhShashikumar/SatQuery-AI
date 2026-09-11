"""Server-side AI provider credentials and connection status.

Local GeoVision installations use the macOS Keychain for API keys.  Deployed
instances should use environment variables, which are intentionally read-only
from the web UI.  Neither mechanism returns the raw credential to the client.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests


GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-3.5-flash").strip()
GEMINI_SUPPORTED_MODELS = tuple(
    model.strip() for model in os.getenv(
        "GEMINI_SUPPORTED_MODELS", "gemini-3.5-flash,gemini-3.6-flash"
    ).split(",") if model.strip()
)

PROVIDERS = {
    "openai": {"label": "OpenAI", "model": "gpt-4.1-mini", "supported": True},
    "gemini": {"label": "Google Gemini", "model": GEMINI_DEFAULT_MODEL, "supported": True},
    "anthropic": {"label": "Anthropic Claude", "model": None, "supported": False},
}
KEYCHAIN_SERVICE = "GeoVision.AIProvider"
KEYCHAIN_ACCOUNT = "local-user"


class ProviderSettingsError(RuntimeError):
    """A recoverable provider-configuration error appropriate for the UI."""


def mask_key(api_key: str) -> str:
    """Show only a stable, non-sensitive hint of an API key."""
    if len(api_key) < 8:
        return "••••••••"
    prefix = api_key[:3] if api_key.startswith("sk-") else api_key[:2]
    return "{}{}{}".format(prefix, "*" * 32, api_key[-4:])


@dataclass(frozen=True)
class ProviderCredentials:
    provider: str
    api_key: Optional[str]
    model: Optional[str]
    source: Optional[str]


class ProviderSettingsStore:
    """Keeps provider metadata in a private state file and secrets in Keychain."""

    def __init__(self) -> None:
        self.state_path = Path.home() / ".config" / "geovision" / "provider.json"

    def _load_local_environment(self) -> None:
        """Keep compatibility with a local, gitignored server .env file."""
        env_file = Path(__file__).with_name(".env")
        if not env_file.is_file():
            return
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

    def _environment_credentials(self) -> Optional[ProviderCredentials]:
        self._load_local_environment()
        provider = os.getenv("VISION_PROVIDER", "").strip().lower()
        gemini_key = os.getenv("GEMINI_API_KEY")
        api_key = os.getenv("VISION_API_KEY") or os.getenv("OPENAI_API_KEY") or gemini_key
        if not provider and gemini_key:
            provider = "gemini"
        elif not provider and api_key:
            provider = "openai"
        if not provider or not api_key:
            return None
        return ProviderCredentials(
            provider=provider,
            api_key=api_key,
            model=(os.getenv("VISION_MODEL") if provider != "gemini" else os.getenv("GEMINI_MODEL")) or PROVIDERS.get(provider, {}).get("model"),
            source="environment",
        )

    def _read_state(self) -> Dict[str, Any]:
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
            return state if isinstance(state, dict) else {}
        except (FileNotFoundError, OSError, ValueError):
            return {}

    def _write_state(self, state: Dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(state, handle)
        temporary.chmod(0o600)
        temporary.replace(self.state_path)

    def _keychain_available(self) -> bool:
        return platform.system() == "Darwin"

    def _read_keychain(self) -> Optional[str]:
        if not self._keychain_available():
            return None
        result = subprocess.run(
            ["security", "find-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None

    def _write_keychain(self, api_key: str) -> None:
        if not self._keychain_available():
            raise ProviderSettingsError(
                "Secure local key storage is unavailable on this system. Use a server environment variable instead."
            )
        result = subprocess.run(
            ["security", "add-generic-password", "-U", "-a", KEYCHAIN_ACCOUNT, "-s", KEYCHAIN_SERVICE, "-w", api_key],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise ProviderSettingsError("GeoVision could not save the API key in macOS Keychain.")

    def _delete_keychain(self) -> None:
        if not self._keychain_available():
            return
        subprocess.run(
            ["security", "delete-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", KEYCHAIN_SERVICE],
            capture_output=True,
            text=True,
            check=False,
        )

    def credentials(self) -> ProviderCredentials:
        # User-configured Keychain credentials take precedence over an optional
        # deployment fallback key; neither raw value ever leaves the backend.
        state = self._read_state()
        keychain_key = self._read_keychain()
        if keychain_key and state.get("provider"):
            return ProviderCredentials(
                provider=str(state.get("provider", "")).lower(),
                api_key=keychain_key,
                model=state.get("model"),
                source="keychain",
            )
        environment = self._environment_credentials()
        if environment:
            return environment
        return ProviderCredentials(provider="", api_key=None, model=None, source=None)

    def public_status(self) -> Dict[str, Any]:
        credentials = self.credentials()
        configured = bool(credentials.api_key and credentials.provider)
        state = self._read_state()
        configured_provider = PROVIDERS.get(credentials.provider, {})
        return {
            "configured": configured,
            "provider": configured_provider.get("label") or None,
            "provider_id": credentials.provider or None,
            "model": credentials.model if configured else None,
            "masked_key": mask_key(credentials.api_key) if credentials.api_key else None,
            "masked_api_key": mask_key(credentials.api_key) if credentials.api_key else None,
            "source": credentials.source,
            "connection_status": state.get("connection_status", "not_configured" if not configured else "connected"),
            "status": state.get("connection_status", "not_configured" if not configured else "connected"),
            "supported": bool(configured_provider.get("supported")),
            "managed_by_environment": credentials.source == "environment",
            "last_tested_at": state.get("last_tested_at"),
            "latency_ms": state.get("latency_ms"),
        }

    def supported_models(self, provider: str) -> Dict[str, Any]:
        provider = provider.strip().lower()
        if provider != "gemini":
            raise ProviderSettingsError("Only Google Gemini models are currently available in this configuration flow.")
        models = [
            {"id": model, "label": model.replace("-", " ").title(), "supports_images": True, "recommended": model == GEMINI_DEFAULT_MODEL}
            for model in GEMINI_SUPPORTED_MODELS
        ]
        return {"provider": "gemini", "default_model": GEMINI_DEFAULT_MODEL, "models": models}

    def save(self, provider: str, api_key: str, model: Optional[str] = None) -> Dict[str, Any]:
        provider = provider.strip().lower()
        clean_key = api_key.strip()
        if provider not in PROVIDERS or not PROVIDERS[provider]["supported"]:
            raise ProviderSettingsError("Choose a provider supported by this application.")
        if not clean_key or "\n" in api_key or "\r" in api_key:
            raise ProviderSettingsError("Enter a complete API key on one line.")
        selected_model = (model or PROVIDERS[provider]["model"] or "").strip()
        if provider == "gemini" and selected_model not in GEMINI_SUPPORTED_MODELS:
            raise ProviderSettingsError("Choose a Gemini model supported by this application.")
        self._write_keychain(clean_key)
        self._write_state({"provider": provider, "model": selected_model, "connection_status": "connected"})
        return self.public_status()

    def record_connection(self, status: str, latency_ms: Optional[int] = None) -> None:
        credentials = self.credentials()
        if credentials.source == "environment":
            return
        state = self._read_state()
        state.update({"provider": credentials.provider, "model": credentials.model, "connection_status": status})
        if latency_ms is not None:
            state["last_tested_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
            state["latency_ms"] = latency_ms
        self._write_state(state)

    def delete(self) -> Dict[str, Any]:
        if self.credentials().source == "environment":
            raise ProviderSettingsError("This deployment is configured through environment variables and cannot be deleted here.")
        self._delete_keychain()
        self._write_state({"connection_status": "not_configured"})
        return self.public_status()

    def test_connection(self) -> Dict[str, Any]:
        credentials = self.credentials()
        if not credentials.api_key or not credentials.provider:
            return {"success": False, "message": "AI analysis requires an API key."}
        provider = PROVIDERS.get(credentials.provider)
        if not provider or not provider["supported"]:
            self._write_state({"provider": credentials.provider, "model": credentials.model, "connection_status": "not_configured"})
            return {"success": False, "message": "This provider is marked for future support."}
        try:
            response = requests.get(
                "https://api.openai.com/v1/models/{}".format(credentials.model or provider["model"]),
                headers={"Authorization": "Bearer {}".format(credentials.api_key)},
                timeout=15,
            )
        except requests.RequestException:
            return {"success": False, "message": "Could not reach the AI provider."}
        if response.status_code == 401:
            self._write_state({"provider": credentials.provider, "model": credentials.model, "connection_status": "invalid"})
            return {"success": False, "message": "Invalid API key."}
        if response.status_code == 429:
            self._write_state({"provider": credentials.provider, "model": credentials.model, "connection_status": "configured"})
            return {"success": False, "message": "OpenAI has no available API quota for this project. Add billing or use a funded API key."}
        if response.status_code == 403:
            self._write_state({"provider": credentials.provider, "model": credentials.model, "connection_status": "configured"})
            return {"success": False, "message": "The API key is valid but does not have access to the configured model."}
        if not response.ok:
            return {"success": False, "message": "Provider request failed. Check key permissions and model access."}
        self._write_state({"provider": credentials.provider, "model": credentials.model, "connection_status": "connected"})
        return {
            "success": True,
            "provider": provider["label"],
            "model": credentials.model or provider["model"],
            "message": "Connection successful.",
        }
