"""Read-only worker readiness check. Never provisions compute or starts inference.

WORLD_GATEWAY_TOKEN=... python preflight.py --url https://worker.example
Only GET /health is requested. --check-config inspects operator configuration
without contacting providers or printing credentials or URLs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class WorkerError(Exception):
    """A controlled error code; upstream URLs, bodies and exceptions stay private."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def validate_url(value: str) -> str:
    if not isinstance(value, str) or any(ord(char) < 33 for char in value):
        raise WorkerError("invalid_worker_url")
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        raise WorkerError("invalid_worker_url") from None
    if (
        not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise WorkerError("invalid_worker_url")
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
        raise WorkerError("remote_worker_requires_https")
    return value.rstrip("/")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a bearer token to a redirect target, even on the same host.
        return None


class WorkerClient:
    def __init__(self, url: str, token: str, timeout: float = 10):
        self.url = validate_url(url)
        if (
            not isinstance(token, str)
            or len(token) < 32
            or not token.isascii()
            or any(not 33 <= ord(char) <= 126 for char in token)
        ):
            raise WorkerError("missing_or_invalid_worker_token")
        if not 0 < timeout <= 120:
            raise WorkerError("invalid_timeout")
        self.token, self.timeout = token, timeout
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, method: str, path: str, data: dict[str, Any] | None = None):
        if (
            not path.startswith("/")
            or path.startswith("//")
            or "?" in path
            or "#" in path
        ):
            raise WorkerError("invalid_request_path")
        headers = {
            "Authorization": "Bearer " + self.token,
            "Accept": "application/json, image/jpeg",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.url + path,
            method=method,
            headers=headers,
            data=None if data is None else json.dumps(data).encode(),
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                limit = 16 * 1024 * 1024 if path.endswith("/frame") else 1024 * 1024
                body = response.read(limit + 1)
                if len(body) > limit:
                    raise WorkerError("worker_response_too_large")
                content_type = (
                    response.headers.get("Content-Type", "").split(";")[0].lower()
                )
                if body and content_type == "application/json":
                    try:
                        body = json.loads(body)
                    except (ValueError, UnicodeError):
                        raise WorkerError("invalid_worker_json") from None
                return response.status, response.headers, body
        except urllib.error.HTTPError as error:
            code = (
                "worker_redirect_refused"
                if 300 <= error.code < 400
                else (
                    "worker_authentication_failed"
                    if error.code in {401, 403}
                    else f"worker_http_{error.code}"
                )
            )
            raise WorkerError(code) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise WorkerError("worker_unreachable_or_tls_failed") from None


def inspect_health(health: Any, model_id: str, token: str = "") -> dict[str, Any]:
    if not isinstance(health, dict) or not isinstance(health.get("models"), list):
        raise WorkerError("invalid_worker_health")
    if health.get("protocolVersion") != 1:
        raise WorkerError("unsupported_worker_protocol")
    model = next(
        (
            item
            for item in health["models"]
            if isinstance(item, dict) and item.get("id") == model_id
        ),
        None,
    )
    if model is None:
        raise WorkerError("selected_model_not_installed")
    ready = (
        health.get("status") in ("ready", "ok", "healthy")
        and model.get("status") == "ready"
    )
    result = {
        "id": model_id,
        "ready": ready,
        "verification": "worker_reported_readiness_without_inference",
        "gpuInferenceVerified": False,
    }
    for source, target in (
        ("version", "sourceRevision"),
        ("checkpointRevision", "checkpointRevision"),
    ):
        value = model.get(source)
        if (
            isinstance(value, str)
            and re.fullmatch(r"[a-fA-F0-9]{7,64}", value)
            and not (token and token in value)
        ):
            result[target] = value
    for key in ("readiness", "servingMode"):
        value = model.get(key)
        if isinstance(value, str) and value in {
            "artifacts-verified-gpu-unverified",
            "resident-chunks",
            "chunked-visual-continuation",
        }:
            result[key] = value
    if not ready:
        result["failure"] = "selected_model_unavailable"
    return result


def configuration_checks(environment: Mapping[str, str]) -> dict[str, Any]:
    """Match control-plane config guards; never probe provider billing or mount state."""
    invalid = []

    def flag(name, default=False):
        value = environment.get(name, str(default)).strip().lower()
        if value not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            invalid.append(name)
        return value in {"true", "1", "yes", "on"}

    def safe_url(name):
        value = environment.get(name, "")
        if not value:
            return "not_configured"
        try:
            validate_url(value)
            if (
                name in {"WORLD_RUNPOD_GATEWAY_URL", "WORLD_MODAL_GATEWAY_URL"}
                and urlsplit(value).scheme != "https"
            ):
                raise WorkerError("remote_worker_requires_https")
            return "valid"
        except WorkerError:
            invalid.append(name)
            return "invalid"

    enabled = flag("WORLD_RUNPOD_ALLOW_PROVISION")
    lifecycle = flag("WORLD_LIFECYCLE_ENABLED", True)
    persistent = flag("WORLD_DATA_PERSISTENT")
    absolute_data = Path(environment.get("WORLD_DATA_DIR", "data/worlds")).is_absolute()
    token = environment.get("WORLD_GATEWAY_TOKEN", "")
    token_valid = (
        len(token) >= 32
        and token.isascii()
        and all(33 <= ord(char) <= 126 for char in token)
    )
    try:
        cap = float(environment.get("WORLD_MAX_WORKER_HOURLY_COST", "nan"))
        cap_valid = math.isfinite(cap) and 0 < cap <= 100
    except ValueError:
        cap_valid = False
    if environment.get("WORLD_MAX_WORKER_HOURLY_COST") and not cap_valid:
        invalid.append("WORLD_MAX_WORKER_HOURLY_COST")
    bounds = {
        "WORLD_RUNPOD_PORT": (1, 65535, 8789),
        "WORLD_MAX_MANAGED_WORKERS": (1, 8, 1),
        "WORLD_SESSION_LEASE_SECONDS": (45, 600, 90),
        "WORLD_WORKER_IDLE_SECONDS": (60, 1800, 300),
        "WORLD_WORKER_STARTUP_SECONDS": (60, 1800, 900),
        "WORLD_WORKER_MAX_LIFETIME_SECONDS": (120, 14400, 3600),
        "WORLD_REAPER_INTERVAL_SECONDS": (5, 60, 15),
    }
    for name, (minimum, maximum, default) in bounds.items():
        try:
            value = int(environment.get(name, str(default)))
            if not minimum <= value <= maximum:
                invalid.append(name)
        except ValueError:
            invalid.append(name)
    guards = {
        "lifecycleEnabled": lifecycle,
        "persistentStorageDeclared": persistent,
        "dataDirectoryAbsolute": absolute_data,
        "hourlyCostCapConfigured": cap_valid,
        "gatewayTokenValid": token_valid,
        "providerKeyPresent": bool(environment.get("WORLD_RUNPOD_API_KEY")),
        "approvedTemplatePresent": bool(environment.get("WORLD_RUNPOD_TEMPLATE_ID")),
        "apiAuthenticationConfigured": bool(environment.get("API_WRITE_TOKEN")),
    }
    complete = all(guards.values()) and not invalid
    result = {
        "scope": "process_environment_only_no_dotenv_loaded",
        "runpod": {
            "gateway": safe_url("WORLD_RUNPOD_GATEWAY_URL"),
            "provisioningEnabled": enabled,
            "provisioningConfiguration": "complete"
            if enabled and complete
            else "incomplete"
            if enabled
            else "disabled",
            "guards": guards,
            "providerCredentialsVerified": False,
            "persistentVolumeVerified": False,
            "lifecycleReaperRunningVerified": False,
        },
        "local": {"gateway": safe_url("WORLD_LOCAL_GATEWAY_URL")},
        "modal": {"gateway": safe_url("WORLD_MODAL_GATEWAY_URL")},
        "reconstruction": {"gateway": safe_url("WORLD_RECONSTRUCTION_GATEWAY_URL")},
    }
    # Existing gateways bypass pod provisioning, but still need both layers of
    # authentication. Do not let disabled provisioning mask an unusable setup.
    if any(environment.get(name) for name in (
        "WORLD_LOCAL_GATEWAY_URL", "WORLD_RUNPOD_GATEWAY_URL", "WORLD_MODAL_GATEWAY_URL"
    )):
        if not token_valid:
            invalid.append("WORLD_GATEWAY_TOKEN")
        if not environment.get("API_WRITE_TOKEN", "").strip():
            invalid.append("API_WRITE_TOKEN")
    reconstruction_url = environment.get("WORLD_RECONSTRUCTION_GATEWAY_URL", "")
    if reconstruction_url and result["reconstruction"]["gateway"] == "valid":
        remote = urlsplit(reconstruction_url).hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }
        if (
            remote
            and not environment.get("WORLD_RECONSTRUCTION_GATEWAY_TOKEN", "").strip()
        ):
            invalid.append("WORLD_RECONSTRUCTION_GATEWAY_TOKEN")
    result["invalidFields"] = invalid
    result["ready"] = not invalid and not (enabled and not complete)
    if enabled and invalid:
        result["runpod"]["provisioningConfiguration"] = "incomplete"
    return result


def run_preflight(
    url: str,
    token: str,
    model_id: str = "astronex-world",
    *,
    timeout: float = 10,
    environment: Mapping[str, str] | None = None,
    client_factory=WorkerClient,
) -> dict[str, Any]:
    report = {
        "schemaVersion": 1,
        "kind": "worlds-worker-preflight",
        "readOnly": True,
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "ready": False,
        "inferenceStarted": False,
        "computeProvisioned": False,
        "checks": [],
    }
    if environment is not None:
        report["configuration"] = configuration_checks(environment)
    try:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", model_id):
            raise WorkerError("invalid_model_id")
        client = client_factory(url, token, timeout=timeout)
        status, _headers, health = client.request("GET", "/health")
        if status != 200:
            raise WorkerError("invalid_worker_health_status")
        report["checks"].append({"name": "authenticated_health", "passed": True})
        report["model"] = inspect_health(health, model_id, token)
        report["ready"] = report["model"]["ready"]
        report["checks"].append(
            {"name": "selected_model_readiness", "passed": report["ready"]}
        )
        if not report["ready"]:
            report["failure"] = "selected_model_unavailable"
        if environment is not None and not report["configuration"]["ready"]:
            report["ready"] = False
            report.setdefault("failure", "operator_configuration_incomplete")
    except WorkerError as error:
        report["failure"] = error.code
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("WORLD_RUNPOD_GATEWAY_URL")
        or os.environ.get("WORLD_LOCAL_GATEWAY_URL", ""),
    )
    parser.add_argument("--model", default="astronex-world")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_preflight(
        args.url,
        os.environ.get("WORLD_GATEWAY_TOKEN", ""),
        args.model,
        timeout=args.timeout,
        environment=os.environ if args.check_config else None,
    )
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        try:
            args.output.write_text(serialized)
        except OSError:
            report["ready"] = False
            report["failure"] = "report_output_unwritable"
            serialized = json.dumps(report, indent=2) + "\n"
    print(serialized, end="")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
