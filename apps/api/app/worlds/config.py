from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config import REPO_ROOT


class ModelProfile(BaseModel):
    """Operator-approved runtime selection; never accepted in browser requests."""

    model_config = ConfigDict(extra="forbid")
    runpod_template_id: str | None = None
    runpod_gpu_type: str | None = None
    local_gateway_url: str | None = None
    runpod_gateway_url: str | None = None
    modal_gateway_url: str | None = None
    lambda_gateway_url: str | None = None
    modal_image: str | None = None
    modal_gpu_type: str | None = None
    modal_hourly_cost: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    lambda_instance_type: str | None = None
    lambda_image_id: str | None = None
    lambda_bootstrap_file: Path | None = None


class WorldsSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WORLD_", extra="ignore", env_file=REPO_ROOT / ".env", env_file_encoding="utf-8"
    )

    data_dir: Path = Path("data/worlds")
    local_gateway_url: str | None = None
    runpod_gateway_url: str | None = None
    modal_gateway_url: str | None = None
    lambda_gateway_url: str | None = None
    model_id: str = "astronex-world"
    model_profiles_json: dict[str, ModelProfile] = Field(default_factory=dict, max_length=8)
    lambda_api_key: SecretStr | None = None
    lambda_allow_provision: bool = False
    lambda_instance_type: str = "gpu_1x_a100_sxm4"
    lambda_region: str | None = None
    lambda_ssh_key_names: list[str] = Field(default_factory=list)
    lambda_image_id: str | None = None
    lambda_bootstrap_file: Path | None = None
    lambda_gateway_template: str | None = None
    modal_allow_provision: bool = False
    modal_token_id: SecretStr | None = None
    modal_token_secret: SecretStr | None = None
    modal_app_name: str = "hexapod-worlds"
    modal_image: str | None = None
    modal_gpu_type: str = "L40S"
    modal_hourly_cost: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    gateway_token: SecretStr | None = None
    runpod_api_key: SecretStr | None = None
    runpod_allow_provision: bool = False
    runpod_template_id: str | None = None
    runpod_gpu_type: str = "NVIDIA L40S"
    runpod_port: int = Field(default=8789, ge=1, le=65535)
    lifecycle_enabled: bool = True
    session_lease_seconds: int = Field(default=90, ge=45, le=600)
    worker_idle_seconds: int = Field(default=300, ge=60, le=1800)
    worker_startup_seconds: int = Field(default=900, ge=60, le=1800)
    worker_max_lifetime_seconds: int = Field(default=3600, ge=120, le=14400)
    reaper_interval_seconds: int = Field(default=15, ge=5, le=60)
    max_managed_workers: int = Field(default=1, ge=1, le=8)
    max_worker_hourly_cost: float | None = Field(default=None, gt=0, le=100, allow_inf_nan=False)
    data_persistent: bool = False

    @classmethod
    def load(cls) -> WorldsSettings:
        # A dedicated file is opt-in; process environment takes precedence over it.
        path = os.getenv("WORLD_ENV_FILE")
        if path and not Path(path).is_file():
            raise RuntimeError("WORLD_ENV_FILE must name an existing readable configuration file.")
        return cls(_env_file=path) if path else cls()

    def for_model(self, model_id: str) -> WorldsSettings:
        if model_id == self.model_id:
            profile = self.model_profiles_json.get(model_id)
        else:
            profile = self.model_profiles_json.get(model_id)
            if profile is None:
                raise ValueError("Model has no operator-approved runtime profile")
        values = profile.model_dump(exclude_none=True) if profile else {}
        if model_id != self.model_id:
            # A model may never inherit a different model's image/template or gateway.
            values = {
                **{
                    key: None
                    for key in (
                        "runpod_template_id",
                        "modal_image",
                        "lambda_image_id",
                        "lambda_bootstrap_file",
                        "local_gateway_url",
                        "runpod_gateway_url",
                        "modal_gateway_url",
                        "lambda_gateway_url",
                    )
                },
                **values,
            }
        return self.model_copy(update={**values, "model_id": model_id})

    def managed(self, provider: str) -> bool:
        return bool(getattr(self, f"{provider}_allow_provision", False)) and not bool(
            getattr(self, f"{provider}_gateway_url", None)
        )

    def gpu_type(self, provider: str) -> str | None:
        return getattr(self, f"{provider}_gpu_type", None) or (
            self.lambda_instance_type if provider == "lambda" else None
        )

    def provisioning_problem(self, provider: str = "runpod") -> str | None:
        if not self.lifecycle_enabled:
            return "Managed provisioning requires the server lifecycle reaper."
        if not self.data_persistent or not self.data_dir.is_absolute():
            return "Set WORLD_DATA_DIR to an absolute persistent volume and WORLD_DATA_PERSISTENT=true."
        if self.max_worker_hourly_cost is None:
            return "Set WORLD_MAX_WORKER_HOURLY_COST before enabling paid provisioning."
        if provider == "runpod" and (not self.runpod_api_key or not self.runpod_template_id):
            return "Configure an approved RunPod template and server API key."
        if provider == "lambda" and not all(
            (
                self.lambda_api_key,
                self.lambda_region,
                self.lambda_ssh_key_names,
                self.lambda_image_id,
                self.lambda_bootstrap_file,
                self.lambda_gateway_template,
            )
        ):
            return "Lambda requires key, region, SSH keys, image, bootstrap file and HTTPS gateway template."
        if provider == "modal" and not all(
            (self.modal_token_id, self.modal_token_secret, self.modal_image, self.modal_hourly_cost)
        ):
            return "Modal requires dedicated credentials, approved registry image and operator hourly estimate."
        if (
            provider == "modal"
            and self.modal_hourly_cost
            and self.modal_hourly_cost > self.max_worker_hourly_cost
        ):
            return "Modal operator hourly estimate exceeds the configured cost ceiling."
        if not self.gateway_token or len(self.gateway_token.get_secret_value()) < 32:
            return "Hosted workers require a WORLD_GATEWAY_TOKEN with at least 32 characters."
        return None

    @property
    def lifecycle_needed(self) -> bool:
        return (
            self.runpod_allow_provision
            or self.lambda_allow_provision
            or self.modal_allow_provision
            or bool(self.model_profiles_json)
            or any(
                (
                    self.local_gateway_url,
                    self.runpod_gateway_url,
                    self.modal_gateway_url,
                    self.lambda_gateway_url,
                )
            )
            or (self.data_dir / "metadata.sqlite3").exists()
        )

    def gateway_url(self, provider: str) -> str | None:
        value = getattr(self, f"{provider}_gateway_url", None)
        if value:
            validate_url(value, local=provider == "local")
        return value


def validate_url(value: str, *, local: bool = False) -> None:
    """Only operator configured targets; no credentials or query-string tokens."""
    if any(char.isspace() or ord(char) < 32 for char in value):
        raise ValueError("Gateway URL cannot contain whitespace or control characters")
    parsed = urlsplit(value)
    # Reading .port validates syntax and range; httpx otherwise raises InvalidURL
    # outside its HTTPError hierarchy, producing an opaque 500 from a config typo.
    _ = parsed.port
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Gateway URL must contain a host and no credentials, query, or fragment")
    loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (local and loopback and parsed.scheme == "http"):
        raise ValueError("Remote gateways require HTTPS; HTTP is allowed only on local loopback")
