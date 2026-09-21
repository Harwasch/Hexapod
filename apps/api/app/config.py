"""Application settings.

Every value can be provided through the environment or a `.env` file at the
repository root. Secrets are never defaulted to real values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", Path(".env")),
        env_file_encoding="utf-8",
        extra="ignore",
        # Without this, pydantic-settings JSON-decodes complex fields (list[str]) inside the
        # dotenv source, before any validator runs -- so the comma-separated
        # API_CORS_ORIGINS that .env.example documents raises SettingsError and nothing can
        # import. `_split_origins` below is what is meant to parse it.
        enable_decoding=False,
    )

    app_name: str = "Digital Twin API"
    environment: str = Field(default="development", alias="APP_ENV")
    database_url: str = "postgresql+psycopg://twin:twin@localhost:5432/twin"
    test_database_url: str = "postgresql+psycopg://twin:twin@localhost:5432/twin_test"
    api_cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    # When false, user-supplied URLs that resolve to loopback/private hosts are rejected.
    allow_private_urls: bool = True

    object_storage_endpoint_url: str | None = None
    object_storage_bucket: str | None = None
    object_storage_access_key: str | None = None
    object_storage_secret_key: str | None = None
    object_storage_region: str = "us-east-1"
    object_storage_public_url: str | None = None

    cesium_ion_server_token: str | None = None
    cesium_ion_api_base: str = "https://api.cesium.com"

    # Mission planning agent. With a key the plan drafter calls Claude through the official
    # SDK; without one it falls back to a rule-based drafter and says so in every draft.
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5"

    # Local 3D Tiles served by the API at /api/v1/tiles (processed captures kept in the repo
    # or produced by a pipeline on this machine). Relative paths are from the repo root.
    tiles_dir: str = "data/tiles"
    # The URL the browser reaches the API at, for seeding absolute tileset URLs.
    public_api_base: str = "http://localhost:8000"

    api_host: str = "0.0.0.0"  # noqa: S104 - container default, documented in DEPLOYMENT.md
    api_port: int = 8000

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}

    @property
    def object_storage_configured(self) -> bool:
        return bool(
            self.object_storage_endpoint_url
            and self.object_storage_bucket
            and self.object_storage_access_key
            and self.object_storage_secret_key
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
