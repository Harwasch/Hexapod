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
        # import. `_split_csv` below is what is meant to parse it.
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

    # The single shared write token (the plan's `API_WRITE_TOKEN`). Unset means this
    # deployment is open for writes, which is how a fresh checkout runs with no
    # configuration at all -- the same way `object_storage_configured` degrades. That is
    # only tolerable outside production, so `create_app` refuses to start when
    # `is_production` and this is empty; see app/main.py.
    api_write_token: str | None = None

    # Signing key for the phone-handoff tokens (app/services/handoff.py). Unset falls back
    # to deriving one from `api_write_token`, and failing that to a random per-process key
    # -- see `handoff.key_for`. Set it explicitly when the API runs as more than one
    # process, or a token minted by one will be refused by the next.
    api_handoff_secret: str | None = None

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
    # The origin the *web app* is served from, for the phone-handoff URL a QR code encodes.
    # localhost is right for development and useless to a real phone, which is the point at
    # which a deployment has to set this.
    public_web_base: str = "http://localhost:5173"

    # --- The worker (app/worker) ----------------------------------------------------
    # Where a run's workdir lives. It survives the run: "retry from this stage" reads the
    # outputs of the stages that already succeeded out of it, and A6's `checkpoint/`
    # contract is only worth anything while the directory is still there.
    worker_workdir: str = "var/worker"
    # "stub" or "local". "local" since A8: Lane 1's stages are real, so a dropped `.ply`
    # or `.spz` becomes a site with no human step, and a worker that ran the stub by
    # default would produce sites pointing at fabricated tilesets. Lane 2 still needs a
    # GPU runner it does not have, so `photo-reconstruct` fails at `train` with a message
    # naming the tier it wanted -- which is the honest answer until B1.
    worker_runner: str = "local"
    # Extra recipes, by name, for tests and for a deployment that ships its own. The
    # pipeline's shipped recipes are found without this.
    worker_recipe_dir: str | None = None
    # Modules the recipe process imports before planning, so their `@stage_impl`s are
    # registered. Comma-separated, like API_CORS_ORIGINS. A deployment that carries its
    # own stages names them here rather than editing the pipeline.
    worker_impl_modules: list[str] = Field(default_factory=list)
    # A0 #2: the claim commits immediately and holds a lease; a worker pushes the expiry
    # forward while it supervises the run. 30 s against a 2 s heartbeat tolerates fourteen
    # missed beats before another worker may take the job.
    worker_lease_s: float = 30.0
    # How often the supervisor beats: it renews the lease, notices a cancellation and
    # drains the running recipe's progress on the same tick. Also the worst-case latency
    # of `POST /jobs/{id}/cancel`.
    worker_poll_s: float = 2.0
    # How long to wait before asking for work again when the queue was empty.
    worker_idle_s: float = 2.0
    # Attempts per stage before the job is dead-lettered. Counts crashes and reclaims as
    # well as ordinary failures, because a job that kills its worker would otherwise be
    # picked up forever by whoever is next.
    worker_max_attempts: int = 3
    # Pause between attempts at the same stage.
    worker_retry_backoff_s: float = 2.0

    api_host: str = "0.0.0.0"  # noqa: S104 - container default, documented in DEPLOYMENT.md
    api_port: int = 8000

    @field_validator("api_cors_origins", "worker_impl_modules", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
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
