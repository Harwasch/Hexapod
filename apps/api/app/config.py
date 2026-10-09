"""Application settings.

Every value can be provided through the environment or a `.env` file at the
repository root. Secrets are never defaulted to real values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def repo_root_for(config_module: PurePosixPath | Path) -> Path:
    """The deployment root, given where this module sits.

    `<repo>/apps/api/app/config.py` -> the repository root, four levels up. Only
    development conveniences hang off it: the root `.env`, and a relative `tiles_dir`
    served by the development-only static mount.

    The fallback is not defensive padding. infra/api.Dockerfile is `WORKDIR /app` +
    `COPY apps/api/ ./`, so in the image this file is `/app/app/config.py` -- which has
    three parents, so `parents[3]` raised `IndexError: 3` **at import**. That image could
    never start: `import app.main` failed before a route or a setting was read, a harder
    failure than the tiles 404 A9 set out to remove, and one that hid it entirely. Found
    by reproducing the layout rather than by reading the Dockerfile. In the image this
    returns `/app`, which is what a deployment root means there.
    """
    parents = config_module.parents
    chosen = parents[3] if len(parents) > 3 else parents[1]
    return Path(chosen)


REPO_ROOT = repo_root_for(Path(__file__).resolve())


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
    # The second bucket: the only one the world can read. Unset means one bucket in both
    # roles, which is how a fresh checkout and the dev MinIO loop run -- and which
    # `create_app` refuses in production, because enabling R2's public access exposes a
    # whole bucket and this one would hold every raw upload. See app/worker/publish.py.
    object_storage_public_bucket: str | None = None

    # The single shared write token (the plan's `API_WRITE_TOKEN`). Unset means this
    # deployment is open for writes, which is how a fresh checkout runs with no
    # configuration at all -- the same way `object_storage_configured` degrades. That is
    # only tolerable outside production, so `create_app` refuses to start when
    # `is_production` and this is empty; see app/main.py.
    api_write_token: str | None = None

    # OIDC workspaces and the legacy single-operator pilot are separate auth modes.
    land_document_workspace_quota_bytes: int = Field(
        default=1024 * 1024 * 1024, ge=20 * 1024 * 1024
    )
    land_auth_mode: Literal["pilot", "oidc"] = "pilot"
    land_oidc_issuer: str | None = None
    land_oidc_audience: str | None = None
    land_oidc_jwks_url: str | None = None

    @model_validator(mode="after")
    def validate_land_identity(self) -> Settings:
        if self.land_auth_mode == "oidc":
            if (
                not self.land_oidc_audience
                or not self.land_oidc_issuer
                or not self.land_oidc_jwks_url
            ):
                raise ValueError("OIDC land access requires issuer, audience and JWKS URL")
            if not self.land_oidc_issuer.startswith(
                "https://"
            ) or not self.land_oidc_jwks_url.startswith("https://"):
                raise ValueError("OIDC issuer and JWKS URL must use HTTPS")
        return self

    # Signing key for the phone-handoff tokens (app/services/handoff.py). Unset falls back
    # to deriving one from `api_write_token`, and failing that to a random per-process key
    # -- see `handoff.key_for`. Set it explicitly when the API runs as more than one
    # process, or a token minted by one will be refused by the next.
    api_handoff_secret: str | None = None
    # The phone key (app/services/phone_key.py): a short shared key typed once on a phone,
    # stored here only as `pbkdf2_sha256$<iterations>$<salt hex>$<digest hex>`. Unset
    # means the phone routes are closed. It unlocks less than the write token: creating a
    # phone capture, uploading to it and processing it.
    api_phone_key_hash: str | None = None
    # How many captures the phone key may create in any 24 hours.
    api_phone_daily_captures: int = 20
    # The request header holding the real client address, for the per-client rate limits
    # (app/services/ratelimit.py). Fly's proxy sets `Fly-Client-IP` on every request, so
    # behind Fly it cannot be forged; without it every client shares the proxy's address.
    # The default is believed only on Fly (FLY_APP_NAME set); elsewhere anyone could send
    # it, and the socket's peer is used. Name another proxy's header to believe that one,
    # or set it empty to always use the socket's peer. An IPv6 client is its /64.
    api_client_ip_header: str = "Fly-Client-IP"

    cesium_ion_server_token: str | None = None
    cesium_ion_api_base: str = "https://api.cesium.com"

    # Mission planning agent. With a key the plan drafter calls Claude through the official
    # SDK; without one it falls back to a rule-based drafter and says so in every draft.
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5"

    # Local 3D Tiles served by the API at /api/v1/tiles. **Development only.** The
    # container image built from infra/api.Dockerfile copies apps/api/ and the two
    # tools/ projects the worker runs, and no data/ at all, so this directory does not
    # exist there and the mount is simply absent -- which is why a deployment's tiles come
    # out of object storage (see `tiles_base_url`) and not from here. Relative paths are
    # from the repo root.
    tiles_dir: str = "data/tiles"
    # Where a capture's published tiles are served from, as a URL prefix that a slug is
    # appended to: `<tiles_base_url>/<slug>/splat/tileset.json`.
    #
    # Unset, it is derived by app/seed/captures.py::tiles_base_url, whose rule is: **a
    # capture is served from the checkout when the checkout has it, and from the bucket
    # otherwise.** Production never reads the checkout; it does not have one. Set this to
    # put a CDN in front of the bucket -- R2's custom domain, which is not the S3 API domain
    # that presigning uses (OBJECT_STORAGE_PUBLIC_URL).
    tiles_base_url: str | None = None
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
    # "stub", "local" or "cloud". "local" since A8: Lane 1's stages are real, so a
    # dropped `.ply` or `.spz` becomes a site with no human step, and a worker that ran
    # the stub by default would produce sites pointing at fabricated tilesets. "cloud"
    # (B1b) adds a GPU runner for the stages that declare `gpu:`, so Lane 2's `train`
    # has somewhere to go; with it unset `photo-reconstruct` still fails at `train` with
    # a message naming the tier it wanted, which stays the honest default.
    worker_runner: str = "local"
    # Where a GPU stage is dispatched, preferred first. Comma-separated, and the **last
    # one is the fallback**: a stage that keeps being preempted is moved down the list
    # rather than retried on the cheap host until the cheap host has cost more. The last
    # entry must not be interruptible, which `Placement.of` enforces at startup.
    # Known names: fake (tests), subprocess (a box you have a shell on), modal (a sketch
    # that has never run -- see tools/pipeline/modal_adapter.py).
    worker_cloud_providers: list[str] = Field(default_factory=list)
    # How many preemptions of one stage before it moves to the next provider.
    worker_preemptions_before_fallback: int = 2
    # The Modal app ModalAdapter looks `run_stage_<tier>` up in: the name infra/modal/app.py
    # deploys under. It defaulted to empty and the adapter then fell back to "twin", which
    # is not the app's name, so the first GPU dispatch would have failed on a lookup.
    worker_modal_app: str = "twin-pipeline"
    # How often a dispatched stage is polled, and how often the machine running it is
    # asked to sync `checkpoint/` back. The second is the one that decides how much work
    # a preemption throws away.
    worker_cloud_poll_s: float = 5.0
    worker_checkpoint_every_s: float = 60.0
    # A directory both the worker and the machine running a dispatched stage can see --
    # an NFS mount, or a workstation whose GPU is in the same box. Unset, a dispatched
    # stage's inputs, checkpoint and outputs move through the bucket.
    worker_cloud_transfer_dir: str | None = None
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
    # Attempts a stage gets on top of `worker_max_attempts` for having been preempted.
    # Being taken off a cheap interruptible box is not the stage failing, so it does not
    # spend the budget meant for one that is -- but the ceiling is still hard.
    worker_max_preemptions: int = 4
    # The most one job may be billed, across every stage and attempt, in dollars: no
    # remote call starts past it and a running one is cancelled when its cost would go
    # over it, and the job is dead-lettered saying so. A typical Lane 2 run is a dollar
    # or two (an L4 hour is ~$0.96 with its reservation); 20 is a runaway guard, not a
    # budget. 0 turns it off.
    worker_job_cost_cap_usd: float = 20.0
    # A remote call whose trainer stops printing progress is cancelled as timed out once
    # it has run this many times what its newest progress line projected (plus 30 min).
    # 0 turns it off; the provider's own limit (six hours on Modal) still applies.
    worker_deadline_factor: float = 2.0
    # A healthchecks.io-style URL pinged for active runs only: `<url>/start` when a job
    # is claimed and every minute while it runs, `<url>` when it succeeds, `<url>/fail`
    # when it fails. Nothing while idle, so the check wants a long period (30 days) and a
    # grace of a few minutes (5): app/worker/alerts.py. Unset, nothing is pinged.
    worker_heartbeat_url: str | None = None
    # Below this many GB free on the workdir's volume the worker does not claim, says so
    # in its log, and evicts finished runs' workdirs older than `worker_evict_after_days`
    # to make room. 0 turns the check off.
    worker_min_free_gb: float = 5.0
    worker_evict_after_days: float = 7.0
    # Pause between attempts at the same stage.
    worker_retry_backoff_s: float = 2.0
    # Jobs one worker process supervises at once, each in its own slot with its own claim,
    # lease, heartbeat and recipe process (app/worker/loop.py). 1 until raised: the
    # worker's README gives the memory each slot costs on the 2 GB machine.
    worker_concurrency: int = Field(default=1, ge=1, le=8)
    # Slots on top of those that only claim a recipe with no `gpu:` stage -- today
    # `splat-ingest` -- so a one-minute ingest does not wait behind a two-hour training
    # run, and two training runs (two GPUs billed, two videos on the 20 GB volume) never
    # share the machine. The worker reads the recipes at start-up to know which qualify.
    worker_cpu_only_slots: int = Field(default=0, ge=0, le=4)
    # An idle worker polls every `worker_idle_s` for this long, then backs off -- doubling
    # a period at a time -- to `worker_idle_max_s`. A queue checked every 2 s forever is a
    # database that never scales to zero.
    worker_idle_backoff_after_s: float = 60.0
    worker_idle_max_s: float = 30.0
    # With nothing running and nothing claimed for this long, the worker exits 0 and its
    # machine stops (fly.toml restarts it only on failure); the API starts it again when
    # it queues a job (app/services/worker_wake.py). 0 polls forever -- which is what a
    # checkout wants, having nothing to start it again: .env.example sets 0.
    worker_idle_exit_s: float = Field(default=900.0, ge=0)

    # --- Waking the worker (app/services/worker_wake.py) ----------------------------
    # A Fly token that may start this app's machines (`fly tokens create deploy`). Unset,
    # queueing a job wakes nothing, which is right for development: there is no machine.
    fly_api_token: str | None = None
    # Set by Fly on every machine; the app whose `worker` machines a new job starts.
    fly_app_name: str | None = None
    # A healthchecks.io-style check URL. Queueing a job onto an idle worker pings
    # `<url>/start` (not while one is running: the job waits for it); the worker pings
    # `<url>` when it claims one, so a job queued and never claimed raises an alert. A long
    # period (30 days), a grace longer than a cold start (10 min). A secret: logs redact it.
    queue_check_url: str | None = None

    api_host: str = "0.0.0.0"  # noqa: S104 - container default, documented in DEPLOYMENT.md
    api_port: int = 8000

    # --- Observability (app/observability.py) ---------------------------------------
    # The level of the API's own `twin.*` loggers; libraries stay at WARNING.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    # `json` (one object per line) or `text`. Unset: json in production, text elsewhere.
    log_format: Literal["json", "text"] | None = None
    # Error reporting. Unset, sentry-sdk is never imported.
    sentry_dsn: str | None = None
    # The share of requests traced for performance. 0 sends errors only.
    sentry_traces_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("log_level", "log_format", mode="before")
    @classmethod
    def _case_insensitive(cls, value: object, info: ValidationInfo) -> object:
        # `LOG_LEVEL=info` and `LOG_FORMAT=JSON` mean what they say; an empty value is unset.
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return "INFO" if info.field_name == "log_level" else None
            return value.upper() if info.field_name == "log_level" else value.lower()
        return value

    @field_validator(
        "api_cors_origins", "worker_impl_modules", "worker_cloud_providers", mode="before"
    )
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}

    @property
    def tiles_source_configured(self) -> bool:
        """Is there anywhere a browser could fetch a capture's tiles from?

        Either an explicit `TILES_BASE_URL` (a CDN, R2's custom domain) or a bucket whose
        public URL `app/seed/captures.tiles_base_url` can derive one from. With neither,
        that function falls back to this API's `/api/v1/tiles` static mount -- which
        production disables outright, and which the container image has no `data/tiles`
        to serve in any case, so every seeded tileset URL would 404.

        `create_app` refuses to start a production deployment in that state rather than
        logging it, for the same reason it refuses one with no write token: a
        misconfiguration that only shows up as a broken globe in someone's browser is
        worse than one that shows up as a deploy that would not go out.
        """
        return bool(self.tiles_base_url) or self.object_storage_configured

    @property
    def publish_bucket_configured(self) -> bool:
        """True where published tiles go somewhere other than the upload bucket."""
        return bool(
            self.object_storage_public_bucket
            and self.object_storage_public_bucket != self.object_storage_bucket
        )

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
