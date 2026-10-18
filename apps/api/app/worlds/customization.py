"""Optional proxy to an already configured training host. Never provisions compute."""

from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.api.deps import RequireWriteToken
from app.config import REPO_ROOT
from app.worlds.config import validate_url
from app.worlds.providers import request, response_json
from app.worlds.router import BoundedWorldsRoute

router = APIRouter(
    prefix="/worlds/customization",
    tags=["worlds customization"],
    dependencies=[RequireWriteToken],
    route_class=BoundedWorldsRoute,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WORLD_CUSTOMIZATION_", extra="ignore", env_file=REPO_ROOT / ".env"
    )
    gateway_url: str | None = None
    gateway_token: SecretStr | None = None

    @classmethod
    def load(cls) -> Settings:
        path = os.getenv("WORLD_ENV_FILE")
        return cls(_env_file=path) if path else cls()


class Profile(BaseModel):
    id: str
    name: str
    modelId: Literal["forge-wm", "sana-wm"]  # noqa: N815
    stage: str
    devices: Literal[8]
    datasetLabel: str  # noqa: N815
    activationCompatible: bool  # noqa: N815


class Capabilities(BaseModel):
    configured: bool
    trainingEnabled: bool = False  # noqa: N815
    maxTrainingSeconds: int = 0  # noqa: N815
    maxConcurrentJobs: int = 0  # noqa: N815
    profiles: list[Profile] = Field(default_factory=list)
    message: str


class Artifact(BaseModel):
    id: str
    name: str
    size: int = Field(ge=0)


class Job(BaseModel):
    id: str
    profileId: str  # noqa: N815
    modelId: str  # noqa: N815
    stage: str
    status: Literal[
        "prepared",
        "running",
        "cancelling",
        "cancelled",
        "completed",
        "failed",
        "unknown",
        "interrupted",
    ]
    createdAt: float  # noqa: N815
    startedAt: float | None = None  # noqa: N815
    finishedAt: float | None = None  # noqa: N815
    maxTrainingSeconds: int  # noqa: N815
    activationCompatible: bool  # noqa: N815
    gpuValidated: bool = False  # noqa: N815
    artifacts: list[Artifact] = Field(default_factory=list)
    installed: list[str] = Field(default_factory=list)
    selectedArtifactId: str | None = None  # noqa: N815
    message: str = ""


class JobList(BaseModel):
    jobs: list[Job]


class Prepare(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profileId: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")  # noqa: N815


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirmTraining: Literal[True]  # noqa: N815


class Select(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifactId: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")  # noqa: N815


def forward(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    config = Settings.load()
    if not config.gateway_url or not config.gateway_token:
        raise HTTPException(
            503,
            "Configure WORLD_CUSTOMIZATION_GATEWAY_URL and "
            "WORLD_CUSTOMIZATION_GATEWAY_TOKEN on the session manager.",
        )
    try:
        validate_url(config.gateway_url, local=True)
    except ValueError:
        raise HTTPException(
            503, "Customization gateway requires HTTPS or local loopback HTTP."
        ) from None
    value = response_json(
        request(
            method,
            config.gateway_url.rstrip("/") + path,
            token=config.gateway_token.get_secret_value(),
            payload=payload,
            timeout=30,
        )
    )
    return value


def job_id(identity: str) -> str:
    import re

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", identity):
        raise HTTPException(422, "Invalid customization job ID.")
    return identity


def parsed_job(value: Any) -> Job:
    try:
        return Job.model_validate(value)
    except ValidationError:
        raise HTTPException(
            502, "The customization gateway returned an invalid job record."
        ) from None


@router.get("/capabilities")
def capabilities() -> Capabilities:
    config = Settings.load()
    if not config.gateway_url or not config.gateway_token:
        return Capabilities(
            configured=False,
            message=(
                "Connect an existing customization worker with operator-approved local dataset "
                "and recipe profiles. No generic LoRA adapter or cloud allocation is provided."
            ),
        )
    try:
        return Capabilities.model_validate(forward("GET", "/capabilities"))
    except ValidationError:
        raise HTTPException(
            502, "The customization gateway returned invalid capabilities."
        ) from None


@router.get("/jobs")
def jobs() -> JobList:
    try:
        return JobList.model_validate(forward("GET", "/jobs"))
    except ValidationError:
        raise HTTPException(502, "The customization gateway returned invalid jobs.") from None


@router.post("/jobs")
def prepare(body: Prepare) -> Job:
    return parsed_job(forward("POST", "/jobs", body.model_dump()))


@router.get("/jobs/{identity}")
def status(identity: str) -> Job:
    return parsed_job(forward("GET", f"/jobs/{job_id(identity)}"))


@router.post("/jobs/{identity}/run")
def run(identity: str, body: Run) -> Job:
    return parsed_job(forward("POST", f"/jobs/{job_id(identity)}/run", body.model_dump()))


@router.post("/jobs/{identity}/cancel")
def cancel(identity: str) -> Job:
    return parsed_job(forward("POST", f"/jobs/{job_id(identity)}/cancel", {}))


@router.post("/jobs/{identity}/install")
def install(identity: str, body: Select) -> Job:
    return parsed_job(forward("POST", f"/jobs/{job_id(identity)}/install", body.model_dump()))


@router.post("/jobs/{identity}/enable")
def enable(identity: str, body: Select) -> Job:
    return parsed_job(forward("POST", f"/jobs/{job_id(identity)}/enable", body.model_dump()))


@router.post("/jobs/{identity}/disable")
def disable(identity: str) -> Job:
    return parsed_job(forward("POST", f"/jobs/{job_id(identity)}/disable", {}))
