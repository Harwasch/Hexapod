"""Back up the actual Fly database privately and rehearse migrations on local PostGIS.

Run only in the explicitly dispatched release workflow. Never logs connection strings,
exports credentials as artifacts, changes the source DB, or allocates GPU resources.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"Required release setting is missing: {name}")
    return value


def run(args: list[str], **kwargs):
    try:
        return subprocess.run(args, check=True, text=True, **kwargs)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Release command {args[0]} failed with exit {error.returncode}; credential-bearing output withheld"
        ) from None


def neon(path: str):
    request = Request(
        "https://console.neon.tech/api/v2" + path,
        headers={
            "Authorization": "Bearer " + required("NEON_API_KEY"),
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def fingerprint(uri: str) -> dict[str, str]:
    parsed = urlsplit(uri)
    return {
        "host": (parsed.hostname or "").replace("-pooler.", "."),
        "database": parsed.path.lstrip("/"),
        "user": parsed.username or "",
    }


def main() -> None:
    config = tomllib.loads((ROOT / "fly.toml").read_text())
    app = config["app"]
    project_name = os.environ.get("NEON_PROJECT_NAME") or "hexapod-twin"
    print("Checking existing Neon project and source database identity", flush=True)
    projects = [
        p for p in neon("/projects?limit=400")["projects"] if p["name"] == project_name
    ]
    if len(projects) != 1:
        raise RuntimeError("Expected exactly one existing Neon project; creating none")
    project = projects[0]["id"]
    branches = [
        b for b in neon(f"/projects/{project}/branches")["branches"] if b.get("default")
    ]
    if len(branches) != 1:
        raise RuntimeError("Expected exactly one default database branch")
    branch = branches[0]["id"]
    databases = neon(f"/projects/{project}/branches/{branch}/databases")["databases"]
    if len(databases) != 1:
        raise RuntimeError("Database selection is ambiguous; refusing to guess")
    database = databases[0]
    query = urlencode(
        {
            "branch_id": branch,
            "database_name": database["name"],
            "role_name": database["owner_name"],
        }
    )
    uri = neon(f"/projects/{project}/connection_uri?{query}")["uri"]
    if not uri or "\n" in uri:
        raise RuntimeError("Invalid database connection response")
    print("::add-mask::" + uri, flush=True)
    machines = json.loads(
        run(
            ["flyctl", "machines", "list", "--app", app, "--json"], capture_output=True
        ).stdout
    )
    api = [
        m
        for m in machines
        if m.get("config", {}).get("metadata", {}).get("fly_process_group") == "app"
    ]
    if len(api) != 1:
        raise RuntimeError(
            "Worlds release requires exactly one existing API machine; refusing to rescale implicitly"
        )
    if api[0]["state"] != "started":
        run(
            ["flyctl", "machine", "start", api[0]["id"], "--app", app],
            capture_output=True,
        )
    print("Verifying the running Fly API database identity", flush=True)
    code = (
        "import json, sys; sys.path.insert(0, '/app'); from urllib.parse import urlsplit; from app.config import get_settings; "
        "u=urlsplit(str(get_settings().database_url)); "
        "print('DATABASE_IDENTITY='+json.dumps({'host':(u.hostname or '').replace('-pooler.','.'),"
        "'database':u.path.lstrip('/'),'user':u.username or ''}))"
    )
    result = run(
        [
            "flyctl",
            "ssh",
            "console",
            "--app",
            app,
            "--machine",
            api[0]["id"],
            "--command",
            "/app/.venv/bin/python -c " + shlex.quote(code),
        ],
        capture_output=True,
    )
    identities = [
        json.loads(line.split("=", 1)[1])
        for line in result.stdout.splitlines()
        if line.startswith("DATABASE_IDENTITY=")
    ]
    if identities != [fingerprint(uri)]:
        raise RuntimeError(
            "Neon database does not match the running API; refusing backup/deployment"
        )
    bucket = os.environ.get("R2_BUCKET") or "twin-assets"
    public_bucket = os.environ.get("R2_PUBLIC_BUCKET") or bucket + "-public"
    if bucket == public_bucket:
        raise RuntimeError("Backup destination must be the private bucket")
    import boto3

    storage = boto3.client(
        "s3",
        endpoint_url="https://"
        + required("CLOUDFLARE_ACCOUNT_ID")
        + ".r2.cloudflarestorage.com",
        region_name="auto",
        aws_access_key_id=required("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=required("R2_SECRET_ACCESS_KEY"),
    )
    storage.head_bucket(Bucket=bucket)
    release = required("GITHUB_RUN_ID") + "-" + required("GITHUB_RUN_ATTEMPT")
    key = "operations/backups/worlds-land/" + release + ".dump"
    with tempfile.TemporaryDirectory(prefix="worlds-land-release-") as directory:
        dump = Path(directory) / "database.dump"
        dump.touch(mode=0o600)
        print("Creating the database backup", flush=True)
        run(
            [
                "pg_dump",
                "--format=custom",
                "--no-owner",
                "--no-privileges",
                "--file",
                str(dump),
            ],
            env={**os.environ, "PGDATABASE": uri},
            capture_output=True,
        )
        digest = hashlib.file_digest(dump.open("rb"), "sha256").hexdigest()
        print("Uploading backup to the private bucket", flush=True)
        storage.upload_file(
            str(dump),
            bucket,
            key,
            ExtraArgs={
                "Metadata": {
                    "sha256": digest,
                    "revision": required("GITHUB_SHA"),
                    "neon-project": project,
                    "neon-branch": branch,
                }
            },
        )
        uploaded = storage.head_object(Bucket=bucket, Key=key)
        if (
            uploaded["ContentLength"] != dump.stat().st_size
            or uploaded["Metadata"]["sha256"] != digest
        ):
            raise RuntimeError("Private backup verification failed")
        print(
            f"Private database backup: s3://{bucket}/{key} (sha256 {digest})",
            flush=True,
        )
        # The rehearsal uses the uploaded bytes, not merely the original dump.
        restored = Path(directory) / "downloaded.dump"
        storage.download_file(bucket, key, str(restored))
        if hashlib.file_digest(restored.open("rb"), "sha256").hexdigest() != digest:
            raise RuntimeError("Downloaded backup checksum mismatch")
        container = "worlds-land-rehearsal-" + release
        try:
            print("Restoring backup into disposable local PostGIS", flush=True)
            run(
                [
                    "docker",
                    "run",
                    "--detach",
                    "--name",
                    container,
                    "--publish",
                    "127.0.0.1::5432",
                    "--env",
                    "POSTGRES_PASSWORD=rehearsal-local",
                    "--env",
                    "POSTGRES_DB=rehearsal",
                    "postgis/postgis:16-3.5",
                ],
                capture_output=True,
            )
            port = (
                run(["docker", "port", container, "5432"], capture_output=True)
                .stdout.strip()
                .rsplit(":", 1)[1]
            )
            local = f"postgresql://postgres:rehearsal-local@127.0.0.1:{port}/rehearsal"
            for attempt in range(60):
                if (
                    subprocess.run(
                        ["pg_isready", "-d", local], capture_output=True, check=False
                    ).returncode
                    == 0
                ):
                    break
                time.sleep(1)
            else:
                raise RuntimeError("Local migration rehearsal database did not start")
            run(
                [
                    "pg_restore",
                    "--dbname",
                    local,
                    "--clean",
                    "--if-exists",
                    "--no-owner",
                    "--no-privileges",
                    "--exit-on-error",
                    str(restored),
                ],
                env={**os.environ, "PGDATABASE": local},
                capture_output=True,
            )
            run(
                [sys.executable, "-m", "alembic", "upgrade", "head"],
                cwd=ROOT / "apps/api",
                env={
                    **os.environ,
                    "ALEMBIC_DATABASE_URL": local.replace(
                        "postgresql://", "postgresql+psycopg://"
                    ),
                },
            )
            revision = run(
                ["psql", "-Atc", "select version_num from alembic_version"],
                env={**os.environ, "PGDATABASE": local},
                capture_output=True,
            ).stdout.strip()
            if revision != "0028":
                raise RuntimeError("Unexpected migration rehearsal head")
            print(
                "Downloaded private backup restores and upgrades successfully to 0028",
                flush=True,
            )
            summary = os.environ.get("GITHUB_STEP_SUMMARY")
            if summary:
                with open(summary, "a") as handle:
                    handle.write(
                        f"\nPrivate database backup: `s3://{bucket}/{key}`\n\n"
                        f"SHA256: `{digest}`. Restore and migration to `0028` passed.\n"
                    )
        finally:
            subprocess.run(
                ["docker", "rm", "--force", container], capture_output=True, check=False
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - never print provider/connection exception secrets
        # HTTP/subprocess exception bodies and arguments may contain connection secrets.
        print(
            f"::error::Release preparation failed ({type(error).__name__}); no deployment was started."
        )
        if isinstance(error, RuntimeError):
            print(str(error))
        sys.exit(1)
