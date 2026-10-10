"""Release safety contracts; all cloud, database and process calls are fake."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "worlds_land_release", Path(__file__).parents[1] / "prepare-worlds-land-release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.fixture
def harness(monkeypatch):
    calls = []
    state = {"host": "ep-test.example", "machines": 1, "backup": None, "metadata": None}
    for name in (
        "NEON_API_KEY",
        "CLOUDFLARE_ACCOUNT_ID",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_SHA",
    ):
        monkeypatch.setenv(name, "test")
    monkeypatch.setenv("NEON_PROJECT_NAME", "test-project")
    monkeypatch.setenv("R2_BUCKET", "private")
    monkeypatch.setenv("R2_PUBLIC_BUCKET", "public")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    def neon(path):
        if path.startswith("/projects?"):
            return {"projects": [{"name": "test-project", "id": "project"}]}
        if path.endswith("/branches"):
            return {"branches": [{"id": "branch", "default": True}]}
        if path.endswith("/databases"):
            return {"databases": [{"name": "db", "owner_name": "user"}]}
        return {"uri": "postgresql://user:private-password@ep-test-pooler.example/db"}

    def run(args, **kwargs):
        calls.append((args, kwargs))
        output = ""
        if args[:3] == ["flyctl", "machines", "list"]:
            output = json.dumps(
                [
                    {
                        "id": "machine",
                        "state": "started",
                        "config": {"metadata": {"fly_process_group": "app"}},
                    }
                ]
                * state["machines"]
            )
        elif args[:3] == ["flyctl", "ssh", "console"]:
            output = "DATABASE_IDENTITY=" + json.dumps(
                {"host": state["host"], "database": "db", "user": "user"}
            )
        elif args[0] == "pg_dump":
            Path(args[args.index("--file") + 1]).write_bytes(
                b"private database snapshot"
            )
        elif args[:2] == ["docker", "port"]:
            output = "127.0.0.1:54329"
        elif args[0] == "pg_restore":
            assert args[args.index("--dbname") + 1].startswith(
                "postgresql://postgres:rehearsal-local@127.0.0.1:"
            )
            assert Path(args[-1]).read_bytes() == state["backup"]
        elif args[0] == "psql":
            output = "0028"
        return SimpleNamespace(stdout=output, returncode=0)

    class Storage:
        def head_bucket(self, **kwargs):
            assert kwargs["Bucket"] == "private"

        def upload_file(self, source, bucket, key, ExtraArgs):
            assert bucket == "private"
            state["backup"] = Path(source).read_bytes()
            state["metadata"] = ExtraArgs["Metadata"]

        def head_object(self, **kwargs):
            return {
                "ContentLength": len(state["backup"]),
                "Metadata": state["metadata"],
            }

        def download_file(self, bucket, key, destination):
            Path(destination).write_bytes(state["backup"])

    monkeypatch.setattr(release, "neon", neon)
    monkeypatch.setattr(release, "run", run)
    monkeypatch.setattr(release.subprocess, "run", run)
    monkeypatch.setitem(
        sys.modules, "boto3", SimpleNamespace(client=lambda *a, **kw: Storage())
    )
    return state, calls


def test_backup_download_restore_and_migration_are_separate_from_production(harness):
    state, calls = harness
    release.main()
    assert state["backup"]
    dump = next(kwargs for args, kwargs in calls if args[0] == "pg_dump")
    assert "ep-test-pooler.example" in dump["env"]["PGDATABASE"]
    migration = next(kwargs for args, kwargs in calls if "alembic" in args)
    assert "127.0.0.1" in migration["env"]["ALEMBIC_DATABASE_URL"]
    assert calls[-1][0][:3] == ["docker", "rm", "--force"]


@pytest.mark.parametrize("change", ["wrong-database", "multiple-api", "public-bucket"])
def test_ambiguous_or_unsafe_release_stops_before_dump(harness, monkeypatch, change):
    state, calls = harness
    if change == "wrong-database":
        state["host"] = "another.example"
    elif change == "multiple-api":
        state["machines"] = 2
    else:
        monkeypatch.setenv("R2_PUBLIC_BUCKET", "private")
    with pytest.raises(RuntimeError):
        release.main()
    assert not any(args[0] == "pg_dump" for args, _ in calls)


def test_failed_restore_does_not_run_migrations_and_removes_local_database(
    harness, monkeypatch
):
    _, calls = harness
    original = release.run

    def fail_restore(args, **kwargs):
        if args[0] == "pg_restore":
            raise subprocess.CalledProcessError(1, args)
        return original(args, **kwargs)

    monkeypatch.setattr(release, "run", fail_restore)
    with pytest.raises(subprocess.CalledProcessError):
        release.main()
    assert not any("alembic" in args for args, _ in calls)
    assert calls[-1][0][:3] == ["docker", "rm", "--force"]
