from __future__ import annotations

from alembic.config import Config
from sqlalchemy import Engine, inspect

from alembic import command


def test_migrations_round_trip(alembic_config: Config, engine: Engine) -> None:
    command.downgrade(alembic_config, "base")
    assert not inspect(engine).has_table("sites")
    command.upgrade(alembic_config, "head")
    inspector = inspect(engine)
    for table in (
        "sites",
        "assets",
        "layers",
        "camera_bookmarks",
        "plans",
        "plan_revisions",
        "captures",
        "capture_files",
        "jobs",
        "job_steps",
        "artifacts",
    ):
        assert inspector.has_table(table)
    index_names = {index["name"] for index in inspector.get_indexes("sites")}
    assert "idx_sites_boundary" in index_names


def test_models_match_migrations(alembic_config: Config, engine: Engine) -> None:
    command.check(alembic_config)
