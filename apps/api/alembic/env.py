from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool, text

from alembic import context
from app.config import get_settings
from app.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
protected_tables: set[tuple[str | None, str]] = {(None, "spatial_ref_sys")}


def _database_url() -> str:
    return os.environ.get("ALEMBIC_DATABASE_URL") or get_settings().database_url


def include_object(
    obj: object, name: str | None, type_: str, reflected: bool, compare_to: object
) -> bool:
    # Extension-owned tables are not application schema. Some PostGIS images put
    # tiger/topology on search_path, so they reflect as schema=None too.
    return not (type_ == "table" and (getattr(obj, "schema", None), name) in protected_tables)


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        extension_rows = connection.execute(
            text("""
            SELECT n.nspname, c.relname, pg_table_is_visible(c.oid)
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_depend d ON d.objid = c.oid AND d.classid = 'pg_class'::regclass
            WHERE d.refclassid = 'pg_extension'::regclass AND d.deptype = 'e'
              AND c.relkind IN ('r', 'p')
        """)
        ).all()
        for schema, name, visible in extension_rows:
            protected_tables.add((schema, name))
            if visible:
                protected_tables.add((None, name))
        # End the catalog query's implicit transaction before Alembic starts its own.
        connection.commit()
        context.configure(
            connection=connection, target_metadata=target_metadata, include_object=include_object
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
