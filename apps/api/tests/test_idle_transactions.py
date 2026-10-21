"""The probe the no-transaction-during-I/O tests share (`tests/idle_transactions.py`)."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, text

from tests.conftest import TEST_DATABASE_URL
from tests.idle_transactions import IdleTransactions


def test_the_probe_counts_this_processs_sessions_and_no_other_clients(engine: Engine) -> None:
    """A session of this process left in a transaction is counted, on any engine; another
    client's, on the same database, is not. Relative to what is there already, so a
    session an earlier test leaked does not decide this one."""
    idle_in_transaction = IdleTransactions(TEST_DATABASE_URL)
    elsewhere = create_engine(TEST_DATABASE_URL, connect_args={"application_name": "psql"})
    try:
        before = idle_in_transaction()
        with elsewhere.connect() as foreign:
            foreign.execute(text("SELECT 1"))  # and now idle in its transaction
            assert idle_in_transaction() == before
            with engine.connect() as ours:
                ours.execute(text("SELECT 1"))
                assert idle_in_transaction() == before + 1
            mine = create_engine(TEST_DATABASE_URL)
            with mine.connect() as also_ours:
                also_ours.execute(text("SELECT 1"))
                assert idle_in_transaction() == before + 1
            mine.dispose()
        assert idle_in_transaction() == before
    finally:
        elsewhere.dispose()
        idle_in_transaction.close()
