"""How many of this test process's sessions are idle in a transaction, asked of Postgres.

Long object-store I/O runs with no transaction open, because Neon (like any Postgres with
`idle_in_transaction_session_timeout`) kills a session left idle inside one and the next
statement on it fails (`runner._end_transaction`). The tests of that sample
`pg_stat_activity` at the moment of each download or upload; this is the sample, in one
place, so that it counts what those tests are about and nothing else:

- Only this process's sessions. `tests/conftest.py` sets `PGAPPNAME`, which libpq takes as
  the default `application_name`, so a connection this process opens -- on any engine, and
  in any worker process it starts -- is named `twin-tests-<pid>` unless it sets a name of
  its own, and the probe counts the sessions whose name is the same as its own. A session
  that sets its own `application_name` is not counted, wherever it comes from. A `psql`
  left in `BEGIN`, or another test run on the same database, is not the code under test.
  (With no `PGAPPNAME` the probe's own name is libpq's default, empty, so it counts the
  sessions that left their name empty; a client that names itself, as `psql` does, is not
  among them.)
- A fresh reading every time. The probe's engine is in autocommit, so each sample is its
  own transaction: Postgres keeps the `pg_stat_activity` it read first for the rest of a
  transaction, and the probe's own session is never idle in one either.
"""

from __future__ import annotations

from sqlalchemy import create_engine, text

COUNT = text(
    "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
    "AND pid <> pg_backend_pid() "
    "AND application_name = current_setting('application_name') "
    "AND state LIKE 'idle in transaction%'"
)


class IdleTransactions:
    """A probe on an engine of its own: call it for the number of this process's sessions
    that are idle in a transaction right now, and close it when the test is done."""

    def __init__(self, url: str) -> None:
        self._engine = create_engine(url, isolation_level="AUTOCOMMIT")

    def __call__(self) -> int:
        with self._engine.connect() as connection:
            return int(connection.execute(COUNT).scalar_one())

    def close(self) -> None:
        self._engine.dispose()
