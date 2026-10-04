"""A stand-in for the `modal` client whose calls outlive the process that spawned them.

`modal/` here is put on the recipe process's `PYTHONPATH` by `tests/test_worker_stops.py`
and nowhere else, so the worker's own `ModalAdapter` -- unchanged -- spawns, polls,
re-attaches to and cancels calls through it. What makes it worth having is the one
property a real Modal call has and an in-process fake does not: a call is a process of
its own (`container.py`, running `remote.execute` exactly as `infra/modal/app.py`
does), with its state on disk, so it keeps running after the recipe process that spawned
it is stopped or killed, and a different process can pick it up by id.
"""
