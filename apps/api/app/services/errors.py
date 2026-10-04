from __future__ import annotations


class NotFoundError(Exception):
    def __init__(self, resource: str, identifier: object) -> None:
        super().__init__(f"{resource} {identifier} not found")
        self.resource = resource
        self.identifier = identifier


class ConflictError(Exception):
    """A request the resource's current state refuses. A 409.

    `code`, where the service gives one, is a machine-readable reason, answered as the
    Problem's `code` so a client can branch on it rather than on the words of `detail`:
    the sidecar attach's `tiles_changed`, `busy` and `not_attachable`
    (`app/services/attach.py`).
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class UnauthorizedError(Exception):
    """A write was attempted without the shared write token.

    Only raised when a token is configured: an unset `API_WRITE_TOKEN` leaves writes
    open, and `create_app` refuses to start in production without one.
    """


class InvalidInputError(ValueError):
    """A request that is well-formed but asks for something that cannot be: a recipe that
    does not exist, a stage a run does not have, a window past the last part. A 422.

    The API used to turn *every* `ValueError` into a 422, which made a bug -- a failed
    `int()`, a pydantic model rejecting a row the database already held -- look like the
    caller's mistake, with the bug's own message as the explanation and no log line at
    all. Now only this is a 422 (with `UrlValidationError` and request validation), and any
    other `ValueError` is a logged 500. It subclasses `ValueError` so that code catching
    `ValueError` around a service call (the phone's Refine) keeps working unchanged.
    """
