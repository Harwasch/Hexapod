from __future__ import annotations


class NotFoundError(Exception):
    def __init__(self, resource: str, identifier: object) -> None:
        super().__init__(f"{resource} {identifier} not found")
        self.resource = resource
        self.identifier = identifier


class ConflictError(Exception):
    pass


class UnauthorizedError(Exception):
    """A write was attempted without the shared write token.

    Only raised when a token is configured: an unset `API_WRITE_TOKEN` leaves writes
    open, and `create_app` refuses to start in production without one.
    """
