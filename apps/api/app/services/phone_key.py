"""The phone key: a short shared key that lets a phone start captures on its own.

The alternative is the handoff QR code, which needs a desktop holding the write token to
create the capture first. This key is typed once on the phone and remembered there, and
it opens the phone routes and the iPhone bundle, question and scoped read routes. It
cannot touch a capture it did not create, delete anything, or reach any other route,
and it may create at most
`api_phone_daily_captures` captures in any 24 hours. A capture it creates may name the
site it is a new epoch of (decision D3) only if a phone capture made that site: never a
seeded, catalog, hand-made or desktop-uploaded one.

Deliberately modest, at the owner's request: shared keys, no accounts, no revocation list.
What keeps it from being worse than that is how it is stored. The configuration holds a
salted PBKDF2 hash, so the repository -- which is public -- carries nothing a key can be
read from, and a leaked hash still costs 200,000 SHA-256 rounds per guess against a key
of about 59 bits. Rotate by appending a hash, then removing the old hash when its builds retire.
"""

from __future__ import annotations

import hashlib
import hmac

from app.config import Settings
from app.services.errors import UnauthorizedError

_SCHEME = "pbkdf2_sha256"


def hash_key(key: str, *, salt: bytes, iterations: int = 200_000) -> str:
    """The stored form of a key. Used to make one, and by the tests."""
    digest = hashlib.pbkdf2_hmac("sha256", key.encode("utf-8"), salt, iterations)
    return f"{_SCHEME}${iterations}${salt.hex()}${digest.hex()}"


def enabled(settings: Settings) -> bool:
    """Whether any phone key (the general one or the iPhone app's) is configured."""
    return bool(settings.api_phone_key_hash or settings.api_phone_app_key_hash)


def check(settings: Settings, supplied: str) -> None:
    """Accept the configured key, or refuse -- the same words whatever is wrong."""
    stored = ",".join(
        value for value in (settings.api_phone_key_hash, settings.api_phone_app_key_hash) if value
    )
    if not stored:
        raise UnauthorizedError("Phone capture is not enabled on this deployment.")
    # Check every configured hash, even after a match, with the same normalization
    # and constant-time digest comparison as the original single-key path.
    typed = supplied.strip().lower()
    accepted = False
    try:
        for entry in stored.split(","):
            scheme, iterations, salt_hex, digest_hex = entry.strip().split("$")
            if scheme != _SCHEME:
                raise ValueError(scheme)
            salt = bytes.fromhex(salt_hex)
            rounds = int(iterations)
            digest = hashlib.pbkdf2_hmac("sha256", typed.encode("utf-8"), salt, rounds).hex()
            accepted |= hmac.compare_digest(digest, digest_hex)
    except (ValueError, OverflowError, TypeError):
        raise UnauthorizedError("Phone capture is misconfigured on this deployment.") from None
    if not typed or not accepted:
        raise UnauthorizedError("That phone key is not right.")
