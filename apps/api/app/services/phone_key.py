"""The phone key: a short shared key that lets a phone start captures on its own.

The alternative is the handoff QR code, which needs a desktop holding the write token to
create the capture first. This key is typed once on the phone and remembered there, and
it opens exactly three routes (app/api/v1/phone.py): create a phone capture, get an
upload token for one, and queue a run over one. It cannot touch a capture it did not
create, delete anything, or reach any other route, and it may create at most
`api_phone_daily_captures` captures in any 24 hours.

Deliberately modest, at the owner's request: one key, no accounts, no revocation list.
What keeps it from being worse than that is how it is stored. The configuration holds a
salted PBKDF2 hash, so the repository -- which is public -- carries nothing a key can be
read from, and a leaked hash still costs 200,000 SHA-256 rounds per guess against a key
of about 59 bits. Rotating it is generating a new key and replacing the hash.
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


def check(settings: Settings, supplied: str) -> None:
    """Accept the configured key, or refuse -- the same words whatever is wrong."""
    stored = settings.api_phone_key_hash
    if not stored:
        raise UnauthorizedError("Phone capture is not enabled on this deployment.")
    try:
        scheme, iterations, salt_hex, digest_hex = stored.split("$")
        if scheme != _SCHEME:
            raise ValueError(scheme)
        salt = bytes.fromhex(salt_hex)
        rounds = int(iterations)
    except ValueError:
        raise UnauthorizedError("Phone capture is misconfigured on this deployment.") from None
    # Normalised the way a person types on a phone: case and surrounding spaces ignored.
    typed = supplied.strip().lower()
    digest = hashlib.pbkdf2_hmac("sha256", typed.encode("utf-8"), salt, rounds).hex()
    if not typed or not hmac.compare_digest(digest, digest_hex):
        raise UnauthorizedError("That phone key is not right.")
