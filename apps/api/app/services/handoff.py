"""The phone-handoff token: upload to *one* capture, for a short while, and nothing else.

Why a signed token and not a table
----------------------------------
Everything this token has to carry -- which capture, until when, and until when it may
be renewed -- is three immutable facts decided at issue time. A row would add a write on
every scan, a read on every part-window, a expiry sweep, and a second source of truth for
"is this still valid", and would buy exactly one thing a signature cannot: revocation
before expiry. At a ten-minute lifetime there is almost nothing to revoke, and the
console can already abort the upload with the write token. So: stateless, signed, and
the secret never leaves the server.

The shape
---------
``h1.<capture-id-hex>.<expires-at>.<renewable-until>.<signature>``

The signature is HMAC-SHA256 over the first four fields joined by dots, base64url without
padding. Every field is inside the MAC, so the capture id and both deadlines are
tamper-evident; changing any byte of any of them invalidates the token.

The capture id is *in* the token, and it is checked against the capture id in the request
path on every single request (`app.api.deps.require_upload_token`). A token issued for
capture A presented against capture B is refused before the route function runs.

Lifetimes
---------
`TOKEN_TTL_SECONDS` (10 minutes) is the handoff itself: unlock the phone, open the
camera, tap the notification, wait for the page, find the file. Every step of that is
seconds, so ten minutes is already generous, and it is the window in which a QR code read
over someone's shoulder is worth anything.

But an upload is not a handoff. A 12 GB video on cellular takes hours, and it talks to the
API at every 256 MiB window, so a ten-minute token would strand it at the first boundary.
Rather than stretch the lifetime to fit the worst upload, each handoff-authorised response
carries a *fresh* token in `X-Handoff-Token`, expiring ten minutes from then; the client
uses the newest one it has seen. `RENEWABLE_FOR_SECONDS` (6 hours) is the ceiling on that
chain, carried in the token and never extended by a renewal, so the sliding window cannot
slide forever: six hours after the QR was drawn on screen, every descendant of that token
is dead and the desktop has to issue a new one.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass

from app.config import Settings
from app.services.errors import UnauthorizedError

#: Version tag, so the format can change without an old token being misread as a new one.
FORMAT_VERSION = "h1"

#: How long one token is good for. See the module docstring.
TOKEN_TTL_SECONDS = 600

#: How long the chain of renewals may run from the first issue. See the module docstring.
RENEWABLE_FOR_SECONDS = 6 * 60 * 60

#: Label mixed into the key derivation so the handoff key is not the write token itself.
_KEY_INFO = b"twin/capture-handoff/v1"

#: Fields in a well-formed token.
_FIELDS = 5


@dataclass(frozen=True, slots=True)
class HandoffClaims:
    """What a verified token asserts. Never constructed from unverified input."""

    capture_id: uuid.UUID
    expires_at: int
    renewable_until: int


def derive_key(secret: str | bytes) -> bytes:
    """A handoff key from a server secret.

    Derived rather than used directly so that the same `API_WRITE_TOKEN` can back both
    checks without a handoff signature being computable from anything an attacker could
    obtain, and so that a future rotation of one does not silently rotate the other.
    """
    raw = secret.encode("utf-8") if isinstance(secret, str) else secret
    return hmac.new(raw, _KEY_INFO, hashlib.sha256).digest()


#: Fallback secret for a deployment that has configured none.
#:
#: A fresh checkout has no `API_WRITE_TOKEN` -- writes are open, which is what makes the
#: repo runnable with no configuration -- and a handoff still has to work there. A random
#: per-process key gives that, and the tokens it signs die when the process restarts,
#: which is the right failure mode for a deployment that has declared no secret at all.
#: Production cannot reach this line: `create_app` refuses to start without a write token.
_EPHEMERAL_SECRET = secrets.token_bytes(32)


def key_for(settings: Settings) -> bytes:
    """The signing key this deployment uses for handoff tokens."""
    return derive_key(settings.api_handoff_secret or settings.api_write_token or _EPHEMERAL_SECRET)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _sign(key: bytes, body: str) -> str:
    return _b64(hmac.new(key, body.encode("ascii"), hashlib.sha256).digest())


def issue(
    key: bytes,
    capture_id: uuid.UUID,
    *,
    now: int,
    ttl: int = TOKEN_TTL_SECONDS,
    renewable_until: int | None = None,
) -> str:
    """Mint a token for one capture.

    `renewable_until` is passed through unchanged by a renewal, which is what stops the
    sliding window from sliding forever.
    """
    ceiling = renewable_until if renewable_until is not None else now + RENEWABLE_FOR_SECONDS
    expires_at = min(now + ttl, ceiling)
    body = f"{FORMAT_VERSION}.{capture_id.hex}.{expires_at}.{ceiling}"
    return f"{body}.{_sign(key, body)}"


def verify(key: bytes, token: str, *, capture_id: uuid.UUID, now: int) -> HandoffClaims:
    """Check a token against one capture at one moment, or refuse it.

    Every refusal is the same `UnauthorizedError` with the same wording: a caller holding
    a bad token learns that it is bad, not *why*, and not whether the capture exists.
    """
    parts = token.split(".")
    if len(parts) != _FIELDS or parts[0] != FORMAT_VERSION:
        raise _refused()
    version, capture_hex, expires_raw, ceiling_raw, signature = parts

    # compare_digest over the recomputed MAC, and before anything is parsed out of the
    # payload: an unsigned field is attacker-controlled and must not reach any logic.
    body = f"{version}.{capture_hex}.{expires_raw}.{ceiling_raw}"
    if not hmac.compare_digest(signature, _sign(key, body)):
        raise _refused()

    try:
        claimed = uuid.UUID(hex=capture_hex)
        expires_at = int(expires_raw)
        ceiling = int(ceiling_raw)
    except ValueError:
        raise _refused() from None

    if now >= expires_at or now >= ceiling:
        raise _refused()
    # The scope check. `capture_id` is the one in the request path, so a token for
    # another capture is refused here rather than by whatever the route would have done.
    if claimed != capture_id:
        raise _refused()
    return HandoffClaims(capture_id=claimed, expires_at=expires_at, renewable_until=ceiling)


def renew(key: bytes, claims: HandoffClaims, *, now: int) -> str | None:
    """The next token in the chain, or None once the ceiling is within reach.

    None rather than an already-dead token: the client keeps using the one it has until
    that expires too, and then the upload stops. Six hours after the QR was drawn, it
    stops for good.
    """
    if now + 1 >= claims.renewable_until:
        return None
    return issue(key, claims.capture_id, now=now, renewable_until=claims.renewable_until)


def _refused() -> UnauthorizedError:
    return UnauthorizedError(
        "This upload link is not valid for this capture, or it has expired. "
        "Scan a fresh QR code from the Captures panel."
    )
