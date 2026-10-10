"""Verify API access tokens; never accept the pilot token in OIDC mode."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import jwt

from app.config import Settings
from app.services.errors import UnauthorizedError


@dataclass(frozen=True)
class Principal:
    id: str
    pilot: bool = False
    expires_at: float | None = None


def principal_id(issuer: str, subject: str) -> str:
    return hashlib.sha256(f"{issuer}\0{subject}".encode()).hexdigest()


@lru_cache(maxsize=8)
def jwks_client(url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(url, lifespan=300, timeout=5)


def verify_token(token: str, settings: Settings) -> Principal:
    if not token or len(token) > 16_384 or not settings.land_oidc_jwks_url:
        raise UnauthorizedError("A valid workspace access token is required.")
    try:
        key = jwks_client(settings.land_oidc_jwks_url).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            key.key,
            algorithms=["RS256", "ES256"],
            audience=settings.land_oidc_audience,
            issuer=settings.land_oidc_issuer,
            options={"require": ["exp", "iat", "sub", "iss", "aud"]},
            leeway=10,
        )
        subject = claims["sub"]
        if not isinstance(subject, str) or not subject or len(subject) > 512:
            raise jwt.InvalidTokenError("Invalid subject")
    except jwt.PyJWTError as error:
        raise UnauthorizedError("The workspace access token is invalid or expired.") from error
    return Principal(principal_id(claims["iss"], subject), expires_at=float(claims["exp"]))
