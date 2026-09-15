"""Validation of user-supplied URLs.

The API never fetches these URLs itself (the browser does), but we still reject
inputs that could never be a legitimate public dataset endpoint, and in
production we refuse loopback / private network hosts so the catalog cannot be
used to point clients at internal services.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from app.config import get_settings

BLOCKED_HOSTS = {"localhost", "0.0.0.0", "::1", "metadata.google.internal"}  # noqa: S104


class UrlValidationError(ValueError):
    pass


def validate_dataset_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise UrlValidationError("only http(s) URLs are allowed")
    if not parts.hostname:
        raise UrlValidationError("URL must include a host")
    if parts.username or parts.password:
        raise UrlValidationError("URLs must not embed credentials")
    if not get_settings().allow_private_urls and _is_private_host(parts.hostname):
        raise UrlValidationError("private or loopback hosts are not allowed")
    return url


def _is_private_host(host: str) -> bool:
    if host.lower() in BLOCKED_HOSTS or host.endswith((".local", ".internal")):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local
