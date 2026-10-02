"""Logs a person can read in `fly logs`, and errors that reach Sentry when it is configured.

Before this module the API configured no logging at all. `twin.api` logged at INFO and
WARNING -- the slow-request line, the production tiles notice -- and Python's last-resort
handler printed WARNING and above with no time, no level and no logger name, and dropped
every INFO line outright. Under uvicorn that meant the API's own account of itself was
mostly missing and the rest was unattributable.

What `configure_logging` sets up, once per process however many apps are created:

* One handler, on the root logger, writing to whatever `sys.stderr` is at the moment of
  the write (so a test that swaps stderr does not leave a handler holding a closed one).
* `LOG_FORMAT=json` -- the default in production -- is one JSON object per line: `time`,
  `level`, `logger`, `message`, any `extra=` fields, and `exception` when there is one.
  `text` -- the default anywhere else -- is the same fields as a line a person reads.
* `LOG_LEVEL` (default INFO) governs this API's own `twin.*` loggers. Everything else --
  SQLAlchemy, botocore, httpx -- is left at WARNING, which is where a library says
  something worth a line.
* In JSON mode uvicorn's own loggers go through the same handler, so access lines and the
  tracebacks of unhandled errors are JSON too, rather than two formats interleaved.

**Nothing secret is written.** Every line goes through `Redactor` before it leaves: bearer
credentials, phone-handoff tokens, the query string of a presigned URL (its signature *is*
the credential, for an hour), PBKDF2 hashes, and the literal values of every secret this
process was configured with. A log line is read by more people, and kept longer, than any
credential is meant to be.

Sentry is opt-in: `init_sentry` does nothing, and imports nothing, unless `SENTRY_DSN` is
set. When it is, it is configured not to send what logs do not either -- no request
bodies, no stack-frame locals (a `Settings` object in a frame is every secret this API
has), and every message and breadcrumb through the same `Redactor`.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, TextIO

from app.config import Settings

#: Loggers that are this API's own, and so follow LOG_LEVEL. Not `app.worker`: the worker
#: is its own process with its own `--log-level` (app/worker/__main__.py).
APP_LOGGERS = ("twin",)
#: uvicorn's loggers, which have handlers and `propagate=False` of their own.
UVICORN_LOGGERS = ("uvicorn", "uvicorn.access")
#: What every other library logs at: warnings and worse.
LIBRARY_LEVEL = logging.WARNING

REDACTED = "[redacted]"

#: The shortest configured secret that is redacted by value. Shorter than this and the
#: "secret" is a test fixture or a placeholder, and replacing it would mangle ordinary text.
_MIN_SECRET_LENGTH = 8

_BEARER = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]+")
_HANDOFF = re.compile(r"\bh1\.[0-9a-f]{32}\.\d+\.\d+\.[A-Za-z0-9_-]+")
_PBKDF2 = re.compile(r"pbkdf2_sha256\$[^\s\"',;]+")
_URL_WITH_QUERY = re.compile(r"(https?://[^\s?#\"'<>]+)\?([^\s#\"'<>]*)")
_SIGNED_QUERY = re.compile(r"(?i)(x-amz-|signature=|awsaccesskeyid=|token=)")


class Redactor:
    """Strips credentials out of text: by shape, and by the values this process holds."""

    def __init__(self, secrets: Iterable[str | None] = ()) -> None:
        # Longest first, so a secret that contains another is replaced whole.
        self._secrets = sorted(
            {s for s in secrets if s and len(s) >= _MIN_SECRET_LENGTH}, key=len, reverse=True
        )

    @classmethod
    def for_settings(cls, settings: Settings) -> Redactor:
        return cls(
            [
                settings.api_write_token,
                settings.api_handoff_secret,
                settings.api_phone_key_hash,
                settings.object_storage_access_key,
                settings.object_storage_secret_key,
                settings.anthropic_api_key,
                settings.cesium_ion_server_token,
                settings.sentry_dsn,
            ]
        )

    def __call__(self, text: str) -> str:
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, REDACTED)
        text = _BEARER.sub(lambda m: f"{m.group(1)} {REDACTED}", text)
        text = _HANDOFF.sub(f"h1.{REDACTED}", text)
        text = _PBKDF2.sub(f"pbkdf2_sha256${REDACTED}", text)
        return _URL_WITH_QUERY.sub(_strip_signed_query, text)


def _strip_signed_query(match: re.Match[str]) -> str:
    """A presigned URL keeps its path -- which object -- and loses its signature."""
    if _SIGNED_QUERY.search(match.group(2)):
        return f"{match.group(1)}?{REDACTED}"
    return match.group(0)


#: The attributes every LogRecord has; anything else on a record came from `extra=`.
#: `color_message` is uvicorn's: the same message again, with ANSI colour codes in it.
_STANDARD_ATTRIBUTES = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()
    | {"message", "asctime", "color_message"}
)


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(record).items()
        if key not in _STANDARD_ATTRIBUTES and not key.startswith("_")
    }


class JsonFormatter(logging.Formatter):
    """One JSON object per line, redacted."""

    def __init__(self, redact: Redactor) -> None:
        super().__init__()
        self._redact = redact

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": self._redact(record.getMessage()),
        }
        for key, value in _extras(record).items():
            payload[key] = self._redact(value) if isinstance(value, str) else value
        if record.exc_info:
            payload["exception"] = self._redact(self.formatException(record.exc_info))
        elif record.exc_text:
            payload["exception"] = self._redact(record.exc_text)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """`time level logger: message key=value ...`, redacted."""

    def __init__(self, redact: Redactor) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")
        self._redact = redact

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extras = _extras(record)
        if extras:
            line += " " + " ".join(f"{key}={value}" for key, value in extras.items())
        return self._redact(line)


class _StderrHandler(logging.StreamHandler[TextIO]):
    """Writes to `sys.stderr` as it is at the time of each write, as `logging.lastResort`
    does. A handler that captured the stream at construction keeps writing to it after a
    test runner has swapped it out and closed it."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def stream(self) -> TextIO:
        return sys.stderr

    @stream.setter
    def stream(self, _: TextIO) -> None:
        pass


#: Marks the handler this module installed, so a second call replaces it, not adds one.
_HANDLER_NAME = "twin"


def log_format(settings: Settings) -> str:
    if settings.log_format:
        return settings.log_format
    return "json" if settings.is_production else "text"


def configure_logging(settings: Settings) -> logging.Handler:
    """Install the handler and levels described in the module docstring. Idempotent."""
    redact = Redactor.for_settings(settings)
    structured = log_format(settings) == "json"
    handler = _StderrHandler()
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(JsonFormatter(redact) if structured else TextFormatter(redact))

    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.get_name() == _HANDLER_NAME:
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(LIBRARY_LEVEL)
    for name in APP_LOGGERS:
        logging.getLogger(name).setLevel(settings.log_level)
    # `logging.config.fileConfig` disables every logger that already exists, and alembic's
    # env.py calls it -- in-process wherever migrations are run from Python, as the test
    # suite does. This function is what decides the API's own loggers, so it re-enables
    # them rather than leaving `twin.api` silenced by whichever ran last.
    for name, known in list(logging.root.manager.loggerDict.items()):
        if isinstance(known, logging.Logger) and name.split(".")[0] in APP_LOGGERS:
            known.disabled = False

    if structured:
        # uvicorn configures these before it imports the app, each with a handler of its
        # own and `propagate=False`. Where it has (that is, under uvicorn), they write
        # through ours instead. In text mode uvicorn's own lines are what a developer
        # expects to see, and are left alone.
        for name in UVICORN_LOGGERS:
            uvicorn_logger = logging.getLogger(name)
            if uvicorn_logger.handlers:
                uvicorn_logger.handlers = [handler]
    return handler


_sentry_dsn: str | None = None


def init_sentry(settings: Settings) -> bool:
    """Start Sentry when `SENTRY_DSN` is set; otherwise do nothing and import nothing.

    Called before the FastAPI app is built, which is when sentry-sdk's FastAPI and
    Starlette integrations (enabled automatically) need to be in place. Once per process
    per DSN: `create_app` runs once per test.
    """
    global _sentry_dsn
    dsn = settings.sentry_dsn
    if not dsn or dsn == _sentry_dsn:
        return bool(dsn)
    import sentry_sdk  # only a deployment that asked for it pays for the import

    redact = Redactor.for_settings(settings)

    def before_send(event: Any, _hint: Any) -> Any:
        return _redact_event(event, redact)

    def before_breadcrumb(crumb: Any, _hint: Any) -> Any:
        if isinstance(crumb.get("message"), str):
            crumb["message"] = redact(crumb["message"])
        return crumb

    sentry_sdk.init(
        dsn=dsn,
        environment=settings.environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        send_default_pii=False,
        # A frame's locals include `settings` -- every secret this API has -- and the
        # write-token check's own arguments. Tracebacks are enough to find a bug.
        include_local_variables=False,
        max_request_body_size="never",
        before_send=before_send,
        before_breadcrumb=before_breadcrumb,
    )
    _sentry_dsn = dsn
    return True


def _redact_event(event: dict[str, Any], redact: Redactor) -> dict[str, Any]:
    logentry = event.get("logentry")
    if isinstance(logentry, dict):
        for key in ("message", "formatted"):
            if isinstance(logentry.get(key), str):
                logentry[key] = redact(logentry[key])
        if isinstance(logentry.get("params"), list):
            logentry["params"] = [
                redact(param) if isinstance(param, str) else param for param in logentry["params"]
            ]
    for exception in (event.get("exception") or {}).get("values") or []:
        if isinstance(exception.get("value"), str):
            exception["value"] = redact(exception["value"])
    request = event.get("request")
    if isinstance(request, dict):
        query = request.get("query_string")
        if isinstance(query, str) and _SIGNED_QUERY.search(query):
            request["query_string"] = REDACTED
        if isinstance(request.get("url"), str):
            request["url"] = redact(request["url"])
    return event
