"""Logging, the 422/500 line, and Sentry -- app/observability.py and app/main.py's handlers."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import observability
from app.config import Settings
from app.main import create_app
from app.observability import JsonFormatter, Redactor, configure_logging, init_sentry
from app.services.errors import InvalidInputError

PRESIGNED = (
    "https://acct.r2.cloudflarestorage.com/twin-assets/captures/abc/video.mp4?uploadId=x"
    "&partNumber=3&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA%2F20260101"
    "&X-Amz-Signature=deadbeef"
)
HANDOFF = "h1." + "a" * 32 + ".1790000000.1790021600.c2lnbmF0dXJlLXNpZ25hdHVyZQ"
WRITE_TOKEN = "write-token-0123456789abcdef"


def _record(message: str, *args: object, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("twin.api", logging.INFO, __file__, 1, message, args, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_redactor_strips_credentials_by_shape_and_by_value() -> None:
    redact = Redactor([WRITE_TOKEN, None, "short"])
    text = redact(
        f"put {PRESIGNED} with Authorization: Bearer abc.def-123 for {HANDOFF}; "
        f"key {WRITE_TOKEN}; hash pbkdf2_sha256$600000$abcd$ef01; plain https://x.dev/a.json?v=2"
    )
    assert "deadbeef" not in text and "AKIA" not in text
    assert (
        "https://acct.r2.cloudflarestorage.com/twin-assets/captures/abc/video.mp4?[redacted]"
        in (text)
    )
    assert "Bearer [redacted]" in text and "abc.def-123" not in text
    assert HANDOFF not in text and "h1.[redacted]" in text
    assert WRITE_TOKEN not in text
    assert "600000$abcd" not in text
    # An ordinary query string is not a credential and is left as it is.
    assert "https://x.dev/a.json?v=2" in text
    # Too short to be a real secret; replacing it would mangle ordinary words.
    assert "short" in redact("a short line")


def test_json_lines_carry_level_logger_extras_and_no_secrets() -> None:
    formatter = JsonFormatter(Redactor([WRITE_TOKEN]))
    line = formatter.format(
        _record("presigned %s for %s", PRESIGNED, WRITE_TOKEN, path="/api/v1/x", attempt=2)
    )
    payload = json.loads(line)
    assert payload["level"] == "info"
    assert payload["logger"] == "twin.api"
    assert payload["path"] == "/api/v1/x" and payload["attempt"] == 2
    assert "deadbeef" not in payload["message"] and WRITE_TOKEN not in payload["message"]
    assert payload["time"].endswith("+00:00")

    try:
        raise RuntimeError(f"failed with Bearer {WRITE_TOKEN}")
    except RuntimeError:
        failed = _record("boom")
        failed.exc_info = sys.exc_info()
    payload = json.loads(formatter.format(failed))
    assert "RuntimeError" in payload["exception"]
    assert WRITE_TOKEN not in payload["exception"]


def test_configure_logging_is_idempotent_and_lets_info_lines_through(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = Settings(log_format="json", log_level="info", api_write_token=WRITE_TOKEN)
    configure_logging(settings)
    configure_logging(settings)
    ours = [h for h in logging.getLogger().handlers if h.get_name() == "twin"]
    assert len(ours) == 1
    assert logging.getLogger("twin").level == logging.INFO

    logging.getLogger("twin.api").info("started with %s", WRITE_TOKEN, extra={"port": 8000})
    logging.getLogger("twin.api").debug("not at INFO")
    logging.getLogger("botocore.credentials").info("libraries stay at WARNING")
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [line["message"] for line in lines] == ["started with [redacted]"]
    assert lines[0]["port"] == 8000

    configure_logging(Settings(log_format="text", log_level="DEBUG"))
    logging.getLogger("twin.api").debug("now at DEBUG")
    assert "DEBUG twin.api: now at DEBUG" in capsys.readouterr().err


def test_log_format_defaults_to_json_in_production_only() -> None:
    assert observability.log_format(Settings(APP_ENV="production")) == "json"
    assert observability.log_format(Settings()) == "text"
    assert observability.log_format(Settings(APP_ENV="production", log_format="TEXT")) == "text"


def _app_raising(exc: Exception) -> TestClient:
    app = create_app()

    @app.get("/boom")
    def boom() -> None:
        raise exc

    return TestClient(app)


def test_an_intentional_invalid_input_is_a_422_with_its_message() -> None:
    response = _app_raising(InvalidInputError("unknown recipe 'x'")).get("/boom")
    assert response.status_code == 422
    assert response.json()["detail"] == "unknown recipe 'x'"


def test_any_other_value_error_is_a_logged_500_that_leaks_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    response = _app_raising(ValueError("invalid literal for int() with base 10: 'x'")).get("/boom")
    assert response.status_code == 500
    body = response.json()
    assert body["title"] == "Internal error"
    assert "int()" not in json.dumps(body)
    logged = capsys.readouterr().err
    assert "unexpected ValueError on GET /boom" in logged
    assert "invalid literal for int()" in logged  # the traceback is in the log, not the body


def test_a_pydantic_error_inside_a_route_is_a_500_not_a_422() -> None:
    """pydantic's ValidationError is a ValueError. Raised by request parsing it is a 422
    (FastAPI's RequestValidationError); raised by code validating data the API holds, it
    is a bug, and used to be reported as the caller's."""
    from pydantic import BaseModel

    class Strict(BaseModel):
        n: int

    try:
        Strict.model_validate({"n": "not a number"})
    except ValueError as error:
        response = _app_raising(error).get("/boom")
    assert response.status_code == 500


def test_sentry_is_not_started_without_a_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(observability, "_sentry_dsn", None)
    assert init_sentry(Settings()) is False


def test_sentry_starts_with_a_dsn_and_sends_no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentry_sdk

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(sentry_sdk, "init", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(observability, "_sentry_dsn", None)
    dsn = "https://public@o0.ingest.sentry.io/1"
    settings = Settings(sentry_dsn=dsn, api_write_token=WRITE_TOKEN, APP_ENV="staging")
    assert init_sentry(settings) is True
    assert init_sentry(settings) is True  # once per DSN, however many apps are created
    assert len(calls) == 1
    options = calls[0]
    assert options["dsn"] == dsn and options["environment"] == "staging"
    assert options["send_default_pii"] is False
    assert options["include_local_variables"] is False
    assert options["max_request_body_size"] == "never"

    event = options["before_send"](
        {
            "logentry": {"message": "token %s", "params": [WRITE_TOKEN]},
            "exception": {"values": [{"type": "ValueError", "value": f"bad {PRESIGNED}"}]},
            "request": {"url": PRESIGNED, "query_string": "X-Amz-Signature=deadbeef"},
        },
        {},
    )
    assert WRITE_TOKEN not in json.dumps(event)
    assert "deadbeef" not in json.dumps(event)
    crumb = options["before_breadcrumb"]({"message": f"Bearer {WRITE_TOKEN}"}, {})
    assert WRITE_TOKEN not in crumb["message"]
