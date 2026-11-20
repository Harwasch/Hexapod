"""Direct dependency/service checks, independent of HTTP threads or PostGIS."""

import time
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import Request, Response
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.api.deps import (
    require_bundle_part_token,
    require_write_or_phone_key,
    restrict_phone_routes,
)
from app.config import Settings
from app.models import Capture, CaptureFile
from app.services import capture_bundles, captures, handoff, phone_access, phone_key
from app.services.errors import UnauthorizedError
from app.services.ratelimit import PHONE_KEY, RateLimited, RateLimits
from tests.test_phone_app_access import ALLOW, KEY, WRITE, operations, settings


def request(config: Settings, token: str = "", method: str = "GET", path: str = "") -> Request:
    return Request(
        {
            "type": "http",
            "method": method,
            "headers": [(b"authorization", f"Bearer {token}".encode())],
            "client": ("127.0.0.1", 1),
            "app": SimpleNamespace(
                state=SimpleNamespace(settings=config, rate_limits=RateLimits())
            ),
            "path": path,
            "route": SimpleNamespace(path=path),
        }
    )


def credential(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_route_guard_is_deny_by_default_for_every_operation() -> None:
    config = settings()
    for method, path in operations():
        req = request(config, KEY, method, path)
        # Reads are never elevated by a phone key, so the guard leaves them alone.
        if (method, path) in ALLOW or method not in {"POST", "PUT", "PATCH", "DELETE"}:
            restrict_phone_routes(req, config)
        else:
            with pytest.raises(UnauthorizedError):
                restrict_phone_routes(req, config)
        # The guard never changes write-token or anonymous access.
        restrict_phone_routes(request(config, WRITE, method, path), config)
        restrict_phone_routes(request(config, "", method, path), config)


def test_combined_dependency_records_credential_and_rejects_handoff() -> None:
    config = settings()
    req = request(config)
    for token, kind in ((KEY, "phone"), (WRITE, "write")):
        require_write_or_phone_key(req, config, credential(token))
        assert req.state.api_credential == kind
    for supplied in (
        None,
        credential("wrong"),
        credential(handoff.issue(handoff.key_for(config), uuid.uuid4(), now=int(time.time()))),
    ):
        with pytest.raises(UnauthorizedError):
            require_write_or_phone_key(req, config, supplied)


def test_new_phone_auth_is_rate_limited_before_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    config = settings()
    req = request(config)
    for _ in range(PHONE_KEY.burst):
        with pytest.raises(UnauthorizedError):
            require_write_or_phone_key(req, config, credential("wrong"))
    check = Mock()
    monkeypatch.setattr(phone_key, "check", check)
    with pytest.raises(RateLimited):
        require_write_or_phone_key(req, config, credential(KEY))
    check.assert_not_called()


@pytest.mark.parametrize("configured_write", [WRITE, None])
def test_phone_parts_require_own_bundle_archive(configured_write: str | None) -> None:
    config = settings()
    config.api_write_token = configured_write
    cid, fid = uuid.uuid4(), uuid.uuid4()
    row = Capture(
        id=cid,
        metadata_={"origin": phone_access.ORIGIN, "bundleFormat": capture_bundles.FORMAT},
        files=[
            CaptureFile(
                id=fid, filename="capture.hexcap", content_type=capture_bundles.CONTENT_TYPE
            )
        ],
    )
    db = Mock(spec=Session)
    db.get.return_value = row
    req = request(config)
    require_bundle_part_token(
        cid, fid, req, config, db, Response(), credential(KEY), credential(KEY)
    )
    assert req.state.api_credential == "phone"
    for metadata, filename in (
        ({}, "capture.hexcap"),
        ({"origin": phone_access.ORIGIN}, "walk.mov"),
        (row.metadata_, "walk.mov"),
    ):
        row.metadata_, row.files[0].filename = metadata, filename
        with pytest.raises(UnauthorizedError):
            require_bundle_part_token(
                cid, fid, req, config, db, Response(), credential(KEY), credential(KEY)
            )


def test_phone_begin_forces_origin_and_resume_refuses_desktop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.schemas.capture import CaptureCreate

    config = settings()
    db = Mock(spec=Session)
    db.scalar.side_effect = [None, 0]
    create = Mock(return_value=Capture())
    monkeypatch.setattr(captures, "create_capture", create)
    bid = uuid.uuid4()
    capture_bundles.begin(
        db,
        bid,
        CaptureCreate(name="Phone", kind="video", metadata={"origin": "spoof"}),
        phone_settings=config,
    )
    assert create.call_args.args[1].metadata["origin"] == phone_access.ORIGIN
    desktop = Capture(id=uuid.uuid4(), metadata_={"bundleFormat": capture_bundles.FORMAT})
    db.scalar.side_effect = None
    db.scalar.return_value = desktop
    db.get.return_value = desktop
    with pytest.raises(UnauthorizedError):
        capture_bundles.begin(
            db, bid, CaptureCreate(name="Phone", kind="video"), phone_settings=config
        )


def test_existing_write_token_surface_contracts() -> None:
    from app.main import create_app
    from tests import test_write_token as contracts

    # These contract checks only inspect client.app.openapi(); no HTTP or DB is needed.
    surface = Mock()
    surface.app = create_app(settings())
    contracts.test_no_mutating_route_is_left_ungated(surface)
    contracts.test_only_the_upload_endpoints_accept_anything_but_the_write_token(surface)
    contracts.test_the_phone_key_opens_the_phone_routes_and_nothing_else(surface)
