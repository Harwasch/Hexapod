"""The shared write token.

The unset case -- no token, writes open -- is what every other test in this suite
runs under, so it is covered by them rather than mocked here. What is asserted here is
the configured case, and the guard that stops the unset case reaching production.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import _db, require_write_token
from app.config import Settings
from app.main import create_app
from app.services.errors import UnauthorizedError
from tests.conftest import site_payload

TOKEN = "correct-horse-battery-staple"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

#: Security schemes that actually gate a write. Every mutating operation in the document
#: must name one of these and nothing else; see the test at the bottom of this file.
GATING_SCHEMES = {"writeToken", "uploadToken"}


@pytest.fixture
def client(db: Session) -> Iterator[TestClient]:
    """An app that actually has a token configured, not a dependency override.

    The test overrides the database session and nothing else, so the dependency under
    test is the real one reading the real setting.
    """
    app = create_app(Settings(api_write_token=TOKEN))

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    with TestClient(app) as test_client:
        yield test_client


def test_reads_stay_open(client: TestClient) -> None:
    assert client.get("/api/v1/sites").status_code == 200
    assert client.get("/api/v1/captures").status_code == 200
    assert client.get("/api/v1/jobs").status_code == 200
    assert client.get("/api/v1/layers").status_code == 200
    assert client.get("/api/v1/health").status_code == 200


def test_a_write_without_a_token_is_refused(client: TestClient) -> None:
    response = client.post("/api/v1/sites", json=site_payload())
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    body = response.json()
    assert body["title"] == "Unauthorized"
    assert body["status"] == 401
    assert client.get("/api/v1/sites").json() == []  # and nothing was created


@pytest.mark.parametrize(
    "header",
    [
        {"Authorization": "Bearer wrong"},
        # A prefix of the real token: the comparison is over the whole value.
        {"Authorization": f"Bearer {TOKEN[:-1]}"},
        {"Authorization": TOKEN},  # no scheme
        {"Authorization": "Basic " + TOKEN},
        {"X-Api-Token": TOKEN},  # the header this API does not read
    ],
)
def test_a_wrong_token_is_refused(client: TestClient, header: dict[str, str]) -> None:
    response = client.post("/api/v1/sites", json=site_payload(), headers=header)
    assert response.status_code == 401


def test_a_non_ascii_configured_token_does_not_crash_the_comparison() -> None:
    """`hmac.compare_digest` raises TypeError on a non-ASCII `str`, and a token read
    from the environment can be anything, so the check encodes both sides first. It
    cannot be exercised over HTTP -- a header value must be ASCII -- so this calls the
    dependency directly.
    """
    settings = Settings(api_write_token="pässwörd")
    good = HTTPAuthorizationCredentials(scheme="Bearer", credentials="pässwörd")
    bad = HTTPAuthorizationCredentials(scheme="Bearer", credentials="something else")
    require_write_token(settings, good)  # no exception: the good token is accepted
    with pytest.raises(UnauthorizedError):
        require_write_token(settings, bad)
    with pytest.raises(UnauthorizedError):
        require_write_token(settings, None)


def test_the_right_token_is_accepted(client: TestClient) -> None:
    response = client.post("/api/v1/sites", json=site_payload(), headers=AUTH)
    assert response.status_code == 201


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("post", "/api/v1/captures", {"name": "x", "kind": "video"}),
        ("post", "/api/v1/captures/{id}/files", {"filename": "a.ply"}),
        ("post", "/api/v1/captures/{id}/files/{id}/parts", {}),
        (
            "post",
            "/api/v1/captures/{id}/files/{id}/complete",
            {"parts": [{"partNumber": 1, "etag": "x"}]},
        ),
        ("post", "/api/v1/captures/{id}/files/{id}/abort", None),
        ("post", "/api/v1/captures/{id}/process", {"recipe": "splat-ingest"}),
        ("post", "/api/v1/captures/{id}/handoff", None),
        ("post", "/api/v1/jobs/{id}/cancel", None),
        ("post", "/api/v1/layers", {"name": "x"}),
        ("post", "/api/v1/assets", {"name": "x"}),
        ("post", "/api/v1/plans", {"projectId": "p"}),
        ("patch", "/api/v1/sites/{id}", {}),
        ("delete", "/api/v1/sites/{id}", None),
        ("delete", "/api/v1/layers/{id}", None),
        ("delete", "/api/v1/assets/{id}", None),
        ("delete", "/api/v1/plans/{id}", None),
        ("put", "/api/v1/plans/{id}", {}),
        ("patch", "/api/v1/plans/{id}/status", {}),
        # Mutate no rows, but spend the account's Anthropic credits per request.
        ("post", "/api/v1/agent/plan-draft", {"goal": "x", "projectId": "p"}),
        ("post", "/api/v1/agent/outline", {"image": "x", "point": [0, 0]}),
    ],
)
def test_every_mutating_route_is_gated(
    client: TestClient, method: str, path: str, body: dict[str, object] | None
) -> None:
    """401 before validation, before the row is looked up: the ids here are nonsense
    on purpose, and a gated route must not tell an anonymous caller whether they exist.
    """
    url = path.replace("{id}", "00000000-0000-0000-0000-000000000000")
    response = client.request(method, url, json=body)
    assert response.status_code == 401, f"{method.upper()} {url} -> {response.status_code}"


def test_the_thumbnail_upload_is_gated(client: TestClient) -> None:
    response = client.post(
        "/api/v1/sites/00000000-0000-0000-0000-000000000000/thumbnail",
        files={"file": ("t.png", b"\x89PNG", "image/png")},
    )
    assert response.status_code == 401


# --- the unset case, and why it cannot reach production ----------------------


def test_without_a_token_writes_are_open(db: Session) -> None:
    """A fresh checkout with no configuration still works -- the way object storage
    degrades when it is not configured."""
    app = create_app(Settings(api_write_token=None))

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    with TestClient(app) as unguarded:
        assert unguarded.post("/api/v1/sites", json=site_payload()).status_code == 201


def test_production_without_a_token_refuses_to_start() -> None:
    with pytest.raises(RuntimeError, match="API_WRITE_TOKEN"):
        create_app(Settings(APP_ENV="production"))


def test_production_with_a_token_starts() -> None:
    assert create_app(Settings(APP_ENV="production", api_write_token=TOKEN)) is not None


def test_no_mutating_route_is_left_ungated(client: TestClient) -> None:
    """The list above is hand-written, so this checks the whole surface instead.

    A new POST/PUT/PATCH/DELETE added without `dependencies=[RequireWriteToken]` is the
    easy mistake: the route works, every existing test passes, and the API is quietly
    open again. This asserts on the generated OpenAPI document rather than FastAPI's
    route objects -- the document is the contract clients actually read, and each gated
    operation carries a `security` entry there, so the check cannot drift from what is
    published.

    A5 made this check *narrower*, not looser. The four upload endpoints now carry
    `uploadToken` instead of `writeToken`, because a phone-handoff token is accepted
    there as well -- so a test that only asked "is `security` non-empty?" would from now
    on be satisfied by any scheme at all, including one that gates nothing. So the
    schemes are named: a mutating operation must be gated by a credential from
    `GATING_SCHEMES`, and adding a scheme to that set is a deliberate act with a reviewer
    attached. `uploadToken` belongs in it because it is a real gate -- it demands the
    write token or an unexpired signature over *this operation's* capture id
    (`app.api.deps.require_upload_token`), never nothing.
    """
    spec = client.app.openapi()  # type: ignore[attr-defined]
    gated: set[str] = set()
    ungated: set[str] = set()
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            if method.upper() not in {"POST", "PUT", "PATCH", "DELETE"}:
                continue
            schemes = {name for entry in operation.get("security", []) for name in entry}
            target = gated if schemes and schemes <= GATING_SCHEMES else ungated
            target.add(f"{method.upper()} {path}")

    assert gated, "no gated operations in the document -- the check is looking in the wrong place"
    assert not ungated, f"mutating operations with no write token: {sorted(ungated)}"


def test_only_the_upload_endpoints_accept_anything_but_the_write_token(
    client: TestClient,
) -> None:
    """`uploadToken` is the one scheme that is not simply the write token, so where it
    appears is worth pinning down: a handoff token must never become a credential for a
    route that was not about putting bytes into one capture."""
    spec = client.app.openapi()  # type: ignore[attr-defined]
    accepts_handoff = {
        f"{method.upper()} {path}"
        for path, operations in spec["paths"].items()
        for method, operation in operations.items()
        if any("uploadToken" in entry for entry in operation.get("security", []))
    }
    assert accepts_handoff == {
        "POST /api/v1/captures/{capture_id}/files",
        "POST /api/v1/captures/{capture_id}/files/{file_id}/parts",
        "POST /api/v1/captures/{capture_id}/files/{file_id}/complete",
        "POST /api/v1/captures/{capture_id}/files/{file_id}/abort",
    }
