from __future__ import annotations

import time
from collections.abc import Iterator

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.config import Settings
from app.main import create_app
from app.services.identity import jwks_client, principal_id
from tests.test_land import BODY

ISSUER = "https://identity.example.test/"
AUDIENCE = "land-api"


@pytest.fixture
def identity_client(db: Session, monkeypatch: pytest.MonkeyPatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
    jwk.update(kid="test", use="sig", alg="RS256")
    jwks_client.cache_clear()
    monkeypatch.setattr(jwt.PyJWKClient, "fetch_data", lambda self: {"keys": [jwk]})
    app = create_app(
        Settings(
            _env_file=None,
            land_auth_mode="oidc",
            land_oidc_issuer=ISSUER,
            land_oidc_audience=AUDIENCE,
            land_oidc_jwks_url=ISSUER + "jwks",
            api_write_token="pilot-token-must-not-bypass-oidc",
        )
    )

    def get_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = get_db

    def token(subject="alice", **overrides):
        now = int(time.time())
        claims = {"sub": subject, "iss": ISSUER, "aud": AUDIENCE, "iat": now, "exp": now + 300}
        claims.update(overrides)
        return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test"})

    with TestClient(app) as client:
        yield client, token
    jwks_client.cache_clear()


def headers(token, workspace=None):
    result = {"Authorization": "Bearer " + token}
    if workspace:
        result["X-Workspace-ID"] = workspace
    return result


def test_workspace_isolation_roles_and_revision_reads(identity_client):
    client, token = identity_client
    alice = headers(token())
    bob = headers(token("bob"))
    first = client.post("/api/v1/workspaces", headers=alice, json={"name": "Ranch"})
    assert first.status_code == 201, first.text
    workspace = first.json()["id"]
    alice = headers(token(), workspace)
    assert client.get("/api/v1/land", headers=headers(token())).status_code == 400
    response = client.post("/api/v1/land", json=BODY, headers=alice)
    assert response.status_code == 201, response.text
    land_id = response.json()["id"]
    assert client.get("/api/v1/workspaces", headers=bob).json() == []
    bob_workspace = client.post("/api/v1/workspaces", headers=bob, json={"name": "Other"}).json()[
        "id"
    ]
    for tail in ("", "/revisions"):
        assert (
            client.get(
                f"/api/v1/land/{land_id}{tail}", headers=headers(token("bob"), bob_workspace)
            ).status_code
            == 404
        )
        assert (
            client.get(
                f"/api/v1/land/{land_id}{tail}", headers=headers(token("bob"), workspace)
            ).status_code
            == 404
        )
    assert client.get("/api/v1/land", headers=headers(token("bob"), bob_workspace)).json() == []
    bob_id = principal_id(ISSUER, "bob")
    assert (
        client.put(
            "/api/v1/workspaces/members",
            headers=alice,
            json={"principalId": bob_id, "role": "viewer"},
        ).status_code
        == 200
    )
    bob = headers(token("bob"), workspace)
    assert client.get(f"/api/v1/land/{land_id}", headers=bob).status_code == 200
    assert client.get(f"/api/v1/land/{land_id}/revisions", headers=bob).status_code == 200
    assert client.post("/api/v1/land", json=BODY, headers=bob).status_code == 403
    assert (
        client.put(
            f"/api/v1/land/{land_id}", headers=bob, json={**BODY, "expectedRevision": 1}
        ).status_code
        == 403
    )
    assert client.delete(f"/api/v1/land/{land_id}", headers=bob).status_code == 403
    assert (
        client.put(
            "/api/v1/workspaces/members", headers=bob, json={"principalId": bob_id, "role": "owner"}
        ).status_code
        == 403
    )
    assert (
        client.put(
            "/api/v1/workspaces/members",
            headers=alice,
            json={"principalId": bob_id, "role": "editor"},
        ).status_code
        == 200
    )
    assert (
        client.put(
            f"/api/v1/land/{land_id}", headers=bob, json={**BODY, "expectedRevision": 1}
        ).status_code
        == 200
    )
    assert client.delete(f"/api/v1/workspaces/members/{bob_id}", headers=alice).status_code == 204
    assert client.get(f"/api/v1/land/{land_id}", headers=bob).status_code == 404
    alice_id = principal_id(ISSUER, "alice")
    assert client.delete(f"/api/v1/workspaces/members/{alice_id}", headers=alice).status_code == 409
    assert (
        client.put(
            "/api/v1/workspaces/members",
            headers=alice,
            json={"principalId": alice_id, "role": "viewer"},
        ).status_code
        == 409
    )


def test_oidc_rejects_invalid_claims_and_pilot_bypass(identity_client):
    client, token = identity_client
    assert client.get("/api/v1/workspaces").status_code == 401
    for invalid in (
        "pilot-token-must-not-bypass-oidc",
        token(exp=int(time.time()) - 60),
        token(aud="another-api"),
        token(iss="https://untrusted.example/"),
        token(sub=""),
        token(iat=int(time.time()) + 600),
    ):
        assert client.get("/api/v1/workspaces", headers=headers(invalid)).status_code == 401
    assert client.get("/api/v1/workspaces/identity", headers=headers(token())).json()[
        "principalId"
    ] == principal_id(ISSUER, "alice")
    assert (
        client.get(
            "/api/v1/land", headers=headers(token(), "00000000-0000-0000-0000-000000000001")
        ).status_code
        == 404
    )


def test_oidc_requires_complete_secure_configuration():
    with pytest.raises(ValueError, match="requires issuer"):
        Settings(_env_file=None, land_auth_mode="oidc")
    with pytest.raises(ValueError, match="HTTPS"):
        Settings(
            _env_file=None,
            land_auth_mode="oidc",
            land_oidc_issuer="http://insecure.test",
            land_oidc_audience="api",
            land_oidc_jwks_url=ISSUER + "jwks",
        )
