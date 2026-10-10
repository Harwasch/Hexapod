from __future__ import annotations

from datetime import timedelta
from typing import cast

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.base import utcnow
from app.models.workspace import WorkspaceInvitation
from app.services.identity import principal_id
from tests.test_workspaces import ISSUER, IdentityClient, headers

pytest_plugins = ["tests.test_workspaces"]


def test_invite_join_retry_scope_and_roles(identity_client: IdentityClient, db: Session) -> None:
    client, token = identity_client
    alice = headers(token())
    workspace = client.post("/api/v1/workspaces", headers=alice, json={"name": " Ranch "}).json()
    assert workspace["name"] == "Ranch"
    alice = headers(token(), workspace["id"])
    response = client.post(
        "/api/v1/workspaces/invitations",
        headers=alice,
        json={"label": "Field team", "role": "editor"},
    )
    assert response.status_code == 201, response.text
    assert "no-store" in response.headers["cache-control"]
    invite = response.json()
    row = db.scalar(select(WorkspaceInvitation))
    assert row is not None and row.token_hash != invite["token"]
    listed = client.get("/api/v1/workspaces/invitations", headers=alice).json()
    assert "token" not in listed[0] and "tokenHash" not in listed[0]
    bob = headers(token("bob"))
    body = {"token": invite["token"]}
    preview = client.post("/api/v1/workspaces/invitations/inspect", headers=bob, json=body)
    assert preview.status_code == 200 and preview.json()["workspaceName"] == "Ranch"
    assert client.get("/api/v1/workspaces", headers=bob).json() == []
    assert (
        client.post("/api/v1/workspaces/invitations/accept", headers=bob, json=body).status_code
        == 409
    )
    assert (
        client.put("/api/v1/workspaces/profile", headers=bob, json={"displayName": " Bob "}).json()[
            "displayName"
        ]
        == "Bob"
    )
    for _ in range(2):
        joined = client.post("/api/v1/workspaces/invitations/accept", headers=bob, json=body)
        assert joined.status_code == 200 and joined.json()["role"] == "editor"
    bob_workspace = headers(token("bob"), workspace["id"])
    assert client.get("/api/v1/workspaces/members", headers=bob_workspace).status_code == 403
    assert (
        client.post(
            "/api/v1/workspaces/invitations", headers=bob_workspace, json={"label": "No"}
        ).status_code
        == 403
    )
    assert client.get("/api/v1/workspaces/members", headers=alice).json()[1]["displayName"] == "Bob"
    assert (
        client.post(
            "/api/v1/workspaces/invitations/accept", headers=headers(token("eve")), json=body
        ).status_code
        == 410
    )
    # Removing a member must prevent invitation replay from restoring access.
    bob_id = principal_id(ISSUER, "bob")
    assert client.delete(f"/api/v1/workspaces/members/{bob_id}", headers=alice).status_code == 204
    assert (
        client.post("/api/v1/workspaces/invitations/accept", headers=bob, json=body).status_code
        == 410
    )


def test_invite_expiry_revocation_and_existing_members(
    identity_client: IdentityClient, db: Session
) -> None:
    client, token = identity_client

    def create_workspace(name: str) -> dict[str, str]:
        return cast(
            dict[str, str],
            client.post("/api/v1/workspaces", headers=headers(token()), json={"name": name}).json(),
        )

    workspace = create_workspace("One")
    other = create_workspace("Two")
    alice = headers(token(), workspace["id"])

    def invite() -> dict[str, str]:
        return cast(
            dict[str, str],
            client.post(
                "/api/v1/workspaces/invitations", headers=alice, json={"label": "Guest"}
            ).json(),
        )

    invitation = invite()
    body = {"token": invitation["token"]}
    # Existing owner is neither demoted nor consumes the link.
    assert (
        client.post("/api/v1/workspaces/invitations/accept", headers=alice, json=body).json()[
            "role"
        ]
        == "owner"
    )
    db.expire_all()
    stored = db.get(WorkspaceInvitation, invitation["id"])
    assert stored is not None and stored.accepted_at is None
    path = "/api/v1/workspaces/invitations/" + invitation["id"]
    assert client.delete(path, headers=headers(token(), other["id"])).status_code == 404
    assert client.delete(path, headers=alice).status_code == 204
    assert (
        client.post(
            "/api/v1/workspaces/invitations/inspect", headers=headers(token("bob")), json=body
        ).status_code
        == 410
    )
    expired = invite()
    stored = db.get(WorkspaceInvitation, expired["id"])
    assert stored is not None
    stored.expires_at = utcnow() - timedelta(seconds=1)
    db.commit()
    assert (
        client.post(
            "/api/v1/workspaces/invitations/accept",
            headers=headers(token("bob")),
            json={"token": expired["token"]},
        ).status_code
        == 410
    )
    assert (
        client.post(
            "/api/v1/workspaces/invitations/inspect", headers=alice, json={"token": "A" * 43}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/v1/workspaces/invitations",
            headers=alice,
            json={"label": "Owner", "role": "owner"},
        ).status_code
        == 422
    )
    assert (
        client.put(
            "/api/v1/workspaces/profile", headers=alice, json={"displayName": "  "}
        ).status_code
        == 422
    )
    assert client.post("/api/v1/workspaces/invitations/accept", json=body).status_code == 401


def test_single_use_invitation_serializes_competing_acceptances(
    identity_client: IdentityClient, db: Session
) -> None:
    from collections.abc import Iterator
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from fastapi import FastAPI

    from app.api.deps import _db

    client, token = identity_client
    workspace = client.post(
        "/api/v1/workspaces", headers=headers(token()), json={"name": "Concurrent"}
    ).json()["id"]
    invite = client.post(
        "/api/v1/workspaces/invitations",
        headers=headers(token(), workspace),
        json={"label": "One person"},
    ).json()
    participants = [headers(token(name)) for name in ("bob", "carol")]
    for who in participants:
        assert (
            client.put(
                "/api/v1/workspaces/profile",
                headers=who,
                json={"displayName": "Synthetic participant"},
            ).status_code
            == 200
        )
    db.rollback()
    bind = db.get_bind()

    def independent_db() -> Iterator[Session]:
        with Session(bind) as session:
            yield session

    app = cast(FastAPI, client.app)
    previous = app.dependency_overrides[_db]
    app.dependency_overrides[_db] = independent_db
    barrier = Barrier(2)

    def accept(who: dict[str, str]) -> int:
        barrier.wait(timeout=10)
        return client.post(
            "/api/v1/workspaces/invitations/accept", headers=who, json={"token": invite["token"]}
        ).status_code

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses = list(pool.map(accept, participants))
        assert sorted(statuses) == [200, 410]
    finally:
        app.dependency_overrides[_db] = previous


def test_pilot_cannot_create_profiles_or_redeem_invites(client: TestClient) -> None:
    assert (
        client.put("/api/v1/workspaces/profile", json={"displayName": "No identity"}).status_code
        == 409
    )
    assert (
        client.post("/api/v1/workspaces/invitations", json={"label": "No identity"}).status_code
        == 409
    )
    for operation in ("inspect", "accept"):
        assert (
            client.post(
                f"/api/v1/workspaces/invitations/{operation}", json={"token": "A" * 43}
            ).status_code
            == 409
        )
