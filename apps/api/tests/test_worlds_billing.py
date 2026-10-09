"""Immutable local billing imports; no cloud account calls or inferred billed amounts."""

from copy import deepcopy

import pytest

from app.worlds.store import Store
from tests import test_worlds as fixtures

setup = fixtures.setup


def owned(config, identity="owned", provider="runpod", provider_id="pod-owned"):
    return Store(config.data_dir).put(
        "worker",
        {
            "id": identity,
            "provider": provider,
            "providerId": provider_id,
            "managed": True,
            "status": "destroyed",
            "estimatedHourlyCost": 2,
            "createdAt": "2026-01-01T00:00:00+00:00",
            "endedAt": "2026-01-01T01:00:00+00:00",
        },
    )


def line(**overrides):
    return {
        "provider": "runpod",
        "workerId": "owned",
        "reference": "invoice-1/line-1",
        "source": "runpod-jan-export.csv",
        "amount": "1.25",
        "currency": "USD",
        "periodStart": "2026-01-01T00:00:00Z",
        "periodEnd": "2026-01-01T00:30:00Z",
        "kind": "charge",
        "category": "compute",
        **overrides,
    }


def submit(client, *rows):
    return client.post("/api/v1/worlds/billing/import", json={"rows": list(rows)})


def test_actual_import_partial_period_delta_and_provenance(setup):
    client, config, calls = setup
    original = deepcopy(owned(config))
    response = submit(client, line())
    assert response.status_code == 200
    value = response.json()
    assert value["inserted"] == 1
    assert value["records"][0]["sourceType"] == "operator-imported"
    assert value["workerTotals"][0]["estimatedComputeCostUSD"] == 1
    assert value["workerTotals"][0]["computeDeltaUSD"] == 0.25
    assert value["providerReportedTotals"] == []
    assert calls == []
    assert Store(config.data_dir).get("worker", "owned") == original
    usage = client.get("/api/v1/worlds/usage").json()
    assert usage["estimatedCostUSD"] == 2
    assert usage["importedActualCostUSD"] == 1.25
    assert usage["billedCostUSD"] is None
    assert usage["computeDeltaUSD"] == 0.25


def test_duplicate_import_is_idempotent_and_conflicts_rollback_batch(setup):
    client, config, _ = setup
    owned(config)
    assert submit(client, line()).json()["inserted"] == 1
    result = submit(client, line(amount=1.250000)).json()
    assert result["duplicates"] == 1
    assert result["inserted"] == 0
    response = submit(client, line(reference="new-line"), line(amount="1.26"))
    assert response.status_code == 409
    assert client.get("/api/v1/worlds/billing").json()["totalRecords"] == 1


def test_explicit_credit_net_and_overlapping_periods_are_counted_once(setup):
    client, config, _ = setup
    owned(config)
    result = submit(
        client, line(), line(reference="invoice-credit", kind="credit", amount="-0.25")
    ).json()
    total = result["workerTotals"][0]
    assert total["importedActual"] == 1
    assert total["estimatedComputeCostUSD"] == 1
    assert total["computeDeltaUSD"] == 0
    assert total["comparedComputeSeconds"] == 1800


def test_provider_id_reconciliation_requires_unique_owned_provider_match(setup):
    client, config, _ = setup
    row = line(providerId="pod-owned")
    del row["workerId"]
    result = submit(client, row).json()
    assert result["unmatchedCount"] == 1
    assert result["workerTotals"] == []
    owned(config)
    result = client.get("/api/v1/worlds/billing").json()
    assert result["unmatchedCount"] == 0
    assert result["records"][0]["workerId"] == "owned"
    owned(config, identity="duplicate-pod")
    assert client.get("/api/v1/worlds/billing").json()["unmatchedCount"] == 1


def test_provider_mismatch_never_assigns_actual_to_worker(setup):
    client, config, _ = setup
    owned(config, provider="lambda")
    value = submit(client, line()).json()
    assert value["unmatchedCount"] == 1
    assert value["records"][0]["workerId"] is None
    assert value["totals"][0]["unmatchedActual"] == 1.25


def test_currencies_and_noncompute_fees_stay_separate(setup):
    client, config, _ = setup
    owned(config)
    result = submit(
        client,
        line(),
        line(reference="storage", category="storage", amount="3"),
        line(reference="euro", currency="EUR", amount="5"),
    ).json()
    groups = {group["currency"]: group for group in result["workerTotals"]}
    assert groups["USD"]["importedActual"] == 4.25
    assert groups["USD"]["computeDeltaUSD"] == 0.25
    assert groups["EUR"]["computeDeltaUSD"] is None
    assert len(result["totals"]) == 2


@pytest.mark.parametrize(
    "override",
    [
        {"amount": "NaN"},
        {"amount": "Infinity"},
        {"amount": True},
        {"amount": "1000000001"},
        {"amount": "-1", "kind": "charge"},
        {"amount": "1", "kind": "credit"},
        {"amount": "0", "kind": "credit"},
        {"amount": "0.0000001"},
        {"currency": "usd"},
        {"currency": "USDD"},
        {"providerId": "also-specified"},
        {"workerId": None},
        {"reference": "   "},
        {"source": ""},
        {"periodStart": "2026-01-01T00:00:00"},
        {"periodEnd": "2025-12-31T00:00:00Z"},
        {"periodEnd": "2028-01-01T00:00:00Z"},
        {"sourceType": "provider-reported"},
    ],
)
def test_invalid_billing_inputs_fail_before_persistence(setup, override):
    client, config, calls = setup
    assert submit(client, line(**override)).status_code == 422
    assert Store(config.data_dir).records("billing") == []
    assert calls == []


def test_billing_routes_require_authentication_and_support_pagination(setup):
    client, config, _ = setup
    owned(config)
    assert (
        client.post(
            "/api/v1/worlds/billing/import",
            json={"rows": [line()]},
            headers={"Authorization": "Bearer wrong"},
        ).status_code
        == 401
    )
    assert (
        client.get("/api/v1/worlds/billing", headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )
    assert submit(client, line(), line(reference="second")).status_code == 200
    value = client.get("/api/v1/worlds/billing?limit=1&offset=1").json()
    assert len(value["records"]) == 1
    assert value["totalRecords"] == 2
    assert value["workerTotals"][0]["lineCount"] == 2


def provider_check(client, identity="owned", **params):
    return client.get(
        f"/api/v1/worlds/billing/runpod/{identity}",
        params={"startTime": "2026-01-01T00:00:00Z", "endTime": "2026-01-02T00:00:00Z", **params},
    )


def test_live_runpod_billing_is_scoped_read_only_and_separate(setup, monkeypatch):
    import httpx

    client, config, _ = setup
    owned(config)
    calls = []

    def request(method, path, payload=None):
        calls.append((method, path))
        return httpx.Response(
            200,
            json=[
                {
                    "podId": "pod-owned",
                    "amount": 2.4,
                    "time": "2026-01-01T00:00:00Z",
                    "timeBilledMs": 3600000,
                    "diskSpaceBilledGb": 20,
                    "secret": "must-not-be-forwarded",
                }
            ],
        )

    monkeypatch.setattr(
        "app.worlds.providers.RunPodProvider.api", lambda self, *a, **kw: request(*a, **kw)
    )
    result = provider_check(client).json()
    assert result["reportedAmountUSD"] == 2.4
    assert result["sourceType"] == "provider-reported"
    assert result["records"][0]["timeBilledMs"] == 3600000
    assert "secret" not in str(result)
    assert calls[0][0] == "GET"
    assert "podId=pod-owned" in calls[0][1]
    assert "grouping=podId" in calls[0][1]
    assert Store(config.data_dir).records("billing") == []


def test_live_billing_empty_is_unknown_and_other_workers_never_queried(setup, monkeypatch):
    import httpx

    client, config, _ = setup
    owned(config)
    calls = []
    monkeypatch.setattr(
        "app.worlds.providers.RunPodProvider.api",
        lambda *a, **kw: calls.append(True) or httpx.Response(200, json=[]),
    )
    assert provider_check(client).json()["reportedAmountUSD"] is None
    owned(config, identity="lambda-owned", provider="lambda")
    assert provider_check(client, "lambda-owned").status_code == 409
    assert provider_check(client, "unknown-id").status_code == 404
    assert provider_check(client, endTime="2026-06-01T00:00:00Z").status_code == 422
    assert len(calls) == 1


@pytest.mark.parametrize(
    "record",
    [
        {"podId": "unrelated", "amount": 2, "time": "2026-01-01T00:00:00Z"},
        {"podId": "pod-owned", "amount": True, "time": "2026-01-01T00:00:00Z"},
        {"podId": "pod-owned", "amount": "NaN", "time": "2026-01-01T00:00:00Z"},
        {"podId": "pod-owned", "amount": 2, "time": "2026-01-01T00:00:00"},
        {"podId": "pod-owned", "amount": 2, "time": "2025-01-01T00:00:00Z"},
    ],
)
def test_live_billing_rejects_mismatched_or_invalid_provider_records(setup, monkeypatch, record):
    import httpx

    client, config, _ = setup
    owned(config)
    monkeypatch.setattr(
        "app.worlds.providers.RunPodProvider.api",
        lambda *a, **kw: httpx.Response(200, json=[record]),
    )
    assert provider_check(client).status_code == 502
