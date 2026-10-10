"""CPU-only readiness and measurement tests; no inference or provider calls."""

import contextlib
import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark
import preflight

TOKEN = "private-worker-secret-not-for-reports-123"
HEALTH = {
    "protocolVersion": 1,
    "status": "ready",
    "models": [
        {
            "id": "astronex-world",
            "status": "ready",
            "version": "a" * 40,
            "checkpointRevision": "b" * 40,
            "readiness": "artifacts-verified-gpu-unverified",
            "servingMode": "resident-chunks",
            "reason": "private reason /home/alice " + TOKEN,
        }
    ],
}


class PreflightTests(unittest.TestCase):
    def test_readiness_only_gets_health_and_does_not_claim_gpu_inference(self):
        client = Mock()
        client.request.return_value = (200, {}, HEALTH)
        factory = Mock(return_value=client)
        result = preflight.run_preflight(
            "https://worker.example", TOKEN, client_factory=factory
        )
        self.assertTrue(result["ready"])
        self.assertFalse(result["inferenceStarted"])
        self.assertFalse(result["model"]["gpuInferenceVerified"])
        client.request.assert_called_once_with("GET", "/health")
        serialized = json.dumps(result)
        for private in [TOKEN, "alice", "worker.example"]:
            self.assertNotIn(private, serialized)

    def test_unavailable_selected_model_and_missing_model_fail(self):
        client = Mock()
        health = {
            **HEALTH,
            "models": [{**HEALTH["models"][0], "status": "unavailable"}],
        }
        client.request.return_value = (200, {}, health)
        result = preflight.run_preflight(
            "https://worker.example", TOKEN, client_factory=lambda *a, **k: client
        )
        self.assertFalse(result["ready"])
        self.assertEqual(result["failure"], "selected_model_unavailable")
        client.request.return_value = (200, {}, {**HEALTH, "models": []})
        result = preflight.run_preflight(
            "https://worker.example", TOKEN, client_factory=lambda *a, **k: client
        )
        self.assertEqual(result["failure"], "selected_model_not_installed")

    def test_url_and_token_validation_precedes_network(self):
        unsafe = [
            "http://remote.example",
            "https://name:secret@worker.example",
            "https://worker.example?token=secret",
            "https://worker.example#secret",
            "https://worker.example/\nsecret",
            "file:///tmp/worker",
            "https://worker.example:invalid",
        ]
        with patch("preflight.urllib.request.build_opener") as opener:
            for url in unsafe:
                with self.subTest(url=url), self.assertRaises(preflight.WorkerError):
                    preflight.WorkerClient(url, TOKEN)
            with self.assertRaises(preflight.WorkerError):
                preflight.WorkerClient("https://worker.example", "short")
            opener.assert_not_called()
        self.assertEqual(
            preflight.validate_url("http://127.0.0.1:8789/"), "http://127.0.0.1:8789"
        )

    def test_redirect_is_refused_and_never_retried(self):
        self.assertIsNone(
            preflight.NoRedirect().redirect_request(
                None, None, 302, "", {}, "https://attacker.example"
            )
        )
        client = preflight.WorkerClient("https://worker.example", TOKEN)
        client.opener = Mock()
        client.opener.open.side_effect = urllib.error.HTTPError(
            "https://worker.example/private", 302, TOKEN, {}, None
        )
        with self.assertRaises(preflight.WorkerError) as error:
            client.request("GET", "/health")
        self.assertEqual(error.exception.code, "worker_redirect_refused")
        client.opener.open.assert_called_once()
        request = client.opener.open.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer " + TOKEN)
        self.assertEqual(request.full_url, "https://worker.example/health")
        self.assertNotIn(TOKEN, str(error.exception))

    def test_bounded_responses_and_auth_errors_are_safe(self):
        client = preflight.WorkerClient("https://worker.example", TOKEN)
        response = Mock()
        response.read.return_value = b"x" * (1024 * 1024 + 1)
        client.opener = Mock()
        client.opener.open.return_value.__enter__ = Mock(return_value=response)
        client.opener.open.return_value.__exit__ = Mock(return_value=False)
        with self.assertRaises(preflight.WorkerError) as error:
            client.request("GET", "/health")
        self.assertEqual(error.exception.code, "worker_response_too_large")
        client.opener.open.side_effect = urllib.error.HTTPError(
            "https://worker.example/private", 401, TOKEN, {}, None
        )
        with self.assertRaises(preflight.WorkerError) as error:
            client.request("GET", "/health")
        self.assertEqual(error.exception.code, "worker_authentication_failed")

    def test_optional_configuration_is_presence_only_and_never_provisions(self):
        result = preflight.configuration_checks(
            {
                "WORLD_RUNPOD_ALLOW_PROVISION": "true",
                "WORLD_RUNPOD_API_KEY": TOKEN,
                "WORLD_RUNPOD_TEMPLATE_ID": "private-template",
                "WORLD_GATEWAY_TOKEN": TOKEN,
                "WORLD_RUNPOD_GATEWAY_URL": "https://worker.example",
                "WORLD_DATA_DIR": "/private-volume/worlds",
                "WORLD_DATA_PERSISTENT": "true",
                "WORLD_MAX_WORKER_HOURLY_COST": "2.5",
                "API_WRITE_TOKEN": TOKEN,
            }
        )
        self.assertEqual(result["runpod"]["provisioningConfiguration"], "complete")
        self.assertFalse(result["runpod"]["providerCredentialsVerified"])
        self.assertNotIn(TOKEN, json.dumps(result))
        self.assertNotIn("private-template", json.dumps(result))
        self.assertNotIn("private-volume", json.dumps(result))
        self.assertFalse(result["runpod"]["persistentVolumeVerified"])

    def test_config_requires_persistence_lifecycle_and_cost_cap(self):
        base = {
            "WORLD_RUNPOD_ALLOW_PROVISION": "true",
            "WORLD_RUNPOD_API_KEY": TOKEN,
            "WORLD_RUNPOD_TEMPLATE_ID": "approved",
            "WORLD_GATEWAY_TOKEN": TOKEN,
            "WORLD_DATA_DIR": "/data/worlds",
            "WORLD_DATA_PERSISTENT": "true",
            "WORLD_MAX_WORKER_HOURLY_COST": "2.5",
            "API_WRITE_TOKEN": TOKEN,
        }
        for field, value in [
            ("WORLD_DATA_DIR", "relative/path"),
            ("WORLD_DATA_PERSISTENT", "false"),
            ("WORLD_LIFECYCLE_ENABLED", "false"),
            ("WORLD_MAX_WORKER_HOURLY_COST", "nan"),
            ("WORLD_MAX_MANAGED_WORKERS", "9"),
            ("API_WRITE_TOKEN", ""),
            ("WORLD_GATEWAY_TOKEN", "too-short"),
        ]:
            with self.subTest(field=field):
                result = preflight.configuration_checks({**base, field: value})
                self.assertFalse(result["ready"])
                self.assertEqual(
                    result["runpod"]["provisioningConfiguration"], "incomplete"
                )

    def test_config_remote_reconstruction_needs_token_and_local_does_not(self):
        remote = preflight.configuration_checks(
            {"WORLD_RECONSTRUCTION_GATEWAY_URL": "https://worker.example"}
        )
        self.assertFalse(remote["ready"])
        self.assertIn("WORLD_RECONSTRUCTION_GATEWAY_TOKEN", remote["invalidFields"])
        local = preflight.configuration_checks(
            {"WORLD_RECONSTRUCTION_GATEWAY_URL": "http://localhost:9000"}
        )
        self.assertTrue(local["ready"])

    def test_existing_gateways_require_authentication_without_provisioning(self):
        for name, url in (
            ("WORLD_LOCAL_GATEWAY_URL", "http://localhost:8789"),
            ("WORLD_RUNPOD_GATEWAY_URL", "https://worker.example"),
            ("WORLD_MODAL_GATEWAY_URL", "https://worker.example"),
        ):
            with self.subTest(provider=name):
                environment = {name: url}
                result = preflight.configuration_checks(environment)
                self.assertFalse(result["ready"])
                self.assertIn("WORLD_GATEWAY_TOKEN", result["invalidFields"])
                self.assertIn("API_WRITE_TOKEN", result["invalidFields"])
                result = preflight.configuration_checks({
                    **environment, "WORLD_GATEWAY_TOKEN": TOKEN, "API_WRITE_TOKEN": TOKEN,
                })
                self.assertTrue(result["ready"])
                self.assertEqual(result["runpod"]["provisioningConfiguration"], "disabled")

    def test_cli_prints_json_and_nonzero_without_configuration(self):
        output = io.StringIO()
        with (
            patch.dict("os.environ", {}, clear=True),
            contextlib.redirect_stdout(output),
        ):
            code = preflight.main([])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(output.getvalue())["ready"])


class Clock:
    def __init__(self):
        self.now = 0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class BenchmarkClient:
    def __init__(
        self, *, error=False, no_frames=False, indexed=True, cleanup_error=False
    ):
        self.error, self.no_frames, self.indexed, self.cleanup_error = (
            error,
            no_frames,
            indexed,
            cleanup_error,
        )
        self.calls = []
        self.frames = 0

    def request(self, method, path, data=None):
        self.calls.append((method, path, data))
        if path == "/health":
            return 200, {}, HEALTH
        if method == "POST":
            return 201, {}, {"status": "loading"}
        if method == "DELETE":
            if self.cleanup_error:
                raise preflight.WorkerError("worker_unreachable_or_tls_failed")
            return 204, {}, b""
        if path.endswith("/frame"):
            if self.no_frames:
                return 204, {}, b""
            self.frames += 1
            return (
                200,
                {
                    "Content-Type": "image/jpeg",
                    **({"X-Frame-Index": str(self.frames)} if self.indexed else {}),
                },
                b"\xff\xd8\xfffixture",
            )
        return (
            200,
            {},
            {
                "status": "error" if self.error else "playing",
                "error": TOKEN,
                "frameIndex": 200,
                "totalGeneratedFrames": 240,
                "totalGenerationSeconds": 20,
                "modelLoadSeconds": 9,
                "generatedFPS": 13,
            },
        )


class BenchmarkTests(unittest.TestCase):
    def run_fixture(self, client, **kwargs):
        clock = Clock()
        return benchmark.run_benchmark(
            client, seconds=1, clock=clock.time, sleep=clock.sleep, **kwargs
        )

    def test_generation_is_distinct_from_sampled_delivery_and_price_is_estimate(self):
        client = BenchmarkClient()
        result = self.run_fixture(client, hourly_price=3.6)
        self.assertTrue(result["passed"])
        self.assertEqual(result["generation"]["generatedFPS"], 12)
        self.assertEqual(result["generation"]["modelLoadSeconds"], 9)
        self.assertLess(result["delivery"]["uniqueIndexedFrames"], 240)
        self.assertEqual(result["pricing"]["kind"], "estimate")
        self.assertAlmostEqual(result["pricing"]["estimatedSessionCostUSD"], 0.001)
        self.assertEqual(client.calls[-1][0], "DELETE")
        self.assertNotIn(TOKEN, json.dumps(result))
        self.assertNotIn("prompt", json.dumps(result))

    def test_error_and_no_frames_fail_and_still_cleanup(self):
        for client, code in [
            (BenchmarkClient(error=True), "model_inference_failed"),
            (BenchmarkClient(no_frames=True), "no_frames_delivered"),
        ]:
            with self.subTest(code=code):
                result = self.run_fixture(client)
                self.assertFalse(result["passed"])
                self.assertEqual(result["failure"], code)
                self.assertEqual(result["cleanup"], "session_deleted")

    def test_unindexed_frames_are_content_lower_bound_not_none_frame(self):
        result = self.run_fixture(BenchmarkClient(indexed=False))
        self.assertEqual(result["delivery"]["uniqueIndexedFrames"], 0)
        self.assertEqual(result["delivery"]["distinctUnindexedImages"], 1)

    def test_cleanup_failure_fails_benchmark(self):
        result = self.run_fixture(BenchmarkClient(cleanup_error=True))
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure"], "session_cleanup_failed")

    def test_long_benchmark_renews_worker_lease_and_still_deletes_session(self):
        clock = Clock()
        client = BenchmarkClient()
        heartbeat_times = []
        request = client.request

        def leased_request(method, path, data=None):
            if path.endswith("/heartbeat"):
                self.assertEqual((method, data), ("POST", {"active": True}))
                heartbeat_times.append(clock.time())
            elif path.endswith("/frame"):
                self.assertTrue(heartbeat_times)
                self.assertLess(clock.time() - heartbeat_times[-1], 30)
            return request(method, path, data)

        client.request = leased_request
        result = benchmark.run_benchmark(
            client, seconds=65, clock=clock.time, sleep=clock.sleep,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(len(heartbeat_times), 4)
        self.assertEqual(client.calls[-1][0], "DELETE")

    def test_invalid_price_seed_and_duration_do_not_start_inference(self):
        client = BenchmarkClient()
        for kwargs in ({"hourly_price": float("nan")}, {"seed": -1}, {"seconds": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(preflight.WorkerError):
                benchmark.run_benchmark(client, **kwargs)
        self.assertEqual(client.calls, [])

    def test_cli_exits_nonzero_for_failed_measurement(self):
        with (
            patch("benchmark.WorkerClient", return_value=BenchmarkClient(error=True)),
            patch(
                "benchmark.run_benchmark",
                return_value={"passed": False, "failure": "model_inference_failed"},
            ),
            patch.object(Path, "write_text"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(benchmark.main(["--url", "https://worker.example"]), 1)


if __name__ == "__main__":
    unittest.main()
