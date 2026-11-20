"""Local-only tests: python3 -m unittest discover -s .github/scripts."""

import importlib.util
import io
import json
from pathlib import Path
import unittest
from contextlib import redirect_stdout
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "phone_smoke", Path(__file__).resolve().parents[2] / "scripts/phone_smoke.py"
)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)

CAPTURE = "11111111-1111-4111-8111-111111111111"
FILE = "22222222-2222-4222-8222-222222222222"
TOKEN = "phone-secret-value"
SIGNED = "https://storage.example/part?X-Amz-Signature=private-signature"


class SSETests(unittest.TestCase):
    def test_api_json_events_and_keepalives(self):
        self.assertTrue(smoke.stream_done(io.BytesIO(
            b': keepalive\r\n\r\ndata: {"type":"status"}\r\n\r\n'
            b'data: {"type":"done"}\r\n\r\n'
        )))

    def test_named_multiline_event(self):
        lines = io.BytesIO(b'event: done\ndata: {\ndata: "ok": true}\n\n')
        self.assertEqual(list(smoke.sse_events(lines)), [("done", '{\n"ok": true}')])

    def test_requires_terminal_framed_done(self):
        for body in (b'', b'data: {"type":"done"}\n',
                     b'data: {"type":"status"}\n\n',
                     b'data: {"type":"done"}\n\ndata: {"type":"status"}\n\n'):
            with self.subTest(body=body):
                self.assertFalse(smoke.stream_done(io.BytesIO(body)))

    def test_malformed_data_is_failure(self):
        with self.assertRaises(ValueError):
            smoke.stream_done(io.BytesIO(b'data: invalid\n\n'))


class OutputTests(unittest.TestCase):
    def test_redacts_token_headers_urls_and_newlines(self):
        output = smoke.redact(
            f'Authorization: Bearer {TOKEN}\n{SIGNED}\nHTTPS://example/secret\n'
            f'Bearer other-secret\r\n{TOKEN.upper()}', TOKEN
        )
        for secret in (TOKEN, TOKEN.upper(), "private-signature", "other-secret",
                       "https://", "HTTPS://", "\n", "\r", "Authorization:"):
            self.assertNotIn(secret, output)

    def test_report_redacts_all_output_fields(self):
        runner = smoke.Smoke("https://api.example", TOKEN)
        runner.step = TOKEN
        output = io.StringIO()
        with redirect_stdout(output):
            runner.report("FAIL", SIGNED)
        self.assertEqual(output.getvalue(), 'FAIL [REDACTED] 0 [URL]\n')
        self.assertTrue(runner.failed)

    def test_exception_details_are_never_printed(self):
        runner = smoke.Smoke("https://api.example", TOKEN)
        runner.api = mock.Mock(side_effect=RuntimeError(TOKEN + SIGNED))
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(runner.run(), 1)
        self.assertNotIn(TOKEN, output.getvalue())
        self.assertNotIn("private-signature", output.getvalue())


class ProtocolTests(unittest.TestCase):
    def test_upload_asks_for_windows_and_stops_before_processing(self):
        runner = smoke.Smoke("https://api.example", TOKEN)
        size = (smoke.FIXTURE / "capture.hexcap").stat().st_size
        calls = []

        def request(step, method, path, body=None, **options):
            calls.append((method, path, body, options))
            runner.step, runner.status = step, 200
            if step == "begin-bundle":
                result = {"id": CAPTURE}
            elif step == "bundle-upload":
                result = {"id": FILE, "bytes": size}
            elif step == "parts-window":
                result = {"partSize": size // 2, "partsTotal": 2,
                          "parts": [{"partNumber": body["firstPartNumber"], "url": SIGNED}]}
            elif step == "put-part":
                return {"ETag": '"etag"'}, b''
            elif step == "complete-upload":
                result = {"status": "complete"}
            else:
                result = {}
            return {}, json.dumps(result).encode()

        runner.request = request
        with redirect_stdout(io.StringIO()):
            runner.upload()
        self.assertEqual(runner.capture_id, CAPTURE)
        parts = [call for call in calls if call[1].endswith('/parts')]
        self.assertEqual([call[2] for call in parts],
                         [{"firstPartNumber": 1, "count": 1},
                          {"firstPartNumber": 2, "count": 1}])
        puts = [call for call in calls if call[3].get("signed")]
        self.assertEqual(b''.join(call[2] for call in puts),
                         (smoke.FIXTURE / "capture.hexcap").read_bytes())
        self.assertFalse(any("process" in call[1] for call in calls))
        complete = next(call for call in calls if call[1].endswith('/complete'))
        self.assertEqual(complete[2]["parts"],
                         [{"partNumber": 1, "etag": '"etag"'},
                          {"partNumber": 2, "etag": '"etag"'}])

    def test_storage_request_has_no_api_credential(self):
        runner = smoke.Smoke("https://api.example", TOKEN)
        response = mock.MagicMock(code=200, headers={"ETag": "etag"})
        response.read.return_value = b''
        runner.opener.open = mock.Mock(return_value=response)
        runner.request("put-part", "PUT", SIGNED, b'bytes', signed=True)
        request = runner.opener.open.call_args.args[0]
        self.assertIsNone(request.get_header("Authorization"))

    def test_only_exact_missing_reasoner_response_skips(self):
        for detail, skip in (("Reasoner is not configured", True), ("storage unavailable", False)):
            runner = smoke.Smoke("https://api.example", TOKEN)
            runner.capture_id = CAPTURE
            runner.status = 503
            runner.request = mock.Mock(return_value=({}, json.dumps({"detail": detail}).encode()))
            output = io.StringIO()
            with redirect_stdout(output):
                if skip:
                    runner.ask()
                    self.assertIn("SKIP", output.getvalue())
                    self.assertIn("ANTHROPIC_API_KEY not set on Fly", output.getvalue())
                else:
                    with self.assertRaises(smoke.SmokeFailure):
                        runner.ask()

    def test_refusals_reject_validation_errors_and_server_failures(self):
        for status in (200, 201, 401, 403, 422, 500):
            runner = smoke.Smoke("https://api.example", TOKEN)
            runner.capture_id = CAPTURE

            def request(step, *args, **kwargs):
                runner.step, runner.status = step, status

            runner.request = request
            with redirect_stdout(io.StringIO()):
                runner.refusals()
            self.assertEqual(runner.failed, status not in (401, 403))


if __name__ == "__main__":
    unittest.main()
