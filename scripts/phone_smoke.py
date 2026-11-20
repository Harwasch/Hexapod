"""Dispatch-only phone credential smoke; stdlib only, never starts processing.

Run: PHONE_API_TOKEN=... python3 scripts/phone_smoke.py --api <origin>
Output contains only controlled notes, statuses and the capture UUID for cleanup.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "apps/ios/Fixtures/bundles/yard-small"


def redact(value, token):
    """Defense in depth: remove credentials, URLs and control characters from notes."""
    value = str(value)
    if token:
        value = re.sub(re.escape(token), "[REDACTED]", value, flags=re.IGNORECASE)
    value = re.sub(r"https?://\S+", "[URL]", value, flags=re.IGNORECASE)
    value = re.sub(r"(?i)\bauthorization\s*:[^\r\n]*", "[REDACTED]", value)
    value = re.sub(r"(?i)\bbearer\s+\S+", "[REDACTED]", value)
    return " ".join(value.split())


def sse_events(lines):
    """Parse SSE frames, including comments, CRLF and multiline data.

    The API carries its event type in JSON data; named SSE events also work.
    An unterminated final frame is discarded per SSE framing rules.
    """
    event, data = "message", []
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = "message", []
        elif not line.startswith(":"):
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "event":
                event = value
            elif field == "data":
                data.append(value)


def stream_done(lines):
    """Success requires the last dispatched event to be done, not just any event."""
    last = None
    for event, data in sse_events(lines):
        last = event if event != "message" else json.loads(data).get("type")
    return last == "done"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    # Never forward the API credential to a redirect target.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class SmokeFailure(Exception):
    pass


class Smoke:
    def __init__(self, origin, token):
        self.base = origin.rstrip("/") + "/api/v1"
        self.token = token
        self.failed = False
        self.capture_id = None
        self.step = "setup"
        self.status = 0
        self.opener = urllib.request.build_opener(NoRedirect())

    def report(self, result, note):
        self.failed |= result == "FAIL"
        print(redact(f"{result} {self.step} {self.status} {note}", self.token), flush=True)

    def request(self, step, method, path, body=None, *, signed=False, stream=False):
        self.step, self.status = step, 0
        headers = {} if signed else {"Authorization": "Bearer " + self.token}
        if body is not None and not isinstance(body, bytes):
            body = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            path if signed else self.base + path,
            data=body, headers=headers, method=method,
        )
        try:
            response = self.opener.open(request, timeout=120)
        except urllib.error.HTTPError as error:
            response = error
        self.status = response.code
        if stream and self.status == 200:
            return response
        with response:
            data = response.read()
        return response.headers, data

    def api(self, step, method, path, body=None, expected=200):
        _, data = self.request(step, method, path, body)
        if self.status != expected:
            raise SmokeFailure("unexpected HTTP status")
        # No response body or exception text is ever logged.
        result = json.loads(data) if data else None
        self.report("PASS", "accepted")
        return result

    def upload(self):
        location = json.loads((FIXTURE / "bundle/manifest.json").read_text())["location"]
        capture = self.api("begin-bundle", "PUT", "/captures/bundles/" + str(uuid.uuid4()), {
            "name": "Smoke test · synthetic yard-small", "kind": "video",
            "metadata": {
                "smokeTest": True, "synthetic": True, "fixture": "yard-small",
                "lat": location["lat"], "lon": location["lon"],
                "uncertaintyM": location["horizontalAccuracyM"],
                "locationAccuracyM": location["horizontalAccuracyM"],
            },
        })
        self.capture_id = str(uuid.UUID(capture["id"]))
        path = "/captures/" + self.capture_id
        archive = FIXTURE / "capture.hexcap"
        total = archive.stat().st_size
        file = self.api("bundle-upload", "PUT", path + "/bundle-upload", {
            "filename": "capture.hexcap", "contentType": "application/x-hex-capture",
            "bytes": total,
        })
        if file["bytes"] != total:
            raise SmokeFailure("archive byte count differs")
        file_path = path + "/files/" + str(uuid.UUID(file["id"]))
        completed, size, next_part = [], None, 1
        with archive.open("rb") as source:
            while True:
                window = self.api("parts-window", "POST", file_path + "/parts", {
                    "firstPartNumber": next_part, "count": 1,
                })
                part_size = window["partSize"]
                if (not isinstance(part_size, int) or part_size <= 0
                        or (size is not None and size != part_size)):
                    raise SmokeFailure("invalid part size")
                size = part_size
                count = (total + size - 1) // size
                if window["partsTotal"] != count or len(window["parts"]) != 1:
                    raise SmokeFailure("invalid parts window")
                part = window["parts"][0]
                if part["partNumber"] != next_part:
                    raise SmokeFailure("unexpected part number")
                parsed = urllib.parse.urlsplit(part["url"])
                if parsed.scheme != "https" or parsed.username or parsed.password:
                    raise SmokeFailure("storage URL must use HTTPS")
                chunk = source.read(min(size, total - source.tell()))
                if not chunk:
                    raise SmokeFailure("empty archive chunk")
                headers, _ = self.request("put-part", "PUT", part["url"], chunk, signed=True)
                if not 200 <= self.status < 300:
                    raise SmokeFailure("storage PUT refused")
                etag = headers.get("ETag")
                if not etag:
                    raise SmokeFailure("storage PUT has no ETag")
                self.report("PASS", "part uploaded with ETag")
                completed.append({"partNumber": next_part, "etag": etag})
                if next_part == count:
                    break
                next_part += 1
        file = self.api("complete-upload", "POST", file_path + "/complete", {"parts": completed})
        if file["status"] != "complete":
            raise SmokeFailure("capture file is not complete")
        self.api("read-capture", "GET", path)

    def ask(self):
        response = self.request("ask", "POST", "/ask", {
            "id": str(uuid.uuid4()), "scope": "scan", "scanId": self.capture_id,
            "keyframes": [], "deviceEntities": [], "question": "What is in this scan?",
        }, stream=True)
        if self.status == 503:
            _, data = response
            if json.loads(data).get("detail") == "Reasoner is not configured":
                self.report("SKIP", "ANTHROPIC_API_KEY not set on Fly")
                return
        if self.status != 200:
            raise SmokeFailure("unexpected HTTP status")
        with response:
            done = stream_done(response)
        if not done:
            raise SmokeFailure("stream did not end with done")
        self.report("PASS", "stream ended with done")

    def refusals(self):
        # Empty bodies make a broken auth guard fail validation before it can mutate
        # sites, create a desktop capture, or enqueue a processing job.
        probes = [("refuse-sites", "POST", "/sites"),
                  ("refuse-desktop-create", "POST", "/captures")]
        if self.capture_id:
            probes.append(("refuse-process", "POST", "/captures/" + self.capture_id + "/process"))
            contract = json.loads((ROOT / "packages/contracts/openapi.json").read_text())
            route = contract["paths"].get("/api/v1/captures/{capture_id}", {})
            if "delete" in route:
                probes.append(("refuse-delete", "DELETE", "/captures/" + self.capture_id))
            else:
                self.step, self.status = "refuse-delete", 0
                self.report("SKIP", "no capture DELETE route")
        for step, method, path in probes:
            try:
                self.request(step, method, path, {})
                self.report("PASS" if self.status in (401, 403) else "FAIL",
                            "refused" if self.status in (401, 403) else "expected 401 or 403")
            except Exception:
                self.report("FAIL", "request failed; details suppressed")

    def run(self):
        try:
            self.api("phone-check", "POST", "/phone/check", expected=204)
        except Exception:
            self.report("FAIL", "credential check failed; details suppressed")
            return 1
        try:
            self.upload()
            self.ask()
        except SmokeFailure as error:
            self.report("FAIL", str(error))
        except Exception:
            self.report("FAIL", "request or response failed; details suppressed")
        self.refusals()
        if self.capture_id:
            print("Capture id: " + self.capture_id, flush=True)
        return int(self.failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True)
    args = parser.parse_args()
    token = os.environ.get("PHONE_API_TOKEN", "").strip()
    try:
        origin = urllib.parse.urlsplit(args.api)
        valid = (origin.scheme in ("https", "http") and origin.hostname
                 and origin.path in ("", "/") and not origin.query and not origin.fragment
                 and not origin.username and not origin.password)
    except ValueError:
        valid = False
    if not token or not valid:
        print("FAIL setup 0 PHONE_API_TOKEN and an API origin are required")
        return 1
    return Smoke(args.api, token).run()


if __name__ == "__main__":
    sys.exit(main())
