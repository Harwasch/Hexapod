"""Authenticated local artifact readiness; no inference, redirects or token output."""

import json
import os
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    try:
        port = int(os.environ.get("PORT", "8790"))
        if not 1 <= port <= 65535:
            return 1
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/health",
            headers={
                "Authorization": "Bearer "
                + os.environ["WORLD_RECONSTRUCTION_GATEWAY_TOKEN"]
            },
        )
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect()
        )
        with opener.open(request, timeout=3) as response:
            body = response.read(65537)
            if len(body) > 65536:
                return 1
            payload = json.loads(body)
            return (
                0
                if isinstance(payload, dict) and payload.get("status") == "ready"
                else 1
            )
    except (OSError, ValueError, KeyError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
