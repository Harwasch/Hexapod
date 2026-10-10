"""Container liveness: authenticated HTTP, with no token in command arguments."""
import os
import urllib.request
request = urllib.request.Request("http://127.0.0.1:" + os.environ.get("PORT", "8789") + "/health", headers={"Authorization": "Bearer " + os.environ["WORLD_GATEWAY_TOKEN"]})
with urllib.request.urlopen(request, timeout=3) as response:
    assert response.status == 200
