"""Checks fly.toml for mistakes that otherwise surface only at deploy time, then prints
the first word of every command it names (one per line) for ci.yml's image job to look
up inside the built image.

Run: python3 .github/scripts/check_fly_toml.py fly.toml > commands.txt
Test: python3 -m unittest discover -s .github/scripts
"""

from __future__ import annotations

import shlex
import sys
import tomllib
from typing import Any

PROVISION_FLAGS = (
    "WORLD_RUNPOD_ALLOW_PROVISION",
    "WORLD_LAMBDA_ALLOW_PROVISION",
    "WORLD_MODAL_ALLOW_PROVISION",
)


def problems(config: dict[str, Any]) -> list[str]:
    found = []
    # TOML gives a bare key to whichever table header precedes it, so the worker's
    # shutdown grace -- which reads most naturally beside the health check -- turns into
    # a field of that check, and Fly never sees it. It was written that way first.
    if "kill_timeout" not in config:
        found.append("kill_timeout is not a top-level key in fly.toml")
    # Paid Worlds workers are torn down by the lease reaper, a thread in the API process.
    # An API the proxy may stop or suspend runs no reaper while it sleeps, and a worker
    # past its deadline bills until the next request.
    env = config.get("env", {})
    allowed = [flag for flag in PROVISION_FLAGS if str(env.get(flag, "false")).lower() == "true"]
    service = config.get("http_service", {})
    if allowed and (
        service.get("auto_stop_machines", "off") != "off"
        or service.get("min_machines_running", 0) < 1
    ):
        found.append(
            f"{', '.join(allowed)} allows paid Worlds provisioning while the API may sleep: "
            'set [http_service] auto_stop_machines = "off" and min_machines_running = 1'
        )
    return found


def commands(config: dict[str, Any]) -> list[str]:
    named = [*config["processes"].values(), config["deploy"]["release_command"]]
    return [shlex.split(command)[0] for command in named]


def main(path: str) -> int:
    with open(path, "rb") as handle:
        config = tomllib.load(handle)
    found = problems(config)
    for problem in found:
        print(f"::error::{problem}", file=sys.stderr)
    if found:
        return 1
    print("\n".join(commands(config)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "fly.toml"))
