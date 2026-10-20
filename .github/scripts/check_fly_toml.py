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
# pydantic's bool parsing, which WorldsSettings uses: these strings in any case and
# nothing else (not even with surrounding spaces). A TOML true/false or 1/0 reads the same.
TRUTHY = {"1", "on", "t", "true", "y", "yes"}
FALSY = {"0", "off", "f", "false", "n", "no"}


def text(value: Any) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def env_value(env: dict[str, Any], name: str) -> Any:
    """pydantic-settings matches environment names case-insensitively."""
    found = [value for key, value in env.items() if key.upper() == name]
    return found[-1] if found else None


def may_sleep(service: dict[str, Any]) -> bool:
    """Whether Fly Proxy may stop or suspend every app machine. `false` means "off"."""
    auto_stop = service.get("auto_stop_machines", "off")
    off = auto_stop is False or text(auto_stop).lower() == "off"
    return not off or service.get("min_machines_running", 0) < 1


def problems(config: dict[str, Any]) -> list[str]:
    found = []
    # TOML gives a bare key to whichever table header precedes it, so the worker's
    # shutdown grace -- which reads most naturally beside the health check -- turns into
    # a field of that check, and Fly never sees it. It was written that way first.
    if "kill_timeout" not in config:
        found.append("kill_timeout is not a top-level key in fly.toml")
    # Paid Worlds workers are torn down by the lease reaper, a thread in the API process.
    # An API the proxy may stop or suspend runs no reaper while it sleeps, and a worker
    # past its deadline bills until the next request. CI sees only fly.toml: a Fly secret
    # overrides [env] unseen, which is why the API refuses too (WORLD_API_MAY_SLEEP).
    env = config.get("env", {})
    allowed = []
    for flag in PROVISION_FLAGS:
        value = env_value(env, flag)
        if value is None:
            continue
        normalized = text(value).lower()
        if normalized in TRUTHY:
            allowed.append(flag)
        elif normalized not in FALSY:
            found.append(f"{flag} = {value!r} is not a boolean the API reads")
    if allowed and may_sleep(config.get("http_service", {})):
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
