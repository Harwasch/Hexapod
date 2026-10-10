#!/usr/bin/env python3
"""Plan/start/stop named local Worlds workers. Never builds, pulls, provisions or infers."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKERS = ROOT / "workers" / "worlds"
MODELS = {
    "ltx-2.5": ("worlds-ltx25:local", "LTX25", "/models/ltx-2.5"),
    "helix-world": ("worlds-helix:local", "HELIXWORLD", "/opt/helixworld/models"),
    "astronex-world": ("worlds-astronex:local", "ASTRONEX", "/models/astronex"),
    "forge-wm": ("worlds-forge:local", "FORGEWM", "/models/forge-wm"),
    "matrix-game-3": ("worlds-matrix:local", "MATRIX_GAME", "/models/matrix-game-3"),
    "sana-wm": ("worlds-sana:local", "SANA_WM", "/models/sana-wm"),
    "hunyuanworld-mirror": (
        "worlds-mirror:local",
        "WORLD_RECONSTRUCTION",
        "/models/hunyuanworld-mirror",
    ),
}
LABEL = "org.hexapod.worlds.local"
DOCKER = ["docker", "--host", "unix:///var/run/docker.sock"]
EXTRA_ENV = (
    "WORLD_TURN_URLS",
    "WORLD_TURN_SECRET",
    "WORLD_STUN_URLS",
    "WORLD_ICE_TRANSPORT_POLICY",
    "WORLD_VIDEO_ENCODER",
    "WORLD_VIDEO_BITRATE",
    "LTX25_RESIDENCY",
    "WORLD_RECONSTRUCTION_BROWSER_ORIGINS",
)


class LocalError(ValueError):
    pass


def run(command, **kwargs):
    if command[0] == "docker":
        environment = dict(kwargs.pop("env", os.environ))
        for key in (
            "DOCKER_CONTEXT",
            "DOCKER_HOST",
            "DOCKER_TLS_VERIFY",
            "DOCKER_CERT_PATH",
        ):
            environment.pop(key, None)
        kwargs["env"] = environment
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30, **kwargs
        )
    except (OSError, subprocess.SubprocessError):
        # Docker stderr can include environment/container metadata; do not echo it.
        raise LocalError(
            f"{command[0]} failed; verify its local installation and configuration"
        ) from None


def inventory():
    result = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    devices = []
    try:
        for row in csv.reader(result.stdout.splitlines()):
            index, uuid, total, free, driver = [field.strip() for field in row]
            devices.append(
                {
                    "index": int(index),
                    "uuid": uuid,
                    "totalMiB": int(total),
                    "freeMiB": int(free),
                    "driver": driver,
                }
            )
    except (ValueError, TypeError):
        raise LocalError(
            "nvidia-smi returned an unsupported inventory format"
        ) from None
    return devices


def selected_gpu(index, minimum_gib):
    for device in inventory():
        if device["index"] == index:
            if device["freeMiB"] < minimum_gib * 1024:
                raise LocalError(
                    "Selected GPU has less free memory than the configured minimum"
                )
            return device
    raise LocalError("Selected GPU index is unavailable")


def valid_name(value):
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", value):
        raise LocalError(
            "Worker names must be lowercase letters/digits/hyphens, starting with a letter"
        )
    return value


def settings(args, environ):
    image, prefix, mount = MODELS[args.model]
    mirror = args.model == "hunyuanworld-mirror"
    token_key = (
        "WORLD_RECONSTRUCTION_GATEWAY_TOKEN" if mirror else "WORLD_GATEWAY_TOKEN"
    )
    port = args.port or (8790 if mirror else 8789)
    if (
        not 1024 <= port <= 65535
        or not 1 <= args.min_vram_gib <= 192
        or not 4 <= args.memory_gib <= 512
        or not 0 <= args.gpu <= 128
    ):
        raise LocalError(
            "Port, minimum VRAM or RAM bound is outside the supported range"
        )
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_dir() or "," in str(weights):
        raise LocalError("Weights must be an existing local directory without commas")
    environment = {key: environ[key] for key in EXTRA_ENV if environ.get(key)}
    environment.update(
        {
            token_key: environ.get(token_key, ""),
            "WORLD_MODEL_ID": args.model,
            "PORT": str(port),
            "WORLD_BIND": "127.0.0.1" if args.mode == "native" else "0.0.0.0",
            "WORLD_IDLE_SECONDS": "300",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "WANDB_MODE": "disabled",
            "DO_NOT_TRACK": "1",
            prefix + "_WEIGHTS": str(weights) if args.mode == "native" else mount,
        }
    )
    if mirror:
        environment.update(
            {
                "WORLD_RECONSTRUCTION_PUBLIC_URL": f"http://127.0.0.1:{port}",
                "WORLD_RECONSTRUCTION_MAX_FRAMES": "12",
                "WORLD_RECONSTRUCTION_TARGET_SIZE": "518",
                "WORLD_RECONSTRUCTION_RETENTION_SECONDS": "3600",
                "WORLD_RECONSTRUCTION_TIMEOUT_SECONDS": "1800",
            }
        )
    if args.mode == "native":
        if not args.source:
            raise LocalError(
                "Native workers require --source pointing to the prepared pinned checkout"
            )
        source = Path(args.source).expanduser().resolve()
        if not source.is_dir():
            raise LocalError("Native source directory does not exist")
        if args.model == "helix-world" and (source / "models").resolve() != weights:
            raise LocalError(
                "HelixWorld requires --weights to resolve to the prepared source/models directory"
            )
        # Preserve the venv executable path: resolving its symlink bypasses the venv.
        python = Path(args.python).expanduser().absolute()
        inference_python = Path(args.inference_python or python).expanduser().absolute()
        if not all(
            path.is_file() and os.access(path, os.X_OK)
            for path in (python, inference_python)
        ):
            raise LocalError(
                "Native Python executables must already exist and be executable"
            )
        environment.update(
            {
                prefix + "_SOURCE": str(source),
                "INFERENCE_PYTHON": str(inference_python),
                prefix + "_PYTHON": str(inference_python),
                "CUDA_VISIBLE_DEVICES": str(args.gpu),
            }
        )
        entry = (
            WORKERS / "reconstruction" / "server.py"
            if mirror
            else WORKERS / "gateway.py"
        )
        command = [str(python), str(entry)]
    else:
        command = [
            *DOCKER,
            "run",
            "--detach",
            "--pull=never",
            "--name",
            "worlds-" + args.name,
            "--label",
            f"{LABEL}={args.name}",
            "--gpus",
            f"device={args.gpu}",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=512",
            f"--memory={args.memory_gib}g",
            f"--memory-swap={args.memory_gib}g",
            "--cpus=8",
            "--shm-size=2g",
            "--tmpfs",
            "/tmp:rw,nosuid,size=8g",
            "--publish",
            f"127.0.0.1:{port}:{port}",
            "--mount",
            f"type=bind,source={weights},target={mount},readonly",
        ]
        if mirror:
            command.extend(
                ["--tmpfs", "/data:rw,nosuid,uid=10001,gid=10001,mode=0700,size=4g"]
            )
        environment.update(
            {"XDG_CACHE_HOME": "/tmp/cache", "HF_HOME": "/tmp/huggingface"}
        )
        for key in sorted(environment):
            command.extend(["--env", key])
        command.append(image)
    return command, environment, token_key, port


def proc_identity(pid):
    try:
        # The command in parentheses may contain spaces, so parse after the final ')'.
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        marker = next(
            (
                item
                for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
                if item.startswith(b"WORLDS_LOCAL_INSTANCE=")
            ),
            b"",
        )
        return {"started": stat[19], "marker": marker.decode("ascii")}
    except (OSError, ValueError, IndexError):
        return None


@contextmanager
def state_lock(directory):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def launch(args, environ):
    valid_name(args.name)
    command, environment, token_key, port = settings(args, environ)
    token = environment[token_key]
    if (
        len(token) < 32
        or not token.isascii()
        or any(not 33 <= ord(c) <= 126 for c in token)
    ):
        raise LocalError(
            f"Set {token_key} to a random printable ASCII token of at least 32 characters"
        )
    device = selected_gpu(args.gpu, args.min_vram_gib)
    directory = Path(args.state_dir).expanduser().resolve()
    with state_lock(directory):
        state = directory / (args.name + ".json")
        if state.exists():
            raise LocalError(
                "This worker name is already recorded; inspect/stop it before reusing the name"
            )
        record = {
            "name": args.name,
            "model": args.model,
            "mode": args.mode,
            "port": port,
            "gpu": device["uuid"],
        }
        base_env = {
            key: environ[key]
            for key in ("PATH", "HOME", "LANG", "LD_LIBRARY_PATH", "CUDA_HOME")
            if key in environ
        }
        if args.mode == "docker":
            # Check the local image explicitly. --pull=never also forbids implicit downloads.
            run([*DOCKER, "image", "inspect", command[-1]])
            record["container"] = run(
                command, env={**base_env, **environment}
            ).stdout.strip()
            if not re.fullmatch(r"[0-9a-f]{64}", record["container"]):
                raise LocalError("Docker returned an unexpected container identifier")
        else:
            data = directory / (args.name + "-data")
            data.mkdir(mode=0o700, exist_ok=True)
            environment["WORLD_RECONSTRUCTION_DATA_DIR"] = str(data)
            environment["WORLDS_LOCAL_INSTANCE"] = secrets.token_hex(24)
            log = directory / (args.name + ".log")
            descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                process = subprocess.Popen(
                    command,
                    env={**base_env, **environment},
                    stdin=subprocess.DEVNULL,
                    stdout=descriptor,
                    stderr=descriptor,
                    start_new_session=True,
                    cwd=WORKERS,
                )
            finally:
                os.close(descriptor)
            identity = proc_identity(process.pid)
            marker = "WORLDS_LOCAL_INSTANCE=" + environment["WORLDS_LOCAL_INSTANCE"]
            deadline = time.monotonic() + 2
            # /proc may briefly expose the pre-exec environment even after Popen
            # returns. Persist only the exact marker owned by this launch.
            while (
                (identity is None or identity["marker"] != marker)
                and process.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
                identity = proc_identity(process.pid)
            if identity is None or identity["marker"] != marker:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                raise LocalError(
                    "Native worker identity was not established at startup; inspect its private local log"
                )
            record.update({"pid": process.pid, "identity": identity, "log": str(log)})
        temporary = state.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(record))
            temporary.chmod(0o600)
            temporary.replace(state)
        except OSError:
            # A failed state write must not orphan an unmanageable paid-capable process.
            if args.mode == "docker":
                run([*DOCKER, "stop", "--time", "20", record["container"]])
                run([*DOCKER, "rm", record["container"]])
            elif proc_identity(record["pid"]) == record["identity"]:
                os.killpg(record["pid"], signal.SIGTERM)
            raise
    return {
        "status": "started",
        "name": args.name,
        "url": f"http://127.0.0.1:{port}",
        "note": "Process started; run authenticated preflight to verify artifacts. No inference requested.",
    }


def stop(args):
    valid_name(args.name)
    directory = Path(args.state_dir).expanduser().resolve()
    with state_lock(directory):
        state = directory / (args.name + ".json")
        if not state.is_file():
            raise LocalError("No named local worker is recorded")
        record = json.loads(state.read_text())
        if record["mode"] == "docker":
            identifier = record["container"]
            if not re.fullmatch(r"[0-9a-f]{64}", identifier):
                raise LocalError("Invalid recorded container identifier")
            inspection = json.loads(run([*DOCKER, "inspect", identifier]).stdout)
            if inspection[0]["Config"]["Labels"].get(LABEL) != args.name:
                raise LocalError(
                    "Container ownership label differs; refusing to stop another service"
                )
            run([*DOCKER, "stop", "--time", "20", identifier])
            run([*DOCKER, "rm", identifier])
        else:
            pid = record["pid"]
            if not isinstance(pid, int) or pid <= 1:
                raise LocalError("Invalid recorded process identifier")
            current = proc_identity(pid)
            if current is not None and current != record["identity"]:
                raise LocalError(
                    "Process identity changed; refusing to signal another process"
                )
            if current:
                os.killpg(pid, signal.SIGTERM)
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline and proc_identity(pid) == current:
                    time.sleep(0.1)
                if proc_identity(pid) == current:
                    os.killpg(pid, signal.SIGKILL)
        state.unlink()
    return {
        "status": "stopped",
        "name": args.name,
        "note": "Docker temporary data was discarded. Native retained outputs follow their configured retention on restart; private local logs remain.",
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("inventory", "plan", "start", "stop"))
    result.add_argument("--name", default="local")
    result.add_argument("--model", choices=tuple(MODELS), default="astronex-world")
    result.add_argument("--mode", choices=("native", "docker"), default="docker")
    result.add_argument("--weights")
    result.add_argument("--source")
    result.add_argument("--python", default=sys.executable)
    result.add_argument("--inference-python")
    result.add_argument("--port", type=int)
    result.add_argument("--gpu", type=int, default=0)
    result.add_argument("--min-vram-gib", type=int, default=8)
    result.add_argument("--memory-gib", type=int, default=64)
    result.add_argument(
        "--state-dir", default=str(Path.home() / ".local/state/hexapod-worlds")
    )
    return result


def main():
    args = parser().parse_args()
    try:
        if args.action == "inventory":
            output = {
                "devices": inventory(),
                "note": "Inventory only; no allocation or inference.",
            }
        elif args.action == "stop":
            output = stop(args)
        else:
            if not args.weights:
                raise LocalError(
                    "Specify --weights with an already prepared checkpoint directory"
                )
            valid_name(args.name)
            if args.action == "plan":
                command, environment, token_key, port = settings(args, os.environ)
                output = {
                    "command": command,
                    "environmentKeys": sorted(environment),
                    "requiredTokenVariable": token_key,
                    "url": f"http://127.0.0.1:{port}",
                    "note": "Plan only; no GPU check, image pull/build, process start or inference.",
                }
            else:
                output = launch(args, os.environ)
        print(json.dumps(output, indent=2))
        return 0
    except (LocalError, OSError, ValueError, KeyError, TypeError):
        # No raw upstream errors, environment values or credentials enter machine reports.
        error = sys.exception()
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": str(error)
                    if isinstance(error, LocalError)
                    else "Local worker operation failed; inspect private local state",
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
