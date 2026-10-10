"""Local launcher safety and real CPU process lifecycle; no GPU work or Docker runs."""

import importlib.util
import json
import os
import socket
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "worlds_local", ROOT / "scripts/worlds-local.py"
)
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)
TOKEN = "local-launcher-test-token-not-a-real-secret"


def arguments(tmp_path, *extra):
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    return local.parser().parse_args(
        [
            "start",
            "--name",
            "test",
            "--weights",
            str(weights),
            "--state-dir",
            str(tmp_path / "state"),
            *extra,
        ]
    )


def test_docker_plan_is_bounded_local_and_contains_no_token(tmp_path):
    args = arguments(tmp_path, "--model", "hunyuanworld-mirror")
    command, environment, key, port = local.settings(
        args,
        {
            "WORLD_RECONSTRUCTION_GATEWAY_TOKEN": TOKEN,
            "RUNPOD_API_KEY": "private-provider",
        },
    )
    assert "--pull=never" in command and "--read-only" in command
    assert command[:3] == ["docker", "--host", "unix:///var/run/docker.sock"]
    assert "127.0.0.1:8790:8790" in command
    assert command[-1] == "worlds-mirror:local"
    assert "readonly" in command[command.index("--mount") + 1]
    assert TOKEN not in json.dumps(command)
    assert environment[key] == TOKEN and port == 8790
    assert "RUNPOD_API_KEY" not in environment


def test_gpu_inventory_rejects_missing_and_insufficient_memory(monkeypatch):
    monkeypatch.setattr(
        local,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="0, GPU-test, 16384, 4096, 550.54\n"
        ),
    )
    assert local.inventory()[0]["totalMiB"] == 16384
    with pytest.raises(local.LocalError, match="less free memory"):
        local.selected_gpu(0, 8)
    with pytest.raises(local.LocalError, match="unavailable"):
        local.selected_gpu(1, 1)


def test_helix_native_preserves_required_model_tree_and_python(tmp_path):
    source = tmp_path / "helix"
    weights = source / "models"
    weights.mkdir(parents=True)
    args = arguments(
        tmp_path,
        "--model",
        "helix-world",
        "--mode",
        "native",
        "--source",
        str(source),
        "--weights",
        str(weights),
        "--inference-python",
        sys.executable,
    )
    _, environment, _, _ = local.settings(args, {})
    assert environment["HELIXWORLD_SOURCE"] == str(source)
    assert environment["INFERENCE_PYTHON"] == sys.executable
    args.weights = str(tmp_path / "weights")
    with pytest.raises(local.LocalError, match="source/models"):
        local.settings(args, {})
    args.mode = "docker"
    command, _, _, _ = local.settings(args, {})
    assert command[-1] == "worlds-helix:local"
    assert (
        "target=/opt/helixworld/models,readonly"
        in command[command.index("--mount") + 1]
    )
    assert "--pull=never" in command


def test_invalid_names_ports_and_tokens_never_start_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        local, "run", lambda *args, **kwargs: pytest.fail("No command expected")
    )
    args = arguments(tmp_path, "--name", "../earth")
    with pytest.raises(local.LocalError):
        local.launch(args, {"WORLD_GATEWAY_TOKEN": TOKEN})
    args = arguments(tmp_path, "--port", "80")
    with pytest.raises(local.LocalError):
        local.launch(args, {"WORLD_GATEWAY_TOKEN": TOKEN})
    args = arguments(tmp_path)
    with pytest.raises(local.LocalError, match="WORLD_GATEWAY_TOKEN"):
        local.launch(args, {})


def test_docker_start_uses_existing_image_and_stop_requires_ownership(
    tmp_path, monkeypatch
):
    args = arguments(tmp_path)
    calls = []
    monkeypatch.setattr(local, "selected_gpu", lambda *args: {"uuid": "GPU-test"})

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        if command[:4] == [*local.DOCKER, "run"]:
            assert kwargs["env"]["WORLD_GATEWAY_TOKEN"] == TOKEN
            return SimpleNamespace(stdout="a" * 64)
        if command[:4] == [*local.DOCKER, "inspect"]:
            return SimpleNamespace(
                stdout=json.dumps(
                    [{"Config": {"Labels": {local.LABEL: "another-worker"}}}]
                )
            )
        return SimpleNamespace(stdout="[]")

    monkeypatch.setattr(local, "run", execute)
    assert local.launch(args, {"WORLD_GATEWAY_TOKEN": TOKEN})["status"] == "started"
    assert calls[0][0] == [*local.DOCKER, "image", "inspect", "worlds-astronex:local"]
    state = Path(args.state_dir) / "test.json"
    assert TOKEN not in state.read_text()
    with pytest.raises(local.LocalError, match="ownership"):
        local.stop(args)
    assert not any(command[:4] == [*local.DOCKER, "stop"] for command, _ in calls)


def test_native_stop_refuses_reused_pid(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    state = Path(args.state_dir)
    state.mkdir()
    (state / "test.json").write_text(
        json.dumps(
            {
                "mode": "native",
                "pid": 23456,
                "identity": {"started": "1", "marker": "old"},
            }
        )
    )
    monkeypatch.setattr(
        local, "proc_identity", lambda pid: {"started": "2", "marker": "other-service"}
    )
    monkeypatch.setattr(
        local.os, "killpg", lambda *args: pytest.fail("Must not signal another process")
    )
    with pytest.raises(local.LocalError, match="identity changed"):
        local.stop(args)


def test_native_mirror_process_starts_authenticated_and_stops_without_inference(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    args = arguments(
        tmp_path,
        "--model",
        "hunyuanworld-mirror",
        "--mode",
        "native",
        "--source",
        str(source),
        "--python",
        sys.executable,
        "--port",
        str(port),
    )
    monkeypatch.setattr(
        local, "selected_gpu", lambda *args: {"uuid": "CPU-test-no-inference"}
    )
    identity = local.proc_identity
    delayed = False

    def delayed_environment(pid):
        nonlocal delayed
        current = identity(pid)
        if current is not None and not delayed:
            delayed = True
            return {**current, "marker": ""}
        return current

    monkeypatch.setattr(local, "proc_identity", delayed_environment)
    state = Path(args.state_dir) / "test.json"
    try:
        assert (
            local.launch(
                args, {**os.environ, "WORLD_RECONSTRUCTION_GATEWAY_TOKEN": TOKEN}
            )["status"]
            == "started"
        )
        assert delayed, "Exercise the transient pre-exec environment observation"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/health",
            headers={"Authorization": "Bearer " + TOKEN},
        )
        deadline = time.monotonic() + 5
        while True:
            try:
                with opener.open(request, timeout=1) as response:
                    assert json.loads(response.read())["status"] == "unavailable"
                    break
            except OSError:
                if time.monotonic() > deadline:
                    pytest.fail("Native CPU worker did not become reachable")
                time.sleep(0.05)
        assert TOKEN not in state.read_text()
        assert not list((Path(args.state_dir) / "test-data").glob("*/inputs"))
    finally:
        if state.exists():
            record = json.loads(state.read_text())
            local.stop(args)
            assert local.proc_identity(record["pid"]) is None
    assert not state.exists()


def test_ltx25_uses_isolated_runtime_and_no_pull_image(tmp_path):
    args = arguments(tmp_path, "--model", "ltx-2.5")
    command, environment, _, _ = local.settings(
        args, {"LTX25_RESIDENCY": "gpu", "HF_TOKEN": "private"}
    )
    assert command[-1] == "worlds-ltx25:local" and "--pull=never" in command
    assert environment["LTX25_WEIGHTS"] == "/models/ltx-2.5"
    assert environment["LTX25_RESIDENCY"] == "gpu"
    assert "HF_TOKEN" not in environment
    source = tmp_path / "source"
    source.mkdir()
    args.mode = "native"
    args.source = str(source)
    args.inference_python = sys.executable
    _, environment, _, _ = local.settings(args, {})
    assert environment["LTX25_SOURCE"] == str(source)
    assert environment["LTX25_PYTHON"] == sys.executable
