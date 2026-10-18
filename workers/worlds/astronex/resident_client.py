"""Bounded JSONL control link to the isolated resident GPU subprocess."""
from __future__ import annotations
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import threading
import time

MAX_RESPONSE = 128 * 1024


class ResidentClient:
    request_limit = 64 * 1024

    def __init__(self, source, weights, env, *, command=None, timeout=1200, stop_event=None):
        self.timeout = timeout
        self.request_lock = threading.Lock()
        self.closed = threading.Event()
        self.buffer = bytearray()
        command = command or [os.environ.get("INFERENCE_PYTHON", sys.executable), str(Path(__file__).with_name("resident.py")),
                              "--source", str(source), "--weights", str(weights)]
        self.process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, start_new_session=True, bufsize=0)
        os.set_blocking(self.process.stdin.fileno(), False)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            hello = self._receive(stop_event or threading.Event(), deadline=time.monotonic() + 30)
            if hello != {"type": "ready", "protocolVersion": 1}:
                raise RuntimeError("The inference process returned an incompatible protocol")
        except BaseException:
            self.close()
            raise

    def _receive(self, stop_event, deadline):
        while True:
            if stop_event.is_set() or self.closed.is_set():
                raise InterruptedError("Resident inference cancelled")
            if time.monotonic() > deadline:
                raise RuntimeError("Resident inference timed out")
            boundary = self.buffer.find(b"\n")
            if boundary > MAX_RESPONSE:
                raise RuntimeError("The inference process exceeded its metadata limit")
            if boundary >= 0:
                line = bytes(self.buffer[:boundary])
                del self.buffer[:boundary + 1]
                try:
                    message = json.loads(line)
                except (ValueError, UnicodeError):
                    raise RuntimeError("The inference process returned invalid metadata") from None
                if not isinstance(message, dict):
                    raise RuntimeError("The inference process returned invalid metadata")
                return message
            if len(self.buffer) > MAX_RESPONSE:
                raise RuntimeError("The inference process exceeded its metadata limit")
            try:
                events = self.selector.select(timeout=0.1)
                for _key, _events in events:
                    data = os.read(self.process.stdout.fileno(), 8192)
                    if not data:
                        if self.closed.is_set() or stop_event.is_set():
                            raise InterruptedError("Resident inference cancelled")
                        raise RuntimeError("The inference process exited unexpectedly; verify the GPU and pinned environment")
                    self.buffer.extend(data)
            except (ValueError, OSError):
                if self.closed.is_set() or stop_event.is_set():
                    raise InterruptedError("Resident inference cancelled") from None
                raise

    def _send(self, payload, stop_event, deadline):
        # Unbuffered pipe writes may be partial and large valid prompts can fill
        # the pipe. Keep timeout/cancellation effective even if the child stops
        # reading after its ready message.
        with selectors.DefaultSelector() as writable:
            writable.register(self.process.stdin, selectors.EVENT_WRITE)
            remaining = memoryview(payload)
            while remaining:
                if self.closed.is_set() or stop_event.is_set():
                    raise InterruptedError("Resident inference cancelled")
                if time.monotonic() > deadline:
                    raise RuntimeError("Resident inference timed out")
                try:
                    if not writable.select(timeout=0.1):
                        continue
                    sent = os.write(self.process.stdin.fileno(), remaining)
                    if sent <= 0:
                        raise RuntimeError("The inference process stopped accepting requests")
                    remaining = remaining[sent:]
                except BlockingIOError:
                    continue
                except (OSError, ValueError):
                    if self.closed.is_set() or stop_event.is_set():
                        raise InterruptedError("Resident inference cancelled") from None
                    raise

    def generate(self, directory, prompt, image, action, seed, quality, stop_event, on_phase=None):
        with self.request_lock:
            if self.closed.is_set() or stop_event.is_set():
                raise InterruptedError("Resident inference cancelled")
            request = {"op": "generate", "directory": str(Path(directory).resolve()), "prompt": prompt,
                       "image": str(Path(image).resolve()) if image is not None else None,
                       "action": action, "seed": seed, "quality": quality}
            payload = (json.dumps(request, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
            if len(payload) > self.request_limit:
                raise ValueError("Resident request too large")
            try:
                deadline = time.monotonic() + self.timeout
                self._send(payload, stop_event, deadline)
                while True:
                    reply = self._receive(stop_event, deadline)
                    if reply.get("type") != "phase":
                        break
                    phase = reply.get("phase")
                    if phase not in ("loading-model", "generating"):
                        raise RuntimeError("The inference process returned an invalid phase")
                    if on_phase:
                        on_phase(phase)
                if reply.get("code") == "gpu-out-of-memory":
                    raise RuntimeError("The GPU ran out of memory. Stop competing GPU workloads or use a larger GPU, then start a new session.")
                if reply.get("type") != "result" or not isinstance(reply.get("result"), dict):
                    raise RuntimeError("Resident inference failed; verify the GPU, pinned environment and checkpoint")
                result = reply["result"]
                self._validate_output(result, Path(directory).resolve())
                return result
            except BaseException:
                # Never leave a GPU process running after timeout, cancellation or
                # protocol failure, even if the HTTP session has already vanished.
                self.close()
                raise

    @staticmethod
    def _validate_output(result, directory):
        raw = result.get("rawFrames")
        paths = result.get("framePaths", [])
        if raw is not None:
            from media import validate_raw_frames
            validate_raw_frames(raw, directory)
            paths = [raw["path"]]
        elif not isinstance(paths, list) or not 1 <= len(paths) <= 64:
            raise RuntimeError("The inference process returned an invalid frame count")
        for raw in [*paths, result.get("continuationPath")]:
            if not isinstance(raw, str):
                raise RuntimeError("The inference process returned an invalid frame path")
            path = Path(raw)
            if not path.is_file() or not path.resolve().is_relative_to(directory):
                raise RuntimeError("The inference process returned a frame outside its private directory")
        if result.get("fps") != 24 or result.get("nativeKVContinuity") is not False:
            raise RuntimeError("The inference process returned incompatible capability metadata")

    def reset(self):
        """A reset acknowledgement is mandatory before lending weights again."""
        with self.request_lock:
            stop = threading.Event()
            deadline = time.monotonic() + 10
            try:
                self._send(b'{"op":"reset"}\n', stop, deadline)
                if self._receive(stop, deadline) != {"type": "reset", "sessionStateCleared": True}:
                    raise RuntimeError("Resident process did not acknowledge session isolation")
            except BaseException:
                self.close()
                raise

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        process = self.process
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        self.selector.close()
        for pipe in (process.stdin, process.stdout):
            if pipe:
                pipe.close()
        self.buffer.clear()
