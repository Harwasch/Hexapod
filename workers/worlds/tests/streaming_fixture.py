"""Loopback-only synthetic transport fixture. Never imported by production code."""
import json
import logging
import math
import os
from pathlib import Path
import secrets
import signal
import sys
import threading
import time
from array import array

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw
from gateway import BoundedHTTPServer, Gateway, Session, handler_for
if os.environ.get('WORLD_TEST_ICE_DEBUG') == '1':
    logging.basicConfig(level=logging.INFO)


class FixtureAdapter:
    def metadata(self):
        return {"id": "synthetic-streaming-test", "status": "ready", "capabilities": {"runtime": {"realtime": False}}}
    capabilities = {"runtime": {"realtime": False}, "nativeActions": ["forward", "stop"], "output": {"audio": os.environ.get("WORLD_TEST_AUDIO") == "1"}}


gateway = Gateway(adapter=FixtureAdapter())
session = Session("synthetic-stream-001", "Explicit synthetic transport test", 1, "balanced", gateway.root / "fixture")
session.directory.mkdir()
session.resumed.set()
gateway.sessions[session.id] = session
pcm = None
if FixtureAdapter.capabilities["output"]["audio"]:
    samples = (int(math.sin(index * math.pi * 2 * 440 / 48000) * 10000) for index in range(48000 * 6))
    pcm = array("h", (value for sample in samples for value in (sample, sample))).tobytes()


def publish():
    index = 0
    while not session.stop.wait(1 / 24):
        image = Image.new("RGB", (640, 360), (10, 30, 80))
        draw = ImageDraw.Draw(image)
        x = index * 5 % 600
        draw.rectangle((x, 100, x + 40, 270), fill=(180, 220, 140))
        draw.text((24, 24), "SYNTHETIC WEBRTC TRANSPORT TEST - NO AI INFERENCE", fill="white")
        with session.lock:
            if pcm is not None and index % 144 == 0:
                session.audio = pcm
                session.audio_index += 1
                session.audio_started_at = time.monotonic()
            session.raw_frame = image.tobytes()
            session.raw_size = image.size
            session.frame_index += 1
        index += 1


session.thread = threading.Thread(target=publish, daemon=True)
session.thread.start()
token = secrets.token_urlsafe(40)
server = BoundedHTTPServer(("127.0.0.1", 0), handler_for(gateway, token))
print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}", "token": token, "sessionId": session.id}), flush=True)


def stop(*_):
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, stop)
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    server.server_close()
    gateway.close()
