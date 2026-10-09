"""Real CPU WebRTC encode/decode and control tests using explicitly synthetic media."""
import asyncio
import base64
import hashlib
import hmac
import io
import json
import os
import math
import time
from array import array
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gateway import Gateway, Session
from streaming import StreamingError, StreamingService, transport_config


class TransportConfigTests(unittest.TestCase):
    def test_turn_credentials_are_expiring_and_do_not_expose_secret(self):
        secret = "turn-secret-" * 4
        config = transport_config("anonymous-session", {"WORLD_TURN_URLS": "turn:relay.example:3478?transport=tcp", "WORLD_TURN_SECRET": secret, "WORLD_ICE_TRANSPORT_POLICY": "relay"}, now=1000)
        server = config["iceServers"][0]
        self.assertTrue(server["username"].startswith("4900:anonymous-session:"))
        expected = base64.b64encode(hmac.new(secret.encode(), server["username"].encode(), hashlib.sha1).digest()).decode()
        self.assertEqual(server["credential"], expected)
        self.assertNotIn(secret, json.dumps(config))
    def test_external_ice_servers_are_opt_in_and_bad_configuration_fails_closed(self):
        self.assertEqual(transport_config("anonymous", {})["iceServers"], [])
        self.assertEqual(transport_config('anonymous', {'WORLD_TURN_CREDENTIAL_TTL_SECONDS': '14700'}, now=1000)['expiresAt'], 15700)
        for env in ({"WORLD_TURN_URLS": "turn:relay.example"}, {"WORLD_STUN_URLS": "https://wrong.example"}, {"WORLD_ICE_TRANSPORT_POLICY": "relay"}, {"WORLD_TURN_CREDENTIAL_TTL_SECONDS": "299"}, {"WORLD_TURN_CREDENTIAL_TTL_SECONDS": "86401"}, {"WORLD_TURN_CREDENTIAL_TTL_SECONDS": "invalid"}):
            with self.assertRaises(StreamingError):
                transport_config("anonymous", env)


@unittest.skipUnless(StreamingService.available(), "Install requirements-streaming.lock to exercise encoded transport")
class EncodedTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiortc import RTCPeerConnection
        from PIL import Image
        self.gateway = Gateway(adapter=SimpleNamespace(capabilities={"nativeActions": ["forward"]}))
        self.session = Session("transport-test-001", "Synthetic transport fixture only", 1, "balanced", self.gateway.root / "fixture")
        self.session.directory.mkdir()
        self.session.resumed.set()
        self.gateway.sessions[self.session.id] = self.session
        image = Image.new("RGB", (320, 180), (10, 30, 240))
        output = io.BytesIO()
        image.save(output, "JPEG")
        self.jpeg = output.getvalue()
        self.peer = RTCPeerConnection()
        self.peer.addTransceiver("video", direction="recvonly")
        self.channel = self.peer.createDataChannel("world-controls-v1", ordered=True)

    async def asyncTearDown(self):
        await self.peer.close()
        await asyncio.to_thread(self.gateway.close)

    async def negotiate(self):
        from aiortc import RTCSessionDescription
        await self.peer.setLocalDescription(await self.peer.createOffer())
        answer = await asyncio.to_thread(self.gateway.streaming.offer, self.session.id, {"sdp": self.peer.localDescription.sdp, "type": "offer"})
        await self.peer.setRemoteDescription(RTCSessionDescription(**answer))

    async def test_encoded_video_and_acknowledged_bidirectional_control(self):
        frame_received = asyncio.get_running_loop().create_future()
        channel_open = asyncio.Event()
        acknowledgement = asyncio.get_running_loop().create_future()

        @self.peer.on("track")
        def track_received(track):
            async def receive():
                try:
                    frame = await track.recv()
                    if not frame_received.done():
                        frame_received.set_result(frame)
                except Exception as error:
                    if not frame_received.done():
                        frame_received.set_exception(error)
            asyncio.create_task(receive())

        @self.channel.on("open")
        def opened():
            channel_open.set()

        @self.channel.on("message")
        def message(value):
            result = json.loads(value)
            if result.get("id") == "command-1" and not acknowledgement.done():
                acknowledgement.set_result(result)

        await self.negotiate()
        if getattr(self, "raw_fixture", False):
            self.assertIn("H264/90000", self.peer.remoteDescription.sdp)
            self.assertNotIn("VP8/90000", self.peer.remoteDescription.sdp)
        await asyncio.wait_for(channel_open.wait(), timeout=10)
        self.channel.send(json.dumps({"id": "command-1", "action": {"type": "native", "action": "forward", "values": {"pressed": True}}}))
        ack = await asyncio.wait_for(acknowledgement, timeout=5)
        self.assertTrue(ack["ok"])
        self.assertEqual(ack["result"]["appliesAt"], "next-clip")
        self.assertEqual(self.session.action, "forward")
        # Several frames let the RTP decoder establish its reorder buffer.
        for index in range(8):
            with self.session.lock:
                if getattr(self, "raw_fixture", False):
                    self.session.raw_frame = bytes([10, 30, 240]) * 320 * 180
                    self.session.raw_size = (320, 180)
                else:
                    self.session.frame = self.jpeg
                self.session.frame_index = index + 1
            await asyncio.sleep(0.06)
        frame = await asyncio.wait_for(frame_received, timeout=5)
        self.assertEqual((frame.width, frame.height), (320, 180))
        self.assertGreater(frame.to_image().getpixel((160, 90))[2], 180)
        await asyncio.to_thread(self.gateway.delete, self.session.id)
        self.assertEqual(self.gateway.streaming.peers, {})
        self.assertEqual(self.gateway.streaming.closers, {})

    async def test_raw_rgb_through_public_h264_packet_track(self):
        self.raw_fixture = True
        with patch.dict(os.environ, {"WORLD_VIDEO_ENCODER": "h264-software"}):
            await self.test_encoded_video_and_acknowledged_bidirectional_control()

    async def test_real_generated_pcm_encodes_as_opus_and_stops_with_session(self):
        self.gateway.adapter.capabilities["output"] = {"audio": True}
        self.peer.addTransceiver("audio", direction="recvonly")
        audio_received = asyncio.get_running_loop().create_future()
        @self.peer.on("track")
        def received(track):
            if track.kind == "audio":
                async def receive():
                    try:
                        audio_received.set_result(await track.recv())
                    except Exception as error:
                        if not audio_received.done():
                            audio_received.set_exception(error)
                asyncio.create_task(receive())
        await self.negotiate()
        samples = array("h", (int(math.sin(index * 2 * math.pi * 440 / 48000) * 10000) for index in range(48000)))
        stereo = array("h", (value for sample in samples for value in (sample, sample)))
        with self.session.lock:
            self.session.audio = stereo.tobytes()
            self.session.audio_index = 1
            self.session.audio_started_at = time.monotonic()
        frame = await asyncio.wait_for(audio_received, timeout=5)
        self.assertEqual(frame.sample_rate, 48000)
        self.assertGreater(frame.samples, 0)
        self.assertTrue(any(bytes(frame.planes[0])))
        self.assertIn("opus/48000/2", self.peer.remoteDescription.sdp)
        await asyncio.to_thread(self.gateway.delete, self.session.id)
        self.assertIsNone(self.session.audio)
        self.assertFalse(self.gateway.streaming.peers)

    async def test_audio_pause_preserves_position_and_resume_advances_timestamp(self):
        from audio_stream import generated_audio_track
        started = time.monotonic()
        self.session.audio = array("h", [1000] * 48000 * 2).tobytes()
        self.session.audio_index = 1
        self.session.audio_started_at = started
        track = generated_audio_track(self.session, started)
        first = await track.recv()
        self.gateway.action(self.session.id, {"type": "pause"})
        pending = asyncio.create_task(track.recv())
        try:
            await asyncio.sleep(0.04)
            self.assertFalse(pending.done())
            self.gateway.action(self.session.id, {"type": "resume"})
            second = await asyncio.wait_for(pending, timeout=1)
            self.assertGreater(second.pts, first.pts)
            self.assertLess(track.offset, 4800, "Pause must not consume samples from the clip")
        finally:
            pending.cancel()
            track.stop()

    async def test_duplicate_cleanup_waits_for_the_same_close_task(self):
        await self.negotiate()
        remote = self.gateway.streaming.peers[self.session.id]
        original = remote.close
        calls = 0
        async def slow_close():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.05)
            await original()
        remote.close = slow_close
        await asyncio.gather(asyncio.to_thread(self.gateway.streaming.close_session, self.session.id), asyncio.to_thread(self.gateway.streaming.close_session, self.session.id))
        self.assertEqual(calls, 1)
        self.assertFalse(self.gateway.streaming.peers)
        self.assertFalse(self.gateway.streaming.closers)

    async def test_second_offer_is_rejected_and_shutdown_removes_negotiation(self):
        await self.negotiate()
        with self.assertRaises(StreamingError) as error:
            await asyncio.to_thread(self.gateway.streaming.offer, self.session.id, {"sdp": self.peer.localDescription.sdp, "type": "offer"})
        self.assertEqual(error.exception.status, 409)
        await asyncio.to_thread(self.gateway.streaming.close)
        self.assertFalse(self.gateway.streaming.thread.is_alive())
        self.assertFalse(self.gateway.streaming.peers)


if __name__ == "__main__":
    unittest.main()
