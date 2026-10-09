"""Authenticated-offer WebRTC video and ordered control channels.

The latest real generated RGB/JPEG is encoded as RTP video; there is no synthetic
frame source, unbounded frame queue, microphone capture, or model-speed claim.
Software encoding is the default, with opt-in NVENC and software fallback.
TURN is required when provider HTTP proxies cannot
expose worker UDP candidates. Credentials are short-lived HMAC TURN credentials.
"""
from __future__ import annotations

import asyncio
import base64
from concurrent.futures import TimeoutError as FutureTimeout
from fractions import Fraction
import hashlib
import hmac
import io
import json
import os
import secrets
import threading
import time
from encoding import encoder_settings, H264PacketEncoder
from media import raw_video_frame


class StreamingError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


def transport_config(session_id, environ=None, now=None):
    env = os.environ if environ is None else environ
    servers = []
    stun = [url.strip() for url in env.get("WORLD_STUN_URLS", "").split(",") if url.strip()]
    turn = [url.strip() for url in env.get("WORLD_TURN_URLS", "").split(",") if url.strip()]
    for url in stun + turn:
        if len(url) > 512 or "@" in url or any(c in url for c in "\r\n\t "):
            raise StreamingError(503, "Invalid server ICE configuration")
    if any(not url.startswith(("stun:", "stuns:")) for url in stun) or any(not url.startswith(("turn:", "turns:")) for url in turn):
        raise StreamingError(503, "Invalid server ICE configuration")
    if stun:
        servers.append({"urls": stun})
    try:
        ttl = int(env.get("WORLD_TURN_CREDENTIAL_TTL_SECONDS", "3900"))
    except ValueError:
        raise StreamingError(503, "TURN credential lifetime must be an integer") from None
    if not 300 <= ttl <= 86400:
        raise StreamingError(503, "TURN credential lifetime must be between 300 and 86400 seconds")
    expires = int(time.time() if now is None else now) + ttl
    if turn:
        secret = env.get("WORLD_TURN_SECRET", "")
        if len(secret) < 32:
            raise StreamingError(503, "TURN URLs require a secret of at least 32 characters")
        username = f"{expires}:{session_id}:{secrets.token_hex(6)}"
        credential = base64.b64encode(hmac.new(secret.encode(), username.encode(), hashlib.sha1).digest()).decode()
        servers.append({"urls": turn, "username": username, "credential": credential})
    policy = env.get("WORLD_ICE_TRANSPORT_POLICY", "all")
    if policy not in ("all", "relay") or policy == "relay" and not turn:
        raise StreamingError(503, "Relay ICE policy requires a configured TURN service")
    try:
        mode, bitrate = encoder_settings(env)
    except ValueError as error:
        raise StreamingError(503, str(error)) from None
    return {"iceServers": servers, "iceTransportPolicy": policy, "expiresAt": expires,
            "controlProtocol": "world-controls-v1", "videoEncoder": "software" if mode == "software" else "pending-probe",
            "requestedEncoder": mode, "videoBitrate": bitrate, "maxMessageBytes": 16384}


class StreamingService:
    """One asyncio media loop per worker; HTTP gateway remains synchronous."""

    def __init__(self, gateway):
        self.gateway = gateway
        self.loop = asyncio.new_event_loop()
        self.peers = {}
        self.closers = {}
        self.closed = False
        self.negotiation_lock = asyncio.Lock()
        self.thread = threading.Thread(target=self._run, name="world-webrtc", daemon=True)
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.close()

    @staticmethod
    def available():
        try:
            import aiortc  # noqa: F401
            import av  # noqa: F401
            return True
        except ImportError:
            return False

    def config(self, session_id):
        self.gateway.get(session_id)
        if not self.available():
            raise StreamingError(503, "WebRTC dependencies are unavailable; authenticated frame transport remains available")
        return transport_config(session_id)

    def offer(self, session_id, body):
        if self.closed:
            raise StreamingError(503, "Worker streaming is shutting down")
        if not isinstance(body, dict) or body.get("type") != "offer" or not isinstance(body.get("sdp"), str) or not 1 <= len(body["sdp"]) <= 131072:
            raise StreamingError(400, "A bounded WebRTC SDP offer is required")
        self.config(session_id)
        pending = asyncio.run_coroutine_threadsafe(self._offer(session_id, body), self.loop)
        try:
            return pending.result(timeout=25)
        except FutureTimeout:
            pending.cancel()
            raise StreamingError(504, "WebRTC negotiation timed out; use frame transport or check TURN") from None
        except StreamingError:
            raise
        except Exception:
            raise StreamingError(400, "WebRTC negotiation failed; check the offer and ICE configuration") from None

    async def _offer(self, session_id, body):
        async with self.negotiation_lock:
            if self.closed:
                raise StreamingError(503, "Worker streaming is shutting down")
            return await self._offer_locked(session_id, body)

    async def _offer_locked(self, session_id, body):
        from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer, RTCSessionDescription, VideoStreamTrack, RTCRtpSender
        from aiortc.mediastreams import MediaStreamError
        from av import VideoFrame
        from PIL import Image

        session = self.gateway.get(session_id)
        config = transport_config(session_id)
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer(**server) for server in config["iceServers"]]))
        # Single observer avoids competing encoders and exposes accidental double joins.
        existing = self.peers.get(session_id)
        if existing and existing.connectionState not in ("closed", "failed", "disconnected"):
            await peer.close()
            raise StreamingError(409, "This session already has a live video connection")
        if existing:
            await self.closers[session_id]()
        self.peers[session_id] = peer
        started = time.monotonic()
        channels = []
        tasks = set()
        mode, bitrate = encoder_settings()
        packet_encoder = H264PacketEncoder(mode, bitrate) if mode != "software" else None

        class LatestFrameTrack(VideoStreamTrack):
            def __init__(self):
                super().__init__()
                self.index = -1
                self.sent = 0
                self.dropped = 0
            async def recv(self):
                while self.readyState == "live" and not session.stop.is_set():
                    with session.lock:
                        raw, size = session.raw_frame, session.raw_size
                        data, index = session.frame, session.frame_index
                    if (data or raw) and index != self.index:
                        if self.index >= 0:
                            self.dropped += max(0, index - self.index - 1)
                        self.index = index
                        # Decode on a thread so media/control ICE heartbeats stay responsive.
                        def decode():
                            if raw is not None:
                                return raw_video_frame(raw, size)
                            with Image.open(io.BytesIO(data)) as image:
                                if image.width * image.height > 3840 * 2160:
                                    raise ValueError("Frame too large")
                                return VideoFrame.from_image(image.convert("RGB"))
                        frame = await asyncio.to_thread(decode)
                        frame.pts = max(1, int((time.monotonic() - started) * 90000))
                        frame.time_base = Fraction(1, 90000)
                        if packet_encoder:
                            frame = await asyncio.to_thread(packet_encoder.encode, frame)
                            if frame is None:
                                continue
                        self.sent += 1
                        return frame
                    await asyncio.sleep(0.01)
                raise MediaStreamError

        track = LatestFrameTrack()
        sender = peer.addTrack(track)
        audio_track = None
        if getattr(self.gateway.adapter, "capabilities", {}).get("output", {}).get("audio"):
            from audio_stream import generated_audio_track
            audio_track = generated_audio_track(session, started)
            peer.addTrack(audio_track)
        for transceiver in peer.getTransceivers():
            # This worker publishes model output; it never accepts a client's
            # camera/microphone media or creates unconsumed inbound decode queues.
            transceiver.direction = "sendonly"
        if packet_encoder:
            codecs = [codec for codec in RTCRtpSender.getCapabilities("video").codecs
                      if codec.mimeType.lower() == "video/h264" and codec.parameters.get("packetization-mode") == "1"]
            for transceiver in peer.getTransceivers():
                if transceiver.sender is sender:
                    transceiver.setCodecPreferences(codecs)

        close_task = None
        async def do_close():
            track.stop()
            if audio_track:
                audio_track.stop()
            cancelled = [task for task in tuple(tasks) if task is not asyncio.current_task()]
            for item in cancelled:
                item.cancel()
            await asyncio.gather(*cancelled, return_exceptions=True)
            for channel in channels:
                channel.close()
            try:
                await peer.close()
            finally:
                if packet_encoder:
                    await asyncio.to_thread(packet_encoder.close)
                if self.peers.get(session_id) is peer:
                    self.peers.pop(session_id, None)
                    self.closers.pop(session_id, None)

        async def close_peer():
            nonlocal close_task
            if close_task is None:
                close_task = asyncio.create_task(do_close())
            await asyncio.shield(close_task)

        self.closers[session_id] = close_peer

        def task(coroutine):
            item = asyncio.create_task(coroutine)
            tasks.add(item)
            item.add_done_callback(tasks.discard)
            return item

        async def connection_deadline():
            await asyncio.sleep(30)
            if peer.connectionState != "connected":
                await close_peer()

        task(connection_deadline())

        @peer.on("connectionstatechange")
        async def state_changed():
            if peer.connectionState in ("closed", "failed"):
                await close_peer()
            elif peer.connectionState == "disconnected":
                async def grace():
                    await asyncio.sleep(5)
                    if peer.connectionState == "disconnected":
                        await close_peer()
                task(grace())

        @peer.on("datachannel")
        def datachannel(channel):
            if channel.label != "world-controls-v1" or channels or not channel.ordered or channel.maxRetransmits is not None or channel.maxPacketLifeTime is not None:
                channel.close()
                return
            channels.append(channel)
            tokens = 60.0
            updated = time.monotonic()

            def reply(value):
                if channel.readyState == "open" and channel.bufferedAmount < 65536:
                    channel.send(json.dumps(value, separators=(",", ":")))

            @channel.on("message")
            def message(value):
                nonlocal tokens, updated
                now = time.monotonic()
                tokens = min(60.0, tokens + (now - updated) * 60)
                updated = now
                if not isinstance(value, str) or len(value.encode()) > 16384 or tokens < 1:
                    channel.close()
                    return
                tokens -= 1
                identity = None
                try:
                    message = json.loads(value)
                    if not isinstance(message, dict) or not isinstance(message.get("id"), str) or not 1 <= len(message["id"]) <= 80 or not isinstance(message.get("action"), dict):
                        raise ValueError("Invalid control envelope")
                    identity = message["id"]
                    result = self.gateway.action(session_id, message["action"])
                    reply({"type": "ack", "id": identity, "ok": True, "result": result})
                except Exception as error:
                    status = getattr(error, "status", 400)
                    reply({"type": "ack", "id": identity, "ok": False, "status": status,
                           "error": "The model rejected this control. Check its advertised capabilities."})

            async def telemetry():
                while channel.readyState != "closed" and not session.stop.is_set():
                    await asyncio.sleep(1)
                    reply({"type": "telemetry", "sentFrames": track.sent, "droppedFrames": track.dropped,
                           "sourceFrameIndex": track.index, "encoder": packet_encoder.encoder if packet_encoder else "software",
                           "encoderFallback": packet_encoder.fallback_reason if packet_encoder else None,
                           "bitrateMode": "fixed" if packet_encoder else "adaptive", "targetBitrate": bitrate if packet_encoder else None,
                           "interactionMode": getattr(self.gateway.adapter, "capabilities", {}).get("runtime", {}).get("interactionMode", "next-clip")})
            task(telemetry())

        try:
            await peer.setRemoteDescription(RTCSessionDescription(sdp=body["sdp"], type="offer"))
            await peer.setLocalDescription(await peer.createAnswer())
            if self.closed:
                raise StreamingError(503, "Worker streaming is shutting down")
            return {"sdp": peer.localDescription.sdp, "type": "answer"}
        except BaseException:
            await close_peer()
            raise

    def close_session(self, session_id):
        if self.closed:
            return
        async def close():
            closer = self.closers.get(session_id)
            if closer:
                await closer()
        asyncio.run_coroutine_threadsafe(close(), self.loop).result(timeout=10)

    def close(self):
        if self.closed:
            return
        self.closed = True
        async def shutdown():
            async with self.negotiation_lock:
                await asyncio.gather(*(closer() for closer in tuple(self.closers.values())))
            # This loop is dedicated to this service; await aiortc background ICE tasks too.
            remaining = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in remaining:
                task.cancel()
            await asyncio.gather(*remaining, return_exceptions=True)
        asyncio.run_coroutine_threadsafe(shutdown(), self.loop).result(timeout=30)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)
