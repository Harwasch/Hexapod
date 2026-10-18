"""Optional public-API H264 packet encoding; no aiortc encoder monkey patches.

Hardware availability is established by encoding the first actual frame. A failed
NVENC attempt falls back once to libx264. Selection never calls external services.
"""
from fractions import Fraction
import os
import threading


def encoder_settings(environ=None):
    env = os.environ if environ is None else environ
    mode = env.get("WORLD_VIDEO_ENCODER", "software")
    if mode not in ("software", "nvenc", "h264-software"):
        raise ValueError("WORLD_VIDEO_ENCODER must be software, nvenc or h264-software")
    try:
        bitrate = int(env.get("WORLD_VIDEO_BITRATE", "3000000"))
    except ValueError:
        raise ValueError("WORLD_VIDEO_BITRATE must be an integer") from None
    if not 250000 <= bitrate <= 12000000:
        raise ValueError("WORLD_VIDEO_BITRATE must be between 250000 and 12000000")
    return mode, bitrate


class H264PacketEncoder:
    """Bounded synchronous encoder, invoked by the track on its decode thread.

Packet tracks cannot consume aiortc's private PLI/REMB state. Send an IDR every
second and expose the configured fixed bitrate rather than claiming adaptation.
"""
    def __init__(self, mode, bitrate, codec_factory=None):
        import av
        self.create = codec_factory or av.CodecContext.create
        self.mode = mode
        self.bitrate = bitrate
        self.codec = None
        self.encoder = "pending"
        self.fallback_reason = None
        self.last_keyframe = None
        self.lock = threading.RLock()
        self.closed = False

    def _open(self, name, frame):
        codec = self.create(name, "w")
        codec.width, codec.height = frame.width, frame.height
        codec.pix_fmt = "yuv420p"
        codec.time_base = Fraction(1, 90000)
        codec.framerate = Fraction(30, 1)
        codec.bit_rate = self.bitrate
        codec.gop_size = 30
        codec.max_b_frames = 0
        codec.profile = "Baseline"
        codec.options = ({"preset": "p4", "tune": "ull", "rc": "cbr", "zerolatency": "1", "forced-idr": "1", "repeat_headers": "1"}
                         if name == "h264_nvenc" else {"preset": "ultrafast", "tune": "zerolatency", "x264-params": "repeat-headers=1:scenecut=0"})
        codec.open()
        return codec

    def encode(self, frame):
        with self.lock:
            if self.closed:
                return None
            return self._encode(frame)

    def _encode(self, frame):
        import av
        if self.last_keyframe is None or frame.pts - self.last_keyframe >= 90000:
            frame.pict_type = av.video.frame.PictureType.I
            self.last_keyframe = frame.pts
        else:
            frame.pict_type = av.video.frame.PictureType.NONE
        preferred = "h264_nvenc" if self.mode == "nvenc" else "libx264"
        try:
            if self.codec is None:
                self.codec = self._open(preferred, frame)
            packets = self.codec.encode(frame)
            self.encoder = "nvenc" if preferred == "h264_nvenc" else "software-h264"
        except Exception:
            if self.mode != "nvenc":
                raise
            # Discard an unusable GPU context. Never retry every frame or silently
            # continue claiming hardware encoding after driver/OOM/codec failure.
            self.codec = None
            self.mode = "h264-software"
            self.fallback_reason = "NVENC unavailable or failed; using software H264"
            self.last_keyframe = None
            return self.encode(frame)
        if not packets:
            return None
        packet = av.Packet(b"".join(bytes(item) for item in packets))
        packet.pts = frame.pts
        packet.dts = frame.pts
        packet.time_base = frame.time_base
        return packet

    def close(self):
        with self.lock:
            self.closed = True
            self.codec = None
