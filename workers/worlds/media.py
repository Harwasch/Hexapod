"""Bounded raw-frame interchange shared by isolated inference and transport."""
from pathlib import Path


def validate_raw_frames(raw, directory):
    if not isinstance(raw, dict) or raw.get("pixelFormat") != "rgb24":
        raise RuntimeError("Unsupported raw frame format")
    for key, minimum, maximum in (("width", 1, 1920), ("height", 1, 1080), ("count", 1, 240)):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise RuntimeError("Raw frame dimensions or count exceed limits")
    expected = raw["width"] * raw["height"] * 3 * raw["count"]
    if expected > 192 * 1024 * 1024 or not isinstance(raw.get("path"), str):
        raise RuntimeError("Raw frame spool exceeds limits")
    path = Path(raw["path"])
    if not path.is_file() or not path.resolve().is_relative_to(Path(directory).resolve()) or path.stat().st_size != expected:
        raise RuntimeError("Raw frame spool is missing, malformed or outside its private directory")
    return path


def raw_video_frame(data, size):
    from av import VideoFrame
    width, height = size
    frame = VideoFrame(width, height, "rgb24")
    plane = frame.planes[0]
    row = width * 3
    if plane.line_size == row:
        plane.update(data)
    else:
        padded = bytearray(plane.buffer_size)
        for y in range(height):
            padded[y * plane.line_size:y * plane.line_size + row] = data[y * row:(y + 1) * row]
        plane.update(padded)
    return frame
