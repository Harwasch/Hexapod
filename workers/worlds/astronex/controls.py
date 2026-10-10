"""Bounded simultaneous camera controls sampled at each model block boundary.

Axis conventions mirror pinned utils/camera_trajectory.py: OpenCV camera space,
0.08 translation units and 3 degrees rotation per latent frame. No model-internal
mid-block steering or persistent KV memory is implied.
"""
import math
import time

AXES = {"forward": ("forward", 1), "backward": ("forward", -1),
        "right": ("right", 1), "left": ("right", -1), "up": ("up", 1), "down": ("up", -1),
        "look_left": ("yaw", -1), "look_right": ("yaw", 1), "look_up": ("pitch", 1), "look_down": ("pitch", -1)}
NATIVE_ACTIONS = [*AXES, "stop", "look", "analog", "move"]


def finite(value, low=-1, high=1):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Control values must be finite and within advertised bounds")
    return float(value)


def validate_motion(value):
    if not isinstance(value, dict) or set(value) - {"forward", "right", "up", "yaw", "pitch"}:
        raise ValueError("Unsupported camera axes")
    return {key: finite(item) for key, item in value.items()}


class CameraControls:
    def __init__(self):
        self.held = {}
        self.analog = {}
        self.mouse = {"yaw": 0.0, "pitch": 0.0}

    def update(self, action, values, now=None):
        now = time.monotonic() if now is None else now
        if not isinstance(values, dict) or "pressed" in values and not isinstance(values["pressed"], bool):
            raise ValueError("Invalid control values")
        if action == "stop":
            self.clear()
        elif action == "look":
            # Browser pointer motion in pixels; capped accumulated angle prevents
            # a burst of stale packets causing an unbounded next-block spin.
            dx, dy = finite(values.get("dx", values.get("x", 0)), -500, 500), finite(values.get("dy", values.get("y", 0)), -500, 500)
            self.mouse["yaw"] = max(-1, min(1, self.mouse["yaw"] + dx / 150))
            self.mouse["pitch"] = max(-1, min(1, self.mouse["pitch"] - dy / 150))
        elif action in ("analog", "move"):
            motion = validate_motion({key: value for key, value in values.items() if key != "pressed"})
            self.analog = {} if values.get("pressed") is False else {key: (value, now + 2) for key, value in motion.items()}
        elif action in AXES:
            if values.get("pressed") is False:
                self.held.pop(action, None)
            else:
                self.held[action] = (finite(values.get("value", 1), 0, 1), now + 2)
        else:
            raise ValueError("Unsupported native action")

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        self.held = {key: value for key, value in self.held.items() if value[1] > now}
        self.analog = {key: value for key, value in self.analog.items() if value[1] > now}
        motion = {key: value[0] for key, value in self.analog.items()}
        for action, (value, _) in self.held.items():
            axis, sign = AXES[action]
            motion[axis] = motion.get(axis, 0) + sign * value
        for axis, value in self.mouse.items():
            motion[axis] = motion.get(axis, 0) + value
        self.mouse = {"yaw": 0.0, "pitch": 0.0}
        return {key: max(-1, min(1, value)) for key, value in motion.items()}

    def clear(self):
        self.held.clear()
        self.analog.clear()
        self.mouse = {"yaw": 0.0, "pitch": 0.0}


def camera_poses(motion, frames):
    """Numeric equivalent of upstream composed motions, also accepting analog."""
    import numpy as np
    motion = validate_motion(motion)
    transform = np.eye(4)
    poses = [transform.copy()]
    yaw, pitch = motion.get("yaw", 0) * math.pi / 60, motion.get("pitch", 0) * math.pi / 60
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    rotation = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    displacement = np.array([motion.get("right", 0), -motion.get("up", 0), motion.get("forward", 0)]) * 0.08
    for _ in range(frames):
        transform[:3, :3] = transform[:3, :3] @ rotation
        transform[:3, 3] += transform[:3, :3] @ displacement
        poses.append(transform.copy())
    return np.linalg.inv(np.array(poses)).astype(np.float32)
