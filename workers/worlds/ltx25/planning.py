"""Bounded, GPU-independent exploration planning for the pinned chunk pipeline."""

import math
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass

PROFILES = {"responsive": (49, 17, 8), "balanced": (73, 25, 8), "smooth": (97, 25, 16)}


@dataclass(frozen=True)
class Window:
    pixel_frames: int
    start_pixel_frame: int
    prev_video_carry_frames: int
    next_video_carry_frames: int
    next_video_blend_frames: int


class Timeline:
    def __init__(self):
        self.end = 0
        self.carry = 0

    def next(self, cadence):
        frames, carry, blend = PROFILES[cadence]
        window = Window(frames, self.end - self.carry, self.carry, carry, blend)
        self.end += frames - self.carry
        self.carry = carry
        return window


def validate_request(request):
    if not isinstance(request, dict) or request.get("op") != "generate":
        raise ValueError("Unsupported operation")
    seed = request.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Invalid seed")
    if request.get("quality") not in ("low-latency", "balanced", "quality"):
        raise ValueError("Invalid quality")
    if not isinstance(request.get("prompt"), str) or len(request["prompt"]) > 8000:
        raise ValueError("Invalid prompt")
    action = request.get("action")
    if not isinstance(action, dict):
        raise TypeError("Exploration state required")
    revision = action.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("Invalid revision")
    settings = action.get("exploration", {})
    if (
        settings.get("mode") not in ("walk", "direct", "cruise")
        or settings.get("cadence") not in PROFILES
    ):
        raise ValueError("Invalid exploration settings")
    speed = settings.get("speed")
    if (
        isinstance(speed, bool)
        or not isinstance(speed, (float, int))
        or not math.isfinite(speed)
        or not 0 <= speed <= 1
    ):
        raise ValueError("Invalid speed")
    motion = action.get("motion")
    if not isinstance(motion, dict) or set(motion) - {
        "forward",
        "right",
        "up",
        "yaw",
        "pitch",
    }:
        raise ValueError("Invalid motion")
    for value in motion.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not -1 <= value <= 1
        ):
            raise ValueError("Invalid motion axis")
    base = action.get("basePrompt", request["prompt"])
    amendments = action.get("promptAmendments", [])
    if (
        not isinstance(base, str)
        or len(base) > 8000
        or not isinstance(amendments, list)
        or len(amendments) > 6
        or any(not isinstance(x, str) for x in amendments)
        or sum(map(len, amendments)) > 3000
    ):
        raise ValueError("Invalid scene amendments")
    return action


def compose_prompt(request):
    action = validate_request(request)
    settings, motion = action["exploration"], action["motion"]
    base = action.get("basePrompt", request["prompt"])
    amendments = action.get("promptAmendments", [])
    pieces = [
        "A continuous first-person view of the same scene. Preserve the existing surroundings, lighting and identities. No cuts or transitions."
    ]
    if amendments:
        pieces.append(
            "Scene directions, newest first; newer directions take precedence: "
            + "; ".join(reversed(amendments))
        )
    directions = []
    # Quantize directions/speed to avoid re-encoding every mouse pixel or stick jitter.
    if settings["mode"] != "direct" and settings["speed"] > 0:
        for axis, positive, negative in [
            ("forward", "moves forward", "moves backward"),
            ("right", "moves right", "moves left"),
            ("up", "rises", "descends"),
        ]:
            if abs(motion.get(axis, 0)) >= 0.12:
                directions.append(positive if motion[axis] > 0 else negative)
    for axis, positive, negative in [
        ("yaw", "turns right", "turns left"),
        ("pitch", "looks upward", "looks downward"),
    ]:
        if abs(motion.get(axis, 0)) >= 0.04:
            directions.append(positive if motion[axis] > 0 else negative)
    pace = (
        "slowly"
        if settings["speed"] < 0.34
        else "steadily"
        if settings["speed"] < 0.67
        else "quickly"
    )
    pieces.append(
        "The camera "
        + (
            pace + " " + " and ".join(directions)
            if directions
            else "holds its position and orientation"
        )
        + ". Natural continuous motion and ambient sound."
    )
    return "\n".join([pieces[-1], *pieces[:-1], "World description: " + base])


def budget_prompt(request, tokenizer, limit=1023):
    """Reserve actual tokenizer space for camera, world and newest scene changes.

    The upstream encoder adds BOS and truncates at 1024. Budget with its packed
    tokenizer first so a long world description cannot hide all live controls.
    """
    import copy

    original = compose_prompt(request)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    if len(encode(original)) <= limit:
        return original, False
    bounded = copy.deepcopy(request)
    action = bounded["action"]
    base_tokens = encode(action.get("basePrompt", bounded["prompt"]))
    action["basePrompt"] = tokenizer.decode(
        base_tokens[: min(384, limit // 3)], skip_special_tokens=True
    )
    amendments = list(action.get("promptAmendments", []))
    action["promptAmendments"] = []
    # Reserve separators/header space and retain the most recent command first.
    available = max(0, limit - len(encode(compose_prompt(bounded))) - 40)
    retained = []
    for amendment in reversed(amendments):
        tokens = encode(amendment)
        if available <= 0:
            break
        kept = tokens[:available]
        retained.append(tokenizer.decode(kept, skip_special_tokens=True))
        available -= len(kept) + 4
    action["promptAmendments"] = list(reversed(retained))
    result = compose_prompt(bounded)
    # Token boundaries can change around joined strings. Since the base is last,
    # this final guard only trims lowest-priority context, never the camera prefix.
    while len(encode(result)) > limit:
        result = result[:-1]
    return result, True


class PromptCache:
    def __init__(self, capacity=4):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.encodings = 0

    def get(self, prompt, encode):
        if prompt not in self.entries:
            self.entries[prompt] = encode(prompt)
            self.encodings += 1
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        self.entries.move_to_end(prompt)
        return self.entries[prompt]


class TransformerResidency:
    """Instance-local lifetime override for the pinned upstream DiffusionStage.

    Never cache its normal context: upstream disposes weights on every exit.
    No global monkeypatch and no reuse between sessions.
    """

    def __init__(self, stage, keep):
        self.stage, self.keep = stage, keep
        self.original = stage._transformer_ctx
        self.model = None
        self.loads = 0
        stage._transformer_ctx = self.context

    @contextmanager
    def context(self, **kwargs):
        if not self.keep:
            self.loads += 1
            with self.original(**kwargs) as model:
                yield model
        else:
            if self.model is None:
                self.model = self.stage._build_transformer(**kwargs)
                self.loads += 1
            yield self.model

    def close(self):
        self.stage._transformer_ctx = self.original
        if self.model is not None:
            self.model.dispose()
            self.model = None
