"""Teacher A: a plant's motion parameters, fitted to video of it swaying.

ADR 0008 decided that video models teach Living Mode **parameters**, not motion: each limb's
frequency (and, relative to the trunk, how much it sways), fitted from clips and written
into the same `motion.json` the runtime already plays, tagged `fitted-generated`. This is
that loop, every stage of it, with the video model behind one interface (`ClipSource`):

1. **stills** -- the plant rendered from a few cameras near where the capture looked
   (`splat_render`), the image a video model is conditioned on;
2. **clips** -- `ClipSource.clips(...)`: frames of the plant swaying, starting from each
   still. `OscillatorClips` makes them on the CPU from a known sidecar (every limb a damped
   oscillator driven by noise, posed through the rig, rendered): the stand-in that proves
   the fit before a model is involved. A video model (Wan 2.2 TI2V-5B, Cosmos-Predict2.5)
   is another `ClipSource`, run on a GPU (infra/modal/world_models.py);
3. **tracks** -- which pixels show which limb, from a label render of the first frame
   (the clip starts from our own still, so frame 0 is ours to label; only pixels at least
   `MIN_PURITY` one limb's count), and how each limb's pixels move: textured points inside them followed by pyramidal Lucas-Kanade with a
   forward-backward check (OpenCV), the median of their motion, in metres at the limb's
   depth. A limb's own motion is what is left once its nearest tracked ancestor's motion
   is regressed out: what the trunk does, every limb on it does too, scaled by lever;
4. **fit** -- per limb, the damped oscillator whose response best explains the Welch
   spectrum of its own motion (averaged over every clip and camera), kept when its
   resonance stands `MIN_PROMINENCE` times above the fitted noise floor; its sway amplitude
   relative to the trunk's. About a minute of footage per camera is needed: with 20 s the
   fit misses by up to 1.4x, with 60 s it is within 5 % on the stand-in's own angles. Damping is never fitted: monocular video does not
   recover it (Wind on Trees, arXiv 2609.17810), and the prior's stays.
5. **sidecar** -- the prior with the fitted frequencies (and relative gains, clamped)
   written in, `motionEvidence: "fitted-generated"`, and a report of every limb.

Nothing here is specific to the stand-in: test L1 (tests/test_teacher_motion.py) renders
clips from the allometric sidecar of the synthetic tree, fits them, and recovers the
frequencies it was given.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from splat_render import Camera, Splats, render

__all__ = [
    "MIN_PROMINENCE",
    "ClipSource",
    "LimbFit",
    "OscillatorClips",
    "Rig",
    "fit_limbs",
    "fitted_sidecar",
    "limb_labels",
    "teach",
    "track_limbs",
]

#: A limb's fitted resonance must peak this many times above the fitted noise floor.
MIN_PROMINENCE = 6.0
#: ...reported up to this (a fit with no noise floor at all).
MAX_PROMINENCE = 1e6
#: Welch segments, seconds: a resolution of ~0.2 Hz, under a limb's resonance width.
SEGMENT_S = 5.0
#: The natural-frequency grid of the resonance fit, Hz.
RESONANCE_STEP_HZ = 0.005
#: The band searched for a limb's frequency, Hz. Trees ring at 0.2-10 Hz (de Langre 2019).
BAND_HZ = (0.25, 8.0)
#: A limb needs this many labelled pixels in frame 0 to be tracked.
MIN_LIMB_PIXELS = 40
#: Only pixels at least this much one limb's are tracked as that limb's: in a crown the
#: foliage of neighbouring limbs overlaps on screen, and a point there sways with both.
MIN_PURITY = 0.8
#: A tracked point is dropped once tracking it back a frame misses by more than this.
FB_TOLERANCE_PX = 0.5
#: Fitted gains relative to the prior's are clamped to this range.
GAIN_CLAMP = (0.25, 4.0)
#: A limb is fitted only from at least this many tracks (clips x cameras that saw it).
MIN_TRACKS = 2
#: A limb is searched from this fraction of the trunk's fitted frequency up.
LIMB_FLOOR = 0.9
#: A limb's peak this close (relative) to a tracked ancestor's is that ancestor's.
INHERITED_TOLERANCE = 0.05


class Rig:
    """The rig's tree and the sidecar's limbs, in the form the teacher uses."""

    def __init__(self, rig: dict, sidecar: dict) -> None:
        nodes = rig["nodes"]
        self.positions = np.array([n["position"] for n in nodes], np.float64)
        self.parents = np.array([n["parent"] for n in nodes], np.int64)
        if np.any(self.parents[1:] >= np.arange(1, len(nodes))):
            raise ValueError("a rig's parents must come before their children")
        columns = sidecar["nodes"]
        self.branch = np.asarray(columns["branch"], np.int64)
        self.frequency = np.asarray(columns["frequencyHz"], np.float64)
        self.damping = np.asarray(columns["damping"], np.float64)
        self.gain = np.asarray(columns["gainRad"], np.float64)
        #: Every limb (oscillator), by its base joint; the trunk's is the root's branch.
        self.limbs = sorted({int(b) for b in self.branch[1:]})
        self.tree = int(self.branch[1]) if len(nodes) > 1 else 0
        #: The limb a limb hangs from (its base joint's parent's limb), -1 for the trunk.
        self.parent_limb = {
            b: (-1 if b == self.tree else int(self.branch[max(int(self.parents[b]), 0)]))
            for b in self.limbs
        }
        for b in self.limbs:  # a limb whose base hangs from the root rides the trunk
            if self.parent_limb[b] in (b, 0) and b != self.tree:
                self.parent_limb[b] = self.tree

    def limb_frequency(self, limb: int) -> float:
        return float(self.frequency[self.branch == limb][0])

    def limb_damping(self, limb: int) -> float:
        return float(self.damping[self.branch == limb][0])

    def pose(self, angles: dict[int, float], axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Joint positions and cumulative rotations with each limb bent by `angles[limb]`
        (radians at unit gain) about `axis`: joint `i` turns by `gain_i * angle` about its
        parent's rest position, as Living Mode's hinge does."""
        n = self.positions.shape[0]
        out = self.positions.copy()
        rotations = np.repeat(np.eye(3)[None], n, axis=0)
        for i in range(1, n):
            p = int(self.parents[i])
            theta = self.gain[i] * angles.get(int(self.branch[i]), 0.0)
            rotations[i] = rotations[p] @ _axis_angle(axis, theta)
            out[i] = out[p] + rotations[i] @ (self.positions[i] - self.positions[p])
        return out, rotations


def _axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
    k = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * (kx @ kx)


def nearest_joints(rig: Rig, positions: np.ndarray, block: int = 4096) -> np.ndarray:
    """Each splat's nearest joint (rigid skinning: enough for a stand-in and for labels)."""
    out = np.empty(positions.shape[0], np.int64)
    for start in range(0, positions.shape[0], block):
        d = np.linalg.norm(positions[start : start + block, None, :] - rig.positions[None], axis=2)
        out[start : start + block] = d.argmin(axis=1)
    return out


def limb_labels(rig: Rig, joints: np.ndarray) -> np.ndarray:
    """Each splat's limb: its joint's branch (the root's splats ride the trunk)."""
    labels = rig.branch[joints].copy()
    labels[joints == 0] = rig.tree
    return labels


class ClipSource(Protocol):
    """Clips of the plant swaying, each starting from a still. `frames[c][k]` is clip `c`'s
    frame `k`, (h, w, 3) uint8, as video is; clip `c` is of still `c % len(stills)`."""

    fps: float

    def clips(
        self, stills: Sequence[np.ndarray], cameras: Sequence[Camera], seeds: Sequence[int]
    ) -> list[list[np.ndarray]]: ...


@dataclass
class OscillatorClips:
    """The stand-in: every limb a damped oscillator at the sidecar's frequency, driven by
    white noise, unit RMS times `amplitude`, posed through the rig and rendered.

    `amplitude` scales every joint's gain: the sidecar's are radians at the reference wind
    (hundredths of a radian), which move a limb tip by under a pixel at a useful distance;
    a generated clip of a tree in a breeze moves by several."""

    splats: Splats
    rig: Rig
    seconds: float = 10.0
    fps: float = 24.0
    amplitude: float = 3.0
    wind_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)  # bends about y: sways along x
    joints: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.joints = nearest_joints(self.rig, self.splats.positions)

    def angles(self, seed: int) -> dict[int, np.ndarray]:
        """Each limb's unit-RMS angle over the clip."""
        frames = round(self.seconds * self.fps)
        n = 4 * frames  # generated long and cropped, so the clip does not wrap
        f = np.fft.rfftfreq(n, 1 / self.fps)
        rng = np.random.default_rng(seed)
        out = {}
        for limb in self.rig.limbs:
            fn, zeta = self.rig.limb_frequency(limb), self.rig.limb_damping(limb)
            h = 1.0 / np.sqrt((fn**2 - f**2) ** 2 + (2 * zeta * fn * f) ** 2 + 1e-12)
            spectrum = h * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
            spectrum[0] = 0
            signal = np.fft.irfft(spectrum, n)[frames : 2 * frames]
            out[limb] = signal / max(float(signal.std()), 1e-12) * self.amplitude
        return out

    def clips(
        self, stills: Sequence[np.ndarray], cameras: Sequence[Camera], seeds: Sequence[int]
    ) -> list[list[np.ndarray]]:
        del stills  # the stand-in renders the scan itself
        axis = np.asarray(self.wind_axis, np.float64)
        out: list[list[np.ndarray]] = []
        for seed in seeds:
            series = self.angles(seed)
            frames_by_camera: list[list[np.ndarray]] = [[] for _ in cameras]
            for k in range(next(iter(series.values())).size):
                posed, rotations = self.rig.pose({b: float(s[k]) for b, s in series.items()}, axis)
                moved = _skin(self.splats, self.rig, self.joints, posed, rotations)
                for c, camera in enumerate(cameras):
                    rgb = render(moved, camera).rgb
                    frames_by_camera[c].append(np.round(rgb * 255).astype(np.uint8))
            out.extend(frames_by_camera)
        return out


def _skin(
    splats: Splats, rig: Rig, joints: np.ndarray, posed: np.ndarray, rotations: np.ndarray
) -> Splats:
    parent = np.maximum(rig.parents[joints], 0)
    pivot_rest = rig.positions[parent]
    pivot_now = posed[parent]
    r = rotations[joints]
    moved = pivot_now + np.einsum("nij,nj->ni", r, splats.positions - pivot_rest)
    root = joints == 0
    moved[root] = splats.positions[root]
    return Splats(moved, splats.rotations, splats.scales, splats.colours, splats.opacities)


def cameras_around(
    rig: Rig, *, count: int = 3, distance_factor: float = 2.2, width: int = 640, height: int = 480
) -> list[Camera]:
    """Cameras at a plant's mid-height, `distance_factor` times its height away, spread
    around it -- where a person filming a tree would stand."""
    base = rig.positions[0]
    top = float(rig.positions[:, 2].max() - base[2])
    target = base + np.array([0.0, 0.0, 0.5 * top])
    out = []
    for k in range(count):
        a = 2 * math.pi * k / count + math.pi / 2  # facing across the wind first
        eye = target + distance_factor * top * np.array([math.cos(a), math.sin(a), 0.1])
        out.append(Camera.look_at(eye, target, fov_deg=40, width=width, height=height))
    return out


def stills(splats: Splats, cameras: Sequence[Camera]) -> list[np.ndarray]:
    return [render(splats, camera).rgb for camera in cameras]


@dataclass
class LimbTrack:
    limb: int
    camera: int
    clip: int
    #: Displacement of the limb's pixels against frame 0, metres, (frames, 2).
    metres: np.ndarray


def _grey(frame: np.ndarray) -> np.ndarray:
    return frame @ np.array([0.299, 0.587, 0.114])


def _to_u8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return np.round(_grey(frame.astype(np.float64))).astype(np.uint8)
    return np.clip(_grey(frame) * 255.0, 0, 255).astype(np.uint8)


def track_limbs(
    clips: list[list[np.ndarray]],
    clip_cameras: Sequence[int],
    label_maps: Sequence[np.ndarray],
    depth_maps: Sequence[np.ndarray],
    cameras: Sequence[Camera],
    limbs: Sequence[int],
    purity_maps: Sequence[np.ndarray] | None = None,
    points_per_limb: int = 40,
) -> list[LimbTrack]:
    """Each limb's displacement in each clip: the median motion of textured points inside
    its pixels in frame 0, followed frame to frame by pyramidal Lucas-Kanade and kept only
    while the backward track returns within `FB_TOLERANCE_PX` of where it started."""
    import cv2

    if purity_maps is None:
        purity_maps = [np.ones(m.shape) for m in label_maps]

    lk = {
        "winSize": (15, 15),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    tracks = []
    for c, frames in enumerate(clips):
        cam_index = clip_cameras[c]
        labels, depth, camera = label_maps[cam_index], depth_maps[cam_index], cameras[cam_index]
        grey = [_to_u8(f) for f in frames]
        starts: dict[int, np.ndarray] = {}
        for limb in limbs:
            mask = ((labels == limb) & (purity_maps[cam_index] >= MIN_PURITY)).astype(np.uint8)
            if int(mask.sum()) < MIN_LIMB_PIXELS:
                continue
            corners = cv2.goodFeaturesToTrack(
                grey[0], points_per_limb, 0.01, 3, mask=mask, blockSize=5
            )
            if corners is not None and len(corners) >= 3:
                starts[limb] = corners.reshape(-1, 2).astype(np.float32)
        if not starts:
            continue
        order = list(starts)
        sizes = [starts[limb].shape[0] for limb in order]
        points = np.concatenate([starts[limb] for limb in order])
        alive = np.ones(points.shape[0], bool)
        path = [points.copy()]
        current = points.copy()
        for k in range(1, len(grey)):
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(grey[k - 1], grey[k], current, None, **lk)
            back, status_b, _ = cv2.calcOpticalFlowPyrLK(grey[k], grey[k - 1], nxt, None, **lk)
            ok = (status.reshape(-1) == 1) & (status_b.reshape(-1) == 1)
            ok &= np.linalg.norm(back - current, axis=1) < FB_TOLERANCE_PX
            alive &= ok
            current = np.where(alive[:, None], nxt, current)
            path.append(current.copy())
        moved = np.stack(path) - path[0][None]  # (frames, points, 2)
        offset = 0
        for limb, size in zip(order, sizes, strict=True):
            sl = slice(offset, offset + size)
            offset += size
            keep = alive[sl]
            if keep.sum() < 3:
                continue
            pixels = np.median(moved[:, sl][:, keep], axis=1)
            metres_per_px = float(np.median(depth[labels == limb])) / camera.focal
            tracks.append(LimbTrack(limb, cam_index, c, pixels * metres_per_px))
    return tracks


@dataclass
class LimbFit:
    limb: int
    frequency_hz: float | None
    prominence: float
    amplitude_m: float
    lever_m: float
    tracks: int
    prior_hz: float
    #: "fitted", "weak" (no peak above `MIN_PROMINENCE`), "inherited" (an ancestor's peak),
    #: "carrier-untracked" (its parent limb was not tracked), "few-tracks" (fewer than
    #: `MIN_TRACKS`) or "untracked".
    status: str = "untracked"
    #: The peak found, whether or not it was kept.
    peak_hz: float | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "limb": self.limb,
            "frequencyHz": None if self.frequency_hz is None else round(self.frequency_hz, 4),
            "priorHz": round(self.prior_hz, 4),
            "prominence": round(self.prominence, 2),
            "amplitudeM": round(self.amplitude_m, 6),
            "leverM": round(self.lever_m, 4),
            "tracks": self.tracks,
            "status": self.status,
            "peakHz": None if self.peak_hz is None else round(self.peak_hz, 4),
        }


def _principal(series: np.ndarray) -> np.ndarray:
    """A 2D displacement series as one signal along its principal direction, detrended."""
    x = series - series.mean(axis=0)
    t = np.arange(x.shape[0])
    for k in range(2):
        x[:, k] -= np.polyval(np.polyfit(t, x[:, k], 1), t)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[0]


def _welch(signals: Sequence[np.ndarray], fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch's power spectrum, averaged over every signal: Hann segments of `SEGMENT_S`
    (or the whole signal, if shorter), half overlapping."""
    seg = min(round(SEGMENT_S * fps), min(s.size for s in signals))
    step = max(seg // 2, 1)
    window = np.hanning(seg)
    spectra = []
    for s in signals:
        for a in range(0, s.size - seg + 1, step):
            x = s[a : a + seg] - s[a : a + seg].mean()
            spectra.append(np.abs(np.fft.rfft(x * window)) ** 2)
    return np.fft.rfftfreq(seg, 1 / fps), np.mean(spectra, axis=0)


def _resonance(signals: Sequence[np.ndarray], fps: float, low: float) -> tuple[float, float, float]:
    """The damped oscillator whose response best explains the signals' spectrum above
    `low`: natural frequency (Hz), damping ratio, and the resonance's peak over the noise
    floor. A white-noise-driven oscillator's power is `a·|H(f)|² + b`, `|H|⁻² =
    (fn² − f²)² + (2ζ·fn·f)²`; `a`, `b` by least squares in relative error, `fn` on a
    `RESONANCE_STEP_HZ` grid and ζ on a log grid. The whole resonance is fitted, not its
    tallest bin: a lightly damped oscillator's periodogram peak wanders by 5-20 % over a
    minute of footage, its fitted shape by under 5 % (measured on the stand-in's own
    angles)."""
    freqs, power = _welch(signals, fps)
    high = min(BAND_HZ[1], fps / 2.5)
    band = (freqs >= low) & (freqs <= high)
    f, p = freqs[band], power[band]
    weight = 1.0 / np.maximum(p, 1e-30) ** 2
    best = (np.inf, low, 0.0, 0.0)
    natural = np.arange(max(low, BAND_HZ[0]), high, RESONANCE_STEP_HZ)
    for zeta in np.geomspace(0.02, 0.5, 24):
        h = 1.0 / (
            (natural[:, None] ** 2 - f[None] ** 2) ** 2
            + (2 * zeta * natural[:, None] * f[None]) ** 2
        )
        a11, a12, a22 = (h * h * weight).sum(1), (h * weight).sum(1), weight.sum()
        b1, b2 = (h * p * weight).sum(1), (p * weight).sum()
        det = a11 * a22 - a12**2
        a = np.maximum((b1 * a22 - a12 * b2) / det, 0.0)
        b = np.maximum((a11 * b2 - a12 * b1) / det, 0.0)
        err = (((a[:, None] * h + b[:, None] - p[None]) ** 2) * weight[None]).sum(1)
        i = int(err.argmin())
        if err[i] < best[0]:
            peak = a[i] * h[i].max()
            ratio = peak / b[i] if b[i] > 0 else MAX_PROMINENCE
            best = (
                float(err[i]),
                float(natural[i]),
                float(zeta),
                float(min(ratio, MAX_PROMINENCE)),
            )
    return best[1], best[2], best[3]


def fit_limbs(rig: Rig, tracks: Sequence[LimbTrack], fps: float) -> list[LimbFit]:
    """Each tracked limb's frequency (its own motion's spectral peak) and amplitude.

    The trunk first, over the whole band. Then every other limb, searched from
    `LIMB_FLOOR` of the trunk's fitted frequency up -- a limb rings no lower than the tree
    it hangs from (Rodriguez, de Langre & Moulia 2008; the allometric rule's own floor) --
    and refused (`inherited`) when its peak is within `INHERITED_TOLERANCE` of a tracked
    ancestor's, or (`carrier-untracked`) when the limb it hangs from was not tracked: in
    both its own motion was not separated from what carries it."""
    by_key = {(t.limb, t.clip): t for t in tracks}
    order = [rig.tree] + [b for b in rig.limbs if b != rig.tree]
    fitted: dict[int, float] = {}
    fits: dict[int, LimbFit] = {}
    for limb in order:
        own = []
        for t in tracks:
            if t.limb != limb:
                continue
            ancestor = _tracked_ancestor(rig, limb, t.clip, by_key)
            own.append(_principal(_own_motion(t.metres, ancestor.metres if ancestor else None)))
        lever = _lever(rig, limb)
        prior = rig.limb_frequency(limb)
        if not own:
            fits[limb] = LimbFit(limb, None, 0.0, 0.0, lever, 0, prior)
            continue
        trunk = fitted.get(rig.tree)
        low = BAND_HZ[0] if limb == rig.tree or trunk is None else LIMB_FLOOR * trunk
        frequency, _zeta, prominence = _resonance(own, fps, low)
        status = "fitted" if prominence >= MIN_PROMINENCE else "weak"
        if status == "fitted" and len(own) < MIN_TRACKS:
            status = "few-tracks"
        parent = rig.parent_limb[limb]
        if status == "fitted" and parent != -1 and not any(t.limb == parent for t in tracks):
            # What carries it was not seen, so its motion cannot be taken out: unknown.
            status = "carrier-untracked"
        if status == "fitted" and limb != rig.tree:
            for ancestor in _ancestors(rig, limb):
                f = fitted.get(ancestor)
                if f is not None and abs(frequency - f) <= INHERITED_TOLERANCE * f:
                    status = "inherited"
                    break
        if status == "fitted":
            fitted[limb] = frequency
        amplitude = float(np.mean([s.std() for s in own]))
        fits[limb] = LimbFit(
            limb,
            frequency if status == "fitted" else None,
            prominence,
            amplitude,
            lever,
            len(own),
            prior,
            status,
            frequency,
        )
    return [fits[limb] for limb in rig.limbs]


def _ancestors(rig: Rig, limb: int) -> list[int]:
    out, parent = [], rig.parent_limb[limb]
    while parent != -1 and parent not in out and parent != limb:
        out.append(parent)
        parent = rig.parent_limb[parent]
    return out


def _tracked_ancestor(
    rig: Rig, limb: int, clip: int, by_key: dict[tuple[int, int], LimbTrack]
) -> LimbTrack | None:
    """The nearest limb up the tree from `limb` that was tracked in `clip`."""
    parent = rig.parent_limb[limb]
    seen = {limb}
    while parent != -1 and parent not in seen:
        found = by_key.get((parent, clip))
        if found is not None:
            return found
        seen.add(parent)
        parent = rig.parent_limb[parent]
    return None


def _own_motion(series: np.ndarray, ancestor: np.ndarray | None) -> np.ndarray:
    """A limb's motion less what its ancestor's explains: the least-squares fit of the
    ancestor's two components (and a constant) removed. A limb carried by a swaying trunk
    moves with it, scaled by how much farther from the trunk's pivot it is."""
    if ancestor is None:
        return series
    design = np.column_stack([ancestor, np.ones(ancestor.shape[0])])
    coef, *_ = np.linalg.lstsq(design, series, rcond=None)
    return series - design @ coef


def _lever(rig: Rig, limb: int) -> float:
    """How far a limb's joints reach from its base's pivot, metres."""
    members = np.flatnonzero(rig.branch == limb)
    pivot = rig.positions[max(int(rig.parents[limb]), 0)]
    return (
        float(np.linalg.norm(rig.positions[members] - pivot, axis=1).max()) if members.size else 0.0
    )


def fitted_sidecar(prior: dict, rig: Rig, fits: Sequence[LimbFit], source: str) -> dict:
    """`prior` with every fitted limb's frequency (and its gain relative to the trunk's)
    written in, tagged `fitted-generated`."""
    out = copy.deepcopy(prior)
    nodes = out["nodes"]
    freq = list(nodes["frequencyHz"])
    gain = list(nodes["gainRad"])
    by_limb = {f.limb: f for f in fits}
    trunk = by_limb.get(rig.tree)
    trunk_angle = (
        trunk.amplitude_m / trunk.lever_m if trunk and trunk.lever_m > 0 and trunk.tracks else None
    )
    prior_trunk = float(rig.gain[rig.branch == rig.tree].sum()) or None
    fitted = 0
    for fit in fits:
        if fit.frequency_hz is None:
            continue
        fitted += 1
        members = np.flatnonzero(rig.branch == fit.limb)
        scale = 1.0
        if (
            fit.limb != rig.tree
            and trunk_angle
            and prior_trunk
            and fit.lever_m > 0
            and rig.gain[members].sum() > 0
        ):
            observed = (fit.amplitude_m / fit.lever_m) / trunk_angle
            predicted = float(rig.gain[members].sum()) / prior_trunk
            scale = float(np.clip(observed / predicted, *GAIN_CLAMP))
        for i in members:
            freq[i] = round(fit.frequency_hz, 4)
            gain[i] = round(gain[i] * scale, 7)
    nodes["frequencyHz"] = freq
    nodes["gainRad"] = gain
    out["motionEvidence"] = "fitted-generated"
    out["provenance"] = dict(out.get("provenance", {}))
    out["provenance"]["fittedGenerated"] = {
        "rule": (
            "each limb's frequency: the damped oscillator best fitting the Welch spectrum of "
            "its motion less its nearest tracked ancestor's, tracked by Lucas-Kanade in clips "
            f"of the plant, kept when its resonance stands {MIN_PROMINENCE:g}x above the noise "
            "floor; its gain scaled by its sway "
            f"relative to the trunk's against the prior's ratio, clamped to {GAIN_CLAMP}; "
            "damping is the prior's (not recoverable from monocular video)"
        ),
        "source": source,
        "status": "fitted",
        "limbsFitted": fitted,
        "limbs": len(fits),
    }
    out["generator"] = "tools/captures/teacher_motion.py"
    return out


@dataclass
class Lesson:
    sidecar: dict
    fits: list[LimbFit]
    cameras: list[Camera]
    report: dict[str, object]


def teach(
    splats: Splats,
    rig_doc: dict,
    prior: dict,
    source: ClipSource,
    *,
    cameras: Sequence[Camera] | None = None,
    seeds: Sequence[int] = (1, 2, 3),
    source_name: str = "oscillator stand-in",
) -> Lesson:
    """The whole loop: stills, clips, tracks, fits, sidecar."""
    rig = Rig(rig_doc, prior)
    cams = list(cameras) if cameras is not None else cameras_around(rig)
    joints = nearest_joints(rig, splats.positions)
    labels = limb_labels(rig, joints)
    firsts = [render(splats, camera, labels=labels) for camera in cams]
    clips = source.clips([f.rgb for f in firsts], cams, seeds)
    clip_cameras = [c % len(cams) for c in range(len(clips))]
    tracks = track_limbs(
        clips,
        clip_cameras,
        [f.label for f in firsts],
        [f.depth for f in firsts],
        cams,
        rig.limbs,
        [f.purity for f in firsts],
    )
    fits = fit_limbs(rig, tracks, source.fps)
    sidecar = fitted_sidecar(prior, rig, fits, source_name)
    fitted = [f for f in fits if f.frequency_hz is not None]
    report: dict[str, object] = {
        "source": source_name,
        "cameras": len(cams),
        "clips": len(clips),
        "fps": source.fps,
        "limbs": len(fits),
        "limbsTracked": sum(1 for f in fits if f.tracks),
        "limbsFitted": len(fitted),
        "fits": [f.to_json() for f in fits],
    }
    return Lesson(sidecar, fits, cams, report)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from splat_render import load_ply

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("source_dir", type=Path, help="a plant's source/ (splat.ply, rig.json)")
    parser.add_argument("out", type=Path, help="the fitted motion.json to write")
    parser.add_argument("--report", type=Path, help="where to write the per-limb report")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--source",
        choices=("oscillator", "wan", "cosmos"),
        default="oscillator",
        help="the CPU stand-in, or a video model on Modal (world_model_client.py)",
    )
    parser.add_argument("--cameras", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args(argv)
    rig_doc = json.loads((args.source_dir / "rig.json").read_text())
    prior = json.loads((args.source_dir / rig_doc.get("motion", "motion.json")).read_text())
    splats = load_ply(args.source_dir / "splat.ply")
    rig = Rig(rig_doc, prior)
    source: ClipSource
    if args.source == "oscillator":
        source, name = OscillatorClips(splats, rig, seconds=args.seconds), "oscillator stand-in"
    else:
        from world_model_client import VideoClips

        clips = VideoClips(model={"wan": "Wan", "cosmos": "Cosmos"}[args.source])
        source, name = clips, clips.name
    lesson = teach(
        splats,
        rig_doc,
        prior,
        source,
        cameras=cameras_around(rig, count=args.cameras, width=args.width, height=args.height),
        seeds=tuple(range(1, args.seeds + 1)),
        source_name=name,
    )
    args.out.write_text(json.dumps(lesson.sidecar, indent=1) + "\n", encoding="utf-8")
    if args.report:
        args.report.write_text(json.dumps(lesson.report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in lesson.report.items() if k != "fits"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
