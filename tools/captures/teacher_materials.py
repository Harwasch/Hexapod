"""The video teacher for materials: per instance stiffness, damping and drag, from a clip.

Step C2 of docs/LIVING_PLAN.md. The browser sways every skinned object with an anchored modal
model (`packages/world/src/skinWind.ts`, ported in `skin_wind.py`) whose only free numbers per
instance are its material (`materials.json`, SCENE_OBJECTS.md §4):

    stiffness c (m/s):  every anchored mode's Ω_i = c · Ω_i(1)      (ω_j = c·√λ_j / scale)
    damping ζ:          every mode's damping ratio
    drag D:             the load, D · |v|v / scale through each handle's mean weight

This fits those three numbers to video of the object moving in the wind, by matching what the
model would show the camera to what the camera saw. Nothing in it knows what the object is.

1. **Observe** (`track_points`, `observe`). The scan rendered from the clip's camera at rest
   labels the object's pixels (`splat_render`, the skin's owner instance per splat). Textured
   points inside them are followed through the clip by pyramidal Lucas-Kanade *against frame 0*
   (no drift over minutes), kept while the backward track returns within `FB_TOLERANCE_PX`;
   the static background's median motion is taken out (a hand-held camera). Each point is
   lifted to the scan by its depth: where it is, so its skin weights `w(x)` (the nearest
   skinned splats') and how a horizontal metre there moves on screen (`J`, 2 × 2).

2. **Predict.** A point's screen motion is `y_p(t) = J_p Σ_j w_j(x_p) q_j(t)`: linear in the
   handles, so the mean over points of each point's power spectrum is a quadratic form in
   them, `Q = mean_p C_pᵀ C_p`. Its eigenvectors make a few channels whose spectra add up to
   exactly that mean (`Observation.channels`). The real wind's realisation is unknown, so the
   model is driven by an ensemble of the same EN 1991-1-4 field at the clip's mean speed and
   bearing (other seeds, `ENSEMBLE_FIELD_MODES` modes each: its expected spectrum), the load
   on each mode computed once (`skin_wind.modal_forces`, not depending on the material), the
   response for any `(c, ζ)` by the integrator's own exact discrete transfer function (an
   FFT, `ModelSpectra`), sampled at the clip's frame times.

3. **Fit** (`fit_material`). Welch spectra of the observed and predicted motion, averaged into
   log-spaced bands, compared in log power with a white tracking-noise floor `b`:
   `mean (log(S(f; c, ζ, D) + b) − log P(f))²` (`loss="whittle"`, the Welch ordinates'
   likelihood, is the alternative: as good on a continuous wind, worse on a line spectrum).
   The resonances' places give `c`, their widths and the share of motion below them `ζ`, the
   level `D` -- and that only with the wind's mean speed known: from a clip of an unknown
   wind, `D` is relative to the speed assumed. A grid over `(c, ζ)` around the prior with
   `D`, `b` profiled, then Nelder-Mead on all three with the model's output bound (`tanh`,
   as the browser draws it).

4. **Write** (`write_materials`). The fitted record in `materials.json`, tagged with its rung
   of the evidence ladder: `fitted-real` for footage of the real object, `fitted-generated` for
   a clip a world model made (or this module's own synthetic clips); other records are kept.

Validation (`validate`, `tests/test_teacher_materials.py`): the synthetic yard, a clip rendered
with known materials from the same model, recovered from a deliberately wrong prior.

    python teacher_materials.py synth TILES SKIN --instance 9 --out /tmp/clip
    python teacher_materials.py fit TILES SKIN --instance 9 --clip /tmp/clip/clip.avi \\
        --camera /tmp/clip/camera.json --strength 0.5 --bearing 60 --materials OUT.json
    python teacher_materials.py validate TILES SKIN --instance 1 --instance 9 --out /tmp/c2
    python teacher_materials.py world TILES SKIN --instance 1 --model Wan ...  # Modal, GPU
    python teacher_materials.py world TILES SKIN --instance 225 --renderer gsplat \\
        --auto-bearing --chain 3 --save /tmp/c2 ...  # what infra/modal/dream.py runs

`world` draws the still the model starts from over a pale sky (`SKY`), from the bearing the
object is seen best from (`--auto-bearing`: `best_bearing`, the most pixels where it is the
nearest thing drawn), and `--chain N` makes each clip N model calls long, each starting
from the last frame of the one before (a model's own clip is ~5 s).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import skin_wind as sw
from splat_render import Camera, SplatIndex, Splats, render

__all__ = [
    "FitResult",
    "ModelSpectra",
    "Observation",
    "Scene",
    "camera_for",
    "fit_material",
    "load_scene",
    "observe",
    "render_clip",
    "track_points",
    "write_materials",
]

#: A tracked point is dropped at a frame where tracking it back misses by more than this.
FB_TOLERANCE_PX = 0.5
#: ... and altogether when it is lost in more than this share of the frames.
MAX_LOST = 0.1
#: Points need at least this share of the object's coverage.
MIN_PURITY = 0.8
#: Pixels shaved off the object's mask: the clip's frame 0 is not the scan's rest pose.
MASK_ERODE_PX = 1
#: Fewest points an instance is fitted from.
MIN_POINTS = 12
#: Seconds the model runs from rest before what it predicts is used (as the browser's does).
WARM_S = 40.0
#: Ensemble seeds of the predicted wind: never a clip's (the realisation is unknown).
ENSEMBLE_SEEDS = (90001, 90002, 90003, 90004)
#: Modes of the ensemble's turbulence fields. The browser's field has 64 (enough for a sway
#: that looks right); its spectrum is a few dozen lines, so one realisation's response is a
#: handful of spikes on the resonance rather than its shape. Real wind is continuous: the
#: prediction is the expected spectrum, estimated with the rigs' 320 modes per field.
ENSEMBLE_FIELD_MODES = 320
#: Spectral bands per decade the fit compares.
BANDS_PER_DECADE = 30
#: The fit's band: from this many Welch bins up ...
LOW_BINS = 2
#: ... to this share of the Nyquist frequency (and the model's 25 Hz field cut-off).
HIGH_NYQUIST = 0.8
#: The resonance must stand this far above the fitted noise floor to be written.
MIN_SIGNAL_TO_NOISE = 3.0
#: The grid's span around the prior's stiffness (each way) and its damping range.
STIFFNESS_SPAN = 6.0
DAMPING_RANGE = (0.02, 0.6)


# ------------------------------------------------------------------------------------ scene


@dataclass
class Scene:
    """A scan's leaf splats with what the skins say about each."""

    splats: Splats
    owner: np.ndarray  # (n,) the instance whose skin moves it, 0 when none
    weights: np.ndarray  # (n, 15) learned weights (dequantised int8), 0 where unskinned
    skins: dict[int, sw.SkinDynamics]  # by instance
    records: dict[int, dict]  # skin.json records by instance
    index: SplatIndex = field(repr=False, default=None)  # type: ignore[assignment]

    def members(self, instance: int) -> np.ndarray:
        return np.flatnonzero(self.owner == instance)

    def moved(self, instance: int, handles: np.ndarray) -> Splats:
        """The splats with `instance` displaced by handle translations (m, 2): `x + Σ w q`."""
        at = self.members(instance)
        m = handles.shape[0]
        w = np.c_[np.ones(at.size), self.weights[at, : m - 1]]
        out = Splats(
            self.splats.positions.copy(),
            self.splats.rotations,
            self.splats.scales,
            self.splats.colours,
            self.splats.opacities,
        )
        out.positions[at, :2] += w @ handles
        return out


def load_scene(tiles_dir: Path, skin_dir: Path) -> Scene:
    """Every leaf gaussian of a tileset (as `splat_render.load_tileset`), each with its skin
    owner and weights from `skin.json` + `skin.bin` (bound by tile checksum)."""
    from rig_tiles import glb_spz
    from skin_scene import QUANT, ROW_BYTES, decode_runs
    from splat_render import _from_columns
    from splat_tiles import unpack_spz
    from synthetic_tree import checksum_positions

    tileset = json.loads((tiles_dir / "tileset.json").read_text(encoding="utf-8"))
    skin_doc = json.loads((skin_dir / "skin.json").read_text(encoding="utf-8"))
    rows = np.frombuffer((skin_dir / "skin.bin").read_bytes(), np.int8).reshape(-1, ROW_BYTES)
    instance_of_skin = {int(s["id"]): int(s["instance"]) for s in skin_doc["skins"]}
    leaves: list[str] = []
    stack = [tileset["root"]]
    while stack:
        tile = stack.pop()
        if tile.get("children"):
            stack.extend(tile["children"])
        else:
            leaves.append(tile["content"]["uri"])
    parts, owners, weights = [], [], []
    for uri in sorted(leaves):
        columns = unpack_spz(glb_spz(tiles_dir / uri))
        positions = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype("<f4")
        positions[positions == 0.0] = 0.0
        entry = skin_doc["tiles"].get(checksum_positions(positions))
        n = positions.shape[0]
        owner = np.zeros(n, np.int64)
        w = np.zeros((n, ROW_BYTES - 1 + 1))
        if entry is not None:
            which = decode_runs(entry["skins"])
            skinned = which > 0
            start = int(entry["row"])
            w[skinned] = rows[start : start + int(skinned.sum())].astype(np.float64) / QUANT
            owner[skinned] = [instance_of_skin[int(s)] for s in which[skinned]]
        parts.append(_from_columns(columns))
        owners.append(owner)
        weights.append(w)
    splats = Splats.concat(parts)
    records = {int(s["instance"]): s for s in skin_doc["skins"]}
    skins = {k: sw.SkinDynamics.from_skin(r) for k, r in records.items() if "dynamics" in r}
    scene = Scene(splats, np.concatenate(owners), np.concatenate(weights), skins, records)
    scene.index = SplatIndex.build(splats)
    return scene


def camera_for(
    scene: Scene,
    instance: int,
    bearing_deg: float,
    *,
    width: int = 320,
    height: int = 240,
    fov_deg: float = 40.0,
    margin: float = 1.15,
) -> Camera:
    """A camera at the object's mid-height, looking at it across the wind (so the sway is
    seen side on), just far enough back that the whole object fits (`margin` to spare)."""
    pos = scene.splats.positions[scene.members(instance)]
    lo, hi = pos.min(0), pos.max(0)
    target = (lo + hi) / 2
    height_m = float(hi[2] - lo[2])
    width_m = float(np.ptp(pos[:, :2], 0).max())
    half_h = math.atan(math.tan(math.radians(fov_deg) / 2) * height / width)
    half_w = math.radians(fov_deg) / 2
    distance = margin * max(0.5 * height_m / math.tan(half_h), 0.5 * width_m / math.tan(half_w))
    distance += 0.5 * width_m
    b = math.radians(bearing_deg)
    across = np.array([math.cos(b), -math.sin(b), 0.0])
    eye = target + across * distance
    return Camera.look_at(eye, target, fov_deg=fov_deg, width=width, height=height)


#: A sky for stills shown to a video model: where the scan has nothing, a world model reads
#: black as night or a void; a pale overcast sky is what it expects above trees.
SKY = (0.78, 0.82, 0.86)


def to_u8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(rgb) * 255), 0, 255).astype(np.uint8)


def draw(scene: Scene, camera: Camera, renderer: object | None = None, **kwargs: object):
    """The scan from `camera`: `splat_render.render` on the CPU, or a `GsplatRenderer`."""
    if renderer is None:
        return render(scene.splats, camera, index=scene.index, **kwargs)  # type: ignore[arg-type]
    return renderer(scene.splats, camera, **kwargs)  # type: ignore[operator]


def visible_pixels(
    scene: Scene, instance: int, camera: Camera, renderer: object | None = None
) -> tuple[int, int]:
    """(pixels where `instance` is the nearest thing drawn, pixels it covers on its own):
    how much of it the camera sees past whatever stands in front of it."""
    only = scene.splats.take(scene.members(instance))
    if renderer is None:
        alone = render(only, camera)
    else:
        alone = renderer(only, camera)  # type: ignore[operator]
    full = draw(scene, camera, renderer)
    covered = (alone.alpha > 0.5) & np.isfinite(alone.depth)
    front = covered & (alone.depth <= full.depth * 1.03 + 0.05)
    return int(front.sum()), int(covered.sum())


def best_bearing(
    scene: Scene,
    instance: int,
    renderer: object | None = None,
    *,
    bearings: Sequence[float] = tuple(range(0, 360, 30)),
    width: int = 256,
    height: int = 144,
) -> tuple[float, list[dict]]:
    """The bearing (`camera_for`'s: the camera looks across a wind blowing toward it) from
    which most of `instance` is seen unoccluded, and every bearing's count."""
    tried = []
    for bearing in bearings:
        camera = camera_for(scene, instance, float(bearing), width=width, height=height)
        seen, covered = visible_pixels(scene, instance, camera, renderer)
        tried.append({"bearing": float(bearing), "visible": seen, "covered": covered})
    best = max(tried, key=lambda t: (t["visible"], t["visible"] / max(t["covered"], 1)))
    return best["bearing"], tried


# ----------------------------------------------------------------------- synthetic clips


@dataclass
class Clip:
    frames: list[np.ndarray]
    fps: float
    camera: Camera
    #: What made it, when known (a synthetic clip): handle translations per frame (T, m, 2).
    handles: np.ndarray | None = None
    truth: dict | None = None


def wind_states(
    model: sw.SkinWindModel,
    wind: sw.SkinWind,
    seed: int,
    times: np.ndarray,
    t0: float = 1000.0,
    field_modes: int = sw.SKIN_WIND_FIELD_MODES,
) -> np.ndarray:
    """The browser's model at `times` (seconds after `t0`), switched on `WARM_S` before `t0`
    from rest: bounded handle translations (T, m, 2)."""
    h = sw.SKIN_WIND_STEP_S
    start = round((t0 - WARM_S) / h)
    steps = round((WARM_S + float(np.max(times))) / h) + 2
    grid = (start + np.arange(steps) + 0.5) * h
    force = sw.modal_forces(
        model, sw.handle_forces(model, sw.SkinWindField(seed, modes=field_modes), wind, grid)
    )
    states = sw.simulate(model.omega, model.damping, force * model.drag * model.scale)
    k = (np.asarray(times) + WARM_S) / h
    lo = np.floor(k + 1e-9).astype(np.int64)
    alpha = np.clip(k - lo, 0, 1)[:, None, None]
    at = states[lo] + alpha * (states[np.minimum(lo + 1, steps)] - states[lo])
    return sw.bounded_handles(model, at)


def render_clip(
    scene: Scene,
    instance: int,
    material: sw.SkinMaterial,
    wind: sw.SkinWind,
    *,
    seed: int,
    field_modes: int = sw.SKIN_WIND_FIELD_MODES,
    seconds: float,
    fps: float,
    camera: Camera,
    log: bool = False,
) -> Clip:
    """A clip of `instance` swaying under the browser's own model with a known material: what
    the fit must recover. Only that instance moves; the rest of the scan is drawn still."""
    model = sw.skin_wind_model(scene.skins[instance], material)
    if model is None:
        raise ValueError(f"instance {instance}: no anchored modes")
    times = np.arange(round(seconds * fps)) / fps
    handles = wind_states(model, wind, seed, times, field_modes=field_modes)
    frames = []
    started = time.perf_counter()
    for k, q in enumerate(handles):
        rgb = render(scene.moved(instance, q), camera, index=scene.index).rgb
        frames.append(np.round(rgb * 255).astype(np.uint8))
        if log and k % 200 == 0:
            print(f"  frame {k}/{len(handles)} ({time.perf_counter() - started:.0f} s)", flush=True)
    truth = {
        "instance": instance,
        "stiffness": material.stiffness,
        "damping": material.damping,
        "drag": material.drag,
        "speedMps": wind.speed_mps,
        "bearingDeg": wind.bearing_deg,
        "seed": seed,
        "fieldModes": field_modes,
    }
    return Clip(frames, fps, camera, handles, truth)


# ------------------------------------------------------------------------------- tracking


def _grey(frame: np.ndarray) -> np.ndarray:
    import cv2

    if frame.dtype != np.uint8:
        frame = np.clip(np.round(frame * 255), 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def track_points(
    frames: Sequence[np.ndarray], mask: np.ndarray, *, max_points: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    """Textured points in `mask` of frame 0, followed against frame 0 itself (each frame's
    guess the last good position): their pixel coordinates in frame 0, OpenCV convention
    (P, 2), and displacements (T, P, 2), NaN where the forward-backward check failed."""
    import cv2

    g0 = _grey(frames[0])
    corners = cv2.goodFeaturesToTrack(
        g0, max_points, 0.01, 3, mask=mask.astype(np.uint8), blockSize=5
    )
    if corners is None:
        return np.zeros((0, 2), np.float32), np.zeros((len(frames), 0, 2))
    p0 = corners.reshape(-1, 2).astype(np.float32)
    lk = {
        "winSize": (15, 15),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        "flags": cv2.OPTFLOW_USE_INITIAL_FLOW,
    }
    disp = np.full((len(frames), p0.shape[0], 2), np.nan)
    disp[0] = 0
    guess = p0.copy()
    for k in range(1, len(frames)):
        gk = _grey(frames[k])
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(g0, gk, p0, guess.copy(), **lk)
        back, st_b, _ = cv2.calcOpticalFlowPyrLK(gk, g0, nxt, p0.copy(), **lk)
        ok = (st.reshape(-1) == 1) & (st_b.reshape(-1) == 1)
        ok &= np.linalg.norm(back - p0, axis=1) < FB_TOLERANCE_PX
        disp[k, ok] = nxt[ok] - p0[ok]
        guess[ok] = nxt[ok]
    return p0, disp


def _fill(disp: np.ndarray) -> np.ndarray:
    """NaN gaps in each point's series filled linearly in time."""
    out = disp.copy()
    t = np.arange(disp.shape[0])
    for p in range(disp.shape[1]):
        for a in range(2):
            s = out[:, p, a]
            bad = np.isnan(s)
            if bad.any() and not bad.all():
                s[bad] = np.interp(t[bad], t[~bad], s[~bad])
    return out


@dataclass
class Observation:
    """What the camera saw of one instance, and how the model's handles would look to it."""

    instance: int
    fps: float
    camera: Camera
    uv: np.ndarray  # (P, 2) frame-0 pixels, OpenCV convention
    points: np.ndarray  # (P, 3) where they are on the scan, tileset frame
    gains: np.ndarray  # (P, 2, 2m): screen motion per handle translation (east, north)
    motion: np.ndarray  # (T, P, 2) pixels, against frame 0
    background_points: int = 0

    @property
    def frames(self) -> int:
        return int(self.motion.shape[0])

    def save(self, path: Path) -> None:
        """Everything a fit needs, so a clip is tracked once (`load`)."""
        np.savez_compressed(
            path,
            instance=self.instance,
            fps=self.fps,
            camera=json.dumps(self.camera.to_json()),
            uv=self.uv,
            points=self.points,
            gains=self.gains,
            motion=self.motion,
            background_points=self.background_points,
        )

    @staticmethod
    def load(path: Path) -> Observation:
        z = np.load(path)
        return Observation(
            int(z["instance"]),
            float(z["fps"]),
            Camera.from_json(json.loads(str(z["camera"]))),
            z["uv"],
            z["points"],
            z["gains"],
            z["motion"],
            int(z["background_points"]),
        )

    def channels(self) -> np.ndarray:
        """(2m, K): handle translations (flattened j-major, east then north) to channels whose
        spectra add up to the mean over points of each point's spectrum."""
        c = self.gains
        q = np.einsum("pak,pal->kl", c, c) / max(c.shape[0], 1)
        values, vectors = np.linalg.eigh((q + q.T) / 2)
        keep = values > 1e-9 * max(float(values.max()), 1e-300)
        return vectors[:, keep] * np.sqrt(values[keep])[None]


def observe(
    scene: Scene, instance: int, frames: Sequence[np.ndarray], fps: float, camera: Camera
) -> Observation:
    """Tracks `instance` in a clip shot by `camera` (posed in the scan's frame) and lifts what
    it tracked onto the scan."""
    import cv2

    rest = render(scene.splats, camera, labels=scene.owner, index=scene.index)
    mask = (rest.label == instance) & (rest.purity >= MIN_PURITY)
    # The label render is grainy: closed first, then shaved.
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if MASK_ERODE_PX > 0:
        closed = cv2.erode(closed, np.ones((2 * MASK_ERODE_PX + 1,) * 2, np.uint8))
    uv, disp = track_points(frames, closed > 0)
    keep = np.isnan(disp[..., 0]).mean(0) <= MAX_LOST
    uv, disp = uv[keep], _fill(disp[:, keep])
    # The static background's motion is the camera's: taken out.
    background = (rest.label == 0) & (rest.alpha > 0.5)
    background_points = 0
    if background.sum() > 200:
        _, bdisp = track_points(frames, background, max_points=150)
        bkeep = np.isnan(bdisp[..., 0]).mean(0) <= MAX_LOST
        background_points = int(bkeep.sum())
        if background_points >= 10:
            disp = disp - np.nanmedian(bdisp[:, bkeep], axis=1)[:, None, :]
    return lift(scene, instance, camera, fps, uv, disp, rest.label, rest.depth, background_points)


def lift(
    scene: Scene,
    instance: int,
    camera: Camera,
    fps: float,
    uv: np.ndarray,
    disp: np.ndarray,
    label: np.ndarray,
    depth_map: np.ndarray,
    background_points: int = 0,
) -> Observation:
    """Tracked points onto the scan: each point's depth (the median of the instance's own
    pixels around it in the rest render), so where it is, its skin weights (the nearest skinned
    splats of the instance, by inverse distance) and how a horizontal metre there moves on
    screen. Points with no depth are dropped."""
    centre_uv = uv.astype(np.float64) + 0.5  # OpenCV's pixel centres are integers
    col = np.clip(np.floor(centre_uv[:, 0]).astype(int), 0, camera.width - 1)
    row = np.clip(np.floor(centre_uv[:, 1]).astype(int), 0, camera.height - 1)
    own = np.where((label == instance) & np.isfinite(depth_map), depth_map, np.nan)
    window = np.lib.stride_tricks.sliding_window_view(
        np.pad(own, 2, constant_values=np.nan), (5, 5)
    )
    around = window[row, col].reshape(len(row), -1)
    seen = np.isfinite(around).any(1)
    depth = np.full(len(row), np.nan)
    depth[seen] = np.nanmedian(around[seen], axis=1)
    ok = np.isfinite(depth)
    uv, disp, centre_uv, depth = uv[ok], disp[:, ok], centre_uv[ok], depth[ok]
    local = np.stack(
        [
            (centre_uv[:, 0] - camera.width / 2) / camera.focal * depth,
            (centre_uv[:, 1] - camera.height / 2) / camera.focal * depth,
            depth,
        ],
        axis=1,
    )
    points = camera.centre + local @ camera.rotation
    members = scene.members(instance)
    m = scene.skins[instance].handles
    dist, nb = cKDTree(scene.splats.positions[members]).query(points, k=min(6, members.size))
    dist, nb = dist.reshape(len(points), -1), nb.reshape(len(points), -1)
    inv = 1.0 / np.maximum(dist, 1e-3)
    learned = scene.weights[members][:, : m - 1]
    w = np.einsum("pk,pkj->pj", inv, learned[nb]) / inv.sum(1, keepdims=True)
    w = np.c_[np.ones(len(points)), w]
    # d(uv)/d(east, north) of the pinhole at each point.
    pc = (points - camera.centre) @ camera.rotation.T
    r, inv_z = camera.rotation, camera.focal / pc[:, 2:3]
    jac = np.stack(
        [
            inv_z * (r[0][None, :2] - (pc[:, 0:1] / pc[:, 2:3]) * r[2][None, :2]),
            inv_z * (r[1][None, :2] - (pc[:, 1:2] / pc[:, 2:3]) * r[2][None, :2]),
        ],
        axis=1,
    )  # (P, 2 screen, 2 ground)
    gains = np.einsum("pj,pab->pajb", w, jac).reshape(len(points), 2, 2 * m)
    return Observation(instance, fps, camera, uv, points, gains, disp, background_points)


# --------------------------------------------------------------------------------- spectra


def _welch(series: np.ndarray, fps: float, segment: int) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch density along axis 0 (Hann, half overlap, mean removed)."""
    from scipy.signal import welch

    return welch(series, fs=fps, nperseg=segment, noverlap=segment // 2, axis=0, detrend="constant")


def segment_frames(frames: int, fps: float) -> int:
    """Welch segments: a quarter of the clip (seven half-overlapping segments)."""
    return max(16, min(frames, frames // 4 if frames >= 64 else frames))


def bands(freqs: np.ndarray, low: float, high: float) -> list[np.ndarray]:
    """Log-spaced bands of Welch bins in `[low, high]`, each at least one bin."""
    edges = np.geomspace(low, high, max(2, round(BANDS_PER_DECADE * math.log10(high / low)) + 1))
    out = []
    which = np.searchsorted(edges, freqs, side="right") - 1
    for b in range(len(edges) - 1):
        sel = np.flatnonzero((which == b) & (freqs >= low) & (freqs <= high))
        if sel.size:
            out.append(sel)
    return out


class ModelSpectra:
    """The model's expected spectrum of an observation, for any material: the modal loads of
    an ensemble of winds computed once, the response for `(c, ζ)` by the integrator's exact
    discrete transfer function."""

    def __init__(
        self,
        source: sw.SkinDynamics,
        observation: Observation,
        wind: sw.SkinWind,
        *,
        seeds: Sequence[int] = ENSEMBLE_SEEDS,
        seconds: float | None = None,
        field_modes: int = ENSEMBLE_FIELD_MODES,
    ) -> None:
        self.unit = sw.skin_wind_model(source, sw.SkinMaterial(1.0, 0.1, 1.0))
        if self.unit is None:
            raise ValueError("no anchored modes")
        self.fps = observation.fps
        self.frames = observation.frames
        record = seconds if seconds is not None else self.frames / self.fps
        h = sw.SKIN_WIND_STEP_S
        self.warm = round(WARM_S / h)
        self.steps = self.warm + round(record / h) + 2
        self.nfft = 1 << math.ceil(math.log2(2 * self.steps))
        self.channels = observation.channels()  # (2m, K)
        self.times = np.arange(round(record * self.fps)) / self.fps
        self.segment = segment_frames(self.frames, self.fps)
        self.spectra = []
        for seed in seeds:
            grid = (1000.0 + np.arange(self.steps) + 0.5) * h
            forces = sw.handle_forces(
                self.unit, sw.SkinWindField(seed, modes=field_modes), wind, grid
            )
            modal = sw.modal_forces(self.unit, forces) * self.unit.drag * self.unit.scale
            self.spectra.append(np.fft.rfft(modal, n=self.nfft, axis=0))  # per unit D
        z = np.exp(2j * np.pi * np.fft.rfftfreq(self.nfft))
        self.z = z

    def modal(self, stiffness: float, damping: float, seed: int = 0) -> np.ndarray:
        """Modal displacements per unit drag at the frame times, one ensemble member,
        (T, r, 2). The transfer function is the integrator's own: with `X = [s, ṡ]`,
        `X_{k+1} = A X_k + (I − A)[f_k / Ω², 0]`, so `s(z) = e₁ᵀ(zI − A)⁻¹(I − A)e₁ f(z) / Ω²`
        (state `k` at grid time `k·h`, the force of step `k` at its middle), the sum taken
        without wrap-around (zero-padded to `nfft`) from rest, as the browser starts."""
        omega = self.unit.omega * stiffness
        coef = sw.transition(omega, damping, sw.SKIN_WIND_STEP_S)
        a11, a12, a21, a22 = (coef[:, i] for i in range(4))
        w2 = omega**2
        z = self.z[:, None]
        det = (z - a11) * (z - a22) - a12 * a21
        transfer = ((z - a22) * (1 - a11) - a12 * a21) / (w2 * det)  # (F, r)
        states = np.fft.irfft(transfer[:, :, None] * self.spectra[seed], n=self.nfft, axis=0)
        k = self.warm + self.times / sw.SKIN_WIND_STEP_S
        lo = np.floor(k + 1e-9).astype(np.int64)
        alpha = np.clip(k - lo, 0, 1)[:, None, None]
        return states[lo] + alpha * (states[lo + 1] - states[lo])

    def handles(
        self, stiffness: float, damping: float, drag: float, seed: int = 0, *, bounded: bool = True
    ) -> np.ndarray:
        """Handle translations at the frame times, one ensemble member, (T, m, 2): bound as
        the browser draws them, or not (then linear in `drag`)."""
        at = self.modal(stiffness, damping, seed) * drag
        if bounded:
            return sw.bounded_handles(self.unit, at)
        return np.einsum("jr,tra->tja", self.unit.shapes, at)

    def psd(
        self, stiffness: float, damping: float, drag: float = 1.0, *, bounded: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Welch frequencies and the ensemble's mean spectrum of the observation, px²/Hz."""
        total = None
        freqs = None
        for seed in range(len(self.spectra)):
            q = self.handles(stiffness, damping, drag, seed, bounded=bounded)
            series = q.reshape(q.shape[0], -1) @ self.channels
            freqs, p = _welch(series, self.fps, self.segment)
            p = p.sum(1)
            total = p if total is None else total + p
        assert freqs is not None and total is not None
        return freqs, total / len(self.spectra)


# ------------------------------------------------------------------------------------- fit


@dataclass
class FitResult:
    instance: int
    material: sw.SkinMaterial
    prior: sw.SkinMaterial
    status: str  # "fitted" | "weak" | "untracked"
    loss: float
    noise_floor: float
    signal_to_noise: float
    points: int
    frames: int
    fps: float
    #: For plots and reports: the fit's band centres, observed, fitted and prior spectra.
    freqs: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    observed: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    fitted: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    prior_spectrum: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    seconds: float = 0.0

    def to_json(self) -> dict[str, object]:
        return {
            "instance": self.instance,
            "status": self.status,
            "stiffness": round(self.material.stiffness, 4),
            "damping": round(self.material.damping, 4),
            "drag": round(self.material.drag, 5),
            "prior": {
                "stiffness": round(self.prior.stiffness, 4),
                "damping": round(self.prior.damping, 4),
                "drag": round(self.prior.drag, 5),
            },
            "loss": round(self.loss, 5),
            "noiseFloor": float(f"{self.noise_floor:.4g}"),
            "signalToNoise": round(self.signal_to_noise, 2),
            "points": self.points,
            "frames": self.frames,
            "fps": self.fps,
            "seconds": round(self.seconds, 1),
        }


def _banded(freqs: np.ndarray, psd: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    return np.array([psd[g].mean() for g in groups])


def _profile(
    model: np.ndarray, observed: np.ndarray, scale_free: bool, loss: str
) -> tuple[float, float, float]:
    """Best `(loss, a, b)` for the spectrum `a·model + b` against `observed`, over a grid of
    the noise floor `b` and of the level `a` (1 unless `scale_free`). `loss` is "whittle", the
    negative log-likelihood of Welch ordinates, `mean(log S + P/S)` (the estimator's own), or
    "log", `mean((log S − log P)²)` over bands."""
    log_p = np.log(observed)
    floor = float(observed.min())
    b = np.r_[0.0, floor * np.geomspace(1e-3, 1.5, 40)]
    if scale_free:
        a0 = math.exp(float(np.mean(log_p - np.log(np.maximum(model, 1e-300)))))
        a = a0 * np.geomspace(0.02, 50, 121)
    else:
        a = np.array([1.0])
    pred = np.maximum(a[:, None, None] * model[None, None, :] + b[None, :, None], 1e-300)
    if loss == "whittle":
        err = (np.log(pred) + observed[None, None, :] / pred).mean(-1)
    else:
        err = ((np.log(pred) - log_p[None, None, :]) ** 2).mean(-1)
    i, j = np.unravel_index(int(err.argmin()), err.shape)
    return float(err[i, j]), float(a[i]), float(b[j])


def fit_material(
    scene: Scene,
    observation: Observation,
    wind: sw.SkinWind,
    prior: sw.SkinMaterial,
    *,
    grid: tuple[int, int] = (33, 10),
    seeds: Sequence[int] = ENSEMBLE_SEEDS,
    loss: str = "log",
    log: bool = False,
) -> FitResult:
    """`(c, ζ, D)` of one instance from its observation (module docstring, step 3)."""
    from scipy.optimize import minimize

    started = time.perf_counter()
    instance = observation.instance
    points = observation.motion.shape[1]
    if points < MIN_POINTS:
        return FitResult(
            instance,
            prior,
            prior,
            "untracked",
            math.inf,
            0,
            0,
            points,
            observation.frames,
            observation.fps,
        )
    spectra = ModelSpectra(scene.skins[instance], observation, wind, seeds=seeds)
    freqs, observed_psd = _welch(observation.motion, observation.fps, spectra.segment)
    observed_psd = observed_psd.mean(1).sum(-1)  # mean over points, both screen axes
    low = freqs[min(LOW_BINS, freqs.size - 1)]
    high = min(HIGH_NYQUIST * observation.fps / 2, sw.FIELD_CUTOFF_HZ)
    shown = bands(freqs, low, high)
    centres = np.array([math.exp(np.log(freqs[g]).mean()) for g in shown])
    groups = (
        [np.array([i]) for i in np.flatnonzero((freqs >= low) & (freqs <= high))]
        if loss == "whittle"
        else shown
    )
    observed = _banded(freqs, observed_psd, groups)

    # 1. Grid over (c, ζ), linear model, D and the noise floor profiled.
    cs = prior.stiffness * np.geomspace(1 / STIFFNESS_SPAN, STIFFNESS_SPAN, grid[0])
    zetas = np.geomspace(*DAMPING_RANGE, grid[1])
    best = (math.inf, prior.stiffness, prior.damping, prior.drag)
    for c in cs:
        for zeta in zetas:
            _, p = spectra.psd(float(c), float(zeta), 1.0, bounded=False)
            value, a, _b = _profile(_banded(freqs, p, groups), observed, True, loss)
            if value < best[0]:
                best = (value, float(c), float(zeta), math.sqrt(a))
    if log:
        print(f"  grid: c {best[1]:.3f} zeta {best[2]:.3f} D {best[3]:.4f} loss {best[0]:.4f}")

    # 2. Nelder-Mead on (log c, log ζ, log D), the model's output bound as drawn.
    def objective(x: np.ndarray) -> float:
        c, zeta, drag = math.exp(x[0]), math.exp(x[1]), math.exp(x[2])
        if not DAMPING_RANGE[0] / 2 <= zeta <= sw.SKIN_MAX_DAMPING:
            return 1e9
        _, p = spectra.psd(c, zeta, drag, bounded=True)
        return _profile(_banded(freqs, p, groups), observed, False, loss)[0]

    x0 = np.log([best[1], best[2], best[3]])
    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"xatol": 2e-3, "fatol": 1e-5, "maxfev": 120},
    )
    c, zeta, drag = (float(v) for v in np.exp(result.x))
    _, p = spectra.psd(c, zeta, drag, bounded=True)
    value, _, floor = _profile(_banded(freqs, p, groups), observed, False, loss)
    fitted_band = _banded(freqs, p, shown)
    snr = float(fitted_band.max() / floor) if floor > 0 else math.inf
    _, prior_p = spectra.psd(prior.stiffness, prior.damping, prior.drag, bounded=True)
    status = "fitted" if snr >= MIN_SIGNAL_TO_NOISE else "weak"
    material = sw.SkinMaterial(c, min(zeta, sw.SKIN_MAX_DAMPING), drag, True, prior.evidence)
    return FitResult(
        instance,
        material,
        prior,
        status,
        value,
        floor,
        snr,
        points,
        observation.frames,
        observation.fps,
        centres,
        _banded(freqs, observed_psd, shown),
        fitted_band + floor,
        _banded(freqs, prior_p, shown) + floor,
        time.perf_counter() - started,
    )


# ----------------------------------------------------------------------------------- write

#: The evidence rung of a clip's source.
EVIDENCE = {"real": "fitted-real", "generated": "fitted-generated"}


def write_materials(path: Path, fits: Sequence[FitResult], evidence: str, source: str) -> dict:
    """Each fitted instance's record into `materials.json` (created if missing), replacing
    any record of the same instance and keeping the others; weak fits are not written."""
    if path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
    else:
        doc = {"format": "hexapod.materials", "version": 1, "materials": []}
    doc.setdefault(
        "model",
        "skin wind v1 (packages/world/src/skinWind.ts): omega_j = stiffness * sqrt(eigenvalue_j)"
        " / scale; every anchored mode damped at `damping`; a handle's acceleration "
        "drag * |v| v / scale",
    )
    records = {int(r["instance"]): r for r in doc.get("materials", [])}
    for fit in fits:
        if fit.status != "fitted":
            continue
        records[fit.instance] = {
            "instance": fit.instance,
            "stiffness": round(fit.material.stiffness, 4),
            "damping": round(fit.material.damping, 4),
            "drag": round(fit.material.drag, 5),
            "wind": True,
            "evidence": evidence,
            "fit": {
                "source": source,
                "generator": "tools/captures/teacher_materials.py",
                "frames": fit.frames,
                "fps": fit.fps,
                "points": fit.points,
                "signalToNoise": round(fit.signal_to_noise, 2),
                "loss": round(fit.loss, 5),
            },
        }
    doc["materials"] = [records[k] for k in sorted(records)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


# ------------------------------------------------------------------------------------ CLI


def prior_of(
    instances_doc: dict | None, materials_doc: dict | None, instance: int
) -> sw.SkinMaterial:
    """The material the browser would use now: the property prior, then any record."""
    props, behaviour = None, None
    for entry in (instances_doc or {}).get("instances", []):
        if int(entry["id"]) == instance:
            props, behaviour = entry.get("properties"), entry.get("behaviour")
    base = sw.material_prior(props, behaviour)
    for record in (materials_doc or {}).get("materials", []):
        if int(record.get("instance", -1)) == instance:
            return sw.merge_material(base, record)
    return base


def read_clip(path: Path) -> tuple[list[np.ndarray], float]:
    """A video file's frames (RGB uint8) and frame rate, or a directory of PNG frames
    (`fps` then comes from `--fps`)."""
    import cv2

    if path.is_dir():
        from PIL import Image

        files = sorted(path.glob("*.png"))
        return [np.asarray(Image.open(f).convert("RGB")) for f in files], 0.0
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise SystemExit(f"{path}: no frames")
    return frames, fps


def write_clip(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    """Frames as a lossless-enough video (MJPG at top quality, which OpenCV writes everywhere)."""
    import cv2

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def plot_fit(path: Path, fit: FitResult, title: str, truth: dict | None = None) -> None:
    """Observed, fitted and prior spectra of one fit (needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    ax.loglog(fit.freqs, fit.observed, color="#0b0b0b", lw=2, label="observed (tracked)")
    ax.loglog(fit.freqs, fit.fitted, color="#2a78d6", lw=2, label="fitted model")
    ax.loglog(fit.freqs, fit.prior_spectrum, color="#eb6834", lw=2, ls="--", label="prior model")
    m = fit.material
    text = f"fit  c {m.stiffness:.2f} m/s, ζ {m.damping:.3f}, D {m.drag:.4f}"
    p = fit.prior
    text += f"\nprior c {p.stiffness:.2f}, ζ {p.damping:.3f}, D {p.drag:.4f}"
    if truth:
        text += (
            f"\ntrue  c {truth['stiffness']:.2f}, ζ {truth['damping']:.3f}, D {truth['drag']:.4f}"
        )
    ax.text(0.02, 0.03, text, transform=ax.transAxes, fontsize=8, family="monospace", va="bottom")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("screen motion PSD (px²/Hz), mean over points")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _wind(args: argparse.Namespace) -> sw.SkinWind:
    speed = args.speed if args.speed is not None else sw.speed_from_strength(args.strength)
    return sw.SkinWind(speed, args.bearing)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tiles", type=Path, help="the tileset directory (tileset.json)")
    parser.add_argument("skin", type=Path, help="the directory of skin.json + skin.bin")
    parser.add_argument("--instance", type=int, action="append", required=True)
    parser.add_argument("--instances", type=Path, help="instances.json (property priors)")
    parser.add_argument("--materials-in", type=Path, help="current materials.json (priors)")
    parser.add_argument("--strength", type=float, default=0.1, help="Living Survey wind 0..1")
    parser.add_argument("--speed", type=float, help="mean wind speed, m/s (over --strength)")
    parser.add_argument("--bearing", type=float, default=0.0, help="downwind, deg from north")


def _priors(args: argparse.Namespace) -> tuple[dict | None, dict | None]:
    inst = json.loads(args.instances.read_text()) if args.instances else None
    mats = None
    path = args.materials_in or (args.skin / "materials.json")
    if path.exists():
        mats = json.loads(path.read_text())
    return inst, mats


def validate(
    scene: Scene,
    instance: int,
    truth: sw.SkinMaterial,
    priors: dict[str, sw.SkinMaterial],
    wind: sw.SkinWind,
    *,
    seconds: float,
    fps: float,
    seed: int = 7,
    field_modes: int = ENSEMBLE_FIELD_MODES,
    width: int = 320,
    height: int = 240,
    out: Path | None = None,
    log: bool = False,
) -> dict[str, object]:
    """Renders a clip with `truth`, fits it from each prior, reports the errors."""
    camera = camera_for(scene, instance, wind.bearing_deg, width=width, height=height)
    started = time.perf_counter()
    clip = render_clip(
        scene,
        instance,
        truth,
        wind,
        seed=seed,
        field_modes=field_modes,
        seconds=seconds,
        fps=fps,
        camera=camera,
        log=log,
    )
    rendered = time.perf_counter() - started
    observation = observe(scene, instance, clip.frames, fps, camera)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        observation.save(out / f"observation_{instance}.npz")
        np.save(out / f"handles_{instance}.npy", clip.handles)
        (out / f"truth_{instance}.json").write_text(json.dumps(clip.truth, indent=1))
    report: dict[str, object] = {
        "instance": instance,
        "truth": clip.truth,
        "frames": len(clip.frames),
        "renderSeconds": round(rendered, 1),
        "points": int(observation.motion.shape[1]),
        "fits": {},
    }
    for name, prior in priors.items():
        fit = fit_material(scene, observation, wind, prior, log=log)
        m = fit.material
        errors = {
            "stiffness": m.stiffness / truth.stiffness - 1,
            "damping": m.damping / truth.damping - 1,
            "drag": m.drag / truth.drag - 1,
        }
        entry = fit.to_json() | {"relativeError": {k: round(v, 4) for k, v in errors.items()}}
        report["fits"][name] = entry  # type: ignore[index]
        if log:
            print(f"  {name}: {json.dumps(entry)}", flush=True)
        if out is not None:
            try:
                plot_fit(
                    out / f"spectrum_{instance}_{name}.png",
                    fit,
                    f"instance {instance}: synthetic clip, {len(clip.frames)} frames at {fps:g} fps"
                    f" (prior: {name})",
                    clip.truth,
                )
            except ImportError:
                pass
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)

    synth = sub.add_parser("synth", help="render a clip with known materials")
    _common(synth)
    synth.add_argument("--stiffness", type=float, required=True)
    synth.add_argument("--damping", type=float, required=True)
    synth.add_argument("--drag", type=float, required=True)
    synth.add_argument("--seconds", type=float, default=120)
    synth.add_argument("--fps", type=float, default=15)
    synth.add_argument("--seed", type=int, default=7)
    synth.add_argument(
        "--field-modes", type=int, default=ENSEMBLE_FIELD_MODES, help="64: the browser's field"
    )
    synth.add_argument("--width", type=int, default=320)
    synth.add_argument("--height", type=int, default=240)
    synth.add_argument("--out", type=Path, required=True)

    fit = sub.add_parser("fit", help="fit materials to a clip and write materials.json")
    _common(fit)
    fit.add_argument("--clip", type=Path, required=True, help="a video, or a directory of PNGs")
    fit.add_argument("--camera", type=Path, required=True, help="camera.json (scan frame)")
    fit.add_argument("--fps", type=float, help="the clip's frame rate (over the file's)")
    fit.add_argument(
        "--source",
        choices=sorted(EVIDENCE),
        required=True,
        help="real footage (fitted-real) or a generated clip (fitted-generated)",
    )
    fit.add_argument("--materials", type=Path, required=True, help="materials.json to update")
    fit.add_argument("--report", type=Path)
    fit.add_argument("--plot", type=Path, help="spectrum plot (PNG; needs matplotlib)")

    val = sub.add_parser("validate", help="synthetic clip with known materials, recovered")
    _common(val)
    val.add_argument("--seconds", type=float, default=180)
    val.add_argument("--fps", type=float, default=15)
    val.add_argument("--out", type=Path, required=True)
    val.add_argument(
        "--field-modes", type=int, default=ENSEMBLE_FIELD_MODES, help="64: the browser's field"
    )

    world = sub.add_parser("world", help="a world-model clip of the scan (Modal GPU), fitted")
    _common(world)
    world.add_argument("--model", choices=("Wan", "Cosmos"), default="Wan")
    world.add_argument("--seeds", type=int, default=2)
    world.add_argument("--width", type=int, default=1280)
    world.add_argument("--height", type=int, default=704)
    world.add_argument("--materials", type=Path, required=True)
    world.add_argument("--report", type=Path)
    world.add_argument("--prompt", help="what the clip shows (default: a gentle breeze)")
    world.add_argument(
        "--chain", type=int, default=1, help="model calls per clip, each from the last frame"
    )
    world.add_argument("--steps", type=int, help="the model's denoising steps")
    world.add_argument(
        "--renderer", choices=("cpu", "gsplat"), default="cpu", help="how the still is drawn"
    )
    world.add_argument(
        "--auto-bearing",
        action="store_true",
        help="per instance, the bearing it is seen best from (over --bearing)",
    )
    world.add_argument("--save", type=Path, help="stills, clips, spectra and tracks here")

    args = parser.parse_args(argv)
    scene = load_scene(args.tiles, args.skin)
    inst_doc, mat_doc = _priors(args)
    wind = _wind(args)

    if args.command == "synth":
        (instance,) = args.instance
        material = sw.SkinMaterial(args.stiffness, args.damping, args.drag)
        camera = camera_for(scene, instance, wind.bearing_deg, width=args.width, height=args.height)
        clip = render_clip(
            scene,
            instance,
            material,
            wind,
            seed=args.seed,
            field_modes=args.field_modes,
            seconds=args.seconds,
            fps=args.fps,
            camera=camera,
            log=True,
        )
        args.out.mkdir(parents=True, exist_ok=True)
        write_clip(args.out / "clip.avi", clip.frames, args.fps)
        (args.out / "camera.json").write_text(json.dumps(camera.to_json(), indent=1))
        (args.out / "truth.json").write_text(json.dumps(clip.truth, indent=1))
        print(json.dumps(clip.truth))
        return 0

    if args.command == "fit":
        frames, fps = read_clip(args.clip)
        fps = args.fps or fps
        if not fps > 0:
            raise SystemExit("the clip's frame rate is unknown: pass --fps")
        camera = Camera.from_json(json.loads(args.camera.read_text()))
        fits = []
        for instance in args.instance:
            observation = observe(scene, instance, frames, fps, camera)
            prior = prior_of(inst_doc, mat_doc, instance)
            result = fit_material(scene, observation, wind, prior, log=True)
            print(json.dumps(result.to_json()), flush=True)
            fits.append(result)
            if args.plot:
                plot_fit(
                    args.plot.with_name(f"{args.plot.stem}_{instance}{args.plot.suffix}"),
                    result,
                    f"instance {instance}: {args.clip.name}",
                )
        write_materials(args.materials, fits, EVIDENCE[args.source], str(args.clip.name))
        if args.report:
            args.report.write_text(json.dumps([f.to_json() for f in fits], indent=1))
        return 0

    if args.command == "validate":
        args.out.mkdir(parents=True, exist_ok=True)
        reports = []
        for instance in args.instance:
            prior = prior_of(inst_doc, mat_doc, instance)
            truth = sw.SkinMaterial(prior.stiffness * 0.8, 0.07, prior.drag * 1.3)
            wrong = sw.SkinMaterial(prior.stiffness * 4, 0.3, prior.drag / 4)
            print(f"instance {instance}: truth {truth}", flush=True)
            reports.append(
                validate(
                    scene,
                    instance,
                    truth,
                    {"prior": prior, "wrong": wrong},
                    wind,
                    seconds=args.seconds,
                    fps=args.fps,
                    field_modes=args.field_modes,
                    out=args.out,
                    log=True,
                )
            )
        (args.out / "validation.json").write_text(json.dumps(reports, indent=1))
        return 0

    if args.command == "world":
        from world_model_client import VideoClips

        options: dict = {"model": args.model, "chain": args.chain, "steps": args.steps}
        if args.prompt:
            options["prompt"] = args.prompt
        source = VideoClips(**options)
        renderer = None
        if args.renderer == "gsplat":
            from splat_render import GsplatRenderer

            renderer = GsplatRenderer()
        fits, reports = [], []
        for instance in args.instance:
            if instance not in scene.skins:
                print(f"instance {instance}: no skin with dynamics, skipped", flush=True)
                continue
            report: dict = {"instance": instance}
            bearing = wind.bearing_deg
            if args.auto_bearing:
                bearing, report["bearings"] = best_bearing(scene, instance, renderer)
            here = sw.SkinWind(wind.speed_mps, bearing)
            camera = camera_for(scene, instance, bearing, width=args.width, height=args.height)
            report["bearing"] = bearing
            report["visible"], report["covered"] = visible_pixels(scene, instance, camera, renderer)
            still = draw(scene, camera, renderer, background=SKY).rgb
            first = len(source.received)
            started = time.perf_counter()
            clips = source.clips([still], [camera], tuple(range(1, args.seeds + 1)))
            report["clipSeconds"] = round(time.perf_counter() - started, 1)
            report["calls"] = source.received[first:]
            frames = [f for clip in clips for f in clip]  # seeds back to back
            observation = observe(scene, instance, frames, source.fps, camera)
            prior = prior_of(inst_doc, mat_doc, instance)
            fit = fit_material(scene, observation, here, prior, log=True)
            fits.append(fit)
            report |= fit.to_json() | {"backgroundPoints": observation.background_points}
            report["clipFrames"] = len(frames)
            report["clipSecondsOfVideo"] = round(len(frames) / source.fps, 2)
            print(json.dumps({k: v for k, v in report.items() if k != "calls"}), flush=True)
            reports.append(report)
            if args.save:
                from world_model_client import encode_png

                folder = args.save / f"instance-{instance}"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "still.png").write_bytes(encode_png(to_u8(still)))
                (folder / "camera.json").write_text(json.dumps(camera.to_json(), indent=1))
                for k, clip in enumerate(clips):
                    write_clip(folder / f"clip-{k}.avi", clip, source.fps)
                observation.save(folder / "observation.npz")
                try:
                    plot_fit(
                        folder / "spectrum.png",
                        fit,
                        f"instance {instance}: {source.name}, {len(frames)} frames "
                        f"at {source.fps:g} fps",
                    )
                except ImportError:
                    pass
        write_materials(args.materials, fits, EVIDENCE["generated"], source.name)
        if args.report:
            args.report.write_text(json.dumps(reports, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
