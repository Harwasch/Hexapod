"""Train the Minnetonka tree: its published photos to a splat, on a Modal GPU. Opt-in.

Living Survey M0 (docs/LIVING_WORLD.md section 9, docs/CAPTURES.md "The real tree"). The
pure parts -- the pinned dataset, metadata, the COLMAP conversion, the pose gate, the
stage params -- are `tools/pipeline/experiments/minnetonka.py`; this is the I/O around
them, run from `.github/workflows/minnetonka-tree.yml` one step per job so each step can
be checked (and re-run) on its own:

    check    ~130 MB. Every photo's first 192 KiB (its DJI metadata) and the published
             COLMAP model; the pose gate on that model. No GPU, no bucket. Minutes.
    prepare  ~14 GB streamed, nothing kept but the frames: each photo is fetched, its
             sha256 checked against the pinned revision, its metadata read, and it is
             shrunk straight from memory, so the runner never holds an original on disk.
             The size is photo-reconstruct's own `max_side: auto` rule (resolution.py),
             measured on the sample of this set it takes of any photo set; m0 was shrunk
             to a fixed 1600 px instead. Frames and meta.json go to the bucket.
    pose     the repository's own `pose` stage (COLMAP 4.2, sequential matching with
             loop closure) on Modal's cpu4, then the pose gate on what it solved. With
             `--poses published` the dataset's model is converted instead, and gated.
             With `--from-tag TAG` nothing is solved: TAG's gated poses are checked
             against these frames (`sfm.poses_serve_frames`) and kept here -- frames
             re-sized from the same photos train on the same poses, since the stage
             extracts its features at `max_image_size` whatever size the frames are and
             gsplat rescales the posed intrinsics to the frames it is given.
    train    the repository's own `train` stage with photo-reconstruct's Standard params
             unchanged (gsplat MCMC, `cap_max: auto`, `converge`, `blocks: auto`, SH 3)
             on a Modal GPU with those poses -- the benchmark's path for a scene with
             known poses (`benchmark.cloud`, `benchmark.execute_remotely`), on the
             recipe's train tier unless `--tier` says otherwise. Refuses poses the gate
             failed. No gaussian count is fixed: the budget is the surface in training
             pixels, up to what the placed GPU holds, and blocks past that. `--sessions`
             trains on some capture sessions only (a day, `2020-07-20`, or one flight,
             `2020-07-20/2`); the poses stay the joint solve, so frame.json and the rig
             are unchanged.
    fetch    what a later job needs back out of the bucket.
    describe the capture's descriptor for `real_tree.py` (site, place, attribution).

Everything a step keeps is under `experiments/minnetonka-tree/<tag>/` in the private
bucket (`OBJECT_STORAGE_*`); the pipeline's own `runs/<id>/` scratch is deleted after each
stage, as the benchmark does.

    uv run --project tools/pipeline --with modal==1.5.5 --with boto3 \\
        python infra/modal/minnetonka.py train --tag m1

`--rehearse` runs everything but Modal and R2: a local directory for the bucket, the test
suite's stand-in trainer (it trains nothing), and COLMAP from this machine. `--mirror DIR`
reads a local copy laid out like the Hub (the tests build one) instead of the network.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PIPELINE = REPO / "tools" / "pipeline"
sys.path.insert(0, str(PIPELINE))

import sfm  # noqa: E402
import stages  # noqa: E402 - also registers every shipped implementation
from cloud import AttemptLedger  # noqa: E402
from experiments import minnetonka as tree  # noqa: E402
from recipe import Recipe  # noqa: E402
from workdir import Workdir  # noqa: E402

PREFIX = "experiments/minnetonka-tree"


def _benchmark() -> Any:
    """`infra/modal/benchmark.py`, whose cloud path this reuses rather than restates."""
    spec = importlib.util.spec_from_file_location(
        "modal_benchmark", Path(__file__).with_name("benchmark.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["modal_benchmark"] = module
    spec.loader.exec_module(module)
    return module


def say(text: str) -> None:
    sys.stdout.write(text.rstrip("\n") + "\n")
    sys.stdout.flush()


def summary(markdown: str) -> None:
    """Into the job summary when there is one, and the log either way."""
    say(markdown)
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(markdown.rstrip("\n") + "\n\n")


# --- where the dataset comes from ---------------------------------------------------------


class HubSource:
    """The Hugging Face Hub at the pinned revision, over plain HTTPS."""

    def __init__(self, attempts: int = 4) -> None:
        self.attempts = attempts

    def _open(self, url: str, headers: dict[str, str] | None = None) -> tuple[bytes, str | None]:
        last: Exception | None = None
        for attempt in range(self.attempts):
            request = urllib.request.Request(url, headers=headers or {})  # noqa: S310
            try:
                with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
                    return response.read(), response.headers.get("Link")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                last = error
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"{url}: {last!r} after {self.attempts} attempts")

    def listing(self, path: str) -> list[dict[str, Any]]:
        url: str | None = tree.tree_url(path)
        entries: list[dict[str, Any]] = []
        while url:
            body, link = self._open(url)
            entries.extend(json.loads(body))
            url = tree.next_page(link)
        return entries

    def read(self, path: str, limit: int | None = None) -> bytes:
        headers = {"Range": f"bytes=0-{limit - 1}"} if limit else None
        return self._open(tree.resolve_url(path), headers)[0]


class MirrorSource:
    """A local copy laid out like the Hub: `<root>/<path>` for bytes and
    `<root>/_listing/<path>.json` for the tree listing. For tests and offline rehearsals."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def listing(self, path: str) -> list[dict[str, Any]]:
        return json.loads((self.root / "_listing" / f"{path}.json").read_text(encoding="utf-8"))

    def read(self, path: str, limit: int | None = None) -> bytes:
        data = (self.root / path).read_bytes()
        return data[:limit] if limit else data


def verified(source: Any, file: tree.RemoteFile) -> bytes:
    data = source.read(file.path)
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != file.size or digest != file.sha256:
        raise ValueError(
            f"{file.path}: {len(data)} bytes, sha256 {digest}; the pinned revision has "
            f"{file.size} bytes, {file.sha256}"
        )
    return data


def colmap_text(source: Any, into: Path) -> Path:
    """The dataset's three COLMAP text files, each checked against its pinned sha256."""
    into.mkdir(parents=True, exist_ok=True)
    for name, (size, sha) in tree.COLMAP_FILES.items():
        target = into / name
        if not (target.is_file() and _sha256(target) == sha):
            file = tree.RemoteFile(path=f"colmap/{name}", size=size, sha256=sha)
            target.write_bytes(verified(source, file))
    return into


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plan(source: Any, groups: Sequence[str]) -> list[tree.RemoteFile]:
    return tree.plan_download(tree.parse_listing(source.listing("images")), groups)


def parallel(
    work: Callable[[tree.RemoteFile], Any], files: Sequence[tree.RemoteFile], workers: int
) -> list[Any]:
    done = [0]
    started = time.monotonic()

    def one(file: tree.RemoteFile) -> Any:
        result = work(file)
        done[0] += 1
        if done[0] % 50 == 0 or done[0] == len(files):
            say(f"  {done[0]}/{len(files)} in {time.monotonic() - started:.0f} s")
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, files))


# --- the bucket ------------------------------------------------------------------------------


def storage(args: argparse.Namespace) -> Any:
    if args.rehearse:
        from adapters import LocalTransfer

        return LocalTransfer(args.work.resolve() / "bucket")
    return _benchmark()._s3_transfer()


def key(tag: str, *parts: str) -> str:
    return "/".join([PREFIX, tag, *parts])


def keep(store: Any, tag: str, path: Path, *parts: str) -> None:
    sent = store.put(key(tag, *parts), path)
    say(f"kept {key(tag, *parts)} ({sent:,} bytes)")


def fetch_kept(store: Any, tag: str, target: Path, *parts: str) -> Path:
    if not store.exists(key(tag, *parts)):
        raise SystemExit(
            f"{key(tag, *parts)} is not in the bucket: run the step that makes it first"
        )
    store.get(key(tag, *parts), target)
    return target


# --- steps -----------------------------------------------------------------------------------


def published_poses(
    text_dir: Path, names: Sequence[str], into: Path, metas: dict[str, tree.DroneMeta]
) -> tuple[tree.FrameFit, dict[str, Any]]:
    """The dataset's own model, for the photos in `names` (original spellings), the four
    suspects dropped and every image renamed to its frame name; then the gate."""
    model = tree.read_text_model(text_dir)
    wanted = set(names) - set(tree.SUSPECT_CAMERAS)
    rename = {
        image.name: tree.frame_name(image.name) for image in model.images if image.name in wanted
    }
    chosen = tree.select_images(model, rename)
    tree.write_binary_model(chosen, into)
    tree.write_json(
        into / "poses.json",
        {
            "source": f"published: {tree.HF_REPO}@{tree.HF_REVISION} colmap/",
            "registered": len(chosen.images),
            "droppedSuspects": sorted(set(names) & set(tree.SUSPECT_CAMERAS)),
            "scale": {"metric": False, "source": "none"},
        },
    )
    model = sfm.read_model(into)
    fit = tree.fit_frame(tree.posed_views(model), metas)
    return fit, tree.frame_document(fit, model, metas)


def gate_markdown(title: str, fit: tree.FrameFit) -> str:
    r = fit.residuals
    g = tree.GATE
    rows = [
        (
            "photos with DJI metadata",
            f"{fit.with_meta}/{fit.posed}",
            f">= {g['metaCoverageMin']:.0%}",
            fit.checks["metaCoverage"],
        ),
        (
            "solved pitch - gimbal pitch, median / p90",
            f"{r['pitchMedianAbsDeg']:.2f} / {r['pitchP90AbsDeg']:.2f} deg",
            f"<= {g['pitchMedianAbsMaxDeg']} / {g['pitchP90AbsMaxDeg']}",
            fit.checks["pitch"],
        ),
        (
            "up from pitch vs up from barometer",
            f"{fit.up_agreement_deg:.2f} deg",
            f"<= {g['upAgreementMaxDeg']}",
            fit.checks["upAgreement"],
        ),
        (
            "barometric heights span",
            f"{r['heightSpanM']:.1f} m",
            f">= {g['heightSpanMinM']}",
            fit.checks["heightSpan"],
        ),
        (
            "barometric height residual, RMS / inliers",
            f"{r['heightRmsM']:.3f} m / {r['heightInlierFraction']:.0%}",
            f"<= {g['heightRmsMaxM']} m / >= {g['heightInlierMin']:.0%}",
            fit.checks["height"],
        ),
        (
            "compass heading residual, median / p90",
            f"{r['headingMedianAbsDeg']:.1f} / {r['headingP90AbsDeg']:.1f} deg",
            f"<= {g['headingMedianAbsMaxDeg']} / {g['headingP90AbsMaxDeg']}",
            fit.checks["heading"],
        ),
    ]
    lines = [
        f"## {title} -- pose gate **{fit.verdict}**",
        "",
        "| check | measured | bar | |",
        "|---|---|---|---|",
        *[f"| {a} | {b} | {c} | {'ok' if d else '**FAIL**'} |" for a, b, c, d in rows],
        "",
        f"Scale {fit.metres_per_unit:.4f} m per model unit (barometric height vs height along "
        f"up; {fit.flights} flight(s), correlation {r['heightCorrelation']:.3f}); heading "
        f"offset {fit.heading_offset_deg:.1f} deg.",
    ]
    return "\n".join(lines) + "\n"


def cmd_check(args: argparse.Namespace) -> int:
    source = MirrorSource(args.mirror) if args.mirror else HubSource()
    out = args.work.resolve() / "check"
    files = plan(source, args.groups)
    say(f"check: {len(files)} photos, reading the first {tree.META_BYTES // 1024} KiB of each")
    heads = parallel(lambda f: source.read(f.path, tree.META_BYTES), files, args.workers)
    metas = {
        tree.frame_name(f.name): tree.read_drone_meta(h) for f, h in zip(files, heads, strict=True)
    }
    tree.write_json(out / "meta.json", {k: v.to_dict() for k, v in sorted(metas.items())})
    fit, document = published_poses(
        colmap_text(source, out / "colmap"), [f.name for f in files], out / "poses", metas
    )
    tree.write_json(out / "gate.json", document)
    summary(gate_markdown(f"Published poses, {', '.join(args.groups)}", fit))
    say(f"check: wrote {out / 'gate.json'}")
    return 0


def frame_size(source: Any, files: Sequence[tree.RemoteFile], asked: str) -> dict[str, Any]:
    """The long side the frames are kept at: photo-reconstruct's own `max_side` rule.

    `auto` is the recipe's normalize rule (`resolution.py`: its base, unless the sharpest
    of a sample spread over the set measurably carry detail above it, then up to its
    ceiling), measured on the same sample of these photos it would take of any photo
    set -- fetched, checked and held in memory, not written to disk. A number is used as
    given, as `max_side: N` is by the stage."""
    if asked != "auto":
        return {"rule": "fixed", "maxSide": int(asked)}
    base, ceiling = tree.frame_size_params()
    picks = [files[i] for i in stages.photo_sample(len(files))]
    samples = [io.BytesIO(verified(source, file)) for file in picks]
    decision = stages.photo_frame_size(samples, base=base, ceiling=ceiling)
    return decision.to_dict()


def cmd_prepare(args: argparse.Namespace) -> int:
    source = MirrorSource(args.mirror) if args.mirror else HubSource()
    work = args.work.resolve() / "prepare"
    frames = work / "frames"
    if frames.exists():
        shutil.rmtree(frames)
    # The whole group is checked against the pins; the four suspects are then not fetched
    # at all -- the dataset's own advice, and their solved focal says those four photos
    # were hard to place, whoever places them.
    files = [f for f in plan(source, args.groups) if f.name not in tree.SUSPECT_CAMERAS]
    decision = frame_size(source, files, args.max_side)
    max_side = int(decision["maxSide"])
    say(
        f"prepare: {len(files)} photos, {tree.planned_bytes(files) / 1e9:.2f} GB, shrunk to "
        f"{max_side} px as they arrive ({decision.get('reason') or decision['rule']}); "
        "originals are never written to disk"
    )

    def one(file: tree.RemoteFile) -> tuple[str, tree.DroneMeta, tuple[int, int]]:
        data = verified(source, file)
        name = tree.frame_name(file.name)
        size = tree.shrink_jpeg(data, frames / name, max_side)
        return name, tree.read_drone_meta(data[: tree.META_BYTES]), size

    results = parallel(one, files, args.workers)
    metas = {name: meta for name, meta, _ in results}
    sizes = sorted({size for _, _, size in results})
    tree.write_json(work / "meta.json", {k: v.to_dict() for k, v in sorted(metas.items())})
    report: dict[str, Any] = {
        "dataset": {
            "hub": tree.HF_REPO,
            "revision": tree.HF_REVISION,
            "github": tree.GITHUB_REPO,
            "commit": tree.GITHUB_COMMIT,
        },
        "groups": list(args.groups),
        "photos": len(files),
        "suspectsNotFetched": list(tree.SUSPECT_CAMERAS),
        "bytesDownloaded": tree.planned_bytes(files),
        "frameSizes": [list(s) for s in sizes],
        # How the size was chosen: photo-reconstruct's `max_side: auto` rule on the same
        # sample of this set it takes of any photo set (resolution.py), or a number given.
        "frameSize": decision,
        "sessions": tree.session_counts(metas),
        "framesBytes": sum(p.stat().st_size for p in frames.iterdir()),
        "gpsMedian": tree.gps_centre(metas.values()),
        "takeoffMslM": tree.takeoff_msl(metas.values()),
    }
    status = 0
    if args.poses == "published":
        fit, document = published_poses(
            colmap_text(source, work / "colmap"), [f.name for f in files], work / "poses", metas
        )
        tree.write_json(work / "frame.json", document)
        report["gate"] = fit.verdict
        summary(gate_markdown("Published poses", fit))
        if fit.verdict != "pass" and not args.force:
            say(
                "prepare: the published poses failed the gate; not keeping them "
                "(--force keeps them)"
            )
            status = 1
    tree.write_json(work / "prepare.json", report)
    summary(
        f"## Prepared {len(files)} photos\n\n{report['bytesDownloaded'] / 1e9:.2f} GB checked "
        f"against the pinned sha256s; frames {sizes}, {report['framesBytes'] / 1e6:.0f} MB."
    )
    if not args.no_keep:
        store = storage(args)
        keep(store, args.tag, frames, "frames")
        keep(store, args.tag, work / "meta.json", "meta.json")
        keep(store, args.tag, work / "prepare.json", "prepare.json")
        if args.poses == "published" and status == 0:
            keep(store, args.tag, work / "poses", "poses")
            keep(store, args.tag, work / "frame.json", "frame.json")
    return status


def _inputs(
    args: argparse.Namespace, store: Any, workdir: Workdir, *, poses: bool
) -> dict[str, tree.DroneMeta]:
    """Frames (and poses) into the run's inputs; meta.json back as `DroneMeta`s."""
    local = args.work.resolve() / "prepare"
    frames = workdir.input_path("frames")
    if (local / "frames").is_dir():
        shutil.copytree(local / "frames", frames, copy_function=os.link, dirs_exist_ok=True)
        meta_path = local / "meta.json"
    else:
        fetch_kept(store, args.tag, frames, "frames")
        meta_path = fetch_kept(store, args.tag, workdir.root / "meta.json", "meta.json")
    if poses:
        fetch_kept(store, args.tag, workdir.input_path("poses"), "poses")
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    return {name: tree.DroneMeta.from_dict(value) for name, value in raw.items()}


def _image_size(path: Path) -> list[int]:
    from PIL import Image

    with Image.open(path) as image:
        return [int(image.size[0]), int(image.size[1])]


def _run_workdir(parent: Path, step: str) -> Workdir:
    """A new run's workdir, named for the second it started; a second run in the same
    second gets `-2`, `-3`, ... rather than the first one's inputs (linking a frame onto
    the one already there failed "File exists")."""
    base = f"tree-{step}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    name, n = base, 1
    while (parent / name).exists():
        n += 1
        name = f"{base}-{n}"
    return Workdir.create(parent / name)


def _prepared(args: argparse.Namespace, store: Any, into: Path, name: str) -> Path:
    """This tag's `prepare.json` or `meta.json`: the prepare job's, or the bucket's."""
    local = args.work.resolve() / "prepare" / name
    return local if local.is_file() else fetch_kept(store, args.tag, into / name, name)


def adopt_poses(args: argparse.Namespace, store: Any) -> int:
    """`pose --from-tag`: another tag's gated poses, checked against these frames."""
    source = args.from_tag
    if source == args.tag:
        raise SystemExit(f"--from-tag {source} is this tag; solve its poses instead")
    work = args.work.resolve() / "adopt"
    if work.exists():
        shutil.rmtree(work)
    poses = fetch_kept(store, source, work / "poses", "poses")
    frame_path = fetch_kept(store, source, work / "frame.json", "frame.json")
    frame = json.loads(frame_path.read_text(encoding="utf-8"))
    prepare = json.loads(_prepared(args, store, work, "prepare.json").read_text(encoding="utf-8"))
    names = sorted(
        json.loads(_prepared(args, store, work, "meta.json").read_text(encoding="utf-8"))
    )
    model = sfm.read_model(poses)
    fit = sfm.poses_serve_frames(model, prepare["frameSizes"], names)
    report: dict[str, Any] = {
        "adoptedFrom": source,
        **fit,
        "points3D": model.points3d,
        "meanReprojectionErrorPx": model.mean_reprojection_error,
        "gate": frame.get("verdict"),
    }
    if store.exists(key(source, "pose.json")):
        solved = json.loads(
            fetch_kept(store, source, work / "source-pose.json", "pose.json").read_text(
                encoding="utf-8"
            )
        )
        report["solved"] = {k: solved.get(k) for k in ("registered", "frames", "cost")}
    tree.write_json(work / "pose.json", report)
    summary(
        f"## Poses from `{source}`, not solved again\n\n"
        f"{fit['registered']}/{fit['frames']} frames posed; the posed cameras are "
        f"{fit['posedSize'][0]}x{fit['posedSize'][1]} and these frames "
        f"{fit['frameSize'][0]}x{fit['frameSize'][1]} (x{fit['pixelScale']:g}): gsplat "
        f"rescales the intrinsics by that, and the extrinsics are the same photos'. Gate "
        f"**{frame.get('verdict')}**, as `{source}` measured it."
    )
    keep(store, args.tag, work / "pose.json", "pose.json")
    if frame.get("verdict") != "pass" and not args.force:
        say(f"pose: {source}'s poses failed the gate; not adopting them (--force does)")
        return 1
    keep(store, args.tag, poses, "poses")
    keep(store, args.tag, frame_path, "frame.json")
    return 0


def cmd_pose(args: argparse.Namespace) -> int:
    store = storage(args)
    if args.from_tag:
        return adopt_poses(args, store)
    driver = _benchmark()
    workdir = _run_workdir(args.work.resolve(), "pose")
    metas = _inputs(args, store, workdir, poses=False)
    overrides: dict[str, Any] = {}
    if args.colmap:
        overrides["colmap"] = args.colmap
    recipe = Recipe.from_dict(
        {
            "name": "minnetonka-pose",
            "version": 1,
            "inputs": ["frames"],
            "stages": [
                {
                    "id": "pose",
                    "impl": "colmap",
                    "params": tree.pose_params(overrides),
                    "gpu": {"tier": args.tier, "preemptible": False},
                }
            ],
        }
    )
    transfer, runner = driver.cloud(args.work.resolve(), app=args.app, rehearse=args.rehearse)
    driver.execute_remotely(recipe, workdir, transfer, runner)
    poses = workdir.out_dir("pose") / "poses"
    model = sfm.read_model(poses)
    fit = tree.fit_frame(tree.posed_views(model), metas)
    frames = sum(1 for p in workdir.input_path("frames").iterdir() if p.is_file())
    ledger = AttemptLedger.read(workdir.attempts_path("pose"))
    report = {
        "registered": model.registered,
        "frames": frames,
        "points3D": model.points3d,
        "meanReprojectionErrorPx": model.mean_reprojection_error,
        "gate": fit.verdict,
        "cost": {"tier": args.tier, "billedSeconds": round(ledger.billed_s, 1), "usd": ledger.usd},
    }
    tree.write_json(workdir.root / "frame.json", tree.frame_document(fit, model, metas))
    tree.write_json(workdir.root / "pose.json", report)
    summary(
        f"## Posed {model.registered}/{frames} frames (COLMAP, `{args.tier}`)\n\n"
        f"{model.points3d:,} points; billed {ledger.billed_s:.0f} s."
    )
    summary(gate_markdown("Solved poses", fit))
    keep(store, args.tag, workdir.root / "pose.json", "pose.json")
    if fit.verdict != "pass" and not args.force:
        say("pose: the solved poses failed the gate; not keeping them (--force keeps them)")
        keep(store, args.tag, workdir.root / "frame.json", "frame-failed.json")
        return 1
    keep(store, args.tag, poses, "poses")
    keep(store, args.tag, workdir.root / "frame.json", "frame.json")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    driver = _benchmark()
    store = storage(args)
    workdir = _run_workdir(args.work.resolve(), "train")
    metas = _inputs(args, store, workdir, poses=True)
    frame_path = fetch_kept(store, args.tag, workdir.root / "frame.json", "frame.json")
    frame = json.loads(frame_path.read_text(encoding="utf-8"))
    if frame.get("verdict") != "pass" and not args.force:
        raise SystemExit(
            f"the poses in {key(args.tag, 'poses')} failed the gate: {frame.get('failed')}"
        )
    frames_dir = workdir.input_path("frames")
    selection: dict[str, Any] | None = None
    if args.sessions:
        # Some capture sessions only: the frames and the posed images both cut to them.
        # The poses stay the joint solve, so frame.json -- and the rig -- still hold.
        chosen = tree.names_in(metas, args.sessions)
        for path in frames_dir.iterdir():
            if path.is_file() and path.name not in chosen:
                path.unlink()
        selection = {
            "sessions": list(args.sessions),
            **sfm.keep_images(workdir.input_path("poses"), chosen),
        }
        say(f"train: {selection}")
    trained_on = {p.name for p in frames_dir.iterdir() if p.is_file()}
    registered = sfm.read_model(workdir.input_path("poses")).registered
    frame_size = _image_size(frames_dir / min(trained_on))
    tier = args.tier or tree.train_tier()
    overrides: dict[str, Any] = {}
    if args.budget_max:
        overrides["budget_max"] = args.budget_max
    if args.roi_m > 0:
        radius = args.roi_m / float(frame["metresPerUnit"])
        overrides["roi"] = {"center": frame["lookAt"], "radius": radius}
    if args.cap_max:
        overrides["cap_max"] = args.cap_max
    params = tree.train_params(overrides)
    if args.rehearse:
        params.update(driver.stand_in_params([str(v) for v in params.get("extra_args") or []]))
        params.pop("converge", None)
    recipe = Recipe.from_dict(
        {
            "name": "minnetonka-train",
            "version": 1,
            "inputs": ["frames", "poses"],
            "stages": [
                {
                    "id": "train",
                    "impl": "gsplat",
                    "params": params,
                    "gpu": {"tier": tier, "preemptible": False},
                }
            ],
        }
    )
    transfer, runner = driver.cloud(args.work.resolve(), app=args.app, rehearse=args.rehearse)
    started = time.monotonic()
    driver.execute_remotely(recipe, workdir, transfer, runner)
    out = workdir.out_dir("train")
    metrics = json.loads((out / "train_metrics.json").read_text(encoding="utf-8"))
    ledger = AttemptLedger.read(workdir.attempts_path("train"))
    report = {
        "params": {k: v for k, v in params.items() if k not in ("trainer", "python")},
        "gaussians": metrics.get("gaussians"),
        "budget": metrics.get("budget"),
        "iterations": metrics.get("iterations"),
        # What real_tree.py reads for the site: how many photos, of which days, and the
        # size the splat was trained at (frame.json's focal is at the posed size).
        "registered": registered,
        "frameSize": frame_size,
        "sessions": (
            counts := tree.session_counts({n: m for n, m in metas.items() if n in trained_on})
        ),
        "days": tree.days(counts),
        "sessionSelection": selection,
        "psnr": metrics.get("psnr"),
        "ssim": metrics.get("ssim"),
        "lpips": metrics.get("lpips"),
        "trainSeconds": metrics.get("trainSeconds"),
        "wallSeconds": round(time.monotonic() - started, 1),
        "cost": {"tier": tier, "billedSeconds": round(ledger.billed_s, 1), "usd": ledger.usd},
        "rehearsal": bool(args.rehearse),
    }
    tree.write_json(out.parent / "train.json", report)
    shutil.copy2(frame_path, out.parent / "frame.json")
    for name in ("trained.ply", "train_metrics.json"):
        keep(store, args.tag, out / name, "train", name)
    keep(store, args.tag, out.parent / "train.json", "train", "train.json")
    usd = report["cost"]["usd"]
    summary(
        (
            "> Rehearsal: the stand-in trained nothing; the numbers are made up.\n\n"
            if args.rehearse
            else ""
        )
        + f"## Trained the tree on `{tier}`\n\n"
        "| gaussians | steps | PSNR | SSIM | LPIPS | train s | billed |\n"
        "|---|---|---|---|---|---|---|\n"
        f"| {report['gaussians']} | {report['iterations']} | {report['psnr']} | {report['ssim']} | "
        f"{report['lpips']} | {report['trainSeconds']} | {ledger.billed_s:.0f} s"
        + (f", ${usd:.2f}" if isinstance(usd, int | float) else "")
        + " |\n\nHeld out: every 8th frame by name, never trained on. "
        + f"Frames {frame_size[0]}x{frame_size[1]}, {registered} posed"
        + (f" (sessions {', '.join(args.sessions)} only)" if args.sessions else "")
        + ". "
        + _budget_line(report["budget"])
    )
    return 0


def _budget_line(budget: Any) -> str:
    """What decided the gaussian count, from train_metrics.json's `budget`."""
    if not isinstance(budget, dict):
        return "No budget recorded."
    raw, cap, clamp = budget.get("raw"), budget.get("capMax"), budget.get("clamp")
    line = f"Budget: {budget.get('mode')} -> {cap}"
    if raw is not None:
        line += f" (density budget {raw}"
        line += f", clamped to the {clamp})" if clamp else ")"
    if clamp == "gpu-memory":
        line += " -- the GPU's memory bound it; a bigger tier would train more"
    return line + "."


def cmd_fetch(args: argparse.Namespace) -> int:
    store = storage(args)
    for item in args.items:
        target = args.into / item
        if args.missing_ok and not store.exists(key(args.tag, *item.split("/"))):
            say(f"not in the bucket, skipped: {key(args.tag, item)}")
            continue
        fetch_kept(store, args.tag, target, *item.split("/"))
        say(f"fetched {key(args.tag, item)} -> {target}")
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    tree.write_json(args.out, tree.capture_descriptor())
    say(f"describe: wrote {args.out}")
    return 0


def cmd_keep(args: argparse.Namespace) -> int:
    keep(storage(args), args.tag, args.path, *args.item.split("/"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", type=Path, default=Path("tree-work"))
    parser.add_argument("--tag", default="m0", help="the bucket prefix a run's steps share")
    parser.add_argument("--rehearse", action="store_true")
    parser.add_argument("--app", default="twin-pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    def dataset(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--groups", type=lambda s: [g for g in s.split(",") if g], default=["The_Tree"]
        )
        sub.add_argument(
            "--mirror", type=Path, help="a local Hub-shaped copy, instead of the network"
        )
        sub.add_argument("--workers", type=int, default=8)

    check = commands.add_parser("check", help="metadata + published poses -> the pose gate")
    dataset(check)
    prepare = commands.add_parser("prepare", help="download, verify, shrink; keep the frames")
    dataset(prepare)
    prepare.add_argument(
        "--max-side",
        default="auto",
        type=lambda s: s if s == "auto" else str(int(s)),
        help="auto (photo-reconstruct's rule, the default) or a long side in pixels",
    )
    prepare.add_argument("--poses", choices=["colmap", "published"], default="colmap")
    prepare.add_argument(
        "--force", action="store_true", help="keep published poses that fail the gate"
    )
    prepare.add_argument("--no-keep", action="store_true", help="do not upload anything")
    pose = commands.add_parser("pose", help="COLMAP on the frames (Modal cpu4), then the gate")
    pose.add_argument("--tier", default="cpu4")
    pose.add_argument("--colmap", choices=["3.9", "4.2"], help="override the recipe's COLMAP")
    pose.add_argument("--force", action="store_true", help="keep solved poses that fail the gate")
    pose.add_argument(
        "--from-tag", default="", help="adopt this tag's gated poses instead of solving"
    )
    train = commands.add_parser("train", help="the train stage on a Modal GPU")
    train.add_argument("--tier", default="", help="the GPU (default: the recipe's train tier)")
    train.add_argument(
        "--budget-max", type=int, default=0, help="an optional ceiling on cap_max auto (0: none)"
    )
    train.add_argument("--cap-max", type=int, default=0, help="a fixed cap instead of auto")
    train.add_argument(
        "--roi-m",
        type=float,
        default=0.0,
        help="a sphere round the orbit's look-at point to train at full density; 0 (the "
        "default) trains the whole scene as the recipe does",
    )
    train.add_argument(
        "--sessions",
        type=lambda s: [d for d in s.split(",") if d],
        default=[],
        help="train on these capture sessions only: a day (2020-07-20) or a flight "
        "(2020-07-20/2); default every photo",
    )
    train.add_argument("--force", action="store_true", help="train on poses the gate failed")
    fetch = commands.add_parser("fetch", help="copy kept items out of the bucket")
    fetch.add_argument("items", nargs="+", help="e.g. train/trained.ply frame.json meta.json")
    fetch.add_argument("--into", type=Path, required=True)
    fetch.add_argument("--missing-ok", action="store_true")
    keeper = commands.add_parser("keep", help="copy a local file or folder into the bucket")
    keeper.add_argument("path", type=Path)
    keeper.add_argument("item", help="where under the tag, e.g. site")
    describe = commands.add_parser("describe", help="the capture descriptor real_tree.py reads")
    describe.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return {
        "check": cmd_check,
        "prepare": cmd_prepare,
        "pose": cmd_pose,
        "train": cmd_train,
        "fetch": cmd_fetch,
        "keep": cmd_keep,
        "describe": cmd_describe,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
