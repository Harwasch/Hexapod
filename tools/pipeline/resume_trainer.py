"""Full-state resume for gsplat 512d366b; imported on Python 3.10.

Native ckpts are evaluation weights saved *before* optimizer.step. These files are
separate, versioned, and saved at the end of the step (including evaluation RNG).
The guarded patch leaves the upstream loop in place. Off means the original loader
and no writes. On keeps four data workers, with sample-keyed crop randomness and a
consumed-batch cursor independent of worker prefetch. Object-storage sync runs outside
the loop; local saves are synchronous and non-fatal.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import random
import re
import shutil
import sys
from collections import OrderedDict
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

VERSION = 4
DEFAULT_EVERY = 5000
MODULES = ("pose_adjust", "pose_perturb", "app_module", "post_processing_module")
OPTIMIZERS = ("optimizers", "pose_optimizers", "app_optimizers", "post_processing_optimizers")
LOADER = """        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )"""


def patch_source(source: str) -> str:
    """Fail closed when the pinned loop moves, also after the quality/detail patches."""
    anchors = (
        "    ckpt: Optional[List[str]] = None",
        LOADER,
        "\n        trainloader_iter = iter(trainloader)\n",
        "        init_step = 0",
        "            # Run post-backward steps after backward and optimizer.",
        '    @torch.no_grad()\n    def eval(self, step: int, stage: str = "val"):',
        "        pbar = tqdm.tqdm(range(init_step, max_steps))",
        "    if cfg.ckpt is not None:",
    )
    for anchor in anchors:
        if source.count(anchor) != 1:
            raise ValueError(f"resume trainer seam moved: expected one {anchor!r}")
    if "_hex_resume" in source:
        raise ValueError("resume trainer already patched")
    source = source.replace(
        anchors[0],
        anchors[0]
        + "\n    resume: Optional[str] = None"
        + "\n    checkpoint_every: int = 0"
        + "\n    checkpoint_dir: Optional[str] = None"
        + "\n    checkpoint_token: str = ''"
        + "\n    checkpoint_final: bool = True"
        + "\n    streaming_export: bool = True",
    )
    # Keep datasets resolving from the trainer's examples directory.
    bootstrap = (
        "import sys as _hex_resume_sys\n"
        f"_hex_resume_sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "import resume_trainer as _hex_resume\n"
        "_hex_resume_sys.path.pop(0)\n"
    )
    futures = list(re.finditer(r"(?m)^from __future__ import [^\n]+\n", source))
    insertion = futures[-1].end() if futures else 0
    source = source[:insertion] + bootstrap + source[insertion:]
    source = source.replace(
        LOADER,
        "        if cfg.checkpoint_every or cfg.resume:\n"
        "            trainloader = _hex_resume.Loader(self.trainset, cfg.batch_size)\n"
        "        else:\n" + "\n".join("    " + line for line in LOADER.splitlines()),
    )
    source = source.replace(
        anchors[2],
        "\n        init_step = _hex_resume.restore(self, schedulers, trainloader)" + anchors[2],
    )
    source = source.replace(anchors[3], "        _hex_resume.configure(self)\n" + anchors[3])
    source = source.replace(
        anchors[6],
        "        pbar = tqdm.tqdm(range(init_step, max_steps), total=max_steps, initial=init_step)",
    )
    source = source.replace(
        anchors[7],
        "    if cfg.resume is not None and cfg.ckpt is not None:\n"
        "        raise ValueError('--resume is training; --ckpt is evaluation only')\n"
        + anchors[7],
    )
    # The last statement in the loop is after eval, compression and viewer updates.
    return source.replace(
        anchors[5],
        "            _hex_resume.save(self, schedulers, trainloader, step + 1)\n\n" + anchors[5],
    )


def checkpoints(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.glob("resume_*.pt") if re.fullmatch(r"resume_\d{9}.pt", p.name)
    )


def configure(runner: Any) -> None:
    """Namespace coarse/block trainers under the same durable run checkpoint root."""
    cfg = runner.cfg
    if cfg.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    if cfg.resume and not cfg.checkpoint_every:
        cfg.checkpoint_every = DEFAULT_EVERY
    if cfg.resume == "latest":
        cfg.checkpoint_dir = str(checkpoint_lane(Path(cfg.checkpoint_dir), Path(cfg.result_dir)))
        latest = newest(Path(cfg.checkpoint_dir))
        cfg.resume = None if latest is None else str(latest)


def checkpoint_lane(directory: Path, result: Path) -> Path:
    suffix = str(result).rsplit("/work/", 1)[-1]
    return (
        directory
        if suffix == "gsplat"
        else directory / hashlib.sha256(suffix.encode()).hexdigest()[:16]
    )


def finished_lane(extra: list[Any], result: Path) -> None:
    """Release only a completed coarse/block trainer's optimizer state."""
    if "--checkpoint-dir" in extra:
        directory = Path(extra[extra.index("--checkpoint-dir") + 1])
        lane = checkpoint_lane(directory, result)
        if lane.is_dir():
            shutil.rmtree(lane)


def resume_step(extra: list[Any], result: Path) -> int | None:
    if "--resume" not in extra:
        return None
    value = extra[extra.index("--resume") + 1]
    if value == "latest":
        directory = Path(extra[extra.index("--checkpoint-dir") + 1])
        path = newest(checkpoint_lane(directory, result))
    else:
        path = Path(value)
    if path is None:
        return None
    if "--checkpoint-token" in extra:
        token = extra[extra.index("--checkpoint-token") + 1]
        try:
            receipt = json.loads(path.with_suffix(".committed").read_text())
        except (OSError, ValueError):
            return None
        if receipt.get("token") != token:
            return None  # The trainer will discard stale state; do not report a resume.
    return int(path.stem.split("_")[-1])


def newest(directory: Path) -> Path | None:
    # Object uploads may complete in any order. Require both the atomically published
    # data object and its size receipt; .part files and orphan receipts are never read.
    paths = [p for p in checkpoints(directory) if committed(p)]
    return paths[-1] if paths else None


def committed(path: Path) -> bool:
    try:
        receipt = json.loads(path.with_suffix(".committed").read_text())
        return bool(path.stat().st_size == receipt["bytes"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def rotate(directory: Path) -> None:
    for path in [p for p in checkpoints(directory) if committed(p)][:-2]:
        path.unlink(missing_ok=True)
        path.with_suffix(".committed").unlink(missing_ok=True)


class SeededDataset:
    """Dataset randomness belongs to a sample key, not an unsaved worker queue."""

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __getitem__(self, key: tuple[int, int]) -> Any:
        index, seed = key
        torch = importlib.import_module("torch")
        np = importlib.import_module("numpy")
        python_rng, numpy_rng, torch_rng = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
        try:
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            return self.dataset[index]
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)


class Batches:
    def __init__(self, loader: Any) -> None:
        self.loader = loader

    def __iter__(self) -> Any:
        loader = self.loader
        # Snapshot only the consumed cursor. DataLoader may prefetch the whole epoch
        # without advancing the checkpoint's position.
        for start in range(loader.cursor, len(loader.order), loader.batch_size):
            yield [
                (loader.order[i], loader.sample_seed(i))
                for i in range(start, min(start + loader.batch_size, len(loader.order)))
            ]


class Loader:
    """Four persistent workers, deterministic samples and a consumed-batch cursor."""

    def __init__(self, dataset: Any, batch_size: int) -> None:
        torch = importlib.import_module("torch")
        self.dataset, self.batch_size = dataset, batch_size
        self.seed = torch.initial_seed()
        self.generator = torch.Generator().manual_seed(self.seed)
        self.order: list[int] = []
        self.cursor = 0
        self.epoch = -1
        self.iterator: Any = None
        self.loader = torch.utils.data.DataLoader(
            SeededDataset(dataset),
            batch_sampler=Batches(self),
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            # Iterator construction must never consume the training RNG on resume.
            generator=torch.Generator().manual_seed(self.seed),
        )

    def sample_seed(self, position: int) -> int:
        key = f"{self.seed}/{self.epoch}/{position}".encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)

    def __iter__(self) -> Loader:
        if self.cursor >= len(self.order):
            torch = importlib.import_module("torch")
            self.order = torch.randperm(len(self.dataset), generator=self.generator).tolist()
            self.cursor = 0
            self.epoch += 1
        self.iterator = iter(self.loader)
        return self

    def __next__(self) -> Any:
        if self.cursor >= len(self.order):
            raise StopIteration
        batch = next(self.iterator)
        self.cursor = min(len(self.order), self.cursor + self.batch_size)
        return batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "cursor": self.cursor,
            "epoch": self.epoch,
            "seed": self.seed,
            "generator": self.generator.get_state(),
            "size": len(self.dataset),
            "batchSize": self.batch_size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["size"] != len(self.dataset) or state["batchSize"] != self.batch_size:
            raise ValueError("resume dataset or batch size changed")
        self.order, self.cursor = state["order"], state["cursor"]
        self.epoch, self.seed = state["epoch"], state["seed"]
        self.generator.set_state(state["generator"].cpu())


def _optimizer_items(runner: Any) -> dict[str, Any]:
    result = {}
    for name in OPTIMIZERS:
        group = getattr(runner, name, [])
        pairs = group.items() if isinstance(group, dict) else enumerate(group)
        result.update({f"{name}/{key}": opt for key, opt in pairs})
    return result


def _identity(runner: Any) -> dict[str, Any]:
    cfg = runner.cfg
    if getattr(runner, "world_size", 1) != 1 or getattr(cfg, "ckpt", None):
        raise ValueError("resume requires one GPU and training mode (never --ckpt)")
    # Dataclass fields are the plan; instance attributes may also hold probe/quality
    # method wrappers. Never serialize those wrappers (or the strategy/cfg itself).
    strategy = cfg.strategy
    settings = (
        {field.name: getattr(strategy, field.name) for field in fields(strategy)}
        if is_dataclass(strategy)
        else {key: value for key, value in vars(strategy).items() if not callable(value)}
    )
    return {
        "maxSteps": cfg.max_steps,
        "strategy": type(cfg.strategy).__name__,
        "strategySettings": plain_data(settings, "strategySettings"),
        "token": cfg.checkpoint_token,
    }


CONVERGE_FIELDS = ("history", "decided_at", "stopped_at", "gain_db", "last", "lpips_skipped")


def plain_data(value: Any, path: str = "checkpoint") -> Any:
    """Normalize state dictionaries; reject objects before they reach pickle.

    No allowlists or custom pickle globals are needed by the restricted loader.
    NumPy appears in scene state; its RNG is encoded separately below.
    """
    torch = importlib.import_module("torch")
    np = importlib.import_module("numpy")
    if torch.is_tensor(value):
        return value.detach()
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return plain_data(value.item(), path)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if type(key) not in (str, bool, int, float):
                raise TypeError(f"{path}: unsupported dictionary key {type(key).__name__}")
            result[key] = plain_data(item, f"{path}/{key}")
        return result
    if isinstance(value, (list, tuple)):
        items = [plain_data(item, f"{path}/{index}") for index, item in enumerate(value)]
        return tuple(items) if isinstance(value, tuple) else items
    raise TypeError(f"{path}: unsupported checkpoint value {type(value).__name__}")


def converge_state(runner: Any) -> dict[str, Any] | None:
    state = getattr(runner, "_hex_converge", None)
    if state is None:
        return None
    return {"rule": state.rule.to_dict(), **{k: getattr(state, k) for k in CONVERGE_FIELDS}}


def capture(runner: Any, schedulers: list[Any], loader: Any, step: int) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    np = importlib.import_module("numpy")
    numpy_rng = np.random.get_state()
    modules = {
        name: getattr(runner, name).state_dict()
        for name in MODULES
        if getattr(runner, name, None) is not None
    }
    state = {
        "version": VERSION,
        "identity": _identity(runner),
        "step": step,
        "splats": runner.splats.state_dict(),
        "requiresGrad": {k: v.requires_grad for k, v in runner.splats.items()},
        "optimizers": {k: opt.state_dict() for k, opt in _optimizer_items(runner).items()},
        "modules": modules,
        "moduleMetadata": {
            name: getattr(weights, "_metadata", {}) for name, weights in modules.items()
        },
        "moduleRequiresGrad": {
            name: {key: p.requires_grad for key, p in module.named_parameters()}
            for name in MODULES
            if (module := getattr(runner, name, None)) is not None
            and hasattr(module, "named_parameters")
        },
        "strategy": runner.strategy_state,
        "scene": {
            name: getattr(runner.scene, name)
            for name in ("id", "signal", "component_names", "component_index")
        }
        if getattr(runner, "scene", None) is not None
        else None,
        "schedulers": [s.state_dict() for s in schedulers],
        "frozen": getattr(runner, "_gaussians_frozen", False),
        "loader": loader.state_dict(),
        "converge": converge_state(runner),
        "rng": {
            "python": random.getstate(),
            "numpy": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    return dict(plain_data(state))


def restore(runner: Any, schedulers: list[Any], loader: Any) -> int:
    cfg = runner.cfg
    if not cfg.resume:
        if cfg.checkpoint_every:
            _identity(runner)
            runner._hex_resume_step = 0
            with contextlib.suppress(OSError):
                report_progress(runner, 0)
        return 0
    torch = importlib.import_module("torch")
    np = importlib.import_module("numpy")
    # No strategy, cfg, module or callable is deserialized. All runtime objects and
    # method patches come from this trainer; only their data state is restored.
    state = torch.load(cfg.resume, map_location="cpu", weights_only=True, mmap=True)
    identity = _identity(runner)
    if state.get("version") != VERSION or state.get("identity") != identity:
        sys.stderr.write(
            f"WARNING: incompatible resume checkpoint or changed plan: {cfg.resume}; "
            f"saved identity={state.get('identity')!r}, current identity={identity!r}. "
            "Discarding this lane's checkpoints and starting at step 0.\n"
        )
        # Do not delete sibling block lanes, including when this is the root lane.
        directory = Path(cfg.resume).parent
        for path in directory.iterdir():
            if path.is_file() and re.fullmatch(
                r"\.?resume_\d{9}\.(pt|committed|part)(\.part)?", path.name
            ):
                path.unlink(missing_ok=True)
        cfg.resume = None
        return restore(runner, schedulers, loader)
    step = int(state["step"])
    if not 0 <= step <= cfg.max_steps:
        raise ValueError("resume step outside schedule")
    if set(state["splats"]) != set(runner.splats):
        raise ValueError("resume splat parameters changed")
    for key, tensor in state["splats"].items():
        # Keep parameter identity: the scene, stage and optimizers all reference it.
        runner.splats[key].data = tensor.to(runner.device)
        runner.splats[key].requires_grad_(state["requiresGrad"][key])
    for name, weights in state["modules"].items():
        # PyTorch stores version metadata as an attribute on its OrderedDict. Save
        # that metadata as plain data too, then reconstruct the runtime container.
        module_state: Any = OrderedDict(weights)
        module_state._metadata = state["moduleMetadata"][name]
        getattr(runner, name).load_state_dict(module_state)
    for name, flags in state.get("moduleRequiresGrad", {}).items():
        params = dict(getattr(runner, name).named_parameters())
        for key, requires_grad in flags.items():
            params[key].requires_grad_(requires_grad)
    optimizers = _optimizer_items(runner)
    if optimizers.keys() != state["optimizers"].keys() or len(schedulers) != len(
        state["schedulers"]
    ):
        raise ValueError("resume optimizer/scheduler groups changed")
    for key, opt in optimizers.items():
        opt.load_state_dict(state["optimizers"][key])
    for scheduler, saved in zip(schedulers, state["schedulers"], strict=True):
        scheduler.load_state_dict(saved)

    # Strategy dictionaries include ADC gradient/radius accumulators and MCMC bins.
    def device(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.to(runner.device)
        if isinstance(value, dict):
            return {k: device(v) for k, v in value.items()}
        if isinstance(value, list):
            return [device(v) for v in value]
        if isinstance(value, tuple):
            return tuple(device(v) for v in value)
        return value

    runner.strategy_state = device(state["strategy"])
    if state.get("scene") is not None:
        for name, value in state["scene"].items():
            setattr(runner.scene, name, device(value))
        runner.scene.validate()
    saved_converge = state.get("converge")
    current_converge = getattr(runner, "_hex_converge", None)
    if (saved_converge is None) != (current_converge is None):
        raise ValueError("resume convergence mode changed")
    if saved_converge is not None and current_converge is not None:
        if saved_converge["rule"] != current_converge.rule.to_dict():
            raise ValueError("resume convergence rule changed")
        for key in CONVERGE_FIELDS:
            setattr(current_converge, key, saved_converge[key])
        if current_converge.decided_at is not None:
            for name in ("ply_steps", "save_steps", "eval_steps"):
                getattr(cfg, name).append(current_converge.decided_at + 2)
    runner._gaussians_frozen = state["frozen"]
    loader.load_state_dict(state["loader"])
    random.setstate(state["rng"]["python"])
    numpy_rng = state["rng"]["numpy"]
    np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
    torch.set_rng_state(state["rng"]["torch"].cpu())
    if state["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["rng"]["cuda"]])
    runner._hex_resume_step = step
    with contextlib.suppress(OSError):
        report_progress(runner, step)
    if step == cfg.max_steps:
        final = Path(cfg.resume).parent / "final-output"
        if not final.is_dir():
            raise ValueError("completed resume checkpoint has no final outputs")
        shutil.copytree(final, cfg.result_dir, dirs_exist_ok=True)
    sys.stdout.write(
        f"resume: restored {cfg.resume}; continuing at step {step} of {cfg.max_steps}\n"
    )
    sys.stdout.flush()
    return step


def report_progress(runner: Any, step: int) -> None:
    cfg = runner.cfg
    if not cfg.checkpoint_every:
        return
    directory = Path(cfg.checkpoint_dir or (Path(cfg.result_dir) / "resume"))
    directory.mkdir(parents=True, exist_ok=True)
    progress = directory / "trainer_progress.json"
    temporary = progress.with_suffix(".part")
    temporary.write_text(
        json.dumps(
            {
                "step": step,
                "maxSteps": cfg.max_steps,
                "resumedFromStep": getattr(runner, "_hex_resume_step", 0),
                "checkpointStep": getattr(
                    runner, "_hex_checkpoint_step", getattr(runner, "_hex_resume_step", 0)
                ),
            }
        )
    )
    temporary.replace(progress)


def save(runner: Any, schedulers: list[Any], loader: Any, step: int) -> Path | None:
    """Checkpoint IO is optional: any failure is reported and training continues."""
    if not runner.cfg.checkpoint_every:
        return None
    try:
        return _save(runner, schedulers, loader, step)
    except Exception as error:
        checkpoint_error(runner, step, error)
        return None


def checkpoint_error(runner: Any, step: int, error: Exception) -> None:
    record = {"step": step, "error": f"{type(error).__name__}: {error}"}
    errors = getattr(runner, "_hex_checkpoint_errors", [])
    errors.append(record)
    runner._hex_checkpoint_errors = errors
    # Keep independent sinks: a full disk/broken stdout must not hide stderr or
    # raise a second exception into the loop. The stage log also retains this diagnostic.
    with contextlib.suppress(Exception):
        sys.stderr.write(
            f"CHECKPOINT ERROR at step {step}: {record['error']}; continuing training; "
            "previous complete checkpoint remains a resume point\n"
        )
        sys.stderr.flush()
    with contextlib.suppress(Exception):
        sys.stdout.write("live-metrics: " + json.dumps({"checkpointErrors": [record]}) + "\n")
        sys.stdout.flush()
    with contextlib.suppress(Exception):
        directory = Path(runner.cfg.result_dir)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "checkpoint_errors.json"
        previous = json.loads(target.read_text())["checkpointErrors"] if target.exists() else []
        partial = target.with_suffix(".part")
        partial.write_text(json.dumps({"checkpointErrors": [*previous, record]}) + "\n")
        partial.replace(target)


def _save(runner: Any, schedulers: list[Any], loader: Any, step: int) -> Path | None:
    cfg = runner.cfg
    every = cfg.checkpoint_every
    if not every:
        return None
    directory = Path(cfg.checkpoint_dir or (Path(cfg.result_dir) / "resume"))
    directory.mkdir(parents=True, exist_ok=True)
    if step == 1 or step % 200 == 0 or step == cfg.max_steps:
        report_progress(runner, step)
    if step == cfg.max_steps and not getattr(cfg, "checkpoint_final", True):
        return None
    if step % every and step != cfg.max_steps:
        return None
    target = directory / f"resume_{step:09d}.pt"
    temporary = directory / ("." + target.name + ".part")
    torch = importlib.import_module("torch")
    if step == cfg.max_steps:
        final = directory / "final-output"
        # The upstream final PLY is exported before the last update; retain those
        # exact bytes if a container dies during the bench's subsequent evaluation.
        for name in ("stats",):
            source = Path(cfg.result_dir) / name
            if source.is_dir():
                shutil.copytree(source, final / name, dirs_exist_ok=True)
        source = Path(cfg.result_dir) / "ply" / f"point_cloud_{step - 1}.ply"
        if source.is_file():
            (final / "ply").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, final / "ply" / source.name)
        for name in ("cfg.yml", "post_processing.json"):
            source = Path(cfg.result_dir) / name
            if source.is_file():
                final.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, final / source.name)
    try:
        with temporary.open("wb") as handle:
            torch.save(capture(runner, schedulers, loader, step), handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
    receipt = target.with_suffix(".committed")
    partial_receipt = receipt.with_suffix(".part")
    partial_receipt.write_text(
        json.dumps({"bytes": target.stat().st_size, "token": cfg.checkpoint_token})
    )
    partial_receipt.replace(receipt)
    rotate(directory)
    runner._hex_checkpoint_step = step

    return target


def main() -> int:
    source = Path(sys.argv[1])
    patched = patch_source(source.read_text())
    compile(patched, str(source), "exec", dont_inherit=True)
    source.write_text(patched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
