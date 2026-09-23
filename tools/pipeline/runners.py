"""The runner seam: where a stage runs, chosen by one fact about the stage.

    Runner.run(stage, workdir) -> StepResult
    |- LocalRunner   calls the registered implementation in this process; stages that need
    |                an external tool shell out through StageContext.run()
    |- StubRunner    fabricates each declared artifact deterministically, so a whole recipe
    |                -- Lane 2 included -- is green on a machine with no GPU and no network
    +- CloudRunner   B1b, in cloud.py: submits the stage to a provider, tails it, and
                     brings its outputs -- and its checkpoint -- back. It lives beside
                     this module rather than in it only because the two protocols it
                     needs (ProviderAdapter, Transfer) are a file's worth of contract

Everything that is not "run the implementation" -- clearing the previous attempt's outputs,
keeping the checkpoint, checking the declared `produces` were actually written, hashing
them, writing the StepResult -- happens once, in BaseRunner, for every runner. A runner
overrides `_invoke` and nothing else, which is what "there is no second code path" means in
practice: a stub cannot drift from the real thing by forgetting a rule.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from artifacts import ArtifactDecl, ArtifactRef, checksum_of, size_of
from contracts import MetricValue, StageContext, StageOutcome, StepResult
from errors import (
    MissingArtifactError,
    NoRunnerError,
    PipelineError,
    StageFailedError,
    UndeclaredArtifactError,
)
from plan import PlannedStage
from workdir import Workdir

__all__ = ["BaseRunner", "LocalRunner", "Runner", "RunnerSet", "StubRunner"]


class Runner(ABC):
    """Runs one stage and returns its StepResult."""

    name: str

    @abstractmethod
    def run(self, stage: PlannedStage, workdir: Workdir, attempt: int = 1) -> StepResult: ...


class BaseRunner(Runner, ABC):
    def run(self, stage: PlannedStage, workdir: Workdir, attempt: int = 1) -> StepResult:
        workdir.prepare_stage(stage.id)
        context = self._context(stage, workdir, attempt)
        started = time.perf_counter()
        try:
            outcome = self._invoke(stage, context)
        except PipelineError:
            raise
        except Exception as exc:
            raise StageFailedError(stage.recipe, stage.id, stage.impl.name, exc) from exc
        duration = time.perf_counter() - started
        self._verify(stage, workdir)
        artifacts = self._collect(stage, workdir)
        checkpointed = any(workdir.checkpoint_dir(stage.id).iterdir())
        result = StepResult(
            stage_id=stage.id,
            impl=stage.impl.name,
            runner=self.name,
            attempt=attempt,
            duration_s=duration,
            artifacts=artifacts,
            metrics=dict(outcome.metrics),
            log_path=workdir.relative(workdir.log_path(stage.id)),
            checkpoint_key=context.checkpoint_key if checkpointed else None,
            gpu_tier=stage.gpu.tier if stage.gpu else None,
            summary=outcome.summary or stage.impl.summary,
        )
        result.write(workdir.step_path(stage.id))
        return result

    @abstractmethod
    def _invoke(self, stage: PlannedStage, context: StageContext) -> StageOutcome: ...

    def _context(self, stage: PlannedStage, workdir: Workdir, attempt: int) -> StageContext:
        log_path = workdir.log_path(stage.id)
        log_path.touch()
        return StageContext(
            recipe=stage.recipe,
            run_id=workdir.root.name,
            stage_id=stage.id,
            impl=stage.impl.name,
            params=dict(stage.params),
            attempt=attempt,
            gpu_tier=stage.gpu.tier if stage.gpu else None,
            out_dir=workdir.out_dir(stage.id),
            work_dir=workdir.work_dir(stage.id),
            checkpoint_dir=workdir.checkpoint_dir(stage.id),
            log_path=log_path,
            checkpoint_key=f"runs/{workdir.root.name}/{stage.id}/checkpoint",
            attempts_path=workdir.attempts_path(stage.id),
            _inputs={name: workdir.root / path for name, path in stage.inputs.items()},
            _produces={decl.name: decl for decl in stage.impl.produces},
        )

    def _verify(self, stage: PlannedStage, workdir: Workdir) -> None:
        """A declared `produces` that was not written fails here, loudly, every time."""
        out = workdir.out_dir(stage.id)
        for decl in stage.impl.produces:
            path = out / decl.name
            missing = not path.exists() or (
                decl.kind == "dir" and (not path.is_dir() or not any(path.rglob("*")))
            )
            if missing:
                raise MissingArtifactError(
                    stage.recipe, stage.id, stage.impl.name, decl.name, workdir.relative(path)
                )
            for member in decl.required_members:
                if not (path / member).exists():
                    raise MissingArtifactError(
                        stage.recipe,
                        stage.id,
                        stage.impl.name,
                        f"{decl.name}/{member}",
                        workdir.relative(path / member),
                    )
        declared = {decl.name.split("/", 1)[0] for decl in stage.impl.produces}
        extra = {entry.name for entry in out.iterdir()} - declared
        if extra:
            raise UndeclaredArtifactError(stage.recipe, stage.id, stage.impl.name, extra)

    def _collect(self, stage: PlannedStage, workdir: Workdir) -> tuple[ArtifactRef, ...]:
        refs = []
        for decl in stage.impl.produces:
            path = workdir.artifact_path(stage.id, decl.name)
            refs.append(
                ArtifactRef(
                    name=decl.name,
                    stage_id=stage.id,
                    path=workdir.relative(path),
                    kind=decl.kind,
                    content_type=decl.content_type,
                    bytes=size_of(path),
                    checksum=checksum_of(path),
                )
            )
        return tuple(refs)


class LocalRunner(BaseRunner):
    """Every CPU stage, wherever the worker happens to run."""

    name = "local"

    def _invoke(self, stage: PlannedStage, context: StageContext) -> StageOutcome:
        return stage.impl.fn(context)


class StubRunner(BaseRunner):
    """Deterministic fake outputs for any implementation, without running it.

    The stub does not know what a splat is. It writes exactly the artifacts the
    implementation declares, in the shape it declares them, with contents derived by hashing
    (recipe, stage, impl, params, artifact, and the checksums of the stage's inputs). Two
    runs over the same inputs are byte-identical; a changed parameter changes the bytes; and
    an implementation that gains an output gains a stub for it with no edit here.
    """

    name = "stub"

    def _invoke(self, stage: PlannedStage, context: StageContext) -> StageOutcome:
        inputs = {name: checksum_of(path) for name, path in sorted(context.inputs.items())}
        context.log(f"stub: {stage.impl.name} over {len(inputs)} input(s)")
        members = 0
        for decl in stage.impl.produces:
            members += self._write(context, stage, decl, inputs)
        metrics: dict[str, MetricValue] = {
            "stub": True,
            "inputs": len(inputs),
            "artifacts": len(stage.impl.produces),
            "files": members,
        }
        return StageOutcome(metrics=metrics, summary=f"stubbed {stage.impl.name}")

    def _write(
        self,
        context: StageContext,
        stage: PlannedStage,
        decl: ArtifactDecl,
        inputs: dict[str, str],
    ) -> int:
        path = context.output(decl.name)
        if decl.kind == "file":
            path.write_bytes(self._content(stage, decl, inputs, path.name))
            return 1
        members = decl.members_to_stub or ("stub.bin",)
        for member in members:
            target = path / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self._content(stage, decl, inputs, member))
        return len(members)

    @staticmethod
    def _content(
        stage: PlannedStage,
        decl: ArtifactDecl,
        inputs: dict[str, str],
        member: str,
    ) -> bytes:
        key = json.dumps(
            {
                "recipe": stage.recipe,
                "stage": stage.id,
                "impl": stage.impl.name,
                "params": _canonical(stage.params),
                "artifact": decl.name,
                "member": member,
                "inputs": inputs,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        if member.endswith(".json"):
            document = {
                "stub": True,
                "artifact": decl.name,
                "member": member,
                "stageId": stage.id,
                "impl": stage.impl.name,
                "digest": digest,
            }
            return (json.dumps(document, indent=1, sort_keys=True) + "\n").encode("utf-8")
        return _expand(digest, decl.stub_bytes)


def _canonical(params: object) -> object:
    """JSON-safe view of params, so an unserialisable value cannot break determinism."""
    try:
        json.dumps(params, sort_keys=True)
    except TypeError:
        return repr(params)
    return params


def _expand(digest: str, length: int) -> bytes:
    out = bytearray()
    block = digest.encode("ascii")
    while len(out) < length:
        block = hashlib.sha256(block).digest()
        out += block
    return bytes(out[:length])


@dataclass(frozen=True)
class RunnerSet:
    """Which runner gets a stage. The only input is whether the stage declares `gpu:`."""

    cpu: Runner
    gpu: Runner | None = None

    def for_stage(self, stage: PlannedStage) -> Runner:
        if stage.gpu is None:
            return self.cpu
        if self.gpu is None:
            raise NoRunnerError(
                f"recipe {stage.recipe!r}, stage {stage.id!r} (impl {stage.impl.name!r}) runs "
                f"remotely on tier {stage.gpu.tier!r} and no remote runner is configured. Use "
                f"StubRunner in CI, or a CloudRunner (cloud.py) with a ProviderAdapter and a "
                f"Transfer"
            )
        return self.gpu

    @staticmethod
    def stubbed() -> RunnerSet:
        runner = StubRunner()
        return RunnerSet(cpu=runner, gpu=runner)

    @staticmethod
    def local() -> RunnerSet:
        """CPU stages run for real; GPU stages have nowhere to go without a provider."""
        return RunnerSet(cpu=LocalRunner(), gpu=None)

    @staticmethod
    def cloud(gpu: Runner) -> RunnerSet:
        """CPU stages here, GPU stages on somebody else's machine. The one routing fact
        is still whether the stage declares `gpu:`."""
        return RunnerSet(cpu=LocalRunner(), gpu=gpu)
