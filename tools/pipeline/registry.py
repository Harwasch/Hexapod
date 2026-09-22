"""The stage registry: one decorator per implementation, and no executor change ever.

Adding `pose: glomap`, `train: opensplat` or `mask: sls` is two edits and neither of them
is here:

    @stage_impl(
        "glomap",
        consumes=("frames",),
        produces=(ArtifactDecl("poses", kind="dir", required_members=("cameras.bin",)),),
    )
    def glomap(ctx: StageContext) -> StageOutcome:
        ...

...and then `impl: glomap` in the recipe. The executor looks implementations up by name and
reads their declared `consumes`/`produces` to validate the chain, so it never learns what
any particular implementation is.

The declaration lives with the implementation rather than in the recipe on purpose: the
implementation is the thing that knows what it reads and writes, and a recipe that could
restate it is a recipe that can lie about it.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from artifacts import ArtifactDecl
from contracts import StageContext, StageOutcome
from errors import DuplicateImplError, RecipeError

__all__ = ["StageFn", "StageImpl", "known_impls", "lookup", "stage_impl"]

StageFn = Callable[[StageContext], StageOutcome]

# Modules that register implementations. Imported on first lookup so that importing the
# registry does not drag in every stage's dependencies.
_BUILTIN_MODULES = ("stages",)
_loaded = False


@dataclass(frozen=True)
class StageImpl:
    """A registered implementation and the artifact contract it declares."""

    name: str
    fn: StageFn
    consumes: tuple[str, ...] = ()
    optional_consumes: tuple[str, ...] = ()
    produces: tuple[ArtifactDecl, ...] = ()
    summary: str = ""

    @property
    def produced_names(self) -> tuple[str, ...]:
        return tuple(decl.name for decl in self.produces)

    def decl(self, name: str) -> ArtifactDecl:
        for decl in self.produces:
            if decl.name == name:
                return decl
        raise KeyError(name)


_REGISTRY: dict[str, StageImpl] = {}


def stage_impl(
    name: str,
    *,
    consumes: Iterable[str] = (),
    optional_consumes: Iterable[str] = (),
    produces: Iterable[ArtifactDecl] = (),
    summary: str = "",
) -> Callable[[StageFn], StageFn]:
    """Register a stage implementation under `name`, returning the function unchanged."""

    def register(fn: StageFn) -> StageFn:
        if name in _REGISTRY:
            existing = _REGISTRY[name].fn
            raise DuplicateImplError(
                f"impl {name!r} is already registered by "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        impl = StageImpl(
            name=name,
            fn=fn,
            consumes=tuple(consumes),
            optional_consumes=tuple(optional_consumes),
            produces=tuple(produces),
            summary=summary or _first_line(fn.__doc__),
        )
        overlap = set(impl.consumes) & set(impl.optional_consumes)
        if overlap:
            raise RecipeError(
                f"impl {name!r} declares {sorted(overlap)} as both required and optional"
            )
        _REGISTRY[name] = impl
        return fn

    return register


def _first_line(doc: str | None) -> str:
    lines = (doc or "").strip().splitlines()
    return lines[0].strip() if lines else ""


def load_builtin_impls() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    for module in _BUILTIN_MODULES:
        importlib.import_module(module)


def lookup(name: str) -> StageImpl | None:
    load_builtin_impls()
    return _REGISTRY.get(name)


def known_impls() -> tuple[str, ...]:
    load_builtin_impls()
    return tuple(sorted(_REGISTRY))
