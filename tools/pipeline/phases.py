"""Where a call's seconds went: named phases, timed, for records and flat metrics.

A GPU call is billed from the moment its container starts to the moment it returns, and
the one number the provider gives back (`Poll.billed_s`) says nothing about what filled
it. A block's part of a fan-out measured 4,191 s billed against 2,967 s of training, and
nothing recorded where the other twenty minutes went. So every call that does more than
run one tool -- a block's part, the head, the join -- times its phases with this, keeps
them in its JSON record (`to_dict`), and reports them flattened into one metric
(`flat`), because step metrics are scalars.

Wall time, `time.monotonic`, rounded to a tenth of a second: these are for finding
minutes, not milliseconds.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager

__all__ = ["ORDER", "Phases", "flat", "ordered", "parse_flat"]

#: The order the phases of a call happen in, for putting back a record whose keys JSON
#: sorted (`blocks.py` writes records with `sort_keys`): a remote call's start and fetch,
#: the stage's preamble and plan, a block's own, the join's, the uploads.
ORDER: tuple[str, ...] = (
    "start",
    "import",
    "fetch",
    "stageDataset",
    "budget",
    "prepare",
    "loadPrior",
    "dataset",
    "seed",
    "train",
    "post",
    "blocks",
    "merge",
    "eval",
    "holdout",
    "stage",
    "stageOther",
    "finalSync",
    "output",
    "upload",
    "rest",
)


class Phases:
    """Seconds per named phase, in the order the phases first ran."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._seconds: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time the body under `name`; a phase run twice adds up. Recorded however the
        body ends, so a phase that raised still shows how long it ran before it did."""
        started = self._clock()
        try:
            yield
        finally:
            self.add(name, self._clock() - started)

    def add(self, name: str, seconds: float) -> None:
        self._seconds[name] = self._seconds.get(name, 0.0) + max(0.0, float(seconds))

    def update(self, other: Mapping[str, float]) -> None:
        for name, seconds in other.items():
            self.add(name, seconds)

    def to_dict(self) -> dict[str, float]:
        return {name: round(seconds, 1) for name, seconds in self._seconds.items()}

    def total(self) -> float:
        return sum(self._seconds.values())

    def flat(self) -> str:
        return flat(self.to_dict())


def flat(phases: Mapping[str, float]) -> str:
    """`name:seconds,...`, the metric form: `dataset:3.2,train:2966.9,post:11.0`."""
    return ",".join(f"{name}:{float(seconds):.1f}" for name, seconds in phases.items())


def ordered(phases: Mapping[str, float]) -> dict[str, float]:
    """`phases` in `ORDER`, anything it does not name after, in its own order."""
    rank = {name: index for index, name in enumerate(ORDER)}
    names = sorted(phases, key=lambda name: rank.get(name, len(ORDER)))
    return {name: phases[name] for name in names}


def parse_flat(text: object) -> dict[str, float]:
    """`flat`'s inverse; anything malformed is skipped rather than raised on, because a
    metric that cannot be read must not fail the call that carried it."""
    out: dict[str, float] = {}
    if not isinstance(text, str):
        return out
    for item in text.split(","):
        name, _, value = item.partition(":")
        try:
            out[name.strip()] = float(value)
        except ValueError:
            continue
    return {name: seconds for name, seconds in out.items() if name}
