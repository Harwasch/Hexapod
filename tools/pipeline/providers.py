"""Where a GPU stage can be sent, and what an hour there costs.

This table used to live in `apps/api/app/services/recipes.py`, where the console read it.
It moved here in B1b because two things now need it and only one of them may import the
other: the API renders it in the New-run form, and `CloudRunner` multiplies a rate by the
seconds a provider billed to fill in `jobs.cost_usd`. `tools/pipeline` must not import
`apps/api`, so the shared fact lives on the side that can be imported from both.

**Only surveyed numbers are in here.** A0 measured an A100 hour at each of the four
providers and nothing else, so that is what the table carries. A tier with no `Rate` is
*unpriced*: a run on it records the seconds it was billed and leaves its cost empty,
which is the honest answer. Inventing an L4 rate to make a column non-null would put a
number nobody has checked in front of somebody deciding how to spend money -- and a wrong
price is worse than a missing one, because it will be believed.

A deployment that knows its own rates -- a contract price, a reserved instance, a box it
already owns -- supplies them through `rates_from_env()` rather than by editing this file,
because that number is true for that deployment and for nobody else.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

__all__ = [
    "PROVIDERS",
    "RATES_ENV_VAR",
    "Provider",
    "Rate",
    "provider",
    "rates_from_env",
    "with_rates",
]

#: `provider:tier=usd_per_hour`, comma separated. See `rates_from_env`.
RATES_ENV_VAR = "PIPELINE_GPU_RATES"


@dataclass(frozen=True)
class Rate:
    """What an hour of one tier at one provider costs, and who says so.

    `source` is not decoration. It is the difference between a figure somebody measured
    and a figure somebody remembered, and the console and the report both say which.
    """

    usd_per_hour: float
    source: str

    def usd_for(self, seconds: float) -> float:
        return self.usd_per_hour * seconds / 3600.0


@dataclass(frozen=True)
class Provider:
    """One GPU host, the tiers it is offered with, and the rates that are known."""

    name: str
    label: str
    tiers: tuple[str, ...]
    #: True where being killed mid-stage is ordinary operation rather than a failure.
    interruptible: bool
    note: str
    #: Tier -> rate. A tier that is offered but not priced is deliberately absent.
    rates: Mapping[str, Rate] = field(default_factory=dict)

    def rate(self, tier: str) -> Rate | None:
        return self.rates.get(tier)

    @property
    def usd_per_hour_a100(self) -> float | None:
        surveyed = self.rate("a100")
        return None if surveyed is None else surveyed.usd_per_hour


#: What the A0 provider survey measured: one A100 hour, at each of four hosts.
A0 = "A0 provider survey"

PROVIDERS: tuple[Provider, ...] = (
    Provider(
        name="modal",
        label="Modal",
        tiers=("l4", "a10g", "a100"),
        interruptible=False,
        note="Per-second billing, scale to zero. The reliable default.",
        rates={"a100": Rate(2.50, A0)},
    ),
    Provider(
        name="runpod-secure",
        label="RunPod Secure",
        tiers=("l4", "a100"),
        interruptible=False,
        note="Ordinary rented pods.",
        rates={"a100": Rate(1.59, A0)},
    ),
    Provider(
        name="runpod-community",
        label="RunPod Community",
        tiers=("l4", "a100"),
        interruptible=True,
        note="Community hosts; cheaper, and killed without warning.",
        rates={"a100": Rate(1.19, A0)},
    ),
    Provider(
        name="vast",
        label="Vast.ai",
        tiers=("l4", "a100"),
        interruptible=True,
        note=(
            "Cheapest sticker, most variable. A0 measured the unverified tier running "
            "20-40% above its listed price once restarts and downtime are priced in."
        ),
        rates={"a100": Rate(0.52, A0)},
    ),
)


def provider(name: str) -> Provider | None:
    for entry in PROVIDERS:
        if entry.name == name:
            return entry
    return None


def rates_from_env(environ: Mapping[str, str] | None = None) -> dict[str, dict[str, Rate]]:
    """Operator-supplied rates: ``PIPELINE_GPU_RATES=modal:l4=0.80,vast:a100=0.44``.

    Anything unparseable is skipped rather than raised on: a malformed rate must not stop
    a run that would otherwise have happened, and the cost it would have priced simply
    stays empty, which is the same outcome as not setting the variable at all.
    """
    raw = (environ if environ is not None else os.environ).get(RATES_ENV_VAR, "")
    parsed: dict[str, dict[str, Rate]] = {}
    for clause in raw.split(","):
        entry = clause.strip()
        if not entry or ":" not in entry or "=" not in entry:
            continue
        key, _, amount = entry.partition("=")
        name, _, tier = key.partition(":")
        try:
            usd = float(amount)
        except ValueError:
            continue
        if usd < 0 or not name.strip() or not tier.strip():
            continue
        parsed.setdefault(name.strip(), {})[tier.strip()] = Rate(usd, RATES_ENV_VAR)
    return parsed


def with_rates(
    overrides: Mapping[str, Mapping[str, Rate]],
    catalogue: tuple[Provider, ...] = PROVIDERS,
) -> tuple[Provider, ...]:
    """The catalogue with an operator's own rates merged over the surveyed ones."""
    if not overrides:
        return catalogue
    merged = []
    for entry in catalogue:
        extra = overrides.get(entry.name)
        merged.append(entry if not extra else replace(entry, rates={**entry.rates, **extra}))
    return tuple(merged)
