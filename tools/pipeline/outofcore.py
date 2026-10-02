"""Exact statistics of a column that is never held whole: medians, percentiles, group counts.

The stages after training need a handful of whole-splat statistics -- the median GSD the
quality bar is relative to, the 2nd/98th percentile footprint `place` recentres on, the
voxels a support mask counts -- and the rule is that no stage holds the splat. So each
statistic is computed from a *stream*: a function that, called, yields the column's values
in chunks, and can be called again for another pass. Memory is a chunk plus a fixed-size
histogram, whatever the number of values.

**Order statistics by radix selection** (`order_statistics`). Each value is mapped to an
unsigned key that sorts as the value does (a float's bits with the sign flipped, or all
bits inverted when negative; an integer's with the sign bit flipped). One pass counts the
top 16 bits of every key into 65,536 bins, which says which bin the k-th value is in and
its rank inside it; a second pass either counts the next 16 bits of the keys in that bin
or, when the bin holds at most `COLLECT` values, collects them and sorts them. A float32
is exact in two passes, a float64 in at most four. Several ranks, and several groups
(`groups=`), share the same passes; with many of them the digit narrows so that a pass's
histograms stay within `HIST_BYTES`, and the keys collected in one pass never exceed
`COLLECT` together -- the memory is a constant, however many values or groups there are.

**numpy's own answer, not an approximation of it** (`median`, `percentile`). `np.median`
and `np.percentile` (method "linear") are functions of the count, the requested position
and at most two order statistics, and those are exact here -- so the result is computed
from them by the same arithmetic numpy 2 uses (`_function_base_impl._quantile` and
`_lerp`: the virtual index `(n - 1) q`, the interpolation weight and the `t >= 0.5`
branch; `_median`: the mean of the one or two middle values, in the array's dtype). The
tests hold both to numpy's own output, bit for bit, over random arrays of every dtype
used here. A NaN anywhere makes the answer NaN, as it does in numpy.

**Counts by group, in partitions** (`group_counts`). A group-by (occupied voxels, cells)
holds one entry per distinct key, which can be as many as there are values. So the keys
are split by a hash into as many partitions as it takes for each to hold at most `BUDGET`
rows, and each partition is one pass: its keys are `np.unique`d a chunk at a time and the
partial counts merged. The caller reduces each partition to what it needs (a share, the
seed voxels) before the next one is read.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "BUDGET",
    "COLLECT",
    "HIST_BYTES",
    "Stream",
    "group_counts",
    "grouped_median",
    "grouped_percentile",
    "median",
    "median_ranks",
    "order_statistics",
    "percentile",
]

#: A re-iterable column: each call yields the values afresh, a chunk at a time. A chunk is
#: a 1-D array, or `(values, groups)` when statistics are asked per group.
Stream = Callable[[], Iterable[Any]]
I64 = npt.NDArray[np.int64]
F64 = npt.NDArray[np.float64]

#: Keys collected and sorted at once, over every rank a pass is finishing: a bin with no
#: more than this is finished by sorting it rather than by counting another digit, which
#: ends the selection a pass sooner. 8 MB of keys at float64.
COLLECT = 1 << 20
#: A pass's digit histograms together: 16 bits (65,536 int64 bins, 0.5 MB) per group or
#: open bin while that fits, fewer bits (more passes) when many groups share a pass --
#: `ground_samples` asks for 64 cells at once.
HIST_BYTES = 8 << 20
#: Rows a `group_counts` partition may hold. Its merge buffer peaks near twice this in
#: (key, count, sum) triples -- ~50 MB -- whatever the number of rows.
BUDGET = 1 << 20

_MAX_DIGIT = 16
_MIN_DIGIT = 8


# ---------------------------------------------------------------------------------------
# Keys that sort as the values do
# ---------------------------------------------------------------------------------------


def _width(dtype: np.dtype[Any]) -> int:
    if dtype.kind not in "fiub":
        raise TypeError(f"order statistics of {dtype} are not supported")
    return 32 if dtype.itemsize <= 4 else 64


def _keys(values: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[np.bool_] | None]:
    """Sortable unsigned keys, and which values were NaN (None for integer types)."""
    dtype = values.dtype
    if dtype.kind == "f":
        if dtype.itemsize < 4:
            values = values.astype(np.float32)
        native = values.astype(values.dtype.newbyteorder("="), copy=False)
        unsigned = np.uint32 if native.dtype.itemsize == 4 else np.uint64
        bits: Any = native.view(unsigned)
        sign: Any = unsigned(1 << (8 * native.dtype.itemsize - 1))
        keys = np.where(bits & sign, ~bits, bits | sign)
        return keys, np.isnan(native)
    if dtype.kind == "b":
        return values.astype(np.uint32), None
    if dtype.kind == "u":
        return values.astype(np.uint32 if dtype.itemsize <= 4 else np.uint64), None
    if dtype.itemsize <= 4:
        return (values.astype(np.int32).view(np.uint32) ^ np.uint32(1 << 31)), None
    return (values.astype(np.int64).view(np.uint64) ^ np.uint64(1 << 63)), None


def _value(key: int, dtype: np.dtype[Any]) -> np.generic:
    """The inverse of `_keys` for one key, as a scalar of the stream's own dtype."""
    width = _width(dtype)
    sign = 1 << (width - 1)
    mask = (1 << width) - 1
    if dtype.kind == "f":
        bits = key & ~sign if key & sign else ~key & mask
        wide = np.float32 if dtype.itemsize <= 4 else np.float64
        unsigned = np.uint32 if width == 32 else np.uint64
        value = np.array([bits], dtype=unsigned).view(wide)[0]
        # The stream's own scalar type (native byte order: a scalar has no other).
        scalar: np.generic = dtype.type(value)
        return scalar
    if dtype.kind in "ub":
        unsigned_value: np.generic = dtype.type(key)
        return unsigned_value
    signed = (key ^ sign) - (1 << width) if (key ^ sign) & sign else key ^ sign
    signed_value: np.generic = dtype.type(signed)
    return signed_value


# ---------------------------------------------------------------------------------------
# Order statistics
# ---------------------------------------------------------------------------------------


@dataclass
class _Request:
    group: int
    #: The rank asked for, and the rank still to find inside the current bin.
    asked: int
    rank: int
    #: How many values share the key's top `bits` bits (`prefix`): the bin's size.
    size: int
    bits: int = 0
    prefix: int = 0
    key: int | None = None


@dataclass
class Selection:
    """What `order_statistics` found: per group id, how many values and NaNs there were,
    and the requested order statistics as scalars of the stream's dtype."""

    dtype: np.dtype[Any]
    counts: dict[int, int]
    nans: dict[int, int]
    values: dict[tuple[int, int], np.generic] = field(default_factory=dict)

    def at(self, group: int, rank: int) -> np.generic:
        return self.values[(group, rank)]


def _split(chunk: Any, lookup: I64 | None) -> tuple[npt.NDArray[Any], npt.NDArray[np.intp] | None]:
    """A chunk's values, and each one's index into the requested groups (rows of other
    groups dropped); None when the stream is not grouped."""
    if lookup is None:
        if isinstance(chunk, tuple):
            # A grouped chunk read as a plain one would mix the group ids into the values.
            raise TypeError("a stream of (values, groups) chunks needs `groups=`")
        return np.asarray(chunk).reshape(-1), None
    values, groups = chunk
    values = np.asarray(values).reshape(-1)
    ids = np.asarray(groups, dtype=np.int64).reshape(-1)
    position = np.searchsorted(lookup, ids)
    position = np.minimum(position, lookup.shape[0] - 1)
    wanted = lookup[position] == ids
    return values[wanted], position[wanted]


def _digit(histograms: int) -> int:
    """Bits per pass: 16 (65,536 bins) for a few histograms, fewer for many, so that all
    of one pass's histograms stay within `HIST_BYTES`."""
    fit = HIST_BYTES // (8 * max(1, histograms))
    return int(min(_MAX_DIGIT, max(_MIN_DIGIT, fit.bit_length() - 1)))


def order_statistics(
    stream: Stream,
    ranks: Callable[[int, int], Sequence[int]],
    *,
    groups: Sequence[int] | None = None,
    collect: int | None = None,
) -> Selection:
    """The exact order statistics `ranks(count, group)` asks for (0-based, NaNs excluded)
    -- per group when `groups` lists the group ids a grouped stream should be read for,
    and for group 0 when the stream is not grouped.

    Memory is one chunk, histograms of at most `HIST_BYTES` together, and at most
    `collect` (`COLLECT`) keys gathered at once over every rank being finished.
    """
    collect = COLLECT if collect is None else collect
    lookup = None if groups is None else np.unique(np.asarray(groups, dtype=np.int64))
    group_count = 1 if lookup is None else int(lookup.shape[0])
    ids = [0] if lookup is None else [int(g) for g in lookup]
    digit = _digit(group_count)
    bins = 1 << digit
    dtype: np.dtype[Any] | None = None
    width = 32
    hist = np.zeros(group_count * bins, dtype=np.int64)
    nans = np.zeros(group_count, dtype=np.int64)
    for chunk in stream():
        values, where = _split(chunk, lookup)
        if dtype is None:
            dtype = values.dtype
            width = _width(dtype)
        if values.size == 0:
            continue
        keys, isnan = _keys(values)
        index = np.zeros(keys.shape[0], dtype=np.intp) if where is None else where
        if isnan is not None and bool(isnan.any()):
            nans += np.bincount(index[isnan], minlength=group_count)
            keys, index = keys[~isnan], index[~isnan]
        top = (keys >> (width - digit)).astype(np.intp)
        hist += np.bincount(index * bins + top, minlength=group_count * bins)
    table = hist.reshape(group_count, bins)
    counts = table.sum(axis=1)
    selection = Selection(
        dtype=np.dtype(np.float64) if dtype is None else dtype,
        counts=dict(zip(ids, (int(c) for c in counts), strict=True)),
        nans=dict(zip(ids, (int(c) for c in nans), strict=True)),
    )
    requests: list[_Request] = []
    for group in range(group_count):
        for rank in sorted(set(ranks(int(counts[group]), ids[group]))):
            if not 0 <= rank < counts[group]:
                raise IndexError(f"rank {rank} of {int(counts[group])} values")
            request = _Request(group=group, asked=rank, rank=rank, size=int(counts[group]))
            _descend(request, table[group], digit, width)
            requests.append(request)
    del hist, table
    if not requests or dtype is None:
        return selection
    while True:
        pending = [r for r in requests if r.key is None]
        if not pending:
            break
        _refine(stream, lookup, pending, width, collect)
    for request in requests:
        assert request.key is not None
        selection.values[(ids[request.group], request.asked)] = _value(request.key, dtype)
    return selection


def _descend(request: _Request, hist: I64, digit: int, width: int) -> None:
    """Move `request` one digit down: which bin its rank falls in, and its rank there."""
    cumulative = np.cumsum(hist)
    chosen = int(np.searchsorted(cumulative, request.rank, side="right"))
    before = int(cumulative[chosen - 1]) if chosen > 0 else 0
    request.rank -= before
    request.size = int(hist[chosen])
    request.prefix = (request.prefix << digit) | chosen
    request.bits += digit
    if request.bits >= width:
        request.key = request.prefix


def _refine(
    stream: Stream, lookup: I64 | None, pending: list[_Request], width: int, collect: int
) -> None:
    """One pass: finish the smallest open bins by collecting and sorting their keys (up to
    `collect` keys in all), and count the next digit of the rest."""
    targets: dict[tuple[int, int, int], list[_Request]] = {}
    for request in pending:
        targets.setdefault((request.group, request.bits, request.prefix), []).append(request)
    gathered: dict[tuple[int, int, int], list[npt.NDArray[Any]]] = {}
    held = 0
    for target in sorted(targets, key=lambda t: targets[t][0].size):
        size = targets[target][0].size
        if held + size > collect:
            break
        gathered[target] = []
        held += size
    counted = [target for target in targets if target not in gathered]
    digit = _digit(len(counted))
    widths = {target: min(digit, width - target[1]) for target in counted}
    hists = {target: np.zeros(1 << widths[target], dtype=np.int64) for target in counted}
    for chunk in stream():
        values, where = _split(chunk, lookup)
        if values.size == 0:
            continue
        keys, isnan = _keys(values)
        index = np.zeros(keys.shape[0], dtype=np.intp) if where is None else where
        if isnan is not None and bool(isnan.any()):
            keys, index = keys[~isnan], index[~isnan]
        for target in targets:
            group, bits, prefix = target
            mask = (keys >> (width - bits)) == prefix if bits else np.ones(keys.shape, bool)
            if lookup is not None:
                mask &= index == group
            chosen = keys[mask]
            if chosen.size == 0:
                continue
            if target in gathered:
                gathered[target].append(chosen)
            else:
                step = widths[target]
                shift = width - bits - step
                below = ((chosen >> shift) & ((1 << step) - 1)).astype(np.intp)
                hists[target] += np.bincount(below, minlength=1 << step)
    for target, requests in targets.items():
        if target in gathered:
            ordered = np.sort(np.concatenate(gathered[target]))
            for request in requests:
                request.key = int(ordered[request.rank])
        else:
            for request in requests:
                _descend(request, hists[target], widths[target], width)


# ---------------------------------------------------------------------------------------
# numpy's median and percentile, from the order statistics
# ---------------------------------------------------------------------------------------


def median_ranks(count: int, _group: int = 0) -> list[int]:
    if count <= 0:
        return []
    if count % 2:
        return [(count - 1) // 2]
    return [count // 2 - 1, count // 2]


def _nan(dtype: np.dtype[Any]) -> np.generic:
    return dtype.type(np.nan) if dtype.kind == "f" else np.float64(np.nan)


def grouped_median(stream: Stream, groups: Sequence[int] | None) -> dict[int, np.generic]:
    """`np.median` of each group's values (see `grouped_percentile` for the stream's
    shape); a group with no values is absent from the result."""
    selection = order_statistics(stream, median_ranks, groups=groups)
    out: dict[int, np.generic] = {}
    for group, count in selection.counts.items():
        if selection.nans[group]:
            out[group] = _nan(selection.dtype)
        elif count:
            middle = [selection.at(group, rank) for rank in median_ranks(count)]
            # `_median` takes the mean of the one or two middle values of the partitioned
            # array: the same mean, over the same values, in the same dtype.
            out[group] = np.mean(np.asarray(middle, dtype=selection.dtype))
    return out


def median(stream: Stream) -> np.generic | None:
    """`np.median` of every value the stream yields, or None when it yields none."""
    return grouped_median(stream, None).get(0)


def _percentile_plan(count: int, q: float) -> tuple[int, int, float]:
    """numpy's linear method: the two ranks it reads and the interpolation weight.

    `_quantile`: the virtual index is `(n - 1) * (q / 100)`; its floor and the next index
    are read, both moved to the last element when the index is at or past it; and the
    weight is the virtual index less the (moved) floor -- so at q = 100 the weight is n,
    applied to a zero difference, exactly as numpy applies it.
    """
    quantile = np.true_divide(q, 100)
    virtual = (count - 1) * quantile
    previous = float(np.floor(virtual))
    following = previous + 1
    if virtual >= count - 1:
        previous = following = -1
    if virtual < 0:
        previous = following = 0
    gamma = float(virtual - previous)
    return int(previous) % count, int(following) % count, gamma


def _lerp(a: np.generic, b: np.generic, t: float, dtype: np.dtype[Any]) -> np.generic:
    """numpy 2's `_lerp`, on 0-d arrays of the stream's dtype."""
    left = np.asarray(a, dtype=dtype)
    right = np.asarray(b, dtype=dtype)
    diff = right - left
    out = np.add(left, diff * t, out=...)
    np.subtract(
        right, diff * (1 - t), out=out, where=t >= 0.5, casting="unsafe", dtype=type(out.dtype)
    )
    return out[()]  # type: ignore[no-any-return]


def grouped_percentile(
    stream: Stream, groups: Sequence[int] | None, qs: Sequence[float]
) -> dict[int, list[np.generic]]:
    """`np.percentile(values, q)` for each q (a Python float, as every caller here passes
    it), per group: a grouped stream yields `(values, group_ids)` and `groups` lists the
    ids wanted; with `groups` None the stream is plain and the answer is under 0. A group
    with no values is absent from the result."""
    plans: dict[int, list[tuple[int, int, float]]] = {}

    def ranks(count: int, group: int) -> list[int]:
        if count == 0:
            return []
        plans[group] = [_percentile_plan(count, q) for q in qs]
        return [rank for low, high, _ in plans[group] for rank in (low, high)]

    selection = order_statistics(stream, ranks, groups=groups)
    out: dict[int, list[np.generic]] = {}
    for group, nans in selection.nans.items():
        if nans:
            out[group] = [_nan(selection.dtype) for _ in qs]
        elif group in plans:
            out[group] = [
                _lerp(selection.at(group, low), selection.at(group, high), t, selection.dtype)
                for low, high, t in plans[group]
            ]
    return out


def percentile(stream: Stream, qs: Sequence[float]) -> list[np.generic] | None:
    """`np.percentile(values, q)` for each q, or None when the stream yields no values."""
    return grouped_percentile(stream, None, qs).get(0)


# ---------------------------------------------------------------------------------------
# Group-by, in partitions
# ---------------------------------------------------------------------------------------

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _partition(keys: I64, parts: int) -> npt.NDArray[np.intp]:
    mixed = (keys.astype(np.uint64) * _GOLDEN) >> np.uint64(32)
    return (mixed % np.uint64(parts)).astype(np.intp)


def _merge(keys: list[I64], counts: list[I64], sums: list[F64]) -> tuple[I64, I64, F64]:
    if not keys:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float64)
    unique, inverse = np.unique(np.concatenate(keys), return_inverse=True)
    inverse = inverse.reshape(-1)
    total = np.bincount(inverse, weights=np.concatenate(counts), minlength=unique.shape[0])
    summed = np.bincount(inverse, weights=np.concatenate(sums), minlength=unique.shape[0])
    return unique.astype(np.int64), np.rint(total).astype(np.int64), summed


def group_counts(
    stream: Stream, *, rows: int, budget: int | None = None
) -> Iterator[tuple[I64, I64, F64]]:
    """Per distinct key: how many rows had it, and the sum of their weights.

    The stream yields `(keys, weights)` chunks (weights may be None: all zero). `rows` is
    an upper bound on how many keys it yields. Each partition comes out as
    `(keys, counts, weight_sums)`, keys sorted within it; together they cover every key
    once.

    One pass first, as one partition: most group-bys here have far fewer distinct keys
    than rows (2 m ground cells, a coarse voxel grid), and are done in it. The moment its
    distinct keys pass `budget` it is abandoned for `rows / budget` partitions of a pass
    each -- so a fine grid costs passes, never memory.
    """
    budget = BUDGET if budget is None else budget
    whole = _partition_pass(stream, 0, 1, budget, abandon=True)
    if whole is not None:
        yield whole
        return
    parts = max(2, math.ceil(rows / max(1, budget)))
    for part in range(parts):
        found = _partition_pass(stream, part, parts, budget, abandon=False)
        assert found is not None
        yield found


def _partition_pass(
    stream: Stream, part: int, parts: int, budget: int, *, abandon: bool
) -> tuple[I64, I64, F64] | None:
    """One pass over partition `part` of `parts`; None when `abandon` and it outgrew
    `budget` distinct keys."""
    keys_seen: list[I64] = []
    counts_seen: list[I64] = []
    sums_seen: list[F64] = []
    held = 0
    for chunk_keys, chunk_weights in stream():
        keys = np.asarray(chunk_keys, dtype=np.int64).reshape(-1)
        weights = (
            np.zeros(keys.shape[0], dtype=np.float64)
            if chunk_weights is None
            else np.asarray(chunk_weights, dtype=np.float64).reshape(-1)
        )
        if parts > 1:
            mine = _partition(keys, parts) == part
            keys, weights = keys[mine], weights[mine]
        if keys.size == 0:
            continue
        unique, inverse, count = np.unique(keys, return_inverse=True, return_counts=True)
        keys_seen.append(unique.astype(np.int64))
        counts_seen.append(count.astype(np.int64))
        sums_seen.append(
            np.bincount(inverse.reshape(-1), weights=weights, minlength=unique.shape[0])
        )
        held += int(unique.shape[0])
        if held > budget:
            merged = _merge(keys_seen, counts_seen, sums_seen)
            if abandon and merged[0].shape[0] > budget:
                return None
            keys_seen, counts_seen, sums_seen = [merged[0]], [merged[1]], [merged[2]]
            held = int(merged[0].shape[0])
    return _merge(keys_seen, counts_seen, sums_seen)
