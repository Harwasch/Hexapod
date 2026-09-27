"""Exact out-of-core statistics: numpy's own answers, bit for bit, from chunked passes."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

import outofcore


def _stream(values: np.ndarray, chunk: int) -> outofcore.Stream:
    def chunks() -> Iterator[np.ndarray]:
        for start in range(0, values.shape[0], chunk):
            yield values[start : start + chunk]

    return chunks


def _same(a: object, b: object) -> None:
    left, right = np.asarray(a), np.asarray(b)
    assert left.dtype == right.dtype, (left.dtype, right.dtype)
    np.testing.assert_array_equal(left, right)
    # Bit for bit: equal floats can still differ in sign of zero.
    assert left.tobytes() == right.tobytes()


def _samples(rng: np.random.Generator, dtype: str, n: int) -> np.ndarray:
    if dtype.startswith("f"):
        # Wide dynamic range, negatives, exact duplicates, zeros and an infinity or two:
        # the cases a radix key has to order correctly.
        values = rng.standard_normal(n) * np.exp(rng.uniform(-20, 20, n))
        values[rng.random(n) < 0.1] = 0.0
        values[rng.random(n) < 0.1] = 1.5
        if n > 50:
            values[:2] = [np.inf, -np.inf]
        return values.astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(max(info.min, -(10**6)), min(info.max, 10**6), n).astype(dtype)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # inf - inf, in numpy too
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "uint8", "uint16"])
@pytest.mark.parametrize("n", [1, 2, 3, 8, 1001, 40_000])
def test_median_and_percentiles_are_numpys_to_the_bit(dtype: str, n: int) -> None:
    rng = np.random.default_rng(n)
    values = _samples(rng, dtype, n)
    for collect in (outofcore.COLLECT, 64):
        original = outofcore.COLLECT
        outofcore.COLLECT = collect
        try:
            stream = _stream(values, 997)
            _same(outofcore.median(stream), np.median(values))
            qs = [0.0, 0.5, 2.0, 5.0, 10.0, 25.0, 50.0, 90.0, 95.0, 98.0, 99.0, 99.5, 100.0]
            got = outofcore.percentile(stream, qs)
            assert got is not None
            for q, value in zip(qs, got, strict=True):
                _same(value, np.percentile(values, q))
        finally:
            outofcore.COLLECT = original


def test_selection_refines_digit_by_digit_when_bins_are_large() -> None:
    """With `collect` tiny, every rank is found by counting digits, four passes deep for
    float64 -- the path a huge, narrow column takes."""
    rng = np.random.default_rng(7)
    values = (1.0 + rng.random(50_000) * 1e-9).astype(np.float64)
    ranks = [0, 1, 17, 25_000, 49_999]
    selection = outofcore.order_statistics(_stream(values, 4096), lambda n, g: ranks, collect=1)
    ordered = np.sort(values)
    for rank in ranks:
        _same(selection.at(0, rank), ordered[rank])


def test_percentile_of_a_2d_column_matches_numpys_axis_0() -> None:
    """`orient` asks `np.percentile(xyz[:, :2], 2.0, axis=0)`: per column, the same."""
    rng = np.random.default_rng(3)
    xy = rng.standard_normal((12_345, 2)).astype(np.float32)
    expected = np.percentile(xy, 2.0, axis=0)
    for axis in range(2):
        got = outofcore.percentile(_stream(np.ascontiguousarray(xy[:, axis]), 1000), [2.0])
        assert got is not None
        _same(got[0], expected[axis])


def test_nan_makes_the_answer_nan_as_numpy_does() -> None:
    values = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    with np.errstate(invalid="ignore"):
        assert np.isnan(np.median(values))
    got = outofcore.median(_stream(values, 2))
    assert got is not None and np.isnan(got)
    percentiles = outofcore.percentile(_stream(values, 2), [50.0])
    assert percentiles is not None and np.isnan(percentiles[0])


def test_nothing_in_the_stream_is_none() -> None:
    empty = np.zeros(0, dtype=np.float32)
    assert outofcore.median(_stream(empty, 10)) is None
    assert outofcore.percentile(_stream(empty, 10), [5.0]) is None


def test_grouped_percentiles_are_each_groups_own() -> None:
    rng = np.random.default_rng(11)
    groups = rng.integers(0, 40, 30_000).astype(np.int64) * 1000
    values = rng.standard_normal(30_000).astype(np.float32)

    def stream() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for start in range(0, values.shape[0], 4096):
            yield values[start : start + 4096], groups[start : start + 4096]

    wanted = [0, 5000, 39000, 123_456]  # the last has no values
    got = outofcore.grouped_percentile(stream, wanted, [5.0])
    assert set(got) == {0, 5000, 39000}
    for group in (0, 5000, 39000):
        _same(got[group][0], np.percentile(values[groups == group], 5.0))


@pytest.mark.parametrize("budget", [outofcore.BUDGET, 1000])
def test_group_counts_cover_every_key_once(budget: int) -> None:
    rng = np.random.default_rng(5)
    keys = rng.integers(-(10**9), 10**9, 60_000).astype(np.int64)
    keys[::3] = keys[1::3][: keys[::3].shape[0]]  # plenty of repeats
    weights = (rng.random(60_000) < 0.3).astype(np.float64)

    def stream() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for start in range(0, keys.shape[0], 5000):
            yield keys[start : start + 5000], weights[start : start + 5000]

    parts = list(outofcore.group_counts(stream, rows=keys.shape[0], budget=budget))
    assert len(parts) == (1 if budget == outofcore.BUDGET else 60)
    got_keys = np.concatenate([p[0] for p in parts])
    got_counts = np.concatenate([p[1] for p in parts])
    got_sums = np.concatenate([p[2] for p in parts])
    order = np.argsort(got_keys)
    expected, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    np.testing.assert_array_equal(got_keys[order], expected)
    np.testing.assert_array_equal(got_counts[order], counts)
    np.testing.assert_array_equal(got_sums[order], np.bincount(inverse, weights=weights))
