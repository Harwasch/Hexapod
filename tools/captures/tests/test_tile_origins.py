"""Per-tile origins (docs/DECISIONS/0011-splat-tile-origins.md): SPZ stores positions as
24-bit fixed point, so at 12 fractional bits a tile spans +/-2047.99976 m and no more. The
packer refuses what does not fit instead of wrapping it, moves a non-root tile that does not
fit to an origin of its own (a translation-only `transform`, relative to its parent), and
gives a tile that still does not fit fewer fractional bits."""

from __future__ import annotations

import gzip

import numpy as np
import pytest

import splat_tiles
from splat_tiles import SPZ_FIXED_MAX, SplatFormatError, pack_spz, unpack_spz


def _pack(xyz: np.ndarray, **kwargs: int) -> bytes:
    n = xyz.shape[0]
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (n, 1))
    return pack_spz(
        xyz.astype(np.float32),
        np.zeros((n, 3), np.float32),
        np.zeros(n, np.float32),
        np.full((n, 3), -3.0, np.float32),
        quat,
        **kwargs,
    )


def _xyz(*rows: tuple[float, float, float]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def test_pack_spz_refuses_a_position_past_the_24_bit_range() -> None:
    # Before the refusal, 3000 m wrapped to 3000 - 4096 = -1096 m without a word.
    with pytest.raises(SplatFormatError, match=r"2047\.99"):
        _pack(_xyz((3000.0, 0.0, 0.0)))


def test_pack_spz_packs_the_last_step_inside_the_range() -> None:
    edge = SPZ_FIXED_MAX / 4096
    xyz = _xyz((edge, -edge, 0.0), (-edge, edge, edge))
    back = unpack_spz(_pack(xyz))
    assert np.array_equal(back["x"], xyz[:, 0])
    assert np.array_equal(back["y"], xyz[:, 1])
    assert np.array_equal(back["z"], xyz[:, 2])


def test_pack_spz_refuses_non_finite_positions() -> None:
    with pytest.raises(SplatFormatError, match="finite"):
        _pack(_xyz((np.nan, 0.0, 0.0)))
    with pytest.raises(SplatFormatError, match="finite"):
        _pack(_xyz((np.inf, 0.0, 0.0)))


def test_pack_spz_writes_the_fractional_bits_it_is_given() -> None:
    blob = _pack(_xyz((3000.5, -3000.5, 1.0)), fractional_bits=10)
    raw = gzip.decompress(blob)
    assert raw[13] == 10
    back = unpack_spz(blob)
    assert float(back["x"][0]) == 3000.5
    assert float(back["y"][0]) == -3000.5


def test_pack_spz_refuses_fractional_bits_outside_the_header_byte() -> None:
    with pytest.raises(ValueError, match="fractional"):
        _pack(_xyz((1.0, 0.0, 0.0)), fractional_bits=24)
    with pytest.raises(ValueError, match="fractional"):
        _pack(_xyz((1.0, 0.0, 0.0)), fractional_bits=-1)


def test_the_default_still_packs_twelve_fractional_bits() -> None:
    assert gzip.decompress(_pack(_xyz((1.0, 2.0, 3.0))))[13] == splat_tiles.SPZ_FRACTIONAL_BITS
