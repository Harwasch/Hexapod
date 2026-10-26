import numpy as np
import pytest

from living_spectra import (
    attach,
    inverse,
    lattice,
    read_sidecar,
    simulate_nodes,
    spectrum,
    write_sidecar,
)


def test_fft_inverse_and_sign():
    t = np.arange(300) / 30
    signal = np.zeros((300, 2, 3))
    signal[:, 0, 0] = 0.2 + 0.3 * np.cos(2 * np.pi * 0.7 * t)
    signal[:, 1, 1] = 0.04 * np.sin(2 * np.pi * 2.3 * t)
    f, c = spectrum(signal, 30)
    np.testing.assert_allclose(inverse(f, c, t), signal, atol=1e-14)
    np.testing.assert_allclose(inverse(f, c, t + 10), signal, atol=1e-14)
    assert c[1, np.argmin(abs(f - 2.3)), 1].imag == pytest.approx(-0.04)


@pytest.mark.parametrize("n", [9, 10])
def test_full_fft_including_dc_and_nyquist(n):
    signal = np.random.default_rng(5).normal(size=(n, 2, 3))
    f, c = spectrum(signal, 30, n // 2 + 1)
    np.testing.assert_allclose(inverse(f, c, np.arange(n) / 30), signal, atol=1e-14)


def test_half_float_roundtrip_and_truncation(tmp_path):
    c = np.array([[[0.125 - 0.25j, 0.5 + 1j, -0.75j]]])
    write_sidecar(tmp_path, {"frequenciesHz": [0.5]}, np.array([123]), c)
    h, indices, decoded = read_sidecar(tmp_path / "spectra.json")
    np.testing.assert_array_equal(indices, [123])
    np.testing.assert_array_equal(decoded, c)
    assert h["motionEvidence"] == "simulated"
    (tmp_path / "spectra.bin").write_bytes(b"bad")
    with pytest.raises(ValueError, match="length"):
        read_sidecar(tmp_path / "spectra.json")


def test_lattice_has_every_interpolation_corner():
    points = np.random.default_rng(2).uniform(-2, 2, (200, 3))
    origin, side, indices, _ = lattice(points, 16, 2000)
    base = np.floor((points - origin) / (side / 15)).astype(int)
    occupied = set(indices)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                cells = base + [dx, dy, dz]
                assert set(cells[:, 0] + 16 * (cells[:, 1] + 16 * cells[:, 2])) <= occupied


def test_hierarchy_pins_root_and_preserves_segments():
    rig = {
        "nodes": [
            {"position": [0, 0, 0], "parent": -1, "band": "trunk"},
            {"position": [0, 0, 1], "parent": 0, "band": "trunk"},
            {"position": [1, 0, 2], "parent": 1, "band": "branch"},
        ]
    }
    rest, parents, rotations, positions = simulate_nodes(rig)
    np.testing.assert_array_equal(positions[:, 0], np.zeros((300, 3)))
    for i in (1, 2):
        np.testing.assert_allclose(
            np.linalg.norm(positions[:, i] - positions[:, parents[i]], axis=1),
            np.linalg.norm(rest[i] - rest[parents[i]]),
            atol=1e-14,
        )
    assert np.ptp(positions[:, 2, 0]) > 0
    np.testing.assert_allclose(
        rotations @ rotations.transpose(0, 1, 3, 2),
        np.broadcast_to(np.eye(3), rotations.shape),
        atol=1e-14,
    )
    ids, weights = attach(np.array([[0, 0, 0], [1, 0, 2]]), rest, parents)
    assert ids.shape == (2, 3)
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    rig["nodes"][0]["parent"] = 2
    with pytest.raises(ValueError, match="parent-first"):
        simulate_nodes(rig)
