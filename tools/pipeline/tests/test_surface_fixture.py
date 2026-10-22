"""The committed cross-language surface fixture is what its generator writes, byte for byte,
and it tells the story it says it tells (experiments/surface_fixture.py)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from captures_bridge import cell_grid
from experiments import surface_fixture

COMMITTED = surface_fixture.OUT
NO = cell_grid.NO_HEIGHT


def test_the_committed_surfaces_are_regenerated_byte_for_byte(tmp_path: Path) -> None:
    surface_fixture.write(tmp_path)
    controls = [f"control/{name}/surface.bin" for name in surface_fixture.CONTROL]
    for name in ("e1/surface.bin", "e2/surface.bin", "surfaces.json", *controls):
        assert (tmp_path / name).read_bytes() == (COMMITTED / name).read_bytes(), name


def test_each_surface_is_on_its_confidence_grid_s_plane() -> None:
    document = json.loads((COMMITTED / "surfaces.json").read_text(encoding="utf-8"))
    for name, epoch in document["epochs"].items():
        blob = (COMMITTED / epoch["file"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == epoch["fileSha256"]
        surface = cell_grid.decode(epoch["descriptor"], blob)
        assert surface_fixture.plane_sha256(surface.planes[0]) == epoch["plane"]["sha256"]
        graded_descriptor, graded = surface_fixture.confidence(name)
        a, b = surface.planes[0], graded.planes[0]
        assert (a.level, a.i0, a.j0, a.ni, a.nj) == (b.level, b.i0, b.j0, b.ni, b.nj)
        assert epoch["descriptor"]["alignedTo"]["sha256"] == graded_descriptor["sha256"]


def test_the_reference_change_finds_what_was_built() -> None:
    """P5's demo rule over the fixture (the top surface, cell for cell): the shed and the
    stack raised, the spoil and the pond lowered, the hedge and the noise unchanged, the
    holes insufficient, the stack weak."""
    change = json.loads((COMMITTED / "surfaces.json").read_text(encoding="utf-8"))["change"]
    assert (change["level"], change["cellM"]) == (2, 1.0)
    tally = Counter((row["class"], row["weak"]) for row in change["changed"])
    # 6 x 4 one-metre cells each; one stack cell is where e1 saw nothing (insufficient);
    # the pond's 10 x 10 less the one cell whose e2 sigma saturates (insufficient).
    assert tally == {("raised", False): 24, ("lowered", False): 24 + 99, ("raised", True): 23}
    # e2's 2 m strip along the west edge (64 cells), e1's 8 unseen cells, e2's saturated
    # sigma (1), and e1's south row past the strip (39) and east column above that row
    # (31), where e2's 0.5 m window covers half of each 1 m cell: a partial child set is
    # insufficient (review B5a).
    assert change["counts"]["insufficient"] == 64 + 8 + 1 + 39 + 31
    assert change["weakChanged"] == 23
    for epoch in ("e1", "e2"):
        channels = [
            c["name"]
            for c in json.loads((COMMITTED / "surfaces.json").read_text())["epochs"][epoch][
                "descriptor"
            ]["channels"]
        ]
        assert channels == ["dsm", "ground", "sigma", "support", "groundSigma"]


def test_the_controls_are_p5s_own_scans_of_round_twos_scenes() -> None:
    """Review B5a's controls and its round 2's: each scan P5's own surface, every cell
    surfaced; ground observed everywhere but under the 40 m crown (g, h), whose core P5
    leaves unobserved (never canopy height)."""
    control = json.loads((COMMITTED / "surfaces.json").read_text(encoding="utf-8"))["control"]
    pairs = [tuple(spec["pair"]) for spec in control["pairs"]]
    assert pairs == [("a", "b"), ("c", "d"), ("e", "f"), ("g", "h"), ("i", "j"), ("k", "l")]
    counts = {name: scan["samplesPerCell"] for name, scan in control["scans"].items()}
    assert (counts["a"], counts["b"], counts["f"]) == (20, 2000, 2000)
    for name, scan in control["scans"].items():
        blob = (COMMITTED / scan["file"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == scan["fileSha256"]
        plane = cell_grid.decode(scan["descriptor"], blob).planes[0]
        assert (plane.data[..., 2] > 0).all(), name
        unobserved = int(scan["descriptor"]["summary"]["groundUnobserved"])
        if name in ("g", "h"):
            assert unobserved > 0 and (plane.data[..., 1][plane.data[..., 4] == 0] == NO).all()
        elif name not in ("i", "j"):
            assert unobserved == 0, name
