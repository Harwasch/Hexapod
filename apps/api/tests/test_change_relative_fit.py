"""Measured pair placement, without a database or private capture data."""

import math
import uuid
from unittest.mock import Mock, patch

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.analysis import change
from app.models import Epoch, EpochMap
from app.schemas.change import ChangeRelativeFitRead
from app.services import change_maps, epochs
from tests.test_change import flat, surface


@pytest.fixture(autouse=True)
def reference_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(epochs, "_reference_keys", lambda *_: ({}, None, "revision-1"))


def epoch() -> Epoch:
    return Epoch(
        id=uuid.uuid4(),
        site_id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        platform="drone",
        status="published",
        alignment={"verdict": "coarse", "uncertaintyM": 5.0},
    )


def link(child: Epoch, reference: Epoch, relative: object = 0.4) -> None:
    child.site_id = reference.site_id
    child.alignment = {
        **(child.alignment or {}),
        "method": "visual",
        "frameRelativeBasis": "visual",
        "frameRelativeUncertaintyM": relative,
        "referenceEpochId": str(reference.id),
        "referenceJobId": str(reference.job_id),
        "referenceRevision": "revision-1",
    }


def pair(
    a: Epoch, b: Epoch, reference: Epoch | None = None
) -> tuple[float, float, ChangeRelativeFitRead | None]:
    db = Mock(spec=Session)
    db.get.return_value = reference
    graded = EpochMap(descriptor={"alignmentCap": {"uncertaintyM": 5.0}})
    return change_maps._pair_placement(
        db, a, b, change_maps._placement(a, graded), change_maps._placement(b, graded)
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("drop,changed", [(250, True), (50, False)])
def test_linked_coarse_pair_uses_measured_fit(reverse: bool, drop: int, changed: bool) -> None:
    a, b = epoch(), epoch()
    link(b, a)
    before, after = (b, a) if reverse else (a, b)
    ua, ub, fit = pair(before, after)
    assert math.hypot(ua, ub) == pytest.approx(0.4)
    assert fit is not None
    assert fit.reference_epoch_id == a.id
    assert fit.uncertainty_m == pytest.approx(0.4)
    fields_a, fields_b = flat(sigma_cm=27), flat(sigma_cm=25, height_cm=1000 - drop)
    if reverse:
        fields_a, fields_b = fields_b, fields_a
    for layer in change.LAYERS:
        found = change.change_planes(
            surface(fields_a), surface(fields_b), ua, ub, layer=layer, slope_guard=True
        )
        counts = found.counts()
        assert counts["raised"] + counts["lowered"] == (400 if changed else 0)


@pytest.mark.parametrize("bad", [None, True, -0.4, 0, float("nan"), float("inf"), "0.4"])
def test_unmeasured_fit_keeps_absolute_rule(bad: object) -> None:
    a, b = epoch(), epoch()
    link(b, a, bad)
    assert pair(a, b) == (5.0, 5.0, None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "gnss-prior-ba"),
        ("referenceEpochId", str(uuid.uuid4())),
        ("referenceJobId", str(uuid.uuid4())),
        ("referenceJobId", None),
        ("frameRelativeBasis", "visual, refuted by its check targets"),
    ],
)
def test_unlinked_or_refuted_fit_keeps_absolute_rule(field: str, value: object) -> None:
    a, b = epoch(), epoch()
    link(b, a)
    b.alignment = {**(b.alignment or {}), field: value}
    assert pair(a, b) == (5.0, 5.0, None)


def test_same_reference_combines_measured_terms_and_requires_same_revision() -> None:
    ref, a, b = epoch(), epoch(), epoch()
    link(a, ref, 0.3)
    link(b, ref, 0.4)
    ua, ub, fit = pair(a, b, ref)
    assert (ua, ub) == (0.3, 0.4)
    assert fit is not None
    assert fit.reference_epoch_id == ref.id
    assert fit.uncertainty_m == pytest.approx(0.5)
    b.alignment = {**(b.alignment or {}), "referenceJobId": str(uuid.uuid4())}
    assert pair(a, b) == (5.0, 5.0, None)


@pytest.mark.parametrize("layer", change.LAYERS)
def test_half_cell_shift_at_step_is_unchanged(layer: change.Layer) -> None:
    a, b = flat(), flat()
    # A half-cell shift of a 4 m step: one boundary cell samples halfway up it.
    for fields in (a, b):
        for channel in ("dsm", "ground"):
            fields[channel][:, 10:] += 400
    for channel in ("dsm", "ground"):
        b[channel][:, 10] -= 200
    found = change.change_planes(surface(a), surface(b), 0, 0.4, layer=layer, slope_guard=True)
    assert found.counts() == {"unchanged": 400, "raised": 0, "lowered": 0, "insufficient": 0}


@pytest.mark.parametrize("swap", [False, True])
def test_slope_budget_uses_steeper_surface_and_full_gradient(swap: bool) -> None:
    a, b = flat(), flat()
    # Isolate propagation from neighbourhood matching and the ground envelope.
    j, i = np.indices((20, 20))
    b["dsm"] += (30 * i + 40 * j).astype(np.int16)
    if swap:
        a, b = b, a
    found = change.change_planes(
        surface(a),
        surface(b),
        0,
        0.4,
        layer="vegetation",
        motion_m=0,
        horizontal=False,
        slope_guard=True,
    )
    expected = 3 * math.sqrt(2 * 0.02**2 + 0.4**2 + (0.5 * 0.4) ** 2)
    assert found.min_detectable_m == pytest.approx(expected)


@pytest.mark.parametrize("layer", change.LAYERS)
def test_slope_without_neighbours_is_insufficient(layer: change.Layer) -> None:
    a, b = flat(ni=1, nj=1), flat(ni=1, nj=1)
    found = change.change_planes(surface(a), surface(b), 0, 0.4, layer=layer, slope_guard=True)
    assert found.counts()["insufficient"] == 1
    assert found.min_detectable_m is None


def test_unlinked_coarse_claiming_centimetres_keeps_b5a_floor() -> None:
    a, b = epoch(), epoch()
    for e in (a, b):
        e.alignment = {"verdict": "coarse", "uncertaintyM": 0.03}
    graded = EpochMap(descriptor={"alignmentCap": {"uncertaintyM": 0.03}})
    placements = [change_maps._placement(e, graded) for e in (a, b)]
    assert change_maps._pair_placement(Mock(spec=Session), a, b, *placements) == (1, 1, None)


@pytest.mark.parametrize("layer", change.LAYERS)
@pytest.mark.parametrize("names", [("a", "b"), ("c", "d"), ("e", "f"), ("i", "j")])
def test_no_change_controls_also_pass_with_relative_fit(
    layer: change.Layer, names: tuple[str, str]
) -> None:
    from tests.test_change import control_surface

    a, b = (control_surface(name) for name in names)
    found = change.change_planes(a, b, 0, 0.4, layer=layer, slope_guard=True)
    counts = found.counts()
    assert counts["raised"] + counts["lowered"] <= math.ceil(sum(counts.values()) / 370)
    assert counts["unchanged"] > 0.9 * sum(counts.values())


def test_drawn_level_retains_slope_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_change import PROJECTION

    a, b = flat(), flat()
    j, i = np.indices((20, 20))
    b["dsm"] += (30 * i + 40 * j).astype(np.int16)
    monkeypatch.setattr(change, "needs_coarser", lambda planes: planes.level == 2)
    found = change.change_map(
        surface(a), surface(b), 0, 0.4, PROJECTION, layer="vegetation", slope_guard=True
    )
    assert found.drawn.level == 3
    expected = 3 * math.sqrt(2 * 0.02**2 + 0.4**2 + 2 * 0.25**2 + (0.5 * 0.4) ** 2)
    assert found.drawn.min_detectable_m == pytest.approx(expected)


def test_same_job_reference_retry_falls_back_to_b5a() -> None:
    a, b = epoch(), epoch()
    link(b, a)
    b.alignment = {**(b.alignment or {}), "referenceRevision": "before-retry"}
    # A retry writes new artifact rows while keeping A's epoch and job IDs.
    with patch.object(epochs, "_reference_keys", return_value=({}, None, "after-retry")):
        assert pair(a, b) == (5.0, 5.0, None)


def test_three_cell_step_shift_needs_slope_guard() -> None:
    a, b = flat(), flat()
    a["dsm"][:, 10:] += 400
    b["dsm"][:, 7:] += 400
    found = change.change_planes(
        surface(a),
        surface(b),
        0,
        0.4,
        layer="vegetation",
        slope_guard=True,
    )
    assert found.counts()["raised"] == found.counts()["lowered"] == 0


def test_shared_reference_retry_invalidates_both_old_fits() -> None:
    ref, a, b = epoch(), epoch(), epoch()
    link(a, ref, 0.3)
    link(b, ref, 0.4)
    with patch.object(epochs, "_reference_keys", return_value=({}, None, "retry")):
        assert pair(a, b, ref) == (5.0, 5.0, None)


@pytest.mark.parametrize("revision", [None, "", 1, True])
def test_missing_or_invalid_reference_revision_keeps_b5a(revision: object) -> None:
    a, b = epoch(), epoch()
    link(b, a)
    b.alignment = {**(b.alignment or {}), "referenceRevision": revision}
    assert pair(a, b) == (5.0, 5.0, None)
