"""`fill_pockets`: free space carved by the cameras' rays, the pocket under a spool's top
flange found as a cluster (and the open air beside the drum left free), and the shape under
the overhang continued into it: opaque from low views, nothing in carved space."""

from __future__ import annotations

import numpy as np
import pytest
from fill_scenes import flanged_spool_scene, spool_cameras

import fill_pockets as fp
import fill_surface as fs
from splat_render import Camera


def _scene():
    scene, masks = flanged_spool_scene()
    cams = spool_cameras()
    vox = fp.free_space(
        scene, cams, np.array([-1.0, -1.0, -0.1]), np.array([1.0, 1.0, 0.9]), divisions=64
    )
    return scene, masks, cams, vox


def test_free_space_and_the_pocket_under_the_flange() -> None:
    _, _, _, vox = _scene()
    counts = np.bincount(vox.state.ravel(), minlength=3)
    assert counts[fp.FREE] > counts[fp.SURFACE] > 0
    # The open air beside the drum, halfway down between the flanges, was seen from the side.
    beside = np.array([[0.45 * np.cos(a), 0.45 * np.sin(a), 0.25] for a in np.linspace(0, 6.2, 24)])
    assert (vox.at(beside) == fp.FREE).mean() > 0.8
    labels, found = fp.pockets(vox)
    assert found
    # The pocket under the top flange: the drum's missing top band, under the overhang.
    band = np.array([[0.33 * np.cos(a), 0.33 * np.sin(a), 0.53] for a in np.linspace(0, 6.2, 24)])
    ijk, inside = vox.cell(band)
    hit = labels[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]
    assert (hit > 0).mean() > 0.5
    pocket = found[int(np.bincount(hit[hit > 0]).argmax()) - 1]
    assert pocket.under > 0.5, pocket.json()
    # The open air beside the drum is not a pocket.
    ijk, inside = vox.cell(beside)
    assert (labels[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]] == 0).mean() > 0.9
    views = fp.pocket_views(vox, labels, pocket, directions=120)
    assert views and views[0]["seenShare"] > 0.3
    # Views that see into it come from low or below.
    assert min(v["elevationDeg"] for v in views) < 20


def test_a_whole_run_ablates_solidity_and_adds_the_shape_layer(tmp_path, monkeypatch) -> None:
    """The flanged spool through `anchor_fill.run_arms` with the stand-in editor: the
    solidity presets clone the arms and reach the distil; the shape stage finds the pocket
    under the top flange, fits the overhang and packages its own layer, with the split pass
    tests and the sheets."""
    pytest.importorskip("torch")
    import json

    import anchor_fill as af

    # Small solidity views: the CPU rasteriser is dense.
    monkeypatch.setattr(af, "SOLID_SIZE", (32, 24))

    from fill_scenes import photos
    from test_anchor_fill import _measured_tileset

    import distill_fill as df
    import generative_fill as gf
    from splat_render import render

    scene, _ = flanged_spool_scene()
    cams = spool_cameras()
    paths = photos(scene, cams, tmp_path / "frames")
    views = [gf.RealView(p.name, c, None, p) for p, c in zip(paths, cams, strict=True)]
    setup, _ = af.make_setup(
        "spool", "a cable spool", scene, views, render, width=96, log=lambda s: None
    )
    options = af.Options(
        arms=("refs", "norefs", "splash"),
        fill_size=(96, 56),
        set_size=(64, 36),
        probe_size=(64, 36),
        quality_width=96,
        azimuths=8,
        anchors=(2, 2),
        propagation=(1, 2),
        seeds=1,
        prop_seeds=1,
        distill=2,
        update_distill=0,
        update_strengths=(),
        distill_width=48,
        lift_stride=4,
        carve_width=96,
        score_width=96,
        solidity=("full",),
        shape=True,
    )
    report = af.run_arms(
        setup,
        af.StandInEditor(),
        None,
        af.OracleDepth(scene, render),
        render,
        af.StandInPerceptual(),
        options,
        _measured_tileset(tmp_path / "scan"),
        tmp_path / "out",
        distil=df.run,
        log=lambda s: None,
    )
    cands = report["candidates"]
    assert {"refs", "norefs", "splash", "refs-full", "norefs-full"} <= set(cands)
    assert "splash-full" not in cands  # the solidity clones are of refs and norefs only
    # The splash arm's anchors went through its gate, every seed with a verdict.
    gate = report["splashGate"]
    assert gate["frames"] >= 1 and gate["kept"] + gate["rejected"] == gate["frames"]
    assert all({"view", "seed", "kept"} <= set(g) for g in gate["seeds"])
    assert all(d.get("iterations") == 2 for d in cands["refs-full"]["distil"])
    shape = report["shape"]
    assert shape["pockets"] and shape["overhang"] is not None, json.dumps(shape)[:2000]
    assert shape["overhang"]["discs"] > 0 and shape["overhang"]["inFreeSpace"] == 0
    summary = shape["tests"]["summary"]
    assert {"before", "refs", "refs-full", "refs+shape", "shape"} <= set(summary)
    assert summary["shape"]["pocketWorst"] < summary["before"]["pocketWorst"]
    assert summary["shape"]["airKept"]
    out = tmp_path / "out"
    assert (out / "anchor-shape" / "inferred" / "tileset.json").exists()
    assert (out / "anchor-refs-full" / "inferred" / "tileset.json").exists()
    for sheet in ("pockets-spool.png", "shape-spool.png", "splash-spool.png"):
        assert (out / "renders" / sheet).exists()


def test_shape_under_the_overhang_is_opaque_low_and_never_in_free_space() -> None:
    scene, masks, _, vox = _scene()
    labels, found = fp.pockets(vox)
    ijk, _ = vox.cell(np.array([[0.33, 0.0, 0.53]]))
    pocket = found[int(labels[ijk[0, 0], ijk[0, 1], ijk[0, 2]]) - 1]
    fit = fp.fit_overhang(scene, pocket, scene.positions[masks["top"]])
    assert fit is not None
    assert abs(fit.radius - 0.3) < 0.02 and abs(fit.rim - 0.6) < 0.05
    assert fit.normal[2] > 0.99
    assert -0.25 < fit.drum_top < -0.15 and -0.05 < fit.underside < -0.005
    discs, report = fp.shape_under_overhang(fit, scene, vox, 0.025)
    assert report["drumDiscs"] > 100 and report["undersideDiscs"] > 100
    assert fp.in_free_space(vox, discs) == 0
    # Drum discs fill the missing band only; underside discs sit just under the top.
    z = discs.positions[:, 2]
    r = np.hypot(discs.positions[:, 0], discs.positions[:, 1])
    drum = np.abs(r - 0.3) < 0.01
    assert z[drum].min() > 0.38 and z[drum].max() < 0.6
    assert np.all(np.abs(z[~drum] - 0.58) < 0.02)
    # Shaded: darker than the wood they were coloured from.
    assert discs.colours.mean() < scene.colours[masks["drum"]].mean()
    # From a low view at the band, the drum's top is opaque with the shape and see-through
    # without it; the air beside the drum stays as see-through as it was.
    both = type(scene).concat([scene, discs])
    eye = np.array([2.2, 0.0, 0.45])
    cam = Camera.look_at(eye, [0.0, 0.0, 0.49], fov_deg=30, width=96, height=72)
    uv, _ = cam.project(np.array([[0.3, 0.0, 0.49]]))
    x, y = int(uv[0, 0]), int(uv[0, 1])
    before = fs.ewa_alpha(scene, cam)
    after = fs.ewa_alpha(both, cam)
    assert before[y, x] < 0.5 and after[y, x] > 0.95
    uv_air, _ = cam.project(np.array([[0.0, 0.45, 0.3]]))
    xa, ya = int(uv_air[0, 0]), int(uv_air[0, 1])
    assert after[ya, xa] <= before[ya, xa] + 0.05
    # The split pass test: the top face passes either way; the pocket fails before and passes
    # with the shape; the air beside the drum stays as clear.
    tests = fp.overhang_tests(fit, {"before": scene, "shape": both}, vox, 0.025, size=(120, 90))
    s = tests["summary"]
    assert s["before"]["topPass"] and s["shape"]["topPass"]
    assert s["before"]["pocketWorst"] > 0.15 and not s["before"]["pocketPass"], s
    assert s["shape"]["pocketPass"], s["shape"]
    assert s["shape"]["airKept"] and s["before"]["airMean"] > 0.2
