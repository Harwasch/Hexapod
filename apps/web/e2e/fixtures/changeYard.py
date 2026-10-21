"""Writes changeYard.json and changeYardCoarse.json: the API's own `/change` answers for the
committed yard fixture.

e2e/changeDrape.spec.ts mocks `GET /sites/{id}/change` with it, so the drape is tested on the
shape and the numbers the API gives for `data/tiles/synthetic-yard-surfaces` (heights) over
`data/tiles/synthetic-yard-epochs` (confidence), both epochs surveyed (u 0.03 m and 0.062 m, as
apps/api tests/test_change_api.py registers them), placed in the yard's frame (the root
transform's origin of data/tiles/synthetic-yard: 46.1335 N, 123.88 W, 0 m). The coarse answer
is the same pair with e2 placed coarsely at ±5 m (a phone epoch, ADR 0018 § API 4), its grid
0.062 m from the frame. It runs the API's analysis and response code, not a copy of it.
Regenerate after the fixture or the API changes, then format:

    cd apps/api && uv run python ../web/e2e/fixtures/changeYard.py
    cd ../.. && corepack pnpm exec prettier --write apps/web/e2e/fixtures/changeYard*.json
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from app.analysis import cell_grid
from app.analysis import change as analysis
from app.analysis.weak_areas import site_projection
from app.services import change_maps

ROOT = Path(__file__).resolve().parents[4]
SURFACES = ROOT / "data" / "tiles" / "synthetic-yard-surfaces"
CONFIDENCE = ROOT / "data" / "tiles" / "synthetic-yard-epochs"
LAT, LON, HEIGHT = 46.1335, -123.88, 0.0
U_FROM, U_TO = 0.03, 0.062
SITE = uuid.UUID("99999999-9999-4999-8999-999999999999")
E1, E2 = (uuid.UUID(f"99999999-9999-4999-8999-00000000000{n}") for n in (1, 2))
A1, A2 = (uuid.UUID(f"99999999-9999-4999-8999-0000000000a{n}") for n in (1, 2))


def grid(folder: Path, document: str, epoch: str, file: str) -> cell_grid.Grid:
    descriptor = json.loads((folder / document).read_text())["epochs"][epoch]["descriptor"]
    return cell_grid.decode(descriptor, (folder / epoch / file).read_bytes())


def answer(u_to: float, to_verdict: str) -> dict[str, object]:
    heights = [grid(SURFACES, "surfaces.json", e, "surface.bin") for e in ("e1", "e2")]
    graded = [grid(CONFIDENCE, "grids.json", e, "confidence.bin") for e in ("e1", "e2")]
    surfaces = [
        analysis.Surface.of(
            g.finest, ground_reach_m=analysis.ground_reach_m(g.descriptor, g.finest.level)
        )
        for g in heights
    ]
    found = {
        layer: analysis.change_map(
            surfaces[0],
            surfaces[1],
            U_FROM,
            u_to,
            site_projection(LAT, LON, HEIGHT),
            layer=layer,
            confidence_a=graded[0],
            confidence_b=graded[1],
        )
        for layer in analysis.LAYERS
    }
    ground = found["ground"].planes
    read = change_maps.ChangeMapRead(
        policy="change-v1",
        derived=True,
        label=change_maps.CHANGE_LABEL,
        calibration="uncalibrated",
        site_id=SITE,
        from_epoch_id=E1,
        to_epoch_id=E2,
        from_asset_id=A1,
        to_asset_id=A2,
        frame="tileset",
        level=ground.level,
        native_level=ground.native_level,
        cell_m=ground.cell_m,
        threshold_sigma=analysis.THRESHOLD_SIGMA,
        uncertainty=change_maps.ChangeUncertaintyRead(
            from_m=U_FROM,
            to_m=u_to,
            from_verdict="survey",
            to_verdict=to_verdict,
            from_frame_m=0.0,
            to_frame_m=0.062,
        ),
        ground=change_maps._layer(found["ground"], change_maps.GROUND_LABEL, "ground"),
        vegetation=change_maps._layer(found["vegetation"], change_maps.VEGETATION_LABEL, "dsm"),
        appearance=change_maps.ChangeAppearanceRead(
            status="not-computed", reason=change_maps.APPEARANCE_REASON
        ),
    )
    dumped: dict[str, object] = read.model_dump(mode="json", by_alias=True)
    return dumped


def main() -> None:
    for name, u_to, verdict in (
        ("changeYard", U_TO, "survey"),
        ("changeYardCoarse", 5.0, "coarse"),
    ):
        out = Path(__file__).with_name(f"{name}.json")
        out.write_text(json.dumps(answer(u_to, verdict), indent=2) + "\n")
        print(out)  # noqa: T201 -- a script: it says what it wrote


if __name__ == "__main__":
    main()
