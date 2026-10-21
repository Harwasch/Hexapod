"""`placement.json`: the similarity `place` applied, written down.

`place_splat` moves the trained splat from COLMAP's frame into east/north/up in up to two
steps -- the georeference's similarity, then (when the scan is recentred) `orient`'s rigid
move -- and records the two as one similarity, so anything that must follow the splat (a
camera, a point picked in the trainer's frame) can be moved exactly as it was.

Composition of similarities: with x' = s1 R1 x + t1 and y = s2 R2 x' + t2,
y = (s1 s2) (R2 R1) x + (s2 R2 t1 + t2).
"""

from __future__ import annotations

from typing import Any

import numpy.typing as npt

from artifacts import ArtifactDecl

PLACEMENT = ArtifactDecl(
    "placement.json",
    content_type="application/json",
    summary=(
        "the similarity place applied, x' = scale * rotation @ x + translation, from the "
        "trained splat's (COLMAP) frame to canonical.ply's"
    ),
)


def compose(
    rotation: npt.ArrayLike,
    translation: npt.ArrayLike | None,
    scale: float,
    then: Any | None,
) -> dict[str, object]:
    """The one similarity equal to `x' = scale * rotation @ x + translation` (translation 0
    when None) followed by `then` (anything with `.rotation`, `.translation` and an optional
    `.scale`, 1 when absent), as `placement.json` records it."""
    import numpy as np  # here, so importing the module stays cheap

    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    t = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
    s = float(scale)
    if then is not None:
        r2 = np.asarray(then.rotation, dtype=np.float64).reshape(3, 3)
        t2 = np.asarray(then.translation, dtype=np.float64).reshape(3)
        s2 = float(getattr(then, "scale", 1.0))
        r, t, s = r2 @ r, s2 * (r2 @ t.reshape(3)) + t2, s * s2
    return {
        "scale": s,
        "rotation": [[float(v) for v in row] for row in r],
        "translation": [float(v) for v in np.asarray(t).reshape(3)],
        "from": "the trained splat's frame (COLMAP)",
        "to": "canonical.ply (east/north/up about the placed origin)",
    }
