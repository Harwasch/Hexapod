"""Parse bench split/scale/metadata evidence. Never infer a train split from test poses."""

from __future__ import annotations

import math
from typing import Any


def names(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("frame selection must contain a list of names")
    result = [item if isinstance(item, str) else item["name"] for item in value]
    if not result or any(not isinstance(n, str) or not n for n in result):
        raise ValueError("nonempty frame names required")
    if len(set(result)) != len(result):
        raise ValueError("duplicate frame name")
    return result


def choose_test_views(test: list[str], maximum: int) -> list[str]:
    """Shared evenly-spaced subset for ordinary bench and calibration rehearsals."""
    if maximum < 0:
        raise ValueError("maximum test views must be nonnegative")
    count = min(maximum or len(test), len(test))
    if count < 2:
        return test[:count]
    return [test[i * (len(test) - 1) // (count - 1)] for i in range(count)]


def spread_test_views(test: list[str]) -> list[str]:
    """Endpoints, then breadth-first interval midpoints: early prefixes span capture time.

    This is a temporal proxy for spatial spread; no unsupported spatial independence
    claim. All saved held-out names appear exactly once.
    """
    ordered = sorted(test)
    if len(ordered) < 3:
        return ordered
    result = [ordered[0], ordered[-1]]
    intervals = [(0, len(ordered) - 1)]
    for left, right in intervals:
        if right - left > 1:
            middle = (left + right) // 2
            result.append(ordered[middle])
            intervals.extend([(left, middle), (middle, right)])
    return result


def select_frames(
    model_names: list[str],
    selection: dict[str, Any],
    split: dict[str, Any],
    result: dict[str, Any],
    test_report: dict[str, Any],
) -> tuple[list[str], list[str], str]:
    """Prefer saved actual train/test names; otherwise reproduce sorted-pool val split."""
    test = names(selection.get("test", split.get("test")))
    training = selection.get("train", selection.get("training"))
    source = "frame_selection.json: explicit train/test"
    if training is None:
        pool = names(selection.get("pool", split.get("pool")))
        params = result.get("trainParams")
        if not isinstance(params, dict) or "test_every" not in params:
            raise ValueError("no actual train names or recorded trainParams.test_every")
        every = params["test_every"]
        if not isinstance(every, int) or every == 1 or every < 0:
            raise ValueError("invalid recorded training validation split")
        training = [n for i, n in enumerate(sorted(pool)) if not every or i % every != 0]
        source = "split.json pool minus recorded sorted-pool validation stride"
    train = names(training)
    known = set(model_names)
    if set(train) & set(test):
        raise ValueError("training/test overlap")
    if (set(train) | set(test)) - known:
        raise ValueError("selected train/test frame absent from dataset")
    for key in ("val", "validation"):
        validation = selection.get(key) or []
        if validation and set(names(validation)) & set(train):
            raise ValueError("validation frame appears in training camera features")
    if test_report.get("perFrame"):
        measured = set(names(test_report["perFrame"]))
        if not measured <= set(test):
            raise ValueError("test.json contains a frame outside the saved held-out test set")
    params = result.get("trainParams") or {}
    if params.get("pose_opt") or params.get("app_opt") or params.get("bilateral_grid"):
        raise ValueError("optimized pose/appearance requires exported effective-camera adapter")
    return train, test, source


def metric_scale(
    evidence: dict[str, Any],
    *,
    attested_scale: float | None = None,
    attested_source: str | None = None,
) -> tuple[float, str]:
    """Accept explicit unit conversion only; generic georeference.scale is NOT metres/unit."""
    found: list[tuple[float, str]] = []

    def walk(value: Any, path: str) -> None:
        if not isinstance(value, dict):
            return
        for key in ("metresPerUnit", "metersPerUnit", "metres_per_unit", "meters_per_unit"):
            if key in value:
                scale = value[key]
                source = value.get("scaleSource", value.get("source"))
                if (
                    isinstance(scale, int | float)
                    and not isinstance(scale, bool)
                    and isinstance(source, str)
                    and source
                    and source.lower() not in {"unknown", "unresolved", "none"}
                ):
                    found.append((float(scale), f"{path}.{key}; {source}"))
        if value.get("metric") is True and value.get("units") in {"m", "metres", "meters"}:
            found.append((1.0, f"{path}: explicit metric metre coordinates"))
        for key, child in value.items():
            if isinstance(child, dict):
                walk(child, f"{path}.{key}")

    walk(evidence, "evidence")
    if attested_scale is not None and attested_source:
        found.append((attested_scale, f"supplied conversion: {attested_source}"))
    if not found or any(not math.isfinite(s) or s <= 0 for s, _ in found):
        raise ValueError(
            "unknown/nonpositive metric scale; supply metres/unit with its evidence and uncertainty"
        )
    if any(not math.isclose(s, found[0][0], rel_tol=1e-9, abs_tol=0) for s, _ in found):
        raise ValueError("conflicting metric scale evidence; refusing 10 cm patches")
    return found[0][0], "; ".join(source for _, source in found)


def frame_metadata(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    frames = document.get("frames", document)
    if isinstance(frames, list):
        records = {r["name"]: r for r in frames if isinstance(r, dict) and "name" in r}
    elif isinstance(frames, dict):
        records = frames
    else:
        return {}
    result = {}
    for name, record in records.items():
        if not isinstance(record, dict):
            continue
        exposure = record.get("exposure") or {}
        ev = exposure.get("ev100") if isinstance(exposure, dict) else exposure
        result[name] = {"blur": record.get("blur"), "exposure": ev}
    return result
