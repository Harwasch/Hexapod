"""Where a run's published copies live: a generation of keys of their own, per publish.

A run writes its outputs to the private bucket under `runs/<job>/<stage>/...`, and those
keys are **not** written once. A phone's Refine re-runs the *same* job from `train`
(`app/api/v1/phone.py`, `REFINE_FROM`), so `package` and `thumbnail` write the same tile
names again with new geometry; "Retry from this stage" does the same. Publishing used to
copy those keys across unchanged, while every non-JSON key under `runs/` was served as
`immutable` for a year (by `outputs.cache_control_for` on the object, and by the tile
proxy, `functions/r2/[[path]].js`, at the edge). A Refine then left browsers and the edge
holding the preview's tiles under a short-cached `tileset.json` that named the new ones:
geometry from two reconstructions in one scan. And a republish that failed half way had
already overwritten some of the live site's tiles.

So a publish never writes a key a viewer may already hold. It copies into a generation of
its own, `runs/<job>/p<generation>/<stage>/<name>`, and the site is repointed at the new
tileset's URL only once every file of it is there (`app/worker/publish.py`). The live
generation is never touched: a republish that fails leaves the site exactly as it was,
and `immutable` is a promise the bytes keep. The generation is a hash of what is being
published -- each object's key, size and ETag -- so publishing the same bytes again (a
retried `register`, a worker that lost its lease after publishing) lands on the same keys
and the browser's cache stays good; where the store cannot vouch for the bytes (no ETag)
it is random instead, which is always safe. A run that is never re-published keeps one
generation; old generations stay in the public bucket until something removes them (see
docs/DEPLOYMENT.md, "Two buckets, one key scheme").

Only the public bucket has generations. With one bucket nothing is copied, a site points
at the run's own keys, and those keys -- rewritable -- are served with the short lifetime.

This module is the layout and nothing else, so that the API (which never imports
`app.worker`; the artifacts view finds references through it) and the worker agree on it.
`functions/r2/[[path]].js` spells `PUBLISHED_KEY` once more, in JavaScript, and
tests/test_worker_outputs.py holds the two to the same pattern.
"""

from __future__ import annotations

import re

#: Hex digits in a generation: 64 bits, unique per job for as many publishes as it gets.
GENERATION_LENGTH = 16

#: `Cache-Control` for an object that is written once: everything inside a generation that
#: `outputs.cache_control_for` calls immutable, and every object the sidecar attach writes
#: (`app/services/attach.py`). A browser that has one never asks again.
IMMUTABLE_CACHE = "public, max-age=31536000, immutable"

_GENERATION = re.compile(rf"[0-9a-f]{{{GENERATION_LENGTH}}}")

#: A key inside a published generation. The only keys that are written exactly once.
PUBLISHED_KEY = re.compile(rf"^runs/[^/]+/p[0-9a-f]{{{GENERATION_LENGTH}}}/")

#: The same segment anywhere in a key or a URL, for taking it back out.
_GENERATION_SEGMENT = re.compile(rf"(^|/)runs/([^/]+)/p[0-9a-f]{{{GENERATION_LENGTH}}}/")


def published_key(key: str, generation: str) -> str:
    """`runs/<job>/<rest>` as `generation` publishes it: `runs/<job>/p<generation>/<rest>`.

    Only a run's own keys are published this way, so anything else is a programming error
    rather than a key to guess a place for.
    """
    if not _GENERATION.fullmatch(generation):
        raise ValueError(f"not a publish generation: {generation!r}")
    root, job, rest = [*key.split("/", 2), "", ""][:3]
    if root != "runs" or not job or not rest:
        raise ValueError(f"only a run's own keys are published into a generation: {key!r}")
    return f"runs/{job}/p{generation}/{rest}"


def is_published(key: str) -> bool:
    """True for a key inside a published generation: written once, and never again."""
    return PUBLISHED_KEY.match(key) is not None


def unpublished(text: str) -> str:
    """A key or URL with its generation taken out: the run's own key it was copied from.

    The artifacts view finds what a site references by looking for an artifact's key
    inside the site's URLs (`app/services/artifacts.py`); a published URL carries the
    generation in the middle of that key, so the search is made on this instead.
    """
    return _GENERATION_SEGMENT.sub(r"\1runs/\2/", text)
