"""The job worker: claim a queued run, execute its recipe, write what happened.

It lives here, inside `apps/api`, rather than in `tools/pipeline`, and the reason is the
direction of the dependencies. This package already owns the ORM models and the session
factory, so the claim loop needs no second copy of the schema and no HTTP round trip to
write a step transition. A6's rule was that *the pipeline* must not depend on `apps/api`
— not the reverse — and that rule is intact: `tools/pipeline` still has no models, no
session and no API client, and is imported here as a plain library through
`app.worker.pipeline_bridge`, the one module that touches `sys.path`.

Nothing in the API imports this package, so `apps/api` still starts without the pipeline
on disk. Run it with `python -m app.worker`.
"""

from __future__ import annotations
