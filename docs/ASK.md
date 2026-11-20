# Ask

`POST /v1/ask` (also `/api/v1/ask`) accepts the unchanged shared `AskRequest`.
Use `Authorization: Bearer <API_WRITE_TOKEN>`. Each SSE `data:` line contains one
`AskEvent`: status, entity, speech, or done. Only a successful, committed answer ends
with done. A status followed by EOF without done means the answer was interrupted.

`scanId` is a catalog capture UUID. Optional `scope` defaults to scan.
`scope: property` requires a valid catalog `siteId`; a supplied scan must belong
to that site. A question without a scan can receive
speech, but cannot supply scene evidence or save entities. The reasoner receives the
question, device observations, earlier entities from that capture, keyframe pixels,
mask references and depth summaries. It can make at most eight turns, four tools per
turn. Tool arguments cannot select URLs, credentials or arbitrary code.

Completed request IDs replay the saved event sequence without another provider call.
Reusing an ID with different content returns 409; an ID already running returns 409
with `code: ask_running`. Failed requests may restart. Entities and the replay ledger
commit together before done; interrupted answers save neither. A phone retries only
before receiving an event, preserving the ID and body. Partial streams surface an
error for the caller. A process killed before cleanup can leave a running row: after
confirming its remote work has stopped, an operator must mark it failed before retry.
There is deliberately no automatic takeover that could duplicate paid work.

## Entities

Authenticated CRUD lives at `/api/v1/captures/{capture_id}/entities` and
`.../{entity_id}`. An optional `site_id` query parameter checks the capture's current
site. Lists accept `limit` (1–200) and an `after` entity UUID cursor. Responses wrap
an unchanged Entity with `id`, `scanId`, and `siteId`.

Migration 0039 adds `scan_entities` and `ask_runs`. World anchors are PostGIS
POINT Z in scan-local metric coordinates (SRID 0); optional geographic anchors are
POINT Z, SRID 4326, longitude/latitude/ellipsoid altitude. The full Entity JSONB
retains extent, mask references, confidence, attributes and source. World coordinates
from different captures must not be compared without registration. A phone location
alone does not georeference an object's world anchor. Capture deletion cascades;
site deletion clears the entity's site reference. Captures can acquire a site after
upload: CRUD checks and responses use the capture's current site.

## Analysis seam

`tools/analysis/hex_analysis/registry.py` defines segmentation, plant identification,
invasive lookup and depth count/measurement behind replaceable interfaces. The API
uses a replaceable adapter; ordinary API import needs no analysis package or provider
client. Descriptor parity is tested. Remote bundle access uses a short-lived signed
read URL, kept outside model context. Masks return as bounded artifacts; the API
stores them in the private capture analysis sidecar. Entity mask refs are relative
to `captures/{scanId}/`, not URLs. Uncommitted sidecars may remain after interruption;
retention/garbage collection is an operational follow-up.

The remote wrapper is `tools/analysis/modal_app.py`. It reuses the pipeline bundle
validator. For v1 video bundles, a KeyframeRef image is `frame-000000.jpg` for index
0 (six-digit padding), pose is `frames.jsonl#0`, and depth must match that frame's
logged depth filename. Still uploads are not in the v1 archive format. At most eight
frames are decoded, one decode thread at a time; image context is resized to 1024 px.

Lifting is isolated in `hex_analysis/lifting.py`: row-major camera-to-world, ARKit
camera -Z forward, image Y down, axial depth in metres. Intrinsics refer to original
RGB dimensions. Integer pixel centres map between depth and RGB using half-pixel
boundaries. Backends must supply boolean instance masks on the depth grid. Invalid,
nonpositive or missing depth is omitted; an entirely empty observation produces no
anchor. Extents bound observed surfaces in world axes, not hidden object volume.
Counting uses only the first requested frame and labels its count accordingly; there
is no unverified cross-view instance association.

The location source ships a tiny, explicitly labelled California fixture. Unknown
coverage stays unknown. A synthetic-only CLI backend now exercises masks, lifting
and entities using the committed yard bundles. Real segmentation and plant inference raise
unavailable until dependencies, weights and alignment are validated. The plant
interface defaults to an open-weight backend seam; the actual backend remains an
open decision. No model accuracy or GPU claim follows from synthetic tests.

## Verification limits

Linux covers contract roundtrips, the mocked tool loop/SDK/adapter, mask handling,
cancellation, lifting and the Cloud SSE parser/client with a fake transport.
`ASK_REASONER=mock` enables deterministic tool planning without a key;
`ASK_MODE=replay` serves two recorded synthetic streams through the normal ledger.
See [standalone testing commands](TESTING-SERVER.md) for each CLI, API tests,
curl replay and the Cloud debug view. HTTP TestClient checks can stall in a sandbox
that blocks cross-thread wakeups; run them in the coordinator environment without
event-loop patches.

PostGIS migration/CRUD/replay tests are in `apps/api/tests/test_ask_db.py` and need a
reachable database. The coordinator must run these and the cross review before merge.
Apple URLSession bytes streaming, strict concurrency in the Apple-only branch,
radio cancellation and XcodeGen app integration require a Mac/iPhone. The Modal
image, provider cancellation, ffmpeg against a real phone bundle, segmentation and
plant inference require a separate provider/device verification; none has been run.

## Integrated property scope

Property requests register `search_property_entities` and `search_property_answers`
alongside the scan analysis tools. The site is bound from the validated request and
cannot be changed by tool arguments. Results carry dates and capture IDs; retrieved
scan-local anchors are evidence, never new observations in the current scan's frame.
Mock and recorded replay modes query the same property memory without paid calls.
Their counts describe the returned page of observations, not deduplicated physical objects.

Migration 0040 stores request context/site on `ask_runs` and creates
`property_entity_memory` / `property_answer_memory` views over the real tables.
Entity scope comes from the capture's current site; observation dates prefer its epoch,
then capture time, then save time. Historical answers are available only through the
write-authenticated Ask tools. Public catalog pins use
`GET /api/v1/properties/{site_id}/entities` (`items`, `nextOffset`).

Catalog registration carries the same run's georef, full placement, verified fixed
ARKit-frame provenance and height datum into capture metadata. Every entity save
runs the v2 georeferencing service. Unknown heading/frame/vertical datum leaves geo
absent. A verified ellipsoid placement produces both payload and PostGIS anchors.
Cross-scan dedupe serializes per site, pages all nearby candidates, and saves identity
proposals under `attrs.identity` in the entity transaction. It retains every dated
observation and never averages anchors. Ambiguous matches require review.
