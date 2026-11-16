# Test server pieces independently

Run from the repository root with the existing environments. These commands need
no installs, network inference, GPU, or provider key. PostGIS is needed only for
the DB gate and HTTP examples. Synthetic outputs are labelled Simulated; they
prove plumbing and geometry, not recognition accuracy.

Prepare scratch space once:

```bash
mkdir -p .tmp-test
export TMPDIR="$PWD/.tmp-test"
export UV_CACHE_DIR="$TMPDIR/uv"
export UV_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

Run commands sequentially. Swift uses one build job; no command launches more than
four workers. Remove scratch space after the last check:
`rm -rf .tmp-test .swift-scratch`.

## API ask, without PostGIS

```bash
(cd apps/api && uv run --no-sync pytest -q tests/test_ask.py)
```

This file uses fake storage, analysis, models and ledger sessions. It checks
contracts, deterministic tool planning, unavailable analysis, SSE framing, auth,
commit-before-done, recorded replay, and cancellation. The standalone handler
checks call the normal route function with its dependencies supplied directly;
the HTTP tests exercise FastAPI's dependency stack through TestClient.

For just the scripted reasoner, without HTTP or a DB:

```bash
(cd apps/api && uv run --no-sync pytest -q tests/test_ask.py -k 'scripted or handler_runs_alone')
```

`AnswerModel`, `Analysis`, and `Answering` are the injection seams. The stateless
`ScriptedAnswerModel` chooses count/measurement for count or height questions,
plant identification for species questions, invasive lookup for invasive
questions, and segmentation otherwise. It speaks only returned tool evidence,
or says analysis is unavailable. It never initializes a provider client.

## API ask, with PostGIS

Use a dedicated test DB containing no important data: the test fixture truncates
its tables. Set `TEST_DATABASE_URL` to that database, then:

```bash
(cd apps/api && ALEMBIC_DATABASE_URL="$TEST_DATABASE_URL" uv run --no-sync alembic upgrade head && uv run --no-sync pytest -q tests/test_ask.py tests/test_ask_db.py)
```

The DB tests check capture scope, XYZ anchors, CRUD, request conflicts, failed-run
restart, replay-mode persistence and exactly-once entities. They also check mock
mode without a key and cleanup after an unsupported replay question. The
coordinator runs this gate against PostGIS before merge.

Static API gate:

```bash
(cd apps/api && uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync mypy .)
```

## Each analysis module

Each command prints an Entities JSON array and writes real PNG masks beneath
`$TMPDIR/analysis`. It loads the committed float16 depth, poses and sibling
`ground_truth.json` from the synthetic bundle. It never changes fixture files.

Segmentation:

```bash
(cd tools/analysis && uv run --no-sync python -m hex_analysis.cli segment --bundle apps/ios/Fixtures/bundles/yard-small --prompt tree)
```

Plant identification:

```bash
(cd tools/analysis && uv run --no-sync python -m hex_analysis.cli plant_id --bundle apps/ios/Fixtures/bundles/yard-small --prompt tree)
```

Invasive lookup:

```bash
(cd tools/analysis && uv run --no-sync python -m hex_analysis.cli invasive_lookup --bundle apps/ios/Fixtures/bundles/yard-small --prompt tree)
```

Count and measurement:

```bash
(cd tools/analysis && uv run --no-sync python -m hex_analysis.cli count_measure --bundle apps/ios/Fixtures/bundles/yard-small --prompt tree)
```

Use `--frame 0` through `--frame 23`, `--prompt shed`, `--prompt all`, or substitute
`yard-limited`. The analysis CLI deliberately permits limited-tracking frames for
isolated geometry tests; this does not waive the pipeline's tracking gate.
`--output <directory>` selects the mask root. Unknown labels return an empty array.

`--backend fixture` is the CLI default. `FixtureSegmenter` derives depth-grid
instance masks from known synthetic geometry, then the ordinary registry lifts
masked pixels into Entities. `FixturePlantIdentifier` returns an explicitly
synthetic identity with an unknown taxon. Invasive coverage stays unknown for
that identity. `--backend real` uses the same registry interfaces and currently
fails explicitly because inference is unconfigured. This CLI consumes synthetic
bundles only; the existing remote bundle loader is the real capture entry point.

The fixture's depth calibration scales intrinsics directly, with integer sample
centres. Its loader explicitly selects `scaled-intrinsics`; the existing loader
retains its `half-pixel` default. Tests verify all surface probes against ground
truth within 0.005 m, and tree/shed counts exactly, on all 24 frames of both
bundles. Tree height and shed height must be within the declared 0.30 m accuracy
limit, approximately two depth rows at 6 m / 40.8 px focal length. Across all 24
frames, the largest selected axial depth is 6.3828125 m; measured maximum height
errors are 0.203231 m for the tree and 0.090401 m for the shed (rounded up).
These record the fixture's silhouette undersampling against the unchanged limit.
These are observed surface bounds,
not an estimate of unseen volume. The recorded tree height is about 2.56 m;
the synthetic object's full ground-truth height is 2.75 m.

Complete analysis gate:

```bash
(cd tools/analysis && uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync mypy . && uv run --no-sync pytest -q)
```

## Mock reasoning via HTTP

With a local PostGIS catalog configured, start one API process:

```bash
(cd apps/api && API_WRITE_TOKEN=fixture-write ANTHROPIC_API_KEY= ASK_MODE=live ASK_REASONER=mock ASK_ANALYSIS_ENABLED=false uv run --no-sync uvicorn app.main:app --host 127.0.0.1 --port 8017)
```

In another terminal, one question needs no capture:

```bash
curl --fail-with-body -N http://127.0.0.1:8017/v1/ask -H 'Authorization: Bearer fixture-write' -H 'Content-Type: application/json' --data '{"id":"mock-1","question":"How tall is the tree?","keyframes":[],"deviceEntities":[]}'
```

Expected: Looking, Analyzing, speech acknowledging unavailable analysis, then done.
The mock reasoner still calls the analysis interface; disabling analysis makes
that unavailable result explicit. Unit tests supply fake analysis to verify the
successful tool-result path. The default reasoner selection stays unchanged;
without mock/replay, a new question needs the configured provider key.

## Recorded replay via HTTP

Stop the mock API and start replay mode:

```bash
(cd apps/api && API_WRITE_TOKEN=fixture-write ANTHROPIC_API_KEY= ASK_MODE=replay ASK_ANALYSIS_ENABLED=false uv run --no-sync uvicorn app.main:app --host 127.0.0.1 --port 8017)
```

Create a dedicated synthetic capture (no upload is required):

```bash
curl --fail-with-body http://127.0.0.1:8017/api/v1/captures -H 'Authorization: Bearer fixture-write' -H 'Content-Type: application/json' --data '{"name":"Simulated yard-small replay","kind":"video"}'
```

Set `SCAN_ID` to the returned capture's `id`, then:

```bash
curl --fail-with-body -N http://127.0.0.1:8017/v1/ask -H 'Authorization: Bearer fixture-write' -H 'Content-Type: application/json' --data "{\"id\":\"yard-height-1\",\"scanId\":\"$SCAN_ID\",\"question\":\"How tall is the tree?\",\"keyframes\":[],\"deviceEntities\":[]}"
```

The supported questions are `How tall is the tree?` and `How many trees?`
(case insensitive). Use a new ID for the second question. Recordings live in
`apps/api/app/fixtures/ask/*.sse`; their geometry comes from yard-small frame 2.
Their entities omit masks because replay does not publish mask artifacts.
The normal ledger commits the events and synthetic entities before emitting done.
Repeat the identical request to test saved-event replay without duplicating
entities. A changed body with the same ID returns 409. Saved answers remain
replayable after changing modes; new unsupported replay questions return 422.
Always use a synthetic capture: replay entities use yard-small local coordinates.

## Cloud alone

On Linux, source the already-installed toolchain if it is not on PATH:
`. ~/hw-tools/swift/env.sh`. On a Mac, use the installed Xcode toolchain. Then:

```bash
swift test --package-path apps/ios/Modules/Cloud --scratch-path "$PWD/.swift-scratch" --cache-path "$TMPDIR/swift-cache" --config-path "$TMPDIR/swift-config" --security-path "$TMPDIR/swift-security" --jobs 1 --disable-sandbox
```

`DeepAnswering` is the client protocol. `FakeDeepAnswering.yardSmall()` loads a
bundled recording; `FakeDeepAnswering(recording:chunkSize:)` accepts any SSE
recording. `RecordedSSETransport` also plugs into `CloudAnswerer` directly. Tests
compare the server and Cloud copies byte for byte, replay both questions with
one-byte chunks and larger chunks, and check missing done, retries and cancellation.

On Apple platforms, open `CloudDebugView`'s SwiftUI preview or embed
`CloudDebugView()` in a debug app. Its default bundled fake runs alone. Turn off
Use bundled fake to enter the local API URL, write token and synthetic capture UUID.
The view can ask, show status/entities/speech, and cancel. Apple URLSession
streaming and the SwiftUI view still need a Mac/device check; Linux compiles their
conditional exclusion and exercises the parser/client through fake transport.

## Shared checks and current limits

```bash
pnpm format:check && pnpm contracts:check
```

The shared phone request now includes `scope`/`siteId`; the API schema still needs
alignment and regenerated OpenAPI/types outside this job's allowed paths. The
examples above use the compatible request shape without those optional fields.
Property-wide analysis is not implemented by the replay or mock backends.

Some restricted Linux environments cannot complete TestClient cross-thread
wakeups. Run the ordinary tests in an unrestricted coordinator environment if
they stall. Do not patch event loops or add sandbox-specific test gates.
