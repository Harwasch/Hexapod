# Worlds and Land personal-testing release

This integrates `codex/worlds-explorer` (`da4e6afc`) and `feat/land-exploration`
(`b0017b78`) on main `99b12fb7`, including the newer concept-first segmentation
release. It preserves both feature histories, registers both API surfaces and
regenerates one OpenAPI/TypeScript contract. The integration PR is not deployment
authorization; the infrastructure owner performs the release separately.

## Testing scope

- Earth remains the existing globe application. Land is an opt-in tool in that
  application; Worlds is the separate `/worlds.html` product on the same host.
- The personal pilot can use `LAND_AUTH_MODE=pilot` and the existing private
  `API_WRITE_TOKEN`. Worlds also requires that token for its protected manager.
  This does not provide multi-user Worlds isolation. Do not expose the shared
  token through a `VITE_` variable, URL or source-controlled configuration.
- Land's saved boundaries, workspaces, records, scenarios and actions use PostGIS.
  Private documents, rasters and derived assets need the configured private object
  store. Land's browser recovery and Worlds' local library are browser/origin local.
- Land research has its own durable worker. Worlds inference has separate GPU
  workers. Neither runs inside the existing capture worker merely because the API
  and frontend have been deployed.
- Worlds authoring and explicitly labelled interaction preview need no GPU.
  Actual generation needs prepared, licensed model artifacts and a validated
  worker. LTX 2.5 GPU execution, visual continuity, latency and cost are still
  unverified for the Worlds adapter. Living View experiments elsewhere in this
  repository do not validate this adapter.

## Deployment owner: prerequisites

1. Back up the database and rehearse the migration against an isolated copy.
   Land adds revisions `0011` through `0028`. Run `alembic heads` on the release
   revision and apply **all** migrations with `uv run alembic upgrade head` from
   `apps/api`; do not stop at an earlier revision mentioned in historical notes.
   Use the configured database connection, and verify the workspace ownership
   backfill before using existing pilot data.
2. Build the combined API image. Land adds Poppler and Tesseract dependencies to
   `infra/api.Dockerfile`; an older image cannot support the new document paths.
   Configure private storage and the existing API token for the API and research
   worker. Keep originals private and verify authenticated retrieval.
3. Build the web application with `VITE_ENABLE_LAND_EXPLORATION=true` for this
   testing environment. For the GitHub deployment workflow, set the
   dispatch input `enable_land` to `true` before dispatching the release.
   With an API release, this also requires a private R2 database backup and a
   successful restore/migration rehearsal before replacing any application machine. It defaults to false. Preserve the existing API origin,
   Earth/imagery configuration and same-domain routes, including `/worlds.html`.
   The integration does not change production feature-flag defaults.
4. Add an independently supervised research process using the API image:
   `python -m app.research`. It needs the same database/private storage settings as
   the API. Configure `ANTHROPIC_API_KEY`/`ANTHROPIC_MODEL` for model-driven research.
   Basic open-data overview does not need a model key. The Fly configuration includes the separate `research` process group.
5. For Worlds, follow [the separate handoff](WORLDS_DEPLOYMENT.md) and
   [environment template](worlds.env.example). Use **one API replica** for the
   current SQLite ledger design. Mount persistent storage on that API machine,
   set `WORLD_DATA_DIR` to an absolute directory on it, and set
   `WORLD_DATA_PERSISTENT=true`. The capture worker keeps its existing `/data` mount; the API uses its own
   `twin_worlds_data` volume at `/worlds-data`. Multiple independent
   ledgers behind a load balancer are unsupported. If the hosting setup requires
   multiple API replicas, resolve the ledger architecture before enabling Worlds
   managed compute rather than treating separate volumes as shared storage.
6. Keep all `WORLD_*_ALLOW_PROVISION` flags false during initial UI/API testing.
   For generation, connect an already prepared model-specific gateway or have the
   infrastructure owner prepare the approved RunPod template/profile, persistent
   checkpoints, GPU limits, token and streaming/TURN configuration. Enable paid
   provisioning only with the owner's separate budget authorization. Provider keys
   stay on the manager; `HF_TOKEN` is only for preparing gated artifacts.

The integration PR alone applies no infrastructure changes. The separately dispatched
release workflow applies the hosting configuration and migrations described above;
paid GPU workers remain disabled. [Land's detailed guide](LAND_EXPLORATION.md) and
[Worlds' model/runtime guide](WORLDS_MODELS.md) describe remaining acceptance limits.

## Personal acceptance walkthrough

1. Open Earth and verify the existing sites, scan renderer, selection and controls.
2. Enable Land, draw/save/reopen a boundary, edit it and verify revision recovery.
   Add a record or asset, then verify that another unauthorized request cannot
   retrieve it. Run one bounded research task and check evidence, live progress,
   cancellation and reopening after a browser refresh.
3. Open `/worlds.html`, create a project, save/reopen it, and export/import a local
   backup. Test the labelled interaction preview and ensure failed real inference
   never silently becomes preview output.
4. With a separately validated GPU gateway, test text/image launch, typed/voice
   changes, navigation, audio, pause/resume, recording and stop. Confirm the worker
   is cleaned up and the manager ledger survives an API restart. For LTX, observe
   queued/applied revisions and measure generated speed separately from 24 fps
   playback. Stop if spending or lifetime limits fail.
5. Test Land and Worlds in the same browser session, then recheck Earth. Their
   private resources and browser storage must retain the intended access rules.

## Rollback

Disable Land's build-time flag and disable new Worlds provisioning if testing must
stop. End active Worlds sessions/workers through the manager and verify cleanup
before replacing the API process. Retain the persistent ledger and private objects.
Restore the previous application revision only after checking schema compatibility.
Do not blindly downgrade the Land migrations: downgrades can delete pilot data.
Use the rehearsed database backup/restore procedure if a schema rollback is needed.

## Integration checks

The normal CI validates the combined API, migrations, generated contracts and web
application. Playwright runs with Land enabled so its opt-in tests do not silently
skip. A separate CPU-only Worlds job validates gateway lifecycle, transport,
reconstruction service and local launcher contracts, plus browser/worker capability
alignment. It downloads neither model checkpoints nor GPU runtimes, provisions no
workers, and does not deploy. Live provider/model acceptance remains separate.
