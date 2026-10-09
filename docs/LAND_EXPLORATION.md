# Land exploration: implementation record

## Product direction

Land exploration is independent of mission planning. The selected land persists while a
person inspects features, follows evidence, compares scenarios, and eventually chooses an
action. Existing selection and planning interactions are not the design constraint.

Coverage: global baseline with deeper U.S. integrations. Data sources: open data only.
Existing configured model and compute services can still be used. A physical outline,
recorded parcel, study boundary, and evidence of legal rights are distinct concepts.

## Implemented increment: independent land selection

The `feat/land-exploration` branch starts from `edd82cdd`. Its isolated checkout is
`/workspace/Hexapod-land`; the original working branch and deployment are untouched.

- A Land tool, with polygon drawing, OpenStreetMap feature picking, traced corridors,
  and GeoJSON import. Imported polygons retain multipart areas and holes.
- Candidate review, naming and notes, direct vertex editing (all rings), undo/redo,
  saving, reopening, exporting, and boundary revision history.
- Land scope stored separately from feature inspection and mission state. The land
  overlay does not consume clicks intended to inspect underlying features.
- Independent PostGIS land records and immutable revision snapshots. Revision writes
  lock the record and compare the expected version, returning 409 on conflicting edits.
- Metric area/perimeter calculations, metric corridor buffering, and union, difference,
  and intersection API operations. Corridor width means total width.
- Generated OpenAPI and TypeScript contracts. Private pilot land reads carry the existing
  server token; public catalog reads remain anonymous.
- Cross-origin PUT allowed by the API so boundary revision requests can reach it.

This increment is not the full research feature. No generated ecological, historical,
legal, or financial findings are shown in place of missing integrations.

## Preview configuration

Apply migration `0011` to an isolated development/preview database and run the API normally.
Set `VITE_ENABLE_LAND_EXPLORATION=true` on the web development server or build. It defaults
to false, including in development, so current production flows remain unchanged until a
deployer explicitly enables it. Set `VITE_API_BASE_URL` to the preview API when using a
separate port; `API_CORS_ORIGINS` is a comma-separated list, not JSON.

The desktop rail exposes **Land**, also available with Shift+L. On phones it is under
**More → Land**. Saved areas require the API; an unsuccessful save leaves the draft intact
and says that it failed. Boundaries are not falsely labelled as saved offline.

The API namespace is `/api/v1/land`:

| Endpoint                 | Behavior                                            |
| ------------------------ | --------------------------------------------------- |
| GET /land                | Bounded list, limit/offset                          |
| POST /land               | Create a record and its first boundary revision     |
| GET /land/{id}           | Boundary, provenance, metrics, and current revision |
| PUT /land/{id}           | Save a revision with `expectedRevision`             |
| GET /land/{id}/revisions | Read prior snapshots                                |
| DELETE /land/{id}        | Delete the area and its snapshots                   |
| POST /land/corridor      | Buffer a supplied centerline in meters              |
| POST /land/operations    | Union, difference, or intersection                  |

## Limits of this increment

- This is a **single-operator pilot** under the deployment's existing shared token.
  All land endpoints, reads included, require it when configured. Workspace identities,
  membership authorization, private artifact storage, and cross-workspace isolation must
  land before multi-user availability. This token is not a multi-user privacy model.
- Selection is manual drawing, mapped features, or GeoJSON. Cadastral providers,
  conversational selection, line-feature snapping, and other import formats are pending.
- OpenStreetMap outlines describe mapped physical features, not legal parcel boundaries.
- Corridor construction uses PostGIS geography buffering with local projection selection.
  Extents over five degrees are rejected rather than treated as accurate continental
  buffers. Antimeridian-crossing land edges require explicitly split MultiPolygons.
- Imports support up to 20,000 vertices and 5 MB in the browser; direct drawing supports
  2,000 points. Geometry validity is checked by the API before saving.
- The current overview contains measured boundary metrics and provenance. The evidence,
  research, raster-analysis, and scenario systems described below remain to be built.
- A selected area's contents and metrics are retained during a session; reloads reopen
  the saved-area library. Draft recovery across browser restarts is pending.
- Land interaction currently uses the existing panel layout. A resizable research
  workspace, mobile sheet expansion controls, and revised primary navigation are pending.

## Validation

Backend: `tests/test_land.py` covers persistence without missions/sites, revision history,
optimistic concurrency, delete cascade, polygon holes, invalid geometry, total corridor
width, end caps, supported extents, boolean operations, private reads, and CORS preflight.
`tests/test_plans.py` checks compatibility with existing plans.

Frontend: `src/__tests__/land.test.ts` covers independent land scope, draft isolation,
undo across selection sessions, and imports preserving holes. Existing token middleware
tests also cover private land reads. Existing layout and environment tests remain in the
validation set.

Run browser checks with the preview flag enabled:

```sh
VITE_ENABLE_LAND_EXPLORATION=true pnpm --filter @twin/web exec playwright test e2e/land.spec.ts --project=chromium
```

These drive the real map for drawing, saving, inspecting underlying ground, revising,
reloading, importing exclusions, and phone layout. Their API fixture is deterministic.
An additional manual browser smoke used a real isolated PostGIS API for import → save →
history → reload → revise, and verified no page exceptions or horizontal phone overflow.
Remote imagery availability is not established by the fixture tests.

## Remaining full implementation

1. Workspace identity, membership and per-resource authorization; persist private artifacts.
2. Finish unified selection: parcel adapters, natural-language grounding, candidate choices,
   geometry combinations in the UI, snapping, other imports, and recoverable drafts.
3. Redesign the larger land workspace: synchronized overview, persistent conversation,
   evidence inspection, mobile expansion, saved views, and scenario comparison.
4. Evidence, findings, investigations, versioned artifacts, and independent land features.
   Every result pins the land revision and source versions; changed boundaries mark results
   stale rather than rewriting their conclusions.
5. Durable research queue/worker with leases, cancellation, retry, budgets, checkpoints,
   authenticated streaming/reconnection, typed agent tools, and sandboxed computation.
6. Source registry and adapters: terrain/geology, soils, hydrology/hazards, imagery/land
   cover, ecology, historical collections, public land records, infrastructure, and energy.
   Validate endpoint coverage and record-level licenses. Distinguish no records from
   unavailable providers. Add COG tiling, raster sampling, zonal statistics, and vector tiles.
7. Open-ended investigation using actual evidence; persistent map/chart/table/document
   outputs, contradiction handling, and readable methods and uncertainty.
8. Domain workflows: historical georeferencing, land-record applicability, validated species
   assessments and restoration scenarios, infrastructure inventory/inspection/change history,
   and reproducible solar or other financial models with editable assumptions.
9. New action composer inheriting land, evidence, constraints, and scenarios. Preserve old
   plan snapshots through adapters. Keep operational approval/dispatch explicit.
10. Agent evaluations, numerical reference cases, live-provider smoke checks, accessibility,
    scale/performance, migration rehearsal, and the production deployment handoff.

## Integration and release

Coordinate changes to AppShell, SelectionManager, command routing, contracts, and Alembic
heads with concurrent branches. `0011` is this branch's revision; reconcile any migration
head conflict before integration. Existing land-unaware clients continue to use the old
API and plan snapshots. Do not backfill or reinterpret existing plans automatically.

The production deployment remains owned by the deployment agent. Deliver commits, schema
and configuration changes, validation results, remaining limits, and rollback instructions
before that handoff. Disabling the web flag hides this increment without deleting saved
land records. Downgrading migration `0011` deletes those records and should only be used
against disposable development databases; production rollback should retain the tables.
