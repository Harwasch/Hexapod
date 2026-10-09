# Land exploration: implementation record

## Current status

The complete feature is **in progress**, not ready for production handoff. The sections
below record successive increments; their historical limits do not supersede later work.
Apply **all migrations through the current Alembic head**, not just the first land migration.

| Capability | Current implementation | Remaining work |
| --- | --- | --- |
| Land selection | Drawing, mapped parcels/features, grounded command previews, metric corridors, composition, geospatial imports, revisions, reviewed-draft and unfinished-drawing recovery | Broader cadastral coverage, snapping/splitting, large/dateline corridor handling |
| Research | Durable worker, source evidence, five overview adapters, bounded public search, typed artifacts/scenarios, isolated terrain/land-cover raster calculation and private map tiles | Time-series/imagery datasets and broader compute tools, agent evaluations and live model validation |
| Workspace | Scoped records, OIDC/PKCE, roles, resizable/mobile panel, keyboard-accessible Discover/Records/Assets/Scenarios/Actions navigation | Membership UI, saved views, deeper accessibility/performance verification |
| Scenarios | Versioned deterministic solar economics and restoration cover/cost comparisons | Roof/shading analysis and imagery/field-derived species cover |
| Inventory | Versioned features, source deduplication, confirmation, map selection, dated inspections and unit-bearing measurements | Batch imports, geometry editing UX, broader detection and asset catalog linkage |
| Historical and rights workflows | Private PDF/text originals, bounded native/OCR page extraction, original-page viewing, exact private citations, record search, dated document relationships, agent retrieval and licensed photo/historical-map discovery | Saved archive imagery/georeferencing and deeper instrument/parcel lineage evaluation |
| Action planning | Versioned drafts, references, exclusions, steps, costs, constraints, explicit approval and private scheduled-mission handoff; agent draft tool | Fleet execution integration, richer step geometry editing, draft recovery and full acceptance evaluation |

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

Apply migrations through the current Alembic head to an isolated development/preview database and run the API normally.
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

## Historical limits of the first increment

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

## Implemented increment: workspaces and sourced research

The next increment adds OIDC workspace authorization with owner/editor/viewer roles,
workspace membership management, and explicit separation from the shared-token pilot.
Migration `0012` backfills existing land into the pilot workspace; it does not grant those
records to the first OIDC user. Browser sign-in uses authorization code with PKCE through
`oidc-client-ts`. Access tokens remain in the browser session; identity/workspace changes
clear the selected land and invalidate the previous workspace's private query cache.

Migration `0013` adds investigations pinned to immutable boundary revisions, research
messages, runs, ordered events, evidence, findings and typed artifacts. Boundary changes
mark earlier investigations stale. Research reads, evidence retrieval, event replay and
streaming are workspace scoped. Finding/artifact citations are validated against evidence
from their investigation, including references nested in timeline entries.

The independent worker (`python -m app.research`) uses database leases, fencing tokens,
checkpoints, bounded recovery attempts, cancellation and step/time/output-token budgets.
A heartbeat runs independently of provider/model calls. Source snapshots are checkpointed
before publishing outputs to keep retried results attached to the same retrieved evidence.
The overview endpoint is idempotent per land revision. Authenticated SSE supports replay;
the current web UI uses bounded event polling and paged investigation reads.

Three adapters currently retrieve real open data: USGS elevation samples, NASA POWER
regional climate/solar climatology, and compatible-license GBIF occurrence records. Each
retains source attribution, relevance limitations and retrieval metadata. GBIF records with
withheld/generalized coordinates or incompatible licenses are excluded; exact occurrence
coordinates are not displayed. Empty data, uncovered locations and failed providers remain
distinct. The model provider can request registered sources and publish validated findings
and chart/table/map/document/timeline artifacts through typed decisions.

The land panel now includes discoveries, investigation history, activity, evidence inspection,
pinning/dismissal, follow-up questions, and visual outputs. Saving a first area starts its
bounded overview. AI investigation clearly reports missing model configuration; it does not
substitute fabricated answers. Map artifact rendering, conversational selection, broader
sources, domain scenario models and the remaining full-plan work are still pending.

Validation for this increment: 28 focused backend tests passed against isolated PostGIS;
38 frontend regression tests, type checking and targeted lint passed. Two selection browser
tests passed. A real browser/API/worker smoke created an area, completed live source
retrieval and rendered three NASA charts, with no page exceptions or horizontal overflow.
Live model calls and live OIDC identity-provider integration remain unverified because no
provider credentials/configuration are attached to this development environment.

Configuration is documented in `.env.example`. Run migrations through `0013` and the
separate research worker in preview. Production deployment remains with the designated
agent. No production resources have been changed.

## Implemented increment: mapped selection, imports and recovery

Mapped candidate lookup now supports Washington's public parcel service and OSM lines and
buildings. Candidate outlines are interactive overlays; selected lines produce corridors
with total width, end caps and a selectable span of mapped vertices. Natural-language
selection is grounded in candidate IDs, with explicit clarification for ambiguity. Local
width/selection commands work without a model; broader interpretation uses the configured
model. A parcel record remains distinct from verified title. Coverage outside the current
parcel adapter is reported explicitly.

Boundary imports support GeoJSON, KML/KMZ, zipped shapefiles and GeoPackages with bounded
uploads, archive traversal checks, polygon/vertex limits, CRS transformation, layer choice,
and explicit repair preview. Multipart outlines and exclusions are retained. Boundary
review adds union, exclusion and intersection drawing. Drafts are recoverable in the same
browser and scoped by identity/workspace; recovery checks the saved revision before editing.
Explicit sign-out removes that identity's stored drafts. Research map artifacts now render
points, lines and polygons as independent overlays. The desktop workspace is resizable and
the phone sheet can expand.

Validation: 39 focused backend tests and 41 frontend tests pass. Desktop/phone selection and
the new mapped transmission-line instruction journey pass in Chromium with software WebGL.
Public parcel/OSM service smoke results establish those adapters at tested locations, not
universal coverage. Full domain workflows and broader analytical infrastructure remain
outstanding. No production changes have been made.

## Implemented increment: public-source discovery

The agent can now issue bounded public-web searches through the configured model provider's
server-side search tool. It retains provider-returned citation excerpts separately from
metadata-only leads, with unresolved land relevance and explicit reuse-rights limitations.
The application does not fetch arbitrary model-generated URLs. Search queries, reserved
budgets and exact result snapshots are checkpointed; replay deduplicates evidence. Search
and time limits are editable in the investigation UI, including disabling public search.
Conversation evidence references now open the source inspector.

New recovery tests also exposed and fixed mutable JSON checkpoint aliasing and stale
identity-map lease checks: checkpoint values are copied, and lease fencing refreshes the
locked database row before authorizing a write. Fifteen research/search tests pass, with
backend type checks and frontend type/lint checks. Live model/search calls remain unverified
without a configured model key. Search discovery does not complete the planned analytical,
historical, rights, restoration, infrastructure or financial workflows.

## Implemented increment: versioned solar and restoration scenarios

Migration `0014` adds private scenarios and immutable revisions pinned to land boundaries.
Users can preview calculations, save/revise assumptions, compare up to three alternatives,
inspect calculation rows, see sensitivity/cost schedules, and export inputs and results.
Boundary changes mark existing results stale. Stable request identifiers deduplicate creates;
revision checks prevent overwriting newer edits. Evidence references must belong to the land.
The agent can create scenarios through the same deterministic calculation service and receives
saved scenario inputs and summaries as research context.

The solar calculator covers supplied usable module area and plane-of-array irradiation,
module efficiency, losses, degradation, self-consumption/export tariffs, escalation,
maintenance/replacement costs, financing, discounted cash flow, sustained equity recovery,
and capital/yield sensitivity. It does not yet determine roof geometry, shade or plane resource
from imagery/terrain. Tilt and azimuth are recorded inputs; they do not transform horizontal
irradiation. The restoration calculator compares supplied mutually exclusive cover classes
with targets and calculates treatment/monitoring budgets with contingency and discounting.
It does not infer species cover or predict ecological establishment.

Validation: four numerical/API/agent scenario tests pass, including immutable snapshots,
private access, idempotency and low-interest financing. Combined feature tests passed apart
from an outdated expectation corrected and rechecked in the scenario suite. Forty-three
frontend regression tests pass; type/lint/build/contracts checks pass. A live browser/API
smoke calculated, saved, reloaded and reopened a solar scenario without page exceptions or
horizontal overflow. The development preview had no imagery catalog; this smoke does not
establish imagery availability. Full domain workflows and deployment remain outstanding.

## Implemented increment: mapped soils and flood-zone intersections

The source registry and bounded overview now include USDA Soil Data Access and FEMA NFHL.
USDA queries a representative point and reports major soil-map-unit components, drainage,
hydrologic group and representative slope with clear map-unit limitations. FEMA polygons
are clipped to the pinned boundary, retain exclusions, and produce cited map/table outputs
with geodesic intersection areas. Overlapping polygons are not summed into a coverage claim.
Empty, uncovered and unavailable results remain distinct; detailed geometry is bounded.

Thirteen source/worker/search tests pass. A live USDA request returned the Alderwood map
unit and component metadata near the development fixture after a maintenance period ended.
FEMA returned HTTP 503 from its public service, so its live integration remains unverified;
fixture tests validate clipping, exclusions and output contracts. The overview preserves
this outage as unavailable rather than interpreting it as no flood risk.

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

## Implemented increment: inventory and inspection records

Migration `0015` introduces private land features, immutable revisions and dated inspections
pinned to the observed feature revision. Features retain geometry, category, identity status,
attributes and source provenance. A source URL/record identifier cannot be duplicated by
create or edit. Stable request keys make create/inspection retries idempotent. Geometry is
validated and normalized to 2D; numeric attributes and measurements must be finite.

The inventory UI finds nearby mapped buildings/lines or records an inspected point, previews
a candidate, saves and confirms identities, revises details, and displays history. Inventory
features remain visible when the panel closes and selecting them reopens their record.
Inspections record observation time, condition, notes and named measurements with units.
Each selected feature has its own inspection form; changing features does not transfer an
unfinished observation to a different asset. Source confirmation does not establish ownership.

Validation: five database/API tests exercise source deduplication on create/edit, conflicts,
workspace scoping, outside-boundary distances, geometry normalization, dated version pins,
inspection idempotency and measurement validation. Two component tests cover complete unit
submission, stable retry keys and duplicate measurement rejection. A browser journey against
the isolated API creates a synthetic mapped asset, confirms it, records an inspection and
reopens the persisted record after reload. A separate real-Cesium check verifies that clicking
the filled polygon reopens its inventory record after the panel is fully closed. Removing
explicit polygon height/reference restores ground-fill rendering and picking; existing
desktop drawing, mobile import and described-corridor browser journeys all pass. Live-source coverage is separate from this fixture.

## Implemented increment: reviewed land actions and private missions

Migration `0016` adds land actions and versioned snapshots. Each snapshot pins its boundary,
optional scenario, inventory revisions and evidence. Exclusions produce a stored effective
work area; explicit step footprints must fall inside that area. Step dependencies must finish
before dependent work starts. Costs retain their basis, unknown costs remain unestimated,
and constraints require an explicit resolution before approval or scheduling.

The Actions workspace starts from a blank objective or a saved solar/restoration scenario.
It provides editable steps, dates/durations, resources, cost assumptions, success measures,
source selection, inventory references and polygon exclusions. Restoration scenario seeds
include treatment and monitoring costs; year-to-day conversion and placeholder durations
are stated for review. Review shows the sequence, unresolved constraints, known and missing
costs, referenced evidence/scenario results, work area and revision history. Approval records
the exact version. A later edit creates a new unapproved revision; earlier approvals and
scheduled missions remain attached to their earlier snapshots.

Scheduling is a separate user action. The existing plan model now has nullable workspace
ownership: legacy public plans keep null ownership and existing behavior. Land-derived
missions carry their workspace, are read only through scoped action routes, and cannot be
listed, read, changed or deleted through public plan endpoints. They begin as scheduled;
this does not dispatch equipment. The mission carries fixed polygon geometry, exclusions,
steps and source-action metadata. Machine-hour estimates remain unknown and are explicitly
identified as unestimated in the legacy plan representation. Downgrading workspace ownership
is blocked while private missions exist, preventing an accidental privacy downgrade.

The research agent's typed `create_action_draft` tool saves only drafts. Its bounded context
includes known scenario and inventory identifiers/revisions; citations are checked before
persistence. Draft writes share the worker's fenced transaction/checkpoint and stable request
key. The tool surface cannot approve, schedule or dispatch actions. Live model behavior still
requires configured credentials; deterministic worker fixtures verify the draft transition.

The land workspace now separates Discover, Assets, Scenarios and Actions into keyboard
navigable tabs that preserve mounted component state while switching. Selecting an inventory
feature on the map opens Assets. Tabs stay accessible while scrolling a long action review.

Validation: action tests cover immutable history, approval/scheduling gates, stale boundaries
and scenario inputs, polygon exclusions, dependency/cost validation, idempotent scheduling,
workspace isolation and public-route privacy. Existing mission tests pass. Component checks
verify that edits submit only writable fields, save does not approve, and stale/unresolved
work cannot be approved. A browser journey against the isolated API verifies draft creation, explicit approval,
scheduling, reload, and a new unapproved revision that retains the previous scheduled mission;
desktop and expanded mobile layouts have no page errors or horizontal overflow. A permanent
Playwright action journey verifies the UI's separate draft/approve/schedule/revise requests.

The combined feature regression run passes 61 backend tests and 47 frontend tests. All five
migration checks pass, including upgrade/downgrade and model/schema comparison. Migration
comparison now recognizes extension-owned tables through PostgreSQL's catalog, preventing
PostGIS tiger/topology tables on the search path from becoming proposed drops. A private-plan
downgrade guard is tested separately. Existing drawing, mobile import and corridor browser
journeys pass, as do web type checks, lint and the production build. This is increment-level
validation; the full domain and release acceptance matrix is still incomplete.

## Full implementation plan

This is the target for the complete feature, not a claim that the remaining systems are
implemented. Deliver in the order below, with reviewable increments; completion means all
acceptance criteria are met. The existing boundary increment is the starting point.

### 1. Design the whole journey and freeze shared contracts

The primary journey is navigate → define land → discover → investigate → compare → act.
Land exploration must work without a mission, machine, site, or operational project.

- Replace the provisional panel with a resizable desktop workspace beside the map and an
  expandable mobile sheet. Keep the land name, area, boundary revision, and research status
  visible while navigating Overview, Investigations, Scenarios, and Assets. Evidence opens
  alongside its finding without losing the conversation, map position, or active layers.
- Prototype four complete journeys: home with neighboring parcels and exclusions; a
  100-foot transmission corridor; ranch restoration; warehouse rooftop solar. Include
  first use, ambiguous selection, loading, incomplete coverage, errors, and returning users.
- Use a shared artifact vocabulary for map layers, charts, tables, timelines, documents,
  images, comparisons, and downloadable analysis. Every artifact has a title, provenance,
  units where relevant, and a way to return to its investigation.
- Keep map scope separate from transient object inspection. A user can click an individual
  pole, historical photograph, or soil polygon without changing their selected land.
- Extend generated OpenAPI/TypeScript contracts before implementing separate consumers.
  Preserve the React/Zustand state and imperative Cesium manager boundary.

Acceptance: prototypes cover all four journeys on desktop and touch; keyboard users can
select, review, inspect evidence, and return without losing context. No exploration step
requires creating a mission. Prototype fixtures are explicitly identified as examples.

### 2. Establish ownership, evidence, and versioned results

Add workspace identities and owner/editor/viewer membership through standards-based OIDC.
Verify issuer, audience, expiry, and signatures server-side. The shared token remains an
explicit local/single-operator mode, never a bypass for multi-user resource authorization.

| Record | Responsibility |
| --- | --- |
| Workspace / Membership | Ownership, permissions, retention and usage limits |
| LandArea / BoundaryRevision | Named study area and immutable geometry/provenance snapshots |
| Investigation / Message | Persistent question, conversation, and linked outputs |
| ResearchRun / RunEvent / ToolCall | Durable execution, progress, budgets, attempts and checkpoints |
| SourceRecord / Evidence | Retrieved material, license, dates, location relevance and excerpts |
| Finding | Supported claim, citations, uncertainty, geometry and user disposition |
| ResearchArtifact | Versioned typed output plus private storage and rendering metadata |
| Scenario / ScenarioRevision | Assumptions, model version, inputs, results and comparison lineage |
| LandFeature / Observation | Stable physical asset identity, geometry, inspections and history |
| ActionDraft | Versioned operational proposal linked to its supporting investigation |

Results pin boundary revision, source version or retrieval snapshot, method/model version,
and assumptions. Boundary changes mark dependent results stale; they never silently rewrite
old conclusions. Recompute creates new results with visible lineage. Distinguish observed
date, publication date, retrieval date, and analysis date.

Authorize all resource reads/writes, event streams, exports, tiles and signed object URLs.
Private caching includes authorization scope. Retention/deletion handles child records and
stored objects without orphaning accessible copies. Keep the existing public world catalog
separate. Backfill existing pilot areas into an explicitly owned workspace during migration.

Acceptance: cross-workspace isolation tests cover guessed IDs, storage URLs, derived tiles,
cache entries and streams; concurrent edits cannot overwrite one another; old results remain
inspectable after a boundary change. Additive migrations preserve existing plan snapshots.

### 3. Complete seamless land selection

Evolve `state/land.ts`, `LandMapController`, and the land API into one selection state machine:
choose method → resolve candidates → preview → refine → save. Preserve recoverable drafts.

- Drawing: polygon, rectangle, multipart, holes, snapping, vertex insertion/removal,
  union/subtract/intersect/split, undo/redo, keyboard controls and touch handles.
- Records: discover supported cadastral services by location; choose adjoining parcels;
  inspect identifiers, dates and sources; resolve overlapping or conflicting boundaries.
- Features: pick mapped lines/buildings/land features; buffer selected lines with adjustable
  total width, endpoints and end caps. Distinguish selecting a whole network from a segment.
- Imports: GeoJSON, KML/KMZ, zipped shapefile and GeoPackage; detect CRS, preview reprojection,
  preserve holes/multipart, report invalid geometry and offer explicit repair previews.
  Bound archive expansion, geometry complexity and parsing resources.
- Conversation: provide the agent camera extent, visible layer IDs, picked features and draft
  geometry. "This line" resolves to actual candidates. If ambiguous, highlight choices on the
  map. Proposed geometry is editable and never silently replaces the saved boundary.
- Imagery-assisted outlining: reuse the existing vision seam with imagery provenance and
  capture date; present an inferred physical boundary separately from a recorded parcel.
- Geometry: specify metric/geodesic behavior for long corridors, polar regions and date-line
  crossings. Replace current extent restrictions only after numerical reference validation.

Acceptance: the corridor journey requires selecting the line and saying its width, not
manually tracing it; a multi-parcel selection with exclusions survives save/reload/import;
failed lookup or save preserves the draft; every boundary retains its source meaning.

### 4. Build discovery and analytical infrastructure

Define provider adapters with separate discover, retrieve, normalize, analyze and render
operations. The registry records geographic/temporal coverage, resolution, licensing,
attribution, authentication requirements, supported operations, rate limits and health.
Verify each selected endpoint and its terms before enabling it. Open data may still require
a free API credential. Lack of coverage, provider failure, and a valid empty result are
different user-visible states.

| Domain | Candidate source families to validate and integrate |
| --- | --- |
| Parcels and rights records | Local government cadastral/recorder services; relevant public federal records |
| Terrain and geology | USGS 3DEP and geological services; open global DEMs and national surveys |
| Soils | USDA SSURGO; SoilGrids with source uncertainty and resolution |
| Water and hazards | Government hydrography/wetlands; FEMA; NOAA and other regional services |
| Imagery and change | Landsat, Sentinel, NAIP where available; open land-cover products |
| Ecology | GBIF, appropriately licensed observation records, habitat and reference ecosystem data |
| History | Library of Congress, public archives, historical maps and local collections |
| Infrastructure | OSM, compatible Overture datasets, public utility/transportation records and user uploads |
| Energy and climate | Open irradiance/weather datasets and reproducible solar modeling inputs |

Implement clipping, overlays, projected measurements, raster sampling and zonal statistics,
time-series comparisons, nodata/cloud masks and resolution-aware aggregation. Use a
rio-tiler/TiTiler service for COG delivery and sampling; tile large vectors rather than
sending unbounded GeoJSON to the browser. Retain canonical data separately from rendering
representations, consistent with the existing architecture.

Cache permitted source snapshots with checksums, ETags, attribution and expiry. Avoid mixing
private derived products with shared public-source caches. Set retrieval and compute limits
by area, pixel count, feature count and time range, with previews of proposed coarsening.

Acceptance: global baseline providers and deeper U.S. adapters return real cited results;
coverage and unavailable sources are visible; numerical fixtures verify area-weighted
statistics, units, CRS, nodata and time alignment. Live smoke checks are separate from CI.

### 5. Build durable, open-ended research

Add an independent research worker/queue. Existing capture jobs require a capture and must
not become the land research data model. Reuse proven lease/retry concepts from `app/worker`
without coupling research lifecycle to reconstruction pipelines.

- A provider abstraction wraps the existing Anthropic integration and typed tool calls.
  Missing model configuration is an explicit unavailable state, never simulated research.
- Tools inspect map context, propose boundaries, discover sources, retrieve evidence, run
  spatial/temporal analyses, create scenarios and publish validated typed artifacts.
- The agent can formulate follow-up hypotheses and discover additional public sources.
  New sources pass fetch policy, license and provenance checks; retrieved text is data and
  cannot change the agent's permissions. Fetching blocks private/internal network targets,
  revalidates redirects, and bounds response size and duration.
- Flexible analytical code runs in a separately isolated, nonprivileged environment with
  approved input files, no application secrets, controlled network access, and CPU/memory/
  storage/time limits. Validate outputs before persistence or rendering; do not execute
  generated code in the API or unrestricted worker process.
- Persist ordered events, tool inputs/results, checkpoints, usage and errors. Support leases
  with fencing, idempotent output writes, bounded retries, cancel/resume and deduplication.
  Publish authenticated SSE with replay cursors and reconnect support.
- Start bounded overview research after saving land. Deeper investigation follows user
  questions with visible progress and configurable budgets. Cancellation retains completed
  useful outputs. Evidence claims include spatial and temporal relevance, not just a URL.

Acceptance: a user can ask an unanticipated question and receive cited map/chart/table
outputs; a worker restart resumes without duplicate findings; cancellation and exhausted
budgets stop further work; unsupported claims and contradictory sources remain explicit.

### 6. Deliver a compelling overview and investigation workspace

Populate overview progressively: boundary measurements first, then physical/ecological
character, meaningful findings, coverage, assets and suggested investigations. Rank findings
by local relevance, evidence strength, novelty and user interests. Avoid presenting every
available dataset as equally useful.

Each finding supports show on map, inspect evidence, ask a follow-up, pin and dismiss.
Evidence inspection shows exact excerpts/pages, source dates, location match and uncertainty.
Common renderers support legends, units, time controls, nodata, side-by-side scenarios and
exports. Saved views preserve camera, active layers, filters and investigation context.
Charts and maps cross-highlight selected features; the research conversation can reference
these selections. Support accessible non-map summaries for essential information.

Acceptance: users can understand why a finding applies to their land and trace it to the
source; reload restores work; mobile navigation and interrupted requests preserve context;
large outputs remain interactive through pagination, tiling and progressive loading.

### 7. Complete the domain workflows

| Workflow | Required behavior and evidence boundary |
| --- | --- |
| Historical discovery | Search former place/owner names and archive metadata; link photographs and events by defensible location/time matches; georeference maps using control points and show residual errors; label nearby/possible matches |
| Rights and restrictions | Extract documents with page references; track parcel lineage, dates and conflicting instruments; show applicability as established, uncertain or unresolved; never turn a nearby record into a claim of current title |
| Geotechnical understanding | Relate terrain, soil, geology, boreholes and hazards; separate regional estimates from measurements on the land; preserve scale and uncertainty in value/use assessments |
| Ecology | Combine habitat, observations, remote sensing and surveys; distinguish observed species from potential habitat; protect obscured sensitive locations and honor record-level licenses |
| Restoration | Estimate cover/change at supported resolution; define reference ecosystem, targets, treatments, costs and monitoring; request field/imagery evidence where species-level estimates are unsupported; compare versioned restoration scenarios |
| Infrastructure | Import/detect candidate features, deduplicate, confirm identities and maintain asset geometry, attributes, inspections and change history; connect to existing renderable catalog assets without conflating the two models |
| Solar and economics | Model usable roof area, exclusions, tilt/orientation, irradiance, shading, system losses, costs, tariffs and financing; calculate reproducible generation/cash flow/NPV/payback and sensitivity; show missing assumptions and user edits |

Domain tools share the evidence/artifact/scenario framework. New analyses register input
schemas, algorithms, provenance requirements and renderers rather than introducing another
special-purpose chat or opaque report. The agent may compose these tools for new use cases.

Acceptance: each workflow has at least one real-source end-to-end example and a deliberately
insufficient-evidence example; deterministic analyses reproduce their recorded outputs.

### 8. Replace the action-planning experience

An explicit "Plan an action" transition creates a draft from a land revision, selected
findings, scenario, constraints, exclusions and success measures. Present editable steps,
map footprints, resources, schedule, estimated costs and unresolved dependencies together.
Review and approve a specific version; execution/dispatch remains a separate explicit step.
Subsequent land edits flag impacted actions and require deliberate reconciliation.

Adapt existing mission APIs/providers and preserve existing plans exactly as stored. Do not
retroactively attach old footprints to a changing live boundary. Allow returning from an
action to its supporting evidence and alternative scenarios.

Acceptance: restoration and infrastructure journeys reach a reviewable action without
reselecting land or re-entering evidence; existing mission regression tests continue to pass.

### 9. Evaluate, integrate, and prepare the deployment handoff

Maintain a traceable acceptance matrix across the four prototype journeys and all domain
workflows. Add agent evaluations for grounded selection, citation support, source recovery,
contradictions, useful presentation and appropriate handling of insufficient evidence.
Exercise stale revisions, provider outages, worker death, duplicate events, offline drafts,
workspace isolation and large-area limits. Test keyboard, touch, focus management, reduced
motion, screen-reader summaries, WebGL recovery and mobile memory pressure.

Measure selection feedback, shell responsiveness, time to first useful finding and large
layer interaction on declared fixtures/devices. Target immediate local interaction feedback
and a usable overview shell within roughly one second; separately report provider-dependent
research latency. Targets are not achieved performance claims.

Before integration, reconcile shared AppShell/selection/command-routing edits, regenerated
contracts and migration heads against the other branches. Keep the feature flag until the
complete acceptance suite and identity/storage configuration are ready. Rehearse migrations
against a disposable copy and document forward recovery; retaining new tables is the normal
rollback path.

The handoff includes commits, configuration names, migration order, worker/tile-service
requirements, source coverage, test results, live-validation limits and rollback steps.
Production deployment remains with the designated deployment agent. This plan does not
authorize this branch to deploy production or modify other agents' branches.

### Dependency order and definition of complete

Implement contracts/UX (1) and ownership/data (2) first. Selection (3), providers/analytics
(4), and the runtime (5) build on those contracts. Integrate the workspace (6), complete
domain workflows (7), then action planning (8); evaluation fixtures and regression checks
accompany every increment, followed by the integration/release work in (9).

Completion requires working selection, persistent research with genuine evidence, reusable
visual outputs, all listed domain workflows, a redesigned action transition, access control,
and verified recovery paths. A polished selection panel or a chat that only returns prose
does not satisfy the full feature. Live identity/model/provider configuration and external
data availability must be reported separately from implementation and fixture-test coverage.

## Implemented increment: private records and page citations

Migration `0017` adds private land documents, original bytes, extracted pages, and dated
relationships between records. The Records tab accepts PDF and UTF-8 text files, preserves
immutable originals, records source/date/parcel metadata, and supports text search, page
reading, original download, and relationships such as amendments or parcel lineage. A
relationship records a user's evidence and does not establish current legal effect.

Uploads use a bounded raw-byte transfer after metadata creation. Each file is limited to
20 MiB; pending uploads reserve storage against `LAND_DOCUMENT_WORKSPACE_QUOTA_BYTES`
(default 1 GiB). Original bytes live in a separate private database table, avoiding the
public asset-storage path and loading blobs only for downloads. Retries reuse the same
request and document; pending transfers can resume after reload. Unfinished reservations
older than one day are reclaimed when another upload is initiated.

PDF/text extraction runs in a disposable process with memory, CPU and wall-time limits.
It does not execute embedded scripts or follow document URLs. Limits are 500 pages,
20,000 extracted characters per page, and 2,000,000 per document. Text form-feed separators
are the only source of page numbers for plain text. Scanned/empty pages, corrupt files,
and truncation are surfaced explicitly, and the original remains available. OCR is not
implemented in this increment.

The research agent can search private records and read up to three exact pages per tool
call. Evidence carries the document ID, page, immutable original SHA-256, actual extracted
passage, source metadata and unresolved spatial applicability. A private citation does not
need a fabricated public URL. Retrieval checkpoints and evidence IDs survive worker
recovery. The page viewer and action evidence inspector open these private citations; an
"Ask about this page" action prepares a question for user review. Research is never started
merely by selecting a document. Private API responses now carry `private, no-store` and
vary by authorization/workspace.

Validation: 22 document/research/action API tests and five migration checks passed, including
immutable retry behavior, bounded uploads, quota reservations, cross-land/workspace denial,
PDF extraction, missing OCR, exact page citations and a typed agent fixture. Fifty relevant
frontend tests, TypeScript, lint and a feature-enabled production build passed. A real local
API browser journey uploaded a two-page synthetic record, read and linked page two, downloaded
the original, prepared a page-specific question, reloaded and found the page through search;
desktop/mobile checks reported no browser errors or horizontal overflow. The migration
rollback test now ends its read transaction before running DDL on another connection.
Live model behavior still needs configured model credentials. No deployment occurred.

## Implemented increment: scanned records and original-page inspection

Migration `0018` stores immutable page/language OCR readings separately from native PDF
text. Originals and earlier citations remain unchanged. Tesseract reads a single PDF page
rendered by Poppler at a maximum dimension of 2,400 pixels; output is capped at 20,000
characters. The disposable process has memory, CPU, file-size and wall-time limits, and
invokes fixed tools with validated scalar arguments. Unsupported engines/languages or
processing limits produce an explicit error while preserving the original.

The original page can be viewed and zoomed in the Records/evidence viewer through a private
PNG endpoint. Image blobs are revoked when the viewer closes. OCR shows its engine version,
original and extracted-text hashes, extraction date, language, truncation and word-confidence
score. The score is explicitly not a calibrated probability of correctness. Users can select
an extraction for a question; machine text stays labeled as unverified. Text search includes
OCR and opens the exact matching extraction. Repeated extraction requests return the existing
immutable reading. The agent's `ocr_document_page` tool returns the same private page evidence
and pins the extraction ID; its writes join the worker's fenced checkpoint transaction.

The API image now includes Poppler and Tesseract with English, Spanish, French and German
language data; the UI exposes only installed supported languages. API CI installs the English
runtime and exercises an image-only PDF. The local runtime has English installed, so only
English OCR has been exercised here. Other languages require their installed data packs.

Validation includes real OCR and private PNG rendering, immutable retries, unchanged original
bytes/native text, searchable extraction IDs, cross-land denial and a typed agent OCR fixture.
The local browser journey uploads an image-only synthetic PDF, views the original, reads it
with OCR, prepares a question pinned to that extraction, and finds it again after reload.
Desktop/mobile checks report no browser errors or horizontal overflow. A duplicate React key
between the image and OCR components was caught in this journey and corrected.

Full API static checking exposed missing annotations in earlier land-feature tests. Those
tests now use typed clients, sessions, model-tool unions and explicit assertions for optional
results; no type-checking exclusions were added. `mypy .` passes all 226 API source files,
and full API lint/format checks pass. Migration checks and the focused frontend tests pass.
This extends the records workflow; it does not complete raster analysis, archives, domain
analysis, fleet integration or the full acceptance/handoff plan.

## Raster input foundation (historical increment)

`app/analysis/raster_io.py` supplies a read-only range file/opener for Rasterio/GDAL.
It accepts only registered public raster hosts, refuses redirects and auxiliary paths,
checks exact byte ranges and source version validators, and shares request/byte/time budgets
across an analysis. A bounded block cache avoids repeated downloads. The application HTTP
client owns HTTPS trust; GDAL does not perform network access through this opener.

A live Copernicus GLO-30 catalog/COG check returned the requested 32×32 window using
1,638,401 bytes and 26 range requests from a 39,786,033-byte source object. Three deterministic
tests cover seeking/cache/byte buffers, changing versions, servers ignoring Range, unapproved
URLs, auxiliary paths and download budgets. This is an input foundation only: zonal terrain
statistics, private output tiling, map display and other raster domains remain pending.


## Implemented increment: terrain raster analysis and map exploration

Migration `0019` adds immutable, private raster metadata/bytes and typed durable analysis
requests. **Analyze terrain** works without model credentials. The agent can request the
same calculation through `analyze_raster`, read its results and continue an investigation
with the saved source evidence. Both paths use the pinned boundary revision. Results from
older boundaries remain inspectable and are labeled stale.

Copernicus GLO-30 source windows are read through the bounded range transport, resampled
bilinearly onto a local metric grid and clipped using polygon/hole cell-center inclusion.
Surface slopes use neighboring source cells before the boundary mask, avoiding false slope
at exclusion edges. Outputs distinguish missing data from valid zero elevation, report actual
grid spacing and per-band coverage, and retain statistics, percentiles and distributions.
The surface model includes buildings/vegetation and uses the source EGM2008 vertical reference;
it is not surveyed bare-earth terrain, a geotechnical assessment or a local survey datum.
Small areas without analysis cell centers explicitly have no reliable area summary.

A fixed subprocess enforces memory, CPU, file-size and wall-time limits. Source requests are
limited to eight tiles, 64 MiB, 256 range requests and a shared time allowance. Source validators
are pinned across repeated opens as well as within each file. Recovery reuses committed raster
bytes, cancellation fences late results, and source evidence/artifacts use idempotent keys.
No generated code executes. Earth Search's explicit match count handles its extra next-page
link after all matching items have already been returned.

Each private Cloud Optimized GeoTIFF carries named/unit-bearing elevation and slope bands,
coordinate reference, no-data values and algorithm/source/datum tags. Workspace-scoped API
routes provide metadata, original download, point sampling and geographic PNG tiles. Every
route checks workspace access, including images, and returns private/no-store responses.
The default raster allowance is 2 GiB per workspace (`LAND_RASTER_WORKSPACE_QUOTA_BYTES`),
with a 16 MiB per-output bound. Analysis extent is limited to 250 km and the requested grid
may be coarsened to its pixel budget; multipart antimeridian analysis remains future work.

The Visuals view supports measurement selection, an explicit legend, per-band coverage,
opacity, framing, point sampling, distributions, sources/limitations and GeoTIFF download.
The Cesium bridge uses the matching geographic tile scheme and bounded map extent, carries
workspace credentials in request headers, replaces providers after token refresh and removes
private layers when the land/workspace context is cleared. At most two analysis layers are
visible together. A displayed layer retains its measurement/opacity when its view reopens.

Browser verification exposed very slow low-zoom virtual-raster reads. Rendering now reprojects
directly into a fixed 256×256 destination with bounded source arrays/warp memory. Local live
requests at zoom levels 0, 1 and 15 completed in 74, 57 and 26 ms respectively. A public
National Mall fixture produced 504 valid samples at 30 m spacing; a real browser loaded 50
terrain tiles, switched bands, sampled a point, downloaded the file and reopened the saved
analysis after reload. Desktop/mobile inspection found no page errors or horizontal overflow.
The preview has no high-resolution basemap catalog, so this does not validate satellite imagery.

Validation: the 78-test land/backend regression run passed before the final export metadata
and tile refinements; the affected raster suite then passed (10 tests, including typed agent
use, recovery, cancellation, quotas and private cross-workspace denial). Five migration checks
and full API lint/format/type checking passed. The final frontend verification passed 57 focused tests across 12 files, including view
state and token-refresh bridge checks; four existing selection/action Playwright journeys
passed. Frontend type checking, targeted lint and the feature-enabled production build
also passed. No live model calls were exercised because credentials remain unconfigured.
This increment completes the initial terrain path, not the full feature. Archives,
land-cover/time-series analysis, deeper ecology/solar, selection refinements and fleet
execution remain in progress or planned. Production deployment remains with its owner.


## Implemented increment: categorical land-cover exploration

The same durable raster workflow now supports ESA WorldCover 2021 v200. **Analyze land cover**
uses its public tile grid and bounded source-window reads, clips an immutable categorical
COG to the pinned boundary and returns class counts, valid-sample fractions and approximate
sampled hectares. It preserves the eleven official broad class codes and colors. Unknown
codes and missing cells are excluded explicitly. Polygon holes use the same cell-center
mask as terrain; the shared grid constructor retains the existing extent/pixel bounds.

Classification is sampled with nearest neighbors, including COG overviews and map rendering.
Integer class codes are never averaged into fictitious categories. Cesium also uses nearest
texture filtering for these maps. The UI shows class names, color keys and proportion bars,
reports missing-data coverage, decodes sampled point classes and exports a georeferenced,
color-mapped categorical TIFF. Numeric means/percentiles are omitted for categorical codes.
The default direct analysis uses 10 m spacing; larger regions may require coarser sampling,
which is clearly labeled rather than presented as exact source-pixel area totals.

The source is CC BY 4.0 and retains the required ESA/Copernicus attribution. 2021 is the
reference year, not a claim about today's land. These classes do not identify species,
native/invasive status or habitat condition. The agent is instructed to preserve those
limits, and the UI explains that field observations and recent imagery are needed. WorldCover
2020/2021 algorithm changes also mean a difference between the products cannot be treated
as observed land-cover change. Time-series comparison is not implemented by this increment.

Validation: a live public National Mall fixture finished in 6.4 seconds with 4,816 valid
10 m samples and complete sampled coverage. The real API/worker/browser journey queued the
analysis, rendered 34 successful tile requests, sampled a class, downloaded the COG and
reopened the result after reload. Desktop/mobile checks found no page errors or horizontal
overflow. The affected backend suite passed 27 tests (including class preservation, unknown
code exclusion, palette correctness and terrain/research regressions); full API typing
passed 240 files and lint/format checks passed. Frontend categorical tests verify class-name
sampling, visible proportions and the absence of meaningless numeric summaries. The final
58-test focused frontend run, type checks, targeted lint and feature-enabled production build
passed. Production
remains unchanged; this extends the ecology evidence foundation rather than supplying species
coverage or a restoration prescription.

## Implemented increment: open archive discovery

A dedicated archive investigation discovers openly licensed Commons photographs near the
land and historical USGS topographic sheets whose catalog footprints overlap it. The
worker saves source metadata, dates, creator attribution, license, spatial relevance and
file-version references. It creates citation-validated gallery artifacts that reopen with
the investigation and can seed a follow-up question referencing the precise evidence.

The galleries support previous/next source, enlarged preview, source/download links,
location/footprint overlays, and responsive viewing. Catalog coordinates may describe a
camera or subject; sheet footprints are approximate coverage, not image georeferencing.
Original-date text is preserved without substituting upload dates. USGS publication years
are distinguished from surveying, revision and scanning dates. Reuse is limited to known
CC BY/BY-SA, CC0 or public-domain records. Search bounds, excluded licenses and incomplete
coverage remain explicit. Remote previews may change; this increment does not archive image
bytes or visually inspect them with the model.

Validation: 13 API archive/research/worker tests, full API lint/format/type checks (243
source files), 10 focused gallery/raster UI tests, full web typing, land UI lint and a
production build passed. A real public National Mall fixture returned 10 photographs and
12 historical sheets, including an 1890 sheet. Browser checks verified both image providers,
gallery navigation, map footprints, source-question focus, saved results after reload and
mobile layout with no overflow or page errors. The unavailable-image fallback was exercised
as well. Chromium required the environment's existing proxy CA in its test trust store;
TLS verification remained enabled. Production/deployment remains untouched.


## Implemented increment: unfinished drawing recovery

Polygon points and corridor centerlines now persist before a reviewed boundary exists.
Recovery restores the drawing mode, points, corridor width and units, and frames the drawing
on the map. Switching feet/meters preserves physical width. Browser data is scoped to the
current identity/workspace, bounded and validated; the prior reviewed-draft format remains
readable. Starting a deliberate new drawing replaces the stored draft, while browsing alone
does not erase it. Cancelling removes it. Changed or deleted saved land can be recovered as a
separate area without overwriting another revision.

Validation: 36 focused land UI tests, full web typing and affected-file lint passed. All six
selection/action browser journeys passed, including two new reload-and-continue drawing
journeys (the original four and two recovery tests were run separately after correcting the
shared fixture's onboarding reset). No API/schema/deployment change is required.
