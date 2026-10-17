# Land exploration: implementation record

## Current status

The complete feature is **in progress**, not ready for production handoff. The sections
below record successive increments; their historical limits do not supersede later work.
Apply **all migrations through the current Alembic head**, not just the first land migration.

| Capability                      | Current implementation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Remaining work                                                                                                        |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Land selection                  | Drawing, mapped parcels/features, grounded command previews, metric corridors, composition, reviewed line splitting, boundary and visible-feature snapping with retained drawing-guide provenance, geospatial imports, revisions, reviewed-draft and unfinished-operation recovery                                                                                                                                                                                                                                                                                                      | Broader cadastral coverage and large/dateline corridor handling                                                       |
| Research                        | Durable worker, resumable authenticated live progress with incremental polling fallback, source evidence, five overview adapters, bounded public search, typed artifacts/scenarios and reproducible general arithmetic recipes, linked map/list feature inspection with source evidence and pinned agent follow-ups, isolated terrain/land-cover calculations, dated quality-masked Sentinel-2 vegetation comparisons and private map tiles                                                                                                                                             | Broader imagery/mosaics and compute tools, agent evaluations and live model validation                                |
| Workspace                       | Scoped records, OIDC/PKCE, roles, owner membership controls, display profiles, expiring single-use invitation links with explicit joining, resizable/mobile panel, keyboard-accessible Discover/Records/Assets/Scenarios/Actions navigation; shared saved camera/research/imagery views with source validation, current inventory context and persistent captured-save recovery                                                                                                                                                                                                         | Broader temporary-overlay snapshots, live identity-provider acceptance, deeper accessibility/performance verification |
| Scenarios and ecology           | Versioned solar economics linked to immutable hourly weather, orientation, temperature, horizon and inverter calculations; restoration cover/cost comparisons; immutable plot-based species surveys, mapped plots, sampling summaries and scenario references; versioned Catalogue of Life name matching, EPA regional context and USDA soil-linked reference candidates; cited species targets with pinned survey baselines, monitoring protocols and explicit conditional response envelopes; recoverable scenario forms with concurrent-revision review and lost-save reconciliation | Verified local reference communities and calibrated ecological forecasting                                            |
| Inventory                       | Versioned features, source deduplication, confirmation, map selection, dated inspections and unit-bearing measurements; direct map placement and geometry editing with multipart/exclusion preservation, undo/redo and metric previews; reviewed, recoverable GeoJSON/CSV batch imports with duplicate identities and atomic receipts; individual draft recovery, reviewed concurrent merges and lost-response reconciliation; agent reads with private revision/page citations, bounded mapped infrastructure discovery and exact-source candidate proposals                           | Broader detection and asset catalog linkage; live provider/model acceptance                                           |
| Historical and rights workflows | Private PDF/text originals, bounded native/OCR page extraction, original-page viewing, exact private citations, record search, dated document relationships, agent retrieval and licensed photo/historical-map discovery and immutable private image snapshots, bounded agent visual inspection and saved control-point map alignment                                                                                                                                                                                                                                                   | Higher-resolution archive masters and deeper instrument/parcel lineage evaluation                                     |
| Action planning                 | Versioned drafts, references, exclusions, steps, costs, constraints, explicit approval and private scheduled-mission handoff; agent draft tool; browser draft recovery, workflow-state conflict review and lost-save reconciliation                                                                                                                                                                                                                                                                                                                                                     | Fleet execution integration, richer step geometry editing and full acceptance evaluation                              |

## Product direction

A saved Land is the ground mission planning runs on. The selected land persists while a
person inspects features, follows evidence, compares scenarios and chooses an action, and the
same land is what plans, scheduled missions and their proof hang off: its boundary and inventory
features become plannable zones, "Plan work on this land" opens the plan card on it, and a
scheduled action opens in Plan. Plans that touch a land are workspace-private (a database check,
the `/workspaces/missions` routes, never browser storage), and their estimates are sized from
the outlines with every assumption stated. Existing selection and planning interactions are not
the design constraint.

The warehouse-roof/solar example illustrates extensibility; it is not a required first-class
product workflow. Dedicated roof reconstruction, roof surveying and fitted panel layouts are
outside this feature's completion criteria. Existing solar calculations are optional tools
within the general investigation/scenario framework. Prioritize open-ended, evidence-based
research over a catalog of special-purpose product flows.

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
  100-foot transmission corridor; ranch restoration; a user-defined investigation using
  reusable analysis tools. Include
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

| Record                            | Responsibility                                                         |
| --------------------------------- | ---------------------------------------------------------------------- |
| Workspace / Membership            | Ownership, permissions, retention and usage limits                     |
| LandArea / BoundaryRevision       | Named study area and immutable geometry/provenance snapshots           |
| Investigation / Message           | Persistent question, conversation, and linked outputs                  |
| ResearchRun / RunEvent / ToolCall | Durable execution, progress, budgets, attempts and checkpoints         |
| SourceRecord / Evidence           | Retrieved material, license, dates, location relevance and excerpts    |
| Finding                           | Supported claim, citations, uncertainty, geometry and user disposition |
| ResearchArtifact                  | Versioned typed output plus private storage and rendering metadata     |
| Scenario / ScenarioRevision       | Assumptions, model version, inputs, results and comparison lineage     |
| LandFeature / Observation         | Stable physical asset identity, geometry, inspections and history      |
| ActionDraft                       | Versioned operational proposal linked to its supporting investigation  |

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

| Domain                     | Candidate source families to validate and integrate                                       |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| Parcels and rights records | Local government cadastral/recorder services; relevant public federal records             |
| Terrain and geology        | USGS 3DEP and geological services; open global DEMs and national surveys                  |
| Soils                      | USDA SSURGO; SoilGrids with source uncertainty and resolution                             |
| Water and hazards          | Government hydrography/wetlands; FEMA; NOAA and other regional services                   |
| Imagery and change         | Landsat, Sentinel, NAIP where available; open land-cover products                         |
| Ecology                    | GBIF, appropriately licensed observation records, habitat and reference ecosystem data    |
| History                    | Library of Congress, public archives, historical maps and local collections               |
| Infrastructure             | OSM, compatible Overture datasets, public utility/transportation records and user uploads |
| Energy and climate         | Open irradiance/weather datasets and reproducible solar modeling inputs                   |

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

| Workflow                   | Required behavior and evidence boundary                                                                                                                                                                                                                          |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Historical discovery       | Search former place/owner names and archive metadata; link photographs and events by defensible location/time matches; georeference maps using control points and show residual errors; label nearby/possible matches                                            |
| Rights and restrictions    | Extract documents with page references; track parcel lineage, dates and conflicting instruments; show applicability as established, uncertain or unresolved; never turn a nearby record into a claim of current title                                            |
| Geotechnical understanding | Relate terrain, soil, geology, boreholes and hazards; separate regional estimates from measurements on the land; preserve scale and uncertainty in value/use assessments                                                                                         |
| Ecology                    | Combine habitat, observations, remote sensing and surveys; distinguish observed species from potential habitat; protect obscured sensitive locations and honor record-level licenses                                                                             |
| Restoration                | Estimate cover/change at supported resolution; define reference ecosystem, targets, treatments, costs and monitoring; request field/imagery evidence where species-level estimates are unsupported; compare versioned restoration scenarios                      |
| Infrastructure             | Import/detect candidate features, deduplicate, confirm identities and maintain asset geometry, attributes, inspections and change history; connect to existing renderable catalog assets without conflating the two models                                       |
| User-defined analysis      | Compose reusable source, geometry, calculation and scenario tools in response to the user’s question; retain assumptions, provenance, units, uncertainty and reproducibility. Solar economics is one optional example, not a dedicated roof-analysis requirement |

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

## Implemented increment: immutable archive-image evidence

Migration `0020` adds private image metadata and separate original/display blobs attached
to existing source evidence. Editors can save the registered archive preview once; repeated
saves reuse that snapshot. The exact downloaded bytes retain their SHA-256, ETag and
Last-Modified values. A separate checksummed PNG applies EXIF orientation, removes embedded
metadata and defines a stable pixel coordinate system for subsequent analysis. Source
attribution and reuse terms remain on the evidence. These are preview snapshots, not the
full-resolution archive masters; the current USGS catalog thumbnail is too small for fine
map-label reading and precise control-point work.

The fixed downloader accepts only saved registered-source image URLs, rejects redirects,
and bounds source bytes/time. Image decoding runs in a fixed subprocess with memory, CPU,
wall-time, pixel and output limits; it accepts only single-frame JPEG/PNG/WebP. Metadata and
bytes require workspace access, writes recheck membership after external work, and storage
quota/duplicate checks serialize together. Deleting the land cascades to these images.
`LAND_ARCHIVE_WORKSPACE_QUOTA_BYTES` defaults to 512 MiB including both copies.

The gallery saves images, displays private snapshots after reload, exposes checksums in
provenance details, and downloads exact original preview bytes. Its object URLs are revoked
when the source is left. Provider outages preserve the original-source link and do not stop
viewing already saved images.

Validation: 15 archive-image/archive/migration tests passed, including schema round trips,
model/schema agreement, original-byte checksums, EXIF orientation, unsupported/oversized
images, redirect refusal, quotas, repeat saves, workspace isolation and role revocation.
Full API lint/format/type checks (248 files), six gallery UI tests, web typing/lint and a
production build passed. A live public National Mall browser journey saved a 960×640 Commons
photo and a 200×245 USGS map preview, verified the downloaded original checksum, and reopened
both saved images with their public hosts deliberately unavailable. Desktop/mobile screenshots
were inspected; no page errors or horizontal overflow were observed. No deployment was made.

## Implemented increment: agent visual inspection of archive evidence

The research agent can request `inspect_archive_image` for source evidence in its current
investigation. The tool saves or reuses the immutable preview, prepares a bounded visual
input, and attaches actual image pixels to subsequent model decisions. Two images can be
attached at once; selecting a third replaces the oldest attached image. Each input records
its evidence ID, canonical snapshot hash, derived-image hash, processor recipe/version and
dimensions. Derivatives preserve aspect ratio, fit within 1568 pixels, flatten transparency
onto white and stay under 2 MiB. The model adapter sends actual JPEG image blocks alongside
the research context. The gallery's “Inspect this image” action prepares a source-specific
question and focuses the conversation.

Image tools validate investigation citations before any retrieval and require a live worker
lease before saving bytes or checkpoints. Recovery reuses saved source pixels and verifies
that the regenerated derivative exactly matches its pinned metadata; changed processing
requires a new run. Model instructions separate visible details from catalog date/location
claims and explicitly address low resolution and untrusted text inside images. No unsupported
visual interpretation is substituted when the model lacks image capability or credentials.

Validation: 31 image/research/archive regression tests passed; 15 affected image tests were
then rerun after tightening streaming deadline checks. These cover actual pixel attachment,
provider payload encoding, citation persistence, cancellation during retrieval, recovery
without redownload, foreign-evidence refusal, attachment count, aspect ratio/transparency,
processor hash mismatches and small-chunk download deadlines. API lint/format/type checks
(249 files), seven gallery UI tests, web typing/lint and build passed. Model responses were
controlled test fixtures: live AI interpretation remains unvalidated because this environment
has no configured model credentials. No production deployment was made.

## Implemented increment: historical-map alignment

Migration `0021` adds immutable image registrations and private georeferenced display
rasters. Each registration pins the saved image SHA-256, four to thirty labelled pixel/map
point pairs, a request key, notes, the affine transform and local projection, densified
footprint, fit diagnostics, and the exact output GeoTIFF hash. Repeating a request returns
the same saved registration; reusing its key with different inputs returns 409. Refining a
saved alignment creates another record and preserves earlier records and source images.
This workflow accepts saved historical map sheets; it does not treat oblique photographs
or catalog bounding boxes as georeferenced imagery.

The fixed calculation fits an affine transform in a regional azimuthal-equidistant
projection. It checks point uniqueness, image bounds, rank/conditioning, geographic extent,
and extrapolation. Fit RMS, maximum residual, leave-one-out RMS and individual errors are
reported along with the image fraction enclosed by the control points. Warnings identify
sparse coverage, large residuals, mirrored fits and low-resolution previews. Small residuals
measure internal agreement, not surveyed positional accuracy, title, or legal boundaries.
Dateline crossings and polar/large regional images require a separate projection workflow.

The raster processor is a fixed subprocess with memory, CPU, file-size and wall-time bounds.
It preserves RGBA transparency and image-edge coordinates while fitting the display raster
to 1024 pixels on its longer side. Workspace storage accounting includes both image
snapshots and aligned rasters under the same serialized quota. The list, metadata, tiles
and GeoTIFF download enforce workspace access; viewers can inspect and check proposed fits,
and editors/owners can save. Private responses do not enter shared caches. Land deletion
cascades through images, registrations and their bytes.

The archive gallery now offers an alignment editor with image point markers, one-shot
Cesium map picks, typed coordinates, editable matches, fit review, notes, and paged saved
alignments. Complete unfinished matches recover from browser storage scoped to identity,
workspace, evidence and image checksum; sign-out clears them with other land drafts.
Late previews cannot overwrite an edited draft. Saved overlays have show/hide, framing,
opacity and download controls and use authenticated geographic tiles in the existing
bounded raster layer stack. The mobile sheet makes room for the map during a pick and
returns to its prior size afterwards. Escape now cancels a one-shot ground pick before
the shell handles another Escape as a panel close.

Validation: 21 focused backend tests passed across numerical fits, actual subprocess
rendering, archive quota integration, private API access, tile/download bytes, immutable
retries, cascades, migration round trips and model/schema agreement. Full API typing
(257 files), lint and formatting passed. Eighteen focused frontend tests cover matches,
fit invalidation, draft recovery, late-response rejection, map-pick cancellation, archive
regressions and private tile routing. Web type checking, targeted lint and production
build passed.

A live browser/API check used deliberately synthetic point correspondences over the public
National Mall fixture. It exercised an actual globe pick, fit review, save, 53 successful
tile responses, opacity changes, GeoTIFF checksum verification, reload/draft recovery and
expanded-mobile cancellation without page errors or horizontal overflow. The saved test
registration is `f571634e-10b3-503e-a7ef-e6b7ccc0eb73`. Its 0.015 m fit RMS and 0.061 m
leave-one-out RMS describe the synthetic correspondences only, not the historical sheet's
true geographic accuracy. The underlying USGS snapshot is only 200 by 245 pixels, so this
check does not establish precise historical feature matching. Screenshots were inspected
at desktop and 390-pixel mobile widths. Production/deployment and the original checkout
remain untouched; the complete feature is still in progress.

All seven land browser journeys also passed, including selection/import/corridor behavior,
mission handoff, unfinished sketch recovery and the new one-Escape cancellation regression.

## Implemented increment: dated satellite vegetation comparisons

Users can choose one to six observation months and create a durable vegetation investigation
for the saved boundary revision. The research agent's typed `analyze_raster` tool can request
one to six chronological, nonoverlapping windows of up to 31 days. This uses public Sentinel-2
Collection 1 L2A imagery through Element 84 Earth Search v1; it requires no imagery API key.
The source collection and public bucket are pinned, and source redirects or arbitrary band
URLs are not accepted. Existing terrain and land-cover retry/checkpoint keys remain compatible.

Each window queries at most 25 scenes, ranks boundary overlap and catalog cloud fraction,
then examines up to three local scene-classification masks. It selects one acquisition with
the most usable land samples among those candidates. This is a bounded single-scene sample,
not a monthly composite, tile mosaic or exhaustive best-image search. Every result exposes
its actual acquisition date, candidate counts, truncation, quality counts and usable coverage.
Missing observations remain missing. A large area spanning satellite tiles may have partial
coverage; the UI and agent must not present that partial sample as complete coverage.

Red and near-infrared bands use their explicit source scale and offset before calculating
NDVI. Nearest samples on a common local metric grid avoid interpolating cloudy reflectance
into clear cells. Grid spacing is at least 20 m, with polygon/hole cell-center masking and
explicit coarsening when dimension limits require it. Only scene classes 4 and 5 contribute;
water, cloud, shadows, snow, defective pixels, unclassified pixels and invalid reflectance
are excluded. NDVI does not identify species, percent species cover, biomass, habitat quality
or restoration success. Season, weather, observation date and processing differences can
change this signal without establishing a cause.

The timeline compares the same cells valid in every selected observation. Each acquisition
also exposes its separate all-usable-cells mean. The first-to-last change layer uses cells
valid at both endpoints and reports that denominator separately. Map colors have fixed
ranges across dates, and exact values remain available in the timeline and point sampling.
Users can switch between observation/change maps, frame them, adjust opacity, reopen saved
results and download a private multiband GeoTIFF. Exports preserve masks, units, acquisition
metadata, calculation method, warnings, source URLs, catalog hashes, source version headers,
reflectance scale/offset and attribution. The existing workspace authorization, raster quota,
immutable storage, cancellation and worker fencing apply.

The fixed processor has a 128 MiB download budget, bounded HTTP request count, 140-second
analysis budget, 165-second subprocess deadline, 120 CPU seconds, 2 GiB memory limit and
16 MiB output limit. Earlier datasets retain their existing shorter processing limits.
This increment needs no database migration beyond the existing raster storage schema.

Validation: 26 backend regression tests passed across actual synthetic raster reads,
reflectance offsets, local mask selection, common-cell comparisons, missing observations,
GeoTIFF metadata, dated worker persistence, private tile/download routes and legacy retry
compatibility. Full API typing (259 files), lint and formatting passed. Eleven focused web
tests cover calendar windows, month editing, common versus per-date means, missing trends,
map band selection and existing raster behavior. Web typing, targeted lint and production
build passed. A live browser/API/worker run over the public National Mall software fixture
selected actual acquisitions from July 2024 and July 2025. It preserved 12 source records,
compared 842 common clear cells (68.3% of boundary grid cells), and produced 90 successful
private map-tile responses. Band switching, opacity, reload, GeoTIFF checksum and embedded
metadata verification passed. The saved raster is `ff7aab11-efb1-52f6-b995-63c9388b0b83`;
its SHA-256 is `66f9cfc1683e9f33e10db61636532b8baa6fd6be8660f4a6650db180fb313ebd`.
The observed +0.1196 NDVI difference is a result for this fixture and these acquisitions,
not evidence of ecological improvement. Desktop and 390-pixel mobile screenshots were
inspected. A discovered native-select/grid overflow was fixed, then both page and panel
horizontal-overflow checks passed with no browser errors. No production deployment was made.
Live AI selection of this tool still requires configured model credentials; this environment
has none attached. The complete feature remains in progress.

## Implemented increment: field species surveys

Migration `0022` adds immutable dated field surveys pinned to a land boundary revision.
Each survey records the observer, assessed vegetation strata, sampling design and method,
plot polygons, species identifications, observations and method notes. Methods support visual
percent cover and point-intercept counts with explicit sampled-point denominators. Plot
geometry must be inside the pinned boundary, including its holes, and plots cannot overlap.
A claimed census must cover the whole boundary. Regional calculations are limited to five
degrees of extent and latitudes below 85 degrees. Plot area uses the WGS84 ellipsoid.

Species percentages and vertical strata may overlap; they are never normalized to a 100%
composition. Summary means use mapped plot-area weights. Unlisted species remain unknown
unless the observer explicitly records a complete inventory for the assessed strata; even
then, the resulting zero is non-detection, not proof of ecological absence. Each species
summary exposes its own assessed area, plot count, range and identification states. These
are descriptive sample means, not statistical whole-land estimates or confidence intervals.
Point-intercept measurements count points with a taxon, not repeated contacts at a point.

Workspace-scoped preview, create, paged list and read routes live under
`/api/v1/land/{land_id}/surveys`. Viewer roles may inspect/preview; recording requires an
editor or owner. Creates are idempotent, request-key reuse with changed observations is a
conflict, and each record carries a checksum over canonical observations and calculation.
Corrections create a new record referring to the prior survey. Land deletion cascades, and
boundary changes mark existing surveys stale without changing their recorded observations.

Research can retrieve bounded pages of species summaries through `read_field_survey` and
cite a private, immutable survey locator and checksum. Model context lists recent survey
identities and observation dates. Survey notes are untrusted source data, and research
instructions distinguish tentative identifications, plot observations and whole-land claims.
Restoration scenarios can pin field-survey references for the same land/boundary revision;
users must still explicitly interpret observations before entering exclusive cover classes.

The Ecology workspace supports plot picking, incomplete-draft recovery,
measurement entry, reviewed saving, mapped plots, visual summaries, correction copies,
JSON import/export, private research citations and scenario reference controls. Numeric
measurements begin empty and changing methods clears them instead of inventing equivalents.

Validation: 32 backend research/document/scenario/survey regressions and two migration
round-trip/model-agreement checks passed. The ten survey tests were rerun after bounding
agent evidence pages; they include long Unicode names/notes, private source citations,
viewer permissions, source pagination, weighted measurements, missing observations,
corrections, cross-land refusal, stale boundaries and restoration references. Full API
lint/format/typing (264 files) passed. Web type checking, affected lint, focused ecology
and scenario tests, and production build passed. All seven existing land browser journeys
also passed after the workspace navigation change.

Live browser/API validation used explicitly synthetic measurements in a 463.3 m² plot
within the public National Mall software fixture. It exercised actual map corner picks,
reload recovery of an unfinished plot and survey, observation entry, review, immutable
save, map rendering, export, reopening and correction copies. The original exported
record's SHA-256 independently verifies as
`4327b635642779cc37a1e91ae0e00f2fd1e0d76ef7b6e224954d5b9298bab367`.
The corrected copy is `27d9e35e-365c-557f-b9e8-766aefcb47bf`. Synthetic 80% herb and 90%
canopy observations remain distinct; their 170% sum is not falsely normalized. Desktop
and 390-pixel mobile screenshots were inspected, with no browser errors or panel overflow.
The configured preview has no high-resolution basemap, so this check establishes plot
interaction/rendering, not image-to-field survey positional accuracy.

These records do not automatically identify species, verify taxonomy or native status,
estimate unsampled whole-land composition, provide ecological succession predictions,
or establish an appropriate restoration reference ecosystem. Live model-provider validation
still requires credentials absent from this environment. Nothing was deployed; the full
land-exploration feature remains in progress.

## Implemented increment: hourly solar generation and linked economics

Migration `0023` adds immutable private solar assessments with a pinned land boundary,
typed equipment/geometry assumptions, weather/model metadata, and checksummed ZIP archives.
The research queue accepts `kind: solar` with a `SolarRequest`; the same durable lease,
cancellation and recovery rules apply. Saved bytes commit before findings/artifacts so a
recovered worker reuses the original calculation. Workspace storage is serialized and
limited by `LAND_SOLAR_WORKSPACE_QUOTA_BYTES` (256 MiB default). Each output is at most
16 MiB. The fixed subprocess has 2 GiB memory, 45 CPU seconds and 110 seconds wall time;
it executes no generated code and retrieves only the fixed NASA POWER hourly endpoint.

The source is a completed calendar year from 2001 onward, explicitly in UTC, with global,
direct and diffuse solar irradiation, temperature and 10-meter wind. Source units and
location/time metadata are checked. Missing, nonfinite, fill and out-of-range inputs remain
missing; a full annual yield is available only when every expected hour is valid. Leap
years retain 8,784 hours. The source's start-of-hour energy values are converted to hourly
mean irradiance and solar geometry is evaluated at the interval midpoint.

The model uses pvlib solar position, Hay-Davies transposition, SAPM cell temperature,
ASHRAE incidence losses, PVWatts DC and inverter output including clipping. Equipment
and mounting parameters are explicit assumptions. An entered full-circle horizon is
periodically interpolated and masks direct/circumsolar light; isotropic diffuse visibility
and incidence loss use one-degree hemisphere quadrature. Additional uniform shading and
other DC system losses are separate inputs. The open-horizon comparison uses identical
weather/equipment while removing the entered horizon and additional shade.

The mapped array zone must be a valid local polygon entirely within the pinned land,
including holes/exclusions. Projected module face area cannot exceed the zone. This is a
necessary area check, not a fitted panel layout or roof survey. The tool does not infer
roof pitch, structural capacity, module strings, local tree/building shadows, snow,
bifacial behavior or an hourly electricity-load profile. One historical year is not a
forecast, typical meteorological year, or interannual uncertainty analysis.

`/api/v1/land/{land_id}/solar-assessments` lists private results; `/preview` checks the
array against the current land before queueing. `/api/v1/land/solar-assessments/{id}` and
`/download` return metadata and the preserved `request.json`, `result.json`, `source.json`
and `hourly.csv`. Metadata includes source/ZIP hashes, versions, exact retrieval URL,
coverage, monthly energy and assumptions/limitations. No public result URLs are created.

Scenarios now accept `solarAssessmentId`. The server verifies workspace, land, boundary,
complete weather coverage and matching physical assumptions. Cash flow uses the saved
annual AC output directly, without double-applying shading/system losses or inventing an
'equivalent' irradiation. Financial assumptions remain editable; physical changes require
a new assessment or an explicit switch to manual resource assumptions. Scenario results
pin the assessment ID/hash and retain the original simple-model behavior for old scenarios.

Research tools `analyze_solar` and `read_solar_assessment` let the agent calculate a new
assessment or inspect an existing one with cited source evidence. Recent saved assessment
identities are available in context. Reading preserves private assumptions and source
hashes without rerunning weather retrieval. Full zone/horizon arrays remain in the private
archive instead of expanding every model context. Live model-provider acceptance remains
unverified; controlled model tests exercise both tools and citations.

The Scenarios workspace supports map corner picking, saved polygon features, GeoJSON
outlines, scoped unfinished-draft/corner recovery, equipment/horizon editing, server review,
durable job status/cancellation, monthly comparisons, array overlays, source downloads,
saved-result reopening, and linked financial scenarios. Failed queue responses retain an
idempotency key for retry/recovery. Discover renders the same assessment and links into
economics. A four-point horizon is available in the form; the typed analysis interface
supports up to 360 measured horizon points.

Scientific references:

- [NASA POWER hourly API](https://power.larc.nasa.gov/docs/services/api/temporal/hourly/)
- [NASA POWER timestamp FAQ](https://power.larc.nasa.gov/docs/faqs/other/)
- [pvlib Hay-Davies](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.haydavies.html)
- [pvlib SAPM cell temperature](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.sapm_cell.html)
- [pvlib PVWatts inverter](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.inverter.pvwatts.html)
- [pvlib ASHRAE incidence model](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.iam.ashrae.html)

Validation: 30 focused backend/model/worker/scenario/raster/migration checks passed,
including independent diffuse energy balance, leap-year units, orientation, incidence,
temperature, horizon wrap/quadrature, clipping, missing weather, ZIP source preservation,
private access, cancel fencing, quota, recovery without recomputation, old raster request
compatibility, financial assumption matching, partial-year refusal and controlled agent
read/analyze tools. Full API typing (273 files), lint and formatting passed. Six focused
frontend solar/scenario tests, full web typing, affected lint and the production build passed.
All seven existing land browser journeys passed after extending their isolated API fixture
to return an empty solar-assessment catalog.

The live browser/API/worker check used a fictional 100 m² module array in a manually
picked 465.93 m² zone within the public National Mall software fixture. Actual map picks,
unfinished-draft reload, reviewed queueing, weather retrieval, saved output, ZIP download,
exact-yield financial saving and reopening worked. All 8,784 source hours of 2024 were
valid; the fictional assumptions produced 25,919.095 kWh. This is software validation,
not a real installation or roof suitability assessment. The saved assessment is
`8f2878d0-c4d8-55a6-bbe5-985b54d904f5`; the downloaded archive independently verifies as
`fee63bce97234411f2bd6c3b00d03763596da4306403952678643af09b53be17`, and its embedded
weather snapshot verifies as `63c9b3c952d666cf84842de555aba8a052ceece95bdc118f684b2a1b27315a99`.
Desktop and 390-pixel phone views were inspected; reopening and map display had no page
exceptions or internal panel overflow. The preview still lacks high-resolution basemap
imagery, so roof alignment/accuracy has not been demonstrated. Nothing was deployed.

## Implemented increment: ecological context and taxonomic name review

The Ecology workspace now runs a durable, model-independent investigation with selectable
EPA regional ecoregions, USDA soil-linked ecological-class associations, GBIF occurrence
samples and up to twenty scientific-name lookups. Users can copy labels from the displayed
field survey into an editable list, supply a kingdom hint, and review them before querying.
Survey observations remain immutable; name matches never silently rewrite field identifications.
Source choices and pending request keys survive browser reloads and lost queue responses.
Results include source coverage, cited findings, tables, saved investigation selection,
cancellation, stale-boundary labels and a handoff to the same investigation in Discover.
Tables scroll within the card on small screens.

`kind: ecology` carries a typed `ecological-context` analysis request through the existing
research queue. Source snapshots are checkpointed before derived outputs. Recovery preserves
the original retrieved data and citations and does not charge a second step for an interrupted
task. Outages, successful empty responses, geographic noncoverage and budget exhaustion remain
distinct. The normal five-source overview stays bounded; the two regional ecological sources
are available as additional registered agent tools. No database migration is needed beyond 0023.

Taxonomic matching explicitly selects Catalogue of Life Extended Release through GBIF's v2
matching API. It records the matching-service index observed immediately before lookup, rather
than labelling it as the latest catalog release. The saved subset includes supplied name,
matched and accepted usages, synonym status, classification, diagnostic match type and score,
notes, issues and up to five alternatives, with explicit truncation flags. Separately licensed
conservation-status fields and formatted source HTML are excluded. GBIF occurrence queries
also explicitly request the COL checklist and retain its identity. Matching a name does not
verify the organism, local presence, cover, native/invasive status or restoration suitability;
fuzzy, higher-rank and ambiguous matches remain visibly uncertain.

EPA Level IV labels use the public ORD map service's December 2011 regional framework,
compiled at 1:250,000. Queries use the actual polygon, including holes, and return context
labels without fabricated area shares or site-scale habitat boundaries. USDA references come
from major components of one soil map unit sampled at a representative point. Map-unit
percentages are not land-wide proportions. Standard ecological-site identifiers may provide
a candidate description link; the linked document has not been fetched or interpreted.
Empty associations are retained as successful source evidence, not restoration impossibility.
Neither source supplies a verified local reference community or a predictive restoration model.

The agent gains `match_taxon` and `read_research_evidence`. Previous evidence metadata from
the investigation is available in context; reading retrieves the saved source snapshot with
its existing citation. Evidence from another investigation is rejected. This lets follow-up
research review earlier results without relying on titles alone or replacing sources with a
fresh retrieval. Controlled model tests validate both tools; live model-provider acceptance
remains outstanding because this environment has no configured model credentials.

Primary technical references:

- [GBIF taxonomy interpretation](https://techdocs.gbif.org/en/data-processing/taxonomy-interpretation)
- [GBIF Catalogue of Life transition](https://data-blog.gbif.org/post/catalogue-of-life-taxonomic-backbone/)
- [GBIF species API](https://techdocs.gbif.org/en/openapi/v1/species)
- [EPA ecoregion framework](https://www.epa.gov/eco-research/level-iii-and-iv-ecoregions-continental-united-states)
- [EPA public ORD service metadata](https://geodata.epa.gov/arcgis/rest/services/ORD/USEPA_Ecoregions_Level_III_and_IV/MapServer)
- [USDA Soil Data Access service](https://sdmdataaccess.sc.egov.usda.gov/WebServiceHelp.aspx)

Live validation used the public National Mall software fixture, not a private parcel.
Investigation `c0993f83-ad33-49e0-b7c6-346b5ffcbc8e`, run
`618f3a07-a8d9-4a73-a16b-a31915b2416a`, completed successfully. It retrieved the Chesapeake
Rolling Coastal Plain regional label, an empty USDA ecological association response,
twenty usable openly licensed biodiversity records, and an exact name match for Quercus alba.
These observations do not establish current species occupancy. The name matching service
reported `COL26.6 XR`, index `315557`, created July 18, 2026; its saved subset hashes to
`21d339955bfcc2b2f00cafda20283a878f6727ab14307dc68ee86345bd42fe62`.
The browser verified reload recovery, source inspection, saved results and the Discover
handoff. A detected mobile overflow was fixed; the subsequent desktop/390-pixel phone check
had no page exceptions or unintended panel overflow. Nothing was deployed.

Validation also covers typed run requests and idempotency, empty-source evidence, bounded
untrusted responses, polygon hole orientation, taxonomic ambiguity and unresolved synonyms,
step budgets, provider outages, interrupted lookup recovery without refetching, saved-evidence
follow-ups and rejection of foreign citations. Existing research and solar API regressions
passed. API typing (279 files), lint and formatting passed. Ten frontend ecology/survey/solar
tests, full web typing, affected lint, the production build and all seven existing land browser
journeys passed. The original deployment checkout remains clean and untouched.

## Implemented increment: species restoration targets and conditional response

Restoration scenarios now optionally include a species plan alongside the existing exclusive
cover-class and cost comparison. Each taxon/stratum target preserves its baseline basis,
independent target range and assessment year, rationale, supporting evidence, named treatment
links, monitoring method and season, and an off-track response. Species and strata may overlap;
their targets are never normalized into the exclusive cover-class table. Treatment names must
exist, and each assessment year must appear in the monitoring schedule. The species plan
adds no second copy of treatment or monitoring costs.

A target can read its baseline from a pinned immutable survey. The API resolves the exact
recorded taxon/stratum summary and retains the survey hash, observed date, identification
status, assessed area and fraction of the land. That mean remains a sampled-plot result;
it is not extrapolated to the property. A missing taxon is rejected rather than interpreted
as zero. Without a linked survey, an explicit assumed baseline or an unknown value is allowed.
Unknown baselines can have proposed targets but cannot generate response curves. Survey
references must belong to this land and the scenario's boundary revision; all nested source
citations are validated against this land, and agent-created citations against the current
investigation. Previously saved scenario revisions and results remain immutable.

An optional response computes a transparent what-if envelope from supplied asymptotic-cover
and annual-rate ranges, after an entered start year:

`C(t) = C0 + (A - C0) * (1 - exp(-r * max(0, t - start)))`.

The envelope evaluates the parameter-range corners, supports increasing or decreasing cover,
and preserves the baseline before the start. It compares the final envelope with the stated
target range. Parameters start blank in the editor. This calculation is **not a fitted
restoration forecast, inferred treatment effect, statistical confidence interval or probability
of success**. There is no automatic parameter calibration, competition/disturbance model or
claim that regional classifications establish the correct local species community. Empirical
site calibration and ecological forecasting remain open work.

The UI supports reference and target source inspection, a paged title-searchable evidence
picker, choosing an actual taxon/layer from a saved survey, editing treatment links and
monitoring criteria, reviewing cover envelopes and numerical rows, saving/revising scenarios,
exporting assumptions/results and reopening baseline observations. Blank baseline values
remain unknown. Form inputs lock while a calculation or save is in flight. The new private
`GET /land/{land_id}/evidence` catalog returns small scoped metadata records and literal title
search results; full evidence is read through the existing authorized evidence endpoint.

The agent can create species plans through the existing typed scenario tool and inspect
immutable revisions with `read_scenario`. Overview, cover classes, treatments, species targets
and species results have bounded, paged reads, revision identifiers, snapshot hashes and
continuation offsets. Large inputs are omitted from the initial saved-scenario context, and
large create responses point to the read tool. An individual read is capped at 29,000 characters;
original source references retain their investigation scope. This avoids expanding every large
scenario into each model turn. No new migration or external service is required.

Validation: analytical half-time and start-delay checks, declining trajectories, zero-rate
limits, envelopes against interior parameter combinations, independent overlapping strata,
unknown baselines, missing taxa, treatment/schedule validation, survey scope/hashes, immutable
results, nested citation isolation and paged private scenario reads passed. Existing scenario,
field-survey and solar API regressions also passed. Controlled agent tests create a cited plan,
read its bounded pages and reject foreign nested citations. API typing covers 283 files;
frontend typing/lint, nine focused scenario/solar/species UI tests, the production build
and all seven existing land browser journeys passed. Live model-provider acceptance remains unverified.

A browser check saved, exported and reopened synthetic scenario
`1b81361b-8ede-553e-aaa2-91c3c501e1a6` on the public National Mall software fixture. Its
80% baseline comes from synthetic sampled plots covering only 0.0962% of the fixture land,
with survey hash `2d20878d89a7d3384104625bde354818afe530745c44113833a3cc32e0ec459a`.
The entered asymptote/rate assumptions give a year-five envelope of 71.7377–88.2623%; these
are software-test assumptions, not measured or predicted National Mall ecology. Exported
results match the saved result. Desktop and 390-pixel phone views had no page exceptions or
unintended panel overflow. The deployment checkout and production remain untouched.

## Implemented increment: asset geometry editing

Assets can now be placed directly on the map and edited as points, lines, polygons or
multipart polygons. The editor exposes each area and exclusion ring, map-picked or
numeric vertices, insertion/removal, and bounded undo/redo. Rings close on submission;
the closing coordinate cannot drift away from the first vertex. Large geometries page
vertex controls and draw handles only for the active page. Incomplete rings remain
unfilled editing paths until validated. Switching tools cancels the editor's map picker
without clearing another tool's picker.

The workspace-scoped geometry preview validates topology, coordinate bounds, the 20,000
vertex limit and unsupported antimeridian crossings, then reports geodesic area, length,
perimeter and relationship to the current land boundary. Previewing does not save an
asset. Every coordinate edit invalidates its preview; stale responses cannot enable
Apply. Applying changes updates the draft. Saving an existing asset requires a revision
note and retains its original source, prior geometry and pinned inspection revisions.
Outside-land assets remain explicit rather than silently clipped. Selected records are
loaded by ID even when outside the currently paged inventory list.

Validation: seven backend inventory tests pass, including independent geodesic length,
hole area subtraction, private scope, no preview mutation and immutable revised geometry.
Six frontend inventory/editor tests pass, including preserved holes/multipart geometry,
undo, invalidated/stale previews, bounded history and canceled picker ownership. API
lint/format/type checking, web type checking/targeted lint and production build pass.
A Chromium/API smoke on the public National Mall software fixture created a synthetic
polygon with an exclusion, edited the exclusion with undo/redo, reviewed it on desktop
and a 390px phone, saved revision 2 and reopened its history. No page exceptions or
unintended layout overflow were observed. Native text-input scrolling is expected for
long names and is not treated as layout overflow. The preview has no high-resolution
basemap; this verifies interaction and geometry persistence, not surveyed location accuracy.

The successful synthetic feature is `21f1cd39-2667-5889-ae11-4e611c4ef5df`.
Local validation artifacts: `/tmp/land-inventory-geometry-browser-result.json`,
`/tmp/land-inventory-geometry-desktop.png`, `/tmp/land-inventory-geometry-mobile.png`.
No schema migration is needed for this increment. Batch import and inventory draft
recovery remain unfinished; the overall feature remains in progress and undeployed.

## Implemented increment: reviewed batch asset imports

Inventory accepts GeoJSON and CSV point files up to 5 MiB. Users review names, categories,
record IDs, per-row parsing errors and dropped-property warnings, select rows, and preview
candidate locations and outside-boundary distances before saving. GeoJSON polygons retain
holes and multipart areas. MultiPoint and MultiLineString inputs explicitly expand into
individual assets with stable part IDs. Unsupported geometries and non-WGS-84 CRS are
reported; no reprojection or inferred coordinate order is claimed. CSV column mapping
supports quoted fields, escaped quotes and embedded newlines. Limits are 200 assets,
20,000 vertices per asset, 100,000 vertices per batch and an 8 MiB normalized API payload.
Topology is validated by the API before preview or save; invalid selected topology must
be corrected in the source or excluded from the reviewed batch.

The default dataset namespace is the SHA-256 fingerprint of the original uploaded file.
A named dataset and explicit record-ID column support deduplication across changed files;
a shared namespace cannot be selected for rows relying on positional fallback IDs.
External namespace comparisons are case-insensitive, record IDs case-sensitive. Existing
source URL/record identities remain supported. Single-asset writes and batch imports
apply the same duplicate checks. Imported features start as candidates and never replace
existing assets or automatically confirm their identity.

Migration **0024** adds private import receipts. A land row lock serializes concurrent
imports and individual feature writes. Selected creates and the receipt commit in one
transaction; late failure rolls everything back. Retry identity derives from the batch
request and row IDs. Reusing a request with changed input conflicts; retrying the original
request returns its original receipt, even after asset edits or a boundary revision.
Different requests for the same dataset records skip existing assets. Receipt history
retains row-to-asset links, dispositions, boundary revision, filename and the client-supplied
original-file fingerprint. The original upload bytes are not retained on the server.

Unfinished imports, mapping, selections, edits and request keys recover in the same browser,
scoped to identity/workspace and land. Editing inputs invalidates the preview. A failed save
keeps the request key for an exact retry; saving clears the recovered draft. Storage quota
failure is visible. Boundary changes require a new review. The selection E2E fixture now
mocks the new import-history endpoint explicitly rather than returning unrelated land rows.

A Chromium/API smoke on the public National Mall software fixture recovered an edited
upload after reload, rejected an invalid row, preserved a polygon hole, expanded multipart
points, identified an outside-land asset, and created five candidates while skipping one
in-file duplicate. Reimporting the same bytes created no assets and skipped all six valid
rows. Desktop and 390px phone checks reported no page errors or unintended layout overflow;
receipt history reopened after reload. The successful receipts are
`e6da938c-250e-5896-af63-ae480c4341fd` and `a23daafc-9f23-5f90-a2fa-63af8159073b`.
Local artifacts: `/tmp/land-inventory-import-browser-result.json`,
`/tmp/land-inventory-import-desktop.png`, `/tmp/land-inventory-import-mobile.png`.

Migration 0024 upgraded, downgraded and upgraded on the isolated test database, and upgraded
the preview database. Twelve focused backend tests cover previews, rollback, exact retries after
edits/boundary changes, duplicate policies, scoped receipts and concurrent imports. Nine
frontend inventory/editor/import tests pass. API lint/format/type checks, web type checking,
targeted lint and the production build pass. All seven land selection/action browser regression
tests pass after updating the import-history fixture. No production deployment was performed.

## Implemented increment: asset draft recovery and concurrent edits

Individual asset drafts now recover after reload, scoped to identity/workspace and land.
The recovery copy includes fields, original revision, revision note and unfinished geometry,
including blank coordinates. Blank values round-trip as unfinished inputs rather than zero.
Undo history stays session-local, while the current working shape persists. Save/discard
clears the recovery copy; local-storage failures are visible and drafts can be downloaded.
Recovery is size-bounded and validates geometry structure and land scope before rendering.
A changed land boundary is reported and geometry still requires a fresh preview.

Recovered revisions and an explicit "Check for newer asset revision" action read the latest
private asset. The merge preserves another editor's changes to fields the current user did
not edit. Conflicting fields require choosing saved or draft values before saving is enabled.
Unfinished geometry counts as a pending edit when the saved geometry changed. The reviewed
merge updates the expected revision; another concurrent write still produces a 409. Newer
saved geometry and fields are reflected in the selected record, and dataset identity is
visible beside the record's source.

A private read-only request lookup returns both the original creation content and current
asset. This lets a recovered creation distinguish an unsaved request from a save whose
response was lost. A matching completed creation opens the current record even if another
editor subsequently revised it; it does not create another asset or revision. If the local
contents changed after that creation, they recover as edits against the original revision.
Lost revision responses are recognized when the current request key and normalized contents
match. Optional source defaults are normalized before comparison, avoiding false mismatches
between an omitted field and the API's explicit default/null representation.

Validation: 13 focused backend inventory/import tests and 13 frontend inventory/import/
geometry/recovery tests pass. API lint/format/type checks, web type checking/targeted lint and
production build pass. All seven land selection/action browser regressions pass. A real
Chromium/API test on the public National Mall software fixture recovered an unfinished line
without changing its blank longitude, completed its preview, reviewed a concurrent name
conflict, saved the merged revision and deliberately dropped a successful creation response.
After reload, recovery opened that saved creation with exactly one POST and no second write.
Desktop and 390px phone checks found no unintended overflow or page exceptions. The
normalization check was also tested against a creation subsequently changed by another editor.

The browser fixture revised asset `71d73d47-6e97-516d-b19b-c4ae2600c4eb` to revision 5 and
recovered creation `79f811f6-8f99-5287-8369-2f6068c9f8e4`. Local artifacts:
`/tmp/land-inventory-recovery-browser-result.json`,
`/tmp/land-inventory-recovery-desktop.png`, `/tmp/land-inventory-recovery-mobile.png`.
This increment needs no new migration beyond 0024. Production remains untouched; the overall
feature still has the remaining capabilities and acceptance work listed at the top.

## Implemented increment: agent inventory research and mapped point assets

The research agent can now search OpenStreetMap building ways, line ways and infrastructure
nodes near an explicit point or the pinned boundary's representative point. Each query is
limited to 2 km and the existing bounded provider response. Point lookup also appears in
Assets as "Poles, towers and equipment". Results retain node/way identities, source URLs,
physical-feature meaning and category tags. A search is not a complete inventory; building
relations are not queried, and provider outages remain distinct from empty results.

Research retains at most 20 complete candidate snapshots of at most 29,000 characters each,
reports omitted candidates and source truncation, and publishes an evidence-linked map.
Oversized geometry is omitted explicitly rather than clipped into a different candidate.
The agent may propose a retained source as an unconfirmed asset: the server checks the
snapshot hash and source identity, takes its exact geometry, validates investigation scope,
and preserves the OpenStreetMap external identity. Existing matching records are returned
unchanged, including confirmed status and revision. Proposals and their tool checkpoint
commit together; this tool cannot confirm, revise, retire or dispatch an asset.

The agent can page private inventory metadata, then read a specific revision's overview,
geometry, attributes or inspections. Geometry pages retain polygon parts, exclusion rings,
vertex indices and closing coordinates. Inspection pages preserve units, feature revision
and an explicit recording cutoff; oversized measurement payloads are marked omitted. Each
specific read creates a private citation containing the exact bounded page, revision and
hash. The source viewer distinguishes that saved revision from its "Open current asset"
action. Inventory records do not independently establish surveyed accuracy, condition or
ownership, and boundary relationship metrics identify the current land revision they use.

Inventory pages and mapped source snapshots are checkpointed before saving evidence. Retry
therefore preserves the retrieved page after a concurrent edit, and resumes publication of
a mapped source without a second provider request or duplicate asset. Cross-land asset
reads and evidence from another investigation cannot be used to copy candidates.

Validation: nine new inventory research tests pass, including scripted agent search/map/
proposal/read, confirmed duplicate preservation, foreign-land/evidence rejection, exact
geometry/hash checks, inspection pagination, source outage/empty distinction, and both
inventory and mapped-source interruption recovery. Existing research, selection, inventory
and batch-import regression suites also pass. API lint, formatting and type checks, web
type checking and targeted lint, generated contracts and the production web build pass.
All seven land browser regressions pass. A desktop and 390px phone browser check verified
the point lookup option, power category inference and editable candidate preview with no
page exceptions or unintended overflow. Its clearly labeled synthetic source was supplied
by a browser route; no inventory record was written. Screenshots were visually inspected.

A live request using the public National Mall software fixture returned `unavailable` from
the source adapter; live point retrieval is not verified. Model decisions were scripted in
tests because this environment has no configured model credentials. These limits are not
claims of successful live AI acceptance. Local validation artifacts:
`/tmp/land-agent-inventory-public-source.json`,
`/tmp/land-agent-inventory-ui-result.json`,
`/tmp/land-agent-inventory-ui-desktop.png`, `/tmp/land-agent-inventory-ui-mobile.png`.
The isolated preview API and worker were restarted with this code. No new migration beyond
0024 is required. Production and the original checkout remain untouched; the full feature
is still in progress.

## Implemented increment: scenario draft recovery

Solar finance and restoration forms now preserve unfinished assumptions in browser storage,
scoped to the current identity/workspace and land. This includes linked hourly assessments,
field-survey and evidence references, species targets, response parameters, and the exact
unfinished monitoring-years text. Blank numeric inputs remain blank instead of becoming
zero. Non-finite draft values use an explicit local serialization marker; they are never
accepted as calculated results. Recovery checks the saved structure and land scope, limits
size, and requires a fresh calculation before saving.

A recoverable draft is offered before creating or editing another option. It can be recovered,
downloaded, deliberately copied into a new scenario, or discarded. Storage failures preserve
the in-memory form and offer a download. Invalid recovery data remains available for download
instead of being silently replaced. Drafts clear after successful saving or explicit discard.
Changing identity/workspace or land remounts the editor under its own recovery key.

The private read-only `GET /land/{land_id}/scenarios/requests/{request_key}` endpoint returns
the revision written by that request and the current scenario. Recovery recognizes a completed
creation or revision even if its response was lost and another editor later changed it, without
issuing another write. Revision requests now carry the browser's stable request key. Ambiguous
reused keys are reported rather than choosing an arbitrary revision.

Recovery and server conflict responses compare the current scenario revision with the draft's
base. A newer saved version is shown alongside the draft's name and boundary reference;
its assumptions can be inspected. Saving over it is disabled. The user may keep local assumptions
as a separate scenario or explicitly load the latest saved assumptions. Neither choice silently
merges ecological or financial assumptions. Boundary references remain pinned until explicitly
changed. Long survey controls and conflict details fit the phone panel.

Validation: five backend scenario tests and eleven frontend scenario/restoration tests pass,
including request lookup scope and original/current revision separation, unfinished species
and monitoring input recovery, lost-save recognition after a subsequent revision, explicit
concurrent-copy behavior, invalid stored data retention and storage-quota failure. API lint,
formatting and type checks, web type checking/targeted lint, generated contracts and the
production build pass. All seven land browser regressions pass.

A real Chromium/API test on the public National Mall software fixture recovered blank
numeric and unfinished monitoring inputs, recalculated the retained ecological plan, received
a concurrent revision conflict, preserved the draft as a separate option, and deliberately
dropped the successful creation response. Reload recovery recognized the saved request with
exactly one POST and no second write. The check exposed and fixed phone overflow from long
survey selectors; desktop and 390px phone checks then found no unintended overflow or page
exceptions, and screenshots were inspected. It advanced synthetic scenario
`1b81361b-8ede-553e-aaa2-91c3c501e1a6` to revision 4 and recovered new synthetic scenario
`6b6f8c35-a051-5f13-a465-d3024d5bf0e5`. Local artifacts:
`/tmp/land-scenario-recovery-browser-result.json`,
`/tmp/land-scenario-recovery-desktop.png`, `/tmp/land-scenario-recovery-mobile.png`.
No new migration is required beyond 0024. The isolated preview API was restarted; production
and the original checkout remain untouched. The overall feature is still in progress.

## Implemented increment: action draft recovery and workflow-state review

Unfinished actions now recover within the current identity/workspace and land. Browser drafts
retain objectives, steps, dependencies, resources, costs, assumptions, pinned scenario/asset/
evidence references, exclusions and complete step footprints. Blank timing fields remain
blank; incomplete local values are not accepted as saved schedules. Recovery validates the
structure and geometry, enforces a bounded file size, and preserves invalid files for download.
Storage failures keep the form editable and offer a download. Approval notes, scheduling notes
and scheduling requests are not part of the recovered draft.

The private read-only `GET /land/{land_id}/actions/requests/{request_key}` lookup returns the
revision saved by a request and the current action. Lost-response recovery compares normalized
editable content, including default fields and trimmed resource/assumption lines. An already
saved draft opens the current action without another write, approval or scheduling operation.
The selected action is also read by ID, so recovery is not limited to the current catalog page.

Recovery checks both revision and workflow state. Approval or mission handoff can change
without a new content revision; these changes now trigger review before saving the local draft.
Concurrent revision conflicts retain local work. Users can preserve their work as a separate
action or explicitly load the latest saved plan into a new draft revision. Current work is
presented as a readable sequence with timing, success measures, costs, resources, prerequisites,
constraints and a map action. Recovery never replays approval or scheduling. Creating a new
revision continues to preserve prior approved/scheduled revisions and their mission records.

Migration 0025 adds non-unique land/request-key indexes to action and scenario revision tables.
Legacy reused keys remain intact; ambiguous recovery requests are reported instead of selecting
an arbitrary revision. The test database passed an upgrade/downgrade/upgrade through 0024/0025,
and both test and isolated preview databases are at 0025 with both indexes verified.

Validation: twelve backend action/scenario tests pass, including private request lookup,
original/current revision separation, retained mission identity, and unchanged approval state
on reads. Five frontend action tests cover unfinished timing/text recovery, lost-save recognition,
workflow-state conflict detection and absence of automatic approval/scheduling requests. API
lint/format/type checks, web type checking/targeted lint, generated contracts and the production
build pass. All seven land browser regressions pass; the action fixture now supports individual
record reads.

A real Chromium/API test created a clearly labeled synthetic draft on the public National Mall
software fixture, recovered unfinished timing and resource lines, detected explicit fixture
approval of the original revision, preserved local work as a separate draft, and deliberately
dropped its successful save response. Recovery made exactly one creation POST and no browser
approval or scheduling requests. The original action remained approved at revision 1 and the
new action remained a draft with no mission. Desktop and 390px phone checks found no page
exceptions or unintended overflow. A further read-only browser check verified the readable
conflict summary and phone layout; its screenshot was inspected.

Original synthetic action: `beaaaa26-ccf7-5a90-bbba-3c91ec7144a7`.
Recovered synthetic draft: `30be2625-cd2c-5613-9129-652d31852ebb`.
Local artifacts: `/tmp/land-action-recovery-browser-result.json`,
`/tmp/land-action-recovery-review-result.json`,
`/tmp/land-action-recovery-desktop.png`, `/tmp/land-action-recovery-mobile.png`.
The isolated preview API was restarted. Production and the original checkout remain untouched;
the full feature is still in progress.

## Implemented increment: workspace people and invitation links

The workspace panel now exposes display profiles and owner-controlled membership management.
Owners can change roles, explicitly remove access, create viewer/editor invitation links, and
revoke unused links. The last owner cannot be removed or demoted; promoting another owner is
an explicit role change. Display names are chosen by the account holder and are not represented
as verified identities. Full account identifiers remain available to distinguish duplicate names.
Workspace catalog refreshes synchronize changed roles and clear inaccessible selected workspaces.
Membership mutations pin their workspace header so switching workspaces cannot redirect a write.

Invitations use 256-bit random tokens, store only SHA-256 hashes, expire after 1–30 days, and
admit one authenticated person. Owners can page through prior invitations. Acceptance and
revocation serialize with membership changes under the workspace lock. Repeating a successful
acceptance after a lost response returns the membership; removing the person prevents reuse.
An existing member keeps their role and does not consume another person's invitation. OIDC
identity is required; the shared pilot token cannot create individual profiles or redeem links.

The browser receives the token in a URL fragment, removes it from the address, and keeps it in
session storage through sign-in. API inspection and acceptance send it in a request body; tokens
are excluded from query keys and invitation listings. A native modal provides focus containment,
keyboard cancellation and a phone layout. The recipient reviews the workspace and role and
explicitly joins. A missing profile is collected before acceptance. No email or external message
is sent. Invitation creation displays its link once; after a lost creation response, the owner
can refresh the list, revoke the uncertain invitation, and create a replacement.

Migration 0026 adds workspace profiles and invitations. The test database passed a 0026 → 0025
→ 0026 roundtrip; both test and isolated preview databases have been upgraded to 0026. The
isolated preview API was restarted. Production and the original checkout remain untouched.

Validation: seven backend workspace tests cover role isolation, last-owner protection, profile
validation, hashed storage, expiry/revocation, retry behavior, removed-member replay, pilot
rejection, and competing acceptances using independent database sessions. Five focused frontend
tests cover explicit joining, redirect persistence, role changes, pinned workspace headers and
removal/role refresh; the existing API-client/capture regression suite also passes (23 combined).
API lint/format/type checks and generated contracts pass. Browser checks with explicitly mocked
OIDC state and workspace responses verified role changes, link creation/revocation and explicit
joining on desktop and a 390px phone, with no page exceptions or unintended horizontal overflow.
Screenshots were inspected. These checks do not establish a live external identity-provider login.
Local artifacts: `/tmp/land-workspace-ui-result.json`, `/tmp/land-workspace-desktop.png`,
`/tmp/land-workspace-mobile.png`, `/tmp/land-workspace-invitation-mobile.png`.
Web type checking, targeted lint and the production build pass, along with all seven land
browser regressions. The build retains the existing PlayCanvas worker externalization warnings.
The complete land exploration feature remains in progress.

## Implemented increment: saved land exploration views

Saved exploration views are workspace-scoped records on each land area, independent of site
camera bookmarks and mission planning. They retain camera position/orientation, workspace tab,
selected investigation/asset/survey/solar record, live inventory visibility, up to twenty research
map references, and up to two raster or historical-map alignment references with band/opacity.
Editors and owners save, rename and delete; viewers can list and open. Listings are paginated.
Saving uses a stable request key and an immutable captured payload for same-session retries;
renames and deletion require the expected version. Removing a view leaves the source records
untouched. Saved views preserve source references rather than accepting replacement geometry
or arbitrary tile URLs from the client.

Opening resolves every referenced source within the same land. Unavailable sources are omitted
with explicit notices. Earlier boundary revisions are identified while the current land boundary
remains visible; old research/raster boundary references are also disclosed. Chosen raster bands,
opacity, bounds and attribution are restored from stored source metadata. Inventory selections
reopen current records and are described as live. Temporary map-editing overlays and unfinished
forms remain separate and are not silently discarded; the save form lists unsupported overlays.
Opening is disabled during a map pick, and a response for land the user has left cannot move
the camera. The camera flight API now accepts an optional roll for saved orientations; ordinary
flights still default to a level horizon.

Migration 0027 adds saved views. Both isolated test and preview databases are at 0027, and the
test database passed a 0027 → 0026 → 0027 roundtrip. Existing OpenAPI schemas and paths are
unchanged; generated contracts add the saved-view models and routes.

Validation: four backend tests cover scope/roles, immutable map references, invalid bands,
missing sources, opacity/bounds, boundary revision notices, idempotent retries and rename/delete
conflicts. Three frontend tests cover captured retries, retained unfinished overlays, restored
layers/camera and stale land-switch responses; five related frontend suites pass (44 tests).
API lint/format/type checks, web type checking/targeted lint, production build and all seven
land browser regressions pass. The build retains the existing PlayCanvas worker externalization
warnings.

A real Chromium/API check used the explicit public National Mall software fixture and its saved
slope raster. It intentionally dropped a successful save response, retried the same request,
verified one stored record, reloaded, restored band 2 at opacity 0.37, and preserved a local name
when a concurrent rename produced a conflict. Desktop and 390px phone checks found no page
exceptions or unintended horizontal overflow; screenshots were inspected. Local artifacts:
`/tmp/land-saved-views-result.json`, `/tmp/land-saved-views-desktop.png`,
`/tmp/land-saved-views-mobile.png`. The initial browser fixture attempt loaded a second Vite
module instance and did not reach save; the corrected fixture uses the app's existing module.
A further real API check saved and opened a combined slope/historical-alignment view, preserving
both layer kinds, bands, opacity and attribution. Its artifact is
`/tmp/land-saved-views-alignment-result.json` (view `4ab66be3-78fa-4f91-9c49-836d9e8be982`).
The browser fixture view is `c1531171-79f3-4fa6-9476-5f85a6dd917d`.

Limits: unfinished view-save capture is retained for the current component session, not yet
recovered after a browser restart. Survey sketches, unsaved asset/import previews, action
footprints and other temporary overlays are explicitly excluded from the saved view; their
underlying records and independent draft recovery remain available. The overall feature is
still in progress and nothing has been deployed.

## Implemented increment: persistent saved-view capture recovery

A captured view save is now written to identity/workspace/land-scoped browser storage before
its creation request. On return, users explicitly recover, download or discard it. Recovery
validates the capture version, land, request identity, finite/bounded camera coordinates, record
references and layer settings within a 64 KB limit. Invalid captures remain downloadable and
cannot silently overwrite a new save. Storage failures keep the captured request available in
memory with a download; another tab's different capture is preserved rather than overwritten.

`POST /land/{land_id}/views/recover` is a read-only reconciliation endpoint. It verifies the
capture against the original request digest and returns the existing view, including a later
rename. A matching saved capture is cleared locally without another creation request or camera
movement. Missing captures are loaded for an explicit retry; altered captures are retained for
review. Viewers may reconcile a previously completed save, but cannot retry an unsaved capture.
The original request key is retained across retries. The captured layer counts and camera are
shown independently of the current map.

Validation: the four saved-view backend tests now also cover missing/reused recovery keys,
changed payload rejection and reconciliation after rename. Six frontend tests cover retries,
read-only recovery, unsaved capture restoration, cross-land/invalid camera rejection, retained
original downloads and navigation races. API lint/format/type checks, web type checking/targeted
lint, generated contracts and production build pass. No additional migration is needed beyond
0027; the isolated preview API was restarted.

A real Chromium/API check on the public National Mall fixture dropped the successful creation
response, renamed the stored view, reloaded the browser, and recovered the capture through the
read-only endpoint. Exactly one creation request and one reconciliation request occurred, and
the recovered local capture was cleared. Desktop and 390px phone checks found no page exceptions
or unintended overflow; screenshots were inspected. Synthetic view:
`7f1ed195-d29c-4f34-9e3d-a7ec12434793`. Local artifacts:
`/tmp/land-view-recovery-result.json`, `/tmp/land-view-recovery-desktop.png`,
`/tmp/land-view-recovery-mobile.png`. The previous increment's restart-recovery limitation is
resolved. Production and the original checkout remain untouched; the full feature remains in
progress.

## Implemented increment: linked research map inspection

Research map outputs now expose searchable, paginated feature lists linked to globe picks.
Selecting a row highlights its geometry without moving the camera; explicit focus controls
frame one feature or the full output. Picking a map feature opens the inspector and reveals
its list page. Zero values, missing values, units and legends remain distinct. Removing a layer
clears its selection; replacing a layer preserves selection only while its feature still exists.

The inspector is independent of the current research page and works for layers restored from
saved views. It retrieves the immutable research output and its citations through the new
workspace-scoped `GET /research/artifacts/{artifact_id}` endpoint, reports an older boundary
revision, and permits inspection of the original source evidence. Saved-view map references
now restore units and legends too. No migration is needed beyond 0027.

Cesium updates highlight styles on existing entities instead of rebuilding geometry. Polygon
holes and multipart areas are retained. Changes unrelated to map layers no longer trigger
this bridge's rendering work. Camera bounds for long lines avoid argument spreading.

Validation: nine focused backend tests pass, including source lookup, boundary staleness and
cross-workspace denial. Six frontend tests cover filtered selection, map-to-list pagination,
zero/missing values, 150,000-point bounds, selection replacement, in-place rendering and
multipart polygon holes. API lint/format/type checks and web type checking pass. A Chromium
check clicked an actual Cesium point, selected another from the list and inspected its source;
desktop and 390px phone checks reported no page errors or unintended overflow. Screenshots
were inspected. The research values and source responses in that browser check were explicitly
synthetic fixtures over the public National Mall test area, not real findings or a live AI run.
Artifacts: `/tmp/land-research-map-result.json`, `/tmp/land-research-map-desktop.png`,
`/tmp/land-research-map-mobile.png`.

The complete feature remains in progress. The original checkout and deployment remain untouched.

Final checks for this increment: all seven land Chromium regression tests pass; targeted web
lint and the production build pass. The build retains the existing PlayCanvas worker-module
externalization warnings. The isolated preview API was restarted with the new source endpoint.

## Implemented increment: reviewed boundary splitting and snapping

`POST /land/split` previews the pieces produced by a drawn cut line. It preserves exclusions
and disconnected land, bounds inputs/results, rejects a cut that does not divide the land,
and returns measured area/perimeter for each part. Explicit `keepParts` selection merges
adjacent retained pieces into a valid boundary. Neither preview nor selection writes a land
record; the user reviews and saves the resulting draft through the normal revision flow.

The editor exposes Split with a line, selectable mapped pieces, focus controls, cancellation,
and undo after applying a choice. Original saved land remains unchanged. Responses arriving
after the line, draft, session or identity changes cannot replace current work. Browser draft
version 3 preserves both unfinished split lines and union/difference/intersection intent;
older drawing/corridor captures remain readable. Restored splits require another preview and
explicit piece selection.

An optional Snap to this boundary control finds nearby corners/edges, including hole rings
and disconnected polygons. A gold ring marks the target within 12 screen pixels. Dragging
excludes the moved vertex and its incident edges so it does not stick to its former position.
This snaps to the current GeoJSON boundary, not unseen parcel boundaries or imagery features.

Validation: 14 land backend tests, 19 focused frontend tests and all seven land Chromium
regressions pass. API type/lint checks, web type/lint checks and production build pass. The
existing PlayCanvas worker-module externalization build warnings remain. The real Chromium/API
check used the public National Mall software fixture: a near-corner click snapped exactly to
[-77.045, 38.891], a two-point cut survived reload, and retaining one of two parts changed the
draft while the saved record remained byte-for-byte unchanged. Desktop and 390px mobile checks
reported no page errors or unintended overflow. The follow-up layout gives selection labels
44px touch targets and uses the standard primary action styling. Screenshots were inspected.
Artifacts: `/tmp/land-split-result.json`, `/tmp/land-split-desktop.png`,
`/tmp/land-split-mobile.png`. No migration is needed; the isolated preview API was restarted.

## Implemented increment: live research progress and optional specialist tools

Research activity now uses the authenticated API client's streaming response, carrying the
same authorization/workspace headers as other private reads. It resumes from the last accepted
sequence, deduplicates replays, catches up incrementally, and cancels on navigation, credential
change or unmount. A buffering/unavailable stream falls back to incremental polling with
bounded retry delays. Access-expired/revoked signals stop reconnection until an explicit retry
or refreshed credentials; revoked progress is cleared. Frames are bounded to 1 MiB, UTF-8 and
split CRLF chunks are handled, and incomplete final frames are replayed rather than accepted.

The client retains up to 2,000 recent activity events, initially displays 100, and lets users
reveal earlier retained updates. Finding/artifact events refresh the investigation with a short
coalescing delay; terminal events refresh it immediately. The main research view's periodic
full-detail fallback is reduced to ten seconds while event delivery supplies prompt updates.
The durable server event history is unchanged.

Following the user's scope correction, dedicated roof analysis is not a completion requirement.
Scenarios now leads with Investigate a possibility and Restoration and cover. The general
entry point moves to the land conversation, preserving an existing question. Solar calculations
remain available under Additional analysis tools and are loaded only when expanded or when
opening a saved assessment. Existing saved scenarios remain usable.

Validation: seven stream/parser/hook tests and eight scenario tests pass, covering fragmented
UTF-8, interrupted frames, cursor consistency, bounded history, reconnection, unsupported-stream
fallback, access revocation and the existing scenario recovery flows. Web type checking passes.
A Chromium check injected clearly synthetic research activity over the public National Mall
fixture; the stream resumed with cursors [1, 2] and the activity appeared exactly once. Initial
StrictMode requests both began at zero; subsequent catch-up reads used cursors 2 and 3. No
research findings or records were written by that test. It also verified the optional solar
entry point and general conversation routing. Desktop/390px phone screenshots were inspected;
no page errors or unintended overflow occurred. Artifacts: `/tmp/land-live-research-result.json`,
`/tmp/land-general-scenarios-desktop.png`, `/tmp/land-live-research-mobile.png`.

Targeted web lint and production build also pass. The existing PlayCanvas worker-module
externalization warnings remain. No schema migration or deployment is part of this increment.

## Implemented increment: grounded map-feature questions

The selected-feature inspector now offers Ask about this feature. It preserves an existing
question, moves to Discover, focuses the question field, and shows a removable feature reference.
Submission sends only an artifact ID and feature index. A pending request retains that reference
and its request key for retry; a newer question or selection is preserved when an earlier request
finishes. Late responses cannot change another land/session's research selection.

The API resolves the reference from a stored map output belonging to the same land, including
outputs from another investigation of that land. It rejects other-land/workspace references,
non-map outputs and invalid indices. In the run transaction it pins the exact feature, value,
unit, method, legend and source boundary revision, and copies the original evidence snapshots
into the destination investigation. Citation validation therefore remains scoped to that
investigation. Retrying a successfully queued request reuses the original snapshot even if the
source later becomes unavailable; changing the focus with the same request key is a conflict.

The worker receives the pinned context and a flag when its source boundary differs from the
investigation's boundary. Small geometries are included directly; larger ones are omitted
explicitly and read through a bounded coordinate-page tool that preserves GeoJSON index paths,
hole rings, closure vertices and disconnected parts. Agent guidance distinguishes research
outputs from verified parcel boundaries or measurements and requires reading source evidence.
Specialist analyses remain optional and driven by the user's question.

Migration 0028 adds nullable focus reference/snapshot columns. The isolated preview API and
research worker now run this implementation. No production deployment or original checkout
change is included.

Validation: 15 research API/worker tests and 16 focused web tests pass. They cover same-land
cross-investigation references, exact evidence copies, citation validation, idempotency,
other-land/workspace rejection, geometry paging and a real worker with a deterministic model
fixture. Web checks cover lost-response retries, preserving a newer question/selection, map
linking and streaming regressions. API lint/format/type checks and web type/lint checks pass.
A real Chromium/Cesium check on the public National Mall software fixture selected mapped
features, inspected a source and submitted the exact focus reference from the phone layout.
The map results and queue response were explicitly synthetic; no research findings were
written by this check. Desktop/390px phone screenshots were inspected, with no page errors or
unintended overflow. Artifacts: `/tmp/land-research-focus-result.json`,
`/tmp/land-research-focus-desktop.png`, `/tmp/land-research-focus-mobile.png`.

## Broader acceptance checkpoint

Before the grounded-follow-up increment, the full web suite passed 1,136 tests with four failures
in the child-process bundle checker caused by sandbox process-spawn denial. All five tests in
that file passed when rerun with the required process permissions; the actual production-bundle
check also passed. This verifies all 1,140 tests across those runs, not one uninterrupted green
run. The full API suite passed 760 tests, failed one orphan-process check and skipped nine
real-S3 tests because TEST_S3_ENDPOINT_URL was unset. The orphan recipe had exited but remained
a zombie under the container's PID 1. Updating the test's process-liveness check to recognize
that state made the targeted test pass; production worker behavior was unchanged. Subsequent
focused test results are recorded above.

Live model and identity-provider acceptance and real-S3 integration remain unverified in this
environment. The full feature remains in progress as described in the current-status table.
Roof reconstruction, surveying or fitted-panel layouts are not completion requirements.

The grounded-follow-up increment also passes all seven land Chromium regressions and the
production build (with the existing PlayCanvas worker-module externalization warnings).
The test database passed a 0028 → 0027 → 0028 migration roundtrip; both isolated test and
preview databases are now at 0028. The final desktop/phone interaction check was rerun after
the focus-chip layout adjustment and again reported no errors or unintended overflow.

## Implemented increment: recoverable research questions and captured retries

General research questions now persist per identity/workspace and land in browser storage,
including their selected map feature, research limits and investigation choice. Reopening the
land restores the draft without submitting it. A recovery notice identifies a changed boundary
revision. An existing question supplied by another workflow takes precedence over the stored
question; its focus is preserved.

Before queueing, the client captures the exact question, limits, focus, investigation, boundary
revision and request key. A lost response leaves an explicit retry record independent of newer
edits to the question. Reloading retains both. Retry checks that the captured investigation
still belongs to this land and revision, then uses the same server idempotency key. A successful
retry removes the record while retaining a newer draft. Users can open the investigation or
explicitly discard the retry record; the interface explains that discarding does not cancel an
already-running investigation. New submission waits until that record is resolved or discarded.

Recovery parsing validates version, land identity, UUIDs, question/label sizes, feature indices,
boundary revisions and research limits, and excludes unknown fields from restored requests.
Unreadable records remain available to download or explicitly discard rather than being silently
overwritten. Browser storage failures are visible; the current in-memory question remains usable.
No credentials, geometry payloads or evidence snapshots are stored in these browser records.

Validation: 21 focused frontend tests, web type checking, targeted lint and production build
pass. Tests cover reload recovery, identical retry payloads/keys, newer drafts, wrong-land
investigations, storage failures and preservation of unreadable records alongside prior map and
streaming regressions. A real Chromium/Cesium check on the public National Mall fixture simulated
a lost queue response, changed the question and time budget, reloaded, and retried. The retry
sent the original 180-second request with the exact same key while the newer 600-second draft
remained intact. Desktop and 390px phone screenshots were inspected; no page errors or unintended
overflow occurred. Synthetic source and queue fixtures were clearly marked and no research run
was created by the browser check. Artifacts: `/tmp/land-question-recovery-result.json`,
`/tmp/land-question-recovery-desktop.png`, `/tmp/land-question-recovery-mobile.png`.
Existing PlayCanvas worker-module externalization build warnings remain. No migration or
production deployment is part of this increment; the original checkout remains untouched.

## Implemented increment: mapped-feature snapping and attributable drawing guides

Boundary drawing, corridor tracing, splitting and vertex edits can now snap to visible research
maps, saved inventory features and explicitly selected drawing guides. Current-boundary snapping
remains independently switchable. A gold ring and source label identify the snap target; snapping
is limited to 12 screen pixels and follows the displayed source data rather than establishing
survey-grade accuracy. Editable previews and temporary split geometry are excluded.

Mapped segments and points use per-layer RBush spatial indexes. Selection-only changes reuse
indexes, hidden/removed layers stop contributing targets, and pointer movement projects nearby
candidates rather than scanning every feature. Point targets, lines, polygon holes and disconnected
polygons are supported. Unsplit antimeridian segments are excluded instead of producing false
cross-world targets. This does not snap to imagery pixels or arbitrary 3D-model edges.

The candidate picker offers Trace using selected features. Reviewed lines become corridor guides;
areas become boundary guides. Clearing the visible guides leaves their source references attached
to the in-progress trace. Optional flat source references retain methods, labels, links, record
IDs, observation dates, attribution and source meanings. The resulting manually traced boundary
remains a study area. References are shown in the review/saved-land provenance view, retained in
boundary revision history and GeoJSON exports, and carried through unfinished-drawing recovery.
Existing version-3 browser drafts remain readable; guide geometry itself is a temporary overlay
and is not restored automatically. Reference nesting is not accepted.

Validation: 39 focused API tests and 27 frontend tests pass, including source preservation across
save/reopen/revisions, inventory/research regressions, holes/multipart/point snapping, hidden
layers, unchanged-geometry reuse and source retention after guide removal and drawing recovery.
The 20,000-point index test projects only the nearby target. All seven land Chromium regressions,
API/web type and lint checks, and the production build pass. The existing PlayCanvas worker-module
externalization build warnings remain; package installation also reports the repository's existing
ESLint/TypeScript peer-version warnings. RBush 4.0.1, already used transitively by Cesium, is now an
explicit web dependency with its type package.

A real Chromium/Cesium check used an explicitly synthetic guide on the public National Mall
software fixture. Near-edge and near-endpoint clicks snapped to the source coordinates; disabling
snapping returned the unsnapped ground point. Removing guides removed their entities, while a
subsequent corridor review retained the exact guide link, attribution and record ID. Desktop,
390px phone and provenance screenshots were inspected, with no page errors or unintended overflow.
The test did not save a land record. Artifacts: `/tmp/land-mapped-snap-result.json`,
`/tmp/land-mapped-snap-desktop.png`, `/tmp/land-mapped-snap-mobile.png`,
`/tmp/land-mapped-snap-provenance.png`. The isolated API and research worker were restarted with
the updated optional source schema. No database migration or production deployment is involved.

## Implemented increment: reproducible general calculations

The agent can now use `calculate` for quantitative questions without adding a dedicated product
flow for each use case. A saved calculation contains its purpose, limitations, labeled cases,
unit-bearing inputs, input origins/bases and citations, ordered formulas, full-precision numeric
results, a canonical request hash and the arithmetic engine version. Examples include comparing
planting costs or converting measurements; this does not introduce a first-class roof feature.

The interpreter supports bounded arithmetic and registered numeric functions, with shared or
per-case inputs and references to earlier formula results. It does not execute arbitrary code.
Recipes allow at most 500 cases, 20 inputs and 20 formulas; each expression has at most 100 AST
nodes. Non-finite/out-of-range values, undeclared or future references, executable Python,
attributes, subscripts and unsupported calls are rejected. Missing inputs and domain/range errors
remain explicit result gaps, with per-cell explanations; a real zero remains zero. Units are
labels, so conversions must be expressed in formulas. Computation verifies arithmetic, not the
agent's transcription, source interpretation or physical assumptions.

Publication recomputes the recipe and rejects altered results, hashes or issues, including the
ordinary publish-artifact path. The saved method is server-authored. Every evidence-origin input
requires citations included in the artifact and scoped to the investigation. Calculations based
entirely on question inputs or declared assumptions may have no citations; other artifact kinds
still require evidence. Lease fencing and existing output keys preserve publication recovery.
`read_calculation` pages saved inputs/results only within the current investigation, and subsequent
runs receive a compact catalog of the latest 20 saved calculations.

Visuals shows a paged results table, optional per-result chart, inspectable input bases and
formulas, gap explanations and exact input-source buttons. Input values and charts follow the
same 25-case page. A JSON download retains the complete recipe, precision, all rows and citations.
Assumption-only artifacts do not show an empty source-inspection button. The existing chart view
was extracted for reuse, preserving its behavior. No new navigation or database migration is
required; generated OpenAPI/TypeScript contracts include the calculation output.

Validation: 44 focused API/worker tests and 15 frontend tests passed. Coverage includes malicious
and unsupported expressions, known arithmetic, scalar broadcasting, formula dependencies,
missing/domain/overflow results, stable hashes, page boundaries, rejected forged outputs,
citation isolation, idempotency, a deterministic calculate/read/complete worker loop, exact input
source selection, synchronized UI pagination, download failures and full-recipe downloads.
API-wide lint/format/type checks, web type/lint checks, the production build and all seven land
Chromium regressions passed. The three calculation UI tests were rerun after test typing fixes.
The existing PlayCanvas worker-module externalization warnings remain.

A Chromium check on the public National Mall software fixture used explicitly synthetic
calculation data without writing research findings. Desktop and 390px phone screenshots were
inspected. It verified zero versus missing values, formulas/gap explanations, per-page charts and
inputs, a 30-row JSON download and no empty citation action. No page errors or unintended overflow
occurred. Artifacts: `/tmp/land-calculation-result.json`, `/tmp/land-calculation-desktop.png`,
`/tmp/land-calculation-mobile.png`. API/worker acceptance used a deterministic model fixture;
live model credentials remain unavailable. The isolated API and worker were restarted, and the
isolated Vite preview was restored on port 5175. The original checkout and deployment are unchanged.

## Implemented increment: research worker presence and "What's here" point facts

### Research worker presence

Land research runs (overviews, terrain, land cover, vegetation, archives, ecology, solar and AI
investigations) are queued by the API and executed by a separate process,
`python -m app.research` (`pnpm dev:research` locally). The API never starts it. Before this
increment the API could not tell an idle worker from a missing one: it only saw per-run leases,
which exist while a run is held, so a run queued with no worker stayed "queued" indefinitely and
said nothing.

Each worker process now upserts one row in `land_research_workers` (migration 0029: id, hostname,
pid, started, last seen, stopped, current run) every 15 s, idle or busy, using the database clock.
The lease keeper refreshes it during long runs; a clean exit, Ctrl+C or SIGTERM marks it stopped.
The worker prunes rows not seen for a week when it starts. Presence writes are advisory and never
break the poll loop. `current_run_id` has no foreign key, so presence never waits on a run lock.

`GET /api/v1/research/status` keeps `modelConfigured`, `model` and `workerRequired` and adds:

- `worker`: `state` is `online` (a non-stopped worker seen within 45 s), `stale` (seen before, not
  lately), `stopped` (the newest worker said it was stopping) or `never`; with `lastSeenAt`,
  `secondsSinceSeen` (database clock, so browser clock skew never shows wrong minutes), `busy`
  (any run in any workspace holds a live lease; it says nothing about whose) and
  `checkInEverySeconds`. Hostnames, pids and worker lists never leave the server.
- `queue`: `queued`, `oldestQueuedAt`, `oldestQueuedSeconds` and `running` (live lease), scoped
  to the caller's workspace.

In the land workspace a queued run explains itself ("Waiting for the research worker: none has
checked in for 14 min. Your request is saved and starts when a worker runs."), and the status is
polled every 10 s while it waits, so the notice clears when a worker starts. Ecology and solar
status rows show the same line. Every disabled research button now says why next to it (connecting,
role, the one queued or running request, AI not configured, a captured request or unreadable saved
question to resolve, an empty question once the box has been used). A queued run's reason is short
("One request runs at a time. Cancel the queued request to start something else."): the worker's
state and age are said once, in the notice beside the run. The Stop button reads "Cancel
queued request" for a queued run. When no worker is online and nothing of this land is queued, a
footnote warns that new requests will wait. In development builds the notice adds
`pnpm dev:research`; production users never see shell commands.

Deployment: `fly.toml` `[processes]` defines only `app` and `worker` (the capture pipeline). It has
no research process, so on the deployed API research runs are queued and never executed; the
status now says so honestly ("none has checked in on this server yet"). Running
`python -m app.research` as its own process group is a deployment decision left open here.

### "What's here": point facts in the inspector

`GET /api/v1/research/place/{fact}?longitude=&latitude=` answers one public fact about one point.
`fact` is `elevation`, `soil`, `land-cover`, `flood-zone` or `ecoregion`. Each reuses a source the
land research already registers, through the same bounded fetches (registry endpoints only, no
redirects, 2 MB cap; a 10 s timeout for point lookups), so no host is added:

| Fact       | Source                                         | Basis                    | Caveat shown                                                                                   |
| ---------- | ---------------------------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------- |
| Elevation  | USGS EPQS (3DEP), public domain                | Elevation model          | Interpolated from the 3DEP model at this point; not a surveyed height.                         |
| Soil       | USDA NRCS Soil Data Access, public domain      | Mapped survey            | Map-unit estimate for the unit containing the point; not a soil test or parcel-wide soil map.  |
| Land cover | ESA WorldCover 2021, CC BY 4.0 (one 10 m cell) | Satellite classification | Broad class for the 10 m cell, classified from 2021 imagery; not today's condition or species. |
| Flood zone | FEMA NFHL layer 28 point query, public domain  | Regulatory map           | Check the effective FIRM and amendments; "No mapped zone" does not establish low flood risk.   |
| Ecoregion  | EPA Level IV ecoregions (2011), CC0            | Regional framework       | Compiled at 1:250,000 for large extents; names the region around the point, not its habitat.   |

Each answer carries `status` (`available`, `empty`, `uncovered` or `unavailable`), a label and
detail, its basis and vintage, the caveat, the exact public query URL, license, attribution and
source resolution. The citation link (`source.linkUrl`) is that query when a browser can open it
(USGS, FEMA and EPA answer a GET with JSON). Soil Data Access takes its SQL only as a POST, and a
WorldCover tile is a 10-20 MB image, so those two link Web Soil Survey and the WorldCover viewer,
and keep the exact query in `data.query` (the SQL) and `data.tileUrl`. Flood zone D reads
"Undetermined flood hazard: not studied" (never "outside the Special Flood Hazard Area"), and a
point FEMA's map leaves out is `empty`, "Not included in FEMA flood mapping", with the same
"does not establish low flood risk" caveat as "No mapped zone". A failure is "could not be retrieved (HTTP 503)" or the exception type, never
a response body, and is never shown as "nothing here". The US-only sources answer `uncovered`
outside the United States without making a request; a missing WorldCover tile (open ocean, HTTP 404) is `uncovered`, not an outage. The flood-zone lookup is a point query with
`returnGeometry=false`: the boundary adapter's envelope query can return a county-wide Zone X
polygon over the 2 MB cap (3.48 MB on a Central Coast parcel), which this avoids; fixing that
adapter is separate work.

Nothing is persisted. Answers are cached in the API process for 6 h, keyed by the point rounded
to about a metre (failures for 2 min; the inspector's Retry asks again past a cached failure).
The route sits behind land identity (`WorkspaceDep`) and its own token bucket (burst 25, 60 per
minute per client; one inspection makes five lookups), so it is not an open proxy to the agencies.
The route is synchronous, so a slow agency holds a server thread for up to the 10 s timeout (15 s
for the WorldCover read). At most 8 lookups ask a source at once across all clients; a lookup
that finds no free slot within 2 s answers `unavailable` ("... was not asked because the server
is busy with other lookups"), uncached, so a hung agency cannot take the threads every other route
needs. Identical lookups in flight are not merged.

In the inspector, any non-site pick (open ground, and Google Photorealistic tiles, which pick as a
layer feature) shows "At this point · looked up live, not saved" with the five rows filling in as
each agency answers. Below them: "Research <land>" when the point is inside a saved land (it opens
that land on Discover), otherwise "Outline this land", which opens the Land tool finding mapped
parcels, fields and lines at the point. The map's right-click menu has the same "Outline land
here", offered like the inspector's only when the workspace is ready and the role can add land.
Both go through `features/land/landEntry.ts`, which clears the active land first (Save would
otherwise revise it with the new outline) and never replaces an unsaved outline
(`hasUnsavedOutline`: a draft, or a sketch with points placed in draw, corridor or split mode, the
same test the browser recovery copy uses): with one open, the button and the menu item read
"Return to your unsaved outline" and only reopen the tool on it. Other entry points (Descent) should use
`landEntry` (`proposeOutline` for a found outline) rather than `useLand.propose`.

The Terrain row above stays: it is the scene's own terrain sample, and the two elevation values
can differ; each names its source and neither is a survey.

Validation: API tests cover presence states and ages, workspace-scoped queue counts, the worker's
idle check-in throttle and stop, pruning, every fact's mapping (including empty, uncovered,
unavailable without bodies, caching and retry), the 422 bounds and the 429 limit. Web tests cover
the copy and precedence of every reason, the queued notice and polling, the Investigate reason,
the inspector rows, retry and CTAs, the menu item and the landEntry safety rules. All five lookups
were checked live at -120.72, 35.53: 236.8 m (1 m DEM, 2019); Corducci and Typic Xerofluvents;
Tree cover (class 10); Zone A, Special Flood Hazard Area, map 06079C; Paso Robles Hills and Valleys.

## Implemented increment: land as mission ground

Migration `0030_plan_land` (written against `0028`; re-point to `0029` when the research
worker-presence migration lands) adds `plans.land_id` (foreign key to `land_areas`, set null on
delete, indexed) and `plans.boundary_revision`, with the check `ck_plans_land_private`: only a
plan that belongs to a workspace may carry land. Missions already scheduled from land actions
are linked on upgrade through `land_action_revisions.mission_id`.

Routes, all workspace-scoped through `WorkspaceDep`:

- `GET /api/v1/land/{id}/ground` returns the land as plannable items: the boundary (`L-` plus
  the first 8 hex of the land id) and every polygon or line inventory feature (`L-` plus the
  feature id's). A line becomes an assumed 10 m work corridor and keeps the line itself; points
  are counted as `skipped`; retired features are left out. Each item says where it came from.
- `GET/POST /api/v1/workspaces/missions`, `GET/PUT /api/v1/workspaces/missions/{id}` and
  `PATCH /api/v1/workspaces/missions/{id}/status` list and act on a workspace's plans. Any role
  reads; owners and editors write. A `land:<id>` project must name that land, the land must be
  in the workspace, and a pinned boundary revision must exist. A mission scheduled from a land
  action refuses revision here (409): the action holds the approved version.
- The public `/api/v1/plans` routes are unchanged and still never list, read or change a
  workspace's plan.

Scheduling a land action now records the land and boundary revision on the mission, uses
`LA-<8 hex of the action id>` area ids (and `-<n>` per step footprint) so two actions never share
a zone id, attributes each area to the approved action revision, and rejects another land's
`land:<id>` project id. Stored missions keep the ids they were written with.

Planning: the planner sizes a zone that comes with an outline from a shapely port of the
console's coverage passes (`services/coverage.py`; holes and MultiPolygons included), using the
assumed machine classes in `services/machine_classes.json`, the one table the web app reads too
(robotic brush mower, seed drill, survey rover, Mavic 3 Enterprise class mapping drone; every label says
"assumed"). Hours are path length over the class speed plus turns, with each zone's basis
sentence on the draft and one combined line under assumptions. A measured work-log rate still
sets an area's hours. Zones without an outline (the simulated demo fleet's) are sized exactly as
before. Outlines are never sent to the model; the measured paths are. An outline that is not
valid (a corner dragged across an edge, OpenStreetMap outers that touch) is repaired with
`make_valid`, or dropped so its zone is sized by acres, rather than failing the draft. Survey
detail scales a drone's flying only, and each drone zone records the height it was flown at.
"Monitor", "patrol", "check", "watch", "count" and "scout" read as surveys as whole words, so a
"county road" or a "checkpoint" does not turn a repair into one.

Console: selecting a saved land engages it as ground. It joins the current project when that
project's zones or machines are on the land (the fleet stays), or becomes the land's own
`land:<id>` project. The plan card and the bar take the land, or the feature a goal names ("the
fence", "the troughs"), before OpenStreetMap, a click or the rough square; a click on the land
takes the land. A land's zones never join another land's own project: with a plan card open
there, selecting a second land waits until the card is approved or closed. Approvals that touch
land ground go to the workspace routes with the land and boundary revision; offline, such a plan
is not saved at all. Plan lists public and workspace plans together (a land's own project asks
only the workspace route, never public `/plans`), Fleet lists what is scheduled, the plan card
shows each zone's path, passes, spacing and hours with the assumed class, and the Land panel
offers "Plan work on this land" (to owners and editors) and "Open in Plan" after scheduling. A
land action's mission is revised as the action, in Land, never with the agent. Browser storage
(`twin.mission.v1`, now version 2) never holds a workspace plan or private zone, and a sign-out
or workspace switch forgets them.

Known limits: a land's own project has no machines, so its plans draft unassigned. The planner
receives land outlines briefly through the write-gated, stateless plan-draft endpoint. The
console's map preview of passes still uses its own preview spacing, not the machine classes.

## Implemented increment: the demo ranch as a saved Land

`uv run python -m app.seed.demo_land` (in `apps/api`, after `python -m app.seed`) saves the demo
ranch, Oak Bench Ranch, as a Land in the pilot workspace, so the joined loop has real ground at
the place the simulated fleet works. Everything comes from `apps/api/app/seed/demo_ranch.json`,
the file the `demo-ranch` site and the web app read, and goes in through the same services as the
Land panel's routes:

- **The Land**: the demo outline as its boundary, with source method `imported`, record id
  `demo-ranch` and meaning `study-area`. The description says that the outline is a demo area,
  not a property line, that the name, fleet, zones and plans are simulated, and that nothing has
  been inspected.
- **Three inventory features**, each a `candidate` with an external reference
  `demo_ranch.json` / its key in the file, clipped to the outline:
  - _Perimeter fence_ (`other`, a line on the outline's east, north and west edges). Its source
    reads "Demo outline fence, simulated", and it carries `simulated: true`. No fence was mapped
    or surveyed.
  - _Figueroa Creek_ (`water`): the USGS NHD flowlines, merged into one line, with the NHD
    credit (public domain).
  - _Oak woodland_ (`vegetation`): the ESA WorldCover 2021 tree cover and shrubland patch, with
    the CC BY 4.0 credit. WorldCover does not map species, so `species` reads "oak (assumed)".

It writes nothing observed: no inspections, findings, evidence, rasters, surveys or research
results. Those come from people and from real runs. `--research` queues the Land's overview, the
same `ensure_overview` behind `POST /land/{id}/overview`, once per boundary revision. It says
when no research worker is running (`pnpm dev:research`).

Running it again is safe. The Land's id is `uuid5(workspace, "demo-land/demo-ranch")`, and each
feature's is the one `land_features.create` derives from a fixed request key. So a rerun finds
them by id, creates what is missing, and brings the outline, the words, the sources and the
geometry back to the file's as new revisions, which keeps any edit in the history. Some things
stay the user's: the Land's and features' names, a feature's status (confirmed or retired), the
evidence they linked, and anything they added. `create_land` takes an optional `identifier`
for this, which the API never passes.

In the console: with the demo ranch open, select Oak Bench Ranch in **Land**. Its ranch zones
overlap the Land, so `groundModeFor` layers it: the project stays the demo ranch, with its fleet
and zones Z-14, Z-21 and Z-08. The fence, the creek and the oak woodland join them as private
zones, and the boundary stays with the Land layer. "Monitor the fence every week" then names
the fence. Two console details came out of checking this in a browser:

- A line feature's chip now sits halfway along its line (`halfwayAlong`). The middle of the
  fence's bounds was in the middle of the ranch.
- A zone's card no longer prints the same source twice when its note and attribution are the
  same, as they are for every Land zone.

Tests: `tests/test_demo_land.py` covers the listed Land, the features and their sources, the
ground's notes, geometry inside the outline, a rerun that changes nothing, a rerun that puts the
file back and keeps the user's choices, a changed file, workspace scoping, the queued overview,
the command's output, and a check that no inspection, evidence, finding, artifact, raster,
survey, document or action exists. `landGround.test.ts` checks that the seeded outline layers
onto the demo ranch's project and where line chips sit.

## Implemented increment: seasonal signatures, bare-earth terrain and NHD flowlines

Three land-scale data sources for the later "Show once, everywhere" and Dress Rehearsal work.
They are computed or read once per boundary revision through the existing research run, raster
processor and private raster storage; no migration.

- **Seasonal 10 m signatures** (`sentinel-2-signature`, "Prepare 10 m signatures" in Explore this
  land). For each of two to six chronological windows (the button asks for 30-day windows centred
  on mid-January, April, July and October of the latest complete year) the processor selects one
  Sentinel-2 L2A scene with the same bounded catalog search and local SCL check as the vegetation
  comparison (`vegetation.select_scene`, at most two quality masks per window) and samples green
  B03, red B04, NIR B08, SWIR B11 and NDVI on the shared 10 m grid. One SCL 4/5 mask per window
  applies to all five bands, so a cell is a full sample in a window or missing; nothing is
  interpolated. B11 and the mask are 20 m sources resampled nearest. The COG holds five float32
  bands per window, band-interleaved without a float predictor (smaller and faster to tile than
  pixel interleaving or PREDICTOR=3 on real data). Metadata carries the dated scenes as a
  `VegetationTimeline` (each window's NDVI band, acquisition date, quality counts, common-cell
  means), which the existing timeline view shows, and a `SignatureInfo` listing every band with
  the common-cell mean and spread that a later similarity's z-scores use. Common cells are clear
  in every band of every window. The map shows the dated NDVI bands; the download and point
  sample hold all of them. A saved view can keep any of a signature's bands (up to 30, the
  six-window ceiling), such as October's NDVI at band 20 of four seasons. Warnings say a similar signature is not a species identification, cover
  estimate or habitat assessment and that "alike" is relative to the variety on this land.
- **Bare-earth terrain** (`usgs-3dep-13`, "Analyze bare-earth terrain"). USGS 3DEP 1/3
  arc-second DTM tiles (`prd-tnm.s3.amazonaws.com`, now in the `raster_io` allowlist, read
  through `RasterOpener`) are warped bilinearly onto the 10 m grid and merged by tile name;
  `terrain.derive` gives elevation (NAVD88 m) and slope (degrees). Each tile is cited with its
  ETag and FGDC metadata link; its file date is not an acquisition date. Outside the United
  States, or where USGS publishes no tile, the run fails with "use cop-dem-glo-30 here", and the
  button is disabled with that reason.
- **NHD flowlines** (`GET /land/{id}/hydrography`). USGS NHD Flowline - Large Scale features
  within 100 m of the boundary, read live by the API (the browser cannot ask NHD), clipped with
  shapely, each with its permanent identifier, GNIS name, FCode and a kind. Natural channels
  are perennial, intermittent, ephemeral, or unclassified when NHD records no permanence (FCode
  46000, common in NHD); `artificial-path` (FType 558) carries a stream through a pond, lake or
  wide river, so a creek check that includes it keeps the creek unbroken; `other` is only canals
  and ditches, connectors, conduits, pipelines and coastline. A creek check should take every
  kind but `other` (and decide about ephemeral), never just perennial and intermittent. The
  query asks for 6-decimal coordinates generalised by 1e-5 degrees (about 1 m, well inside the
  1:24,000 accuracy of about 12 m), which made the same answer 40% smaller (98 KB instead of
  166 KB). It answers "uncovered" outside the US and "unavailable" when it cannot read
  flowlines: an upstream failure says to try again and is asked again next time; an answer past
  the 2 MB response limit says trying again will not help and is kept for ten minutes. It says
  when the 500-record limit was hit and keeps good answers in memory for an hour per land
  revision. Responses are private, no-store.

The research agent's `analyze_raster` text describes both datasets. Processor limits: signature
120 CPU s and 200 s wall time with a 128 MiB / 1,024-request / 180 s read budget; NDVI 165 s;
3DEP 75 CPU s and 90 s with a 64 MiB / 256-request / 60 s budget; all keep the 16 MiB output
limit. A raster run's research time budget must cover its processor: Explore this land sends
300 s for signatures, 240 s for NDVI and 180 s otherwise (`maxSeconds` in
`features/land/rasterRuns.ts`; the default budget is 180 s). The worker stops a processor 15 s
before the run's budget ends, so a raster that finishes is always saved with its evidence and
a budget stop says plainly that no raster was saved; with under 45 s left (a resumed run counts
from its first start) it refuses to start one and says why. An investigation's agent sees
`remainingSeconds` and is told a signature needs about 220 s of it, which a default 180 s
investigation never has, so it points the person to "Prepare 10 m signatures" instead.

Map tiles of a stored raster decode each band once into a 64 MiB in-process cache keyed by the
blob's content hash (`land_rasters.band_array`, for the later similarity too), instead of
reading the whole blob from Postgres and inflating the band for every 256 px tile; tiles that
arrive together wait for the one decode. Rendering one z15 view of the signature's October NDVI
(30 tiles of band 20, 3.35 MB blob) took 3.29 s without it and 1.53 s with it, in process.

Measured on the Oak Bench Ranch demo outline (1,349 ac, 254 x 255 cells at 10 m) with the API,
the research worker and the buttons in a browser, through this environment's proxy: bare-earth
terrain finished 6.0 s after the click (5.3 s processing), read 1.8 MB from one tile (n35w121)
and stored 0.49 MB (elevation 312 to 505 m, mean slope 13.2 degrees, every boundary cell
valid). Seasonal signatures finished 98.7 s after the click (96.9 s processing), read 43.5 MB
from 28 cited files and stored 3.35 MB in 20 bands; scenes acquired 2025-01-15, 2025-04-05,
2025-07-24 and 2025-10-17 left 54,436 of 54,596 cells (99.7%) clear in every window, with
common-cell NDVI 0.29, 0.52, 0.26 and 0.29. The July NDVI band shows the oaks and the Figueroa
Creek corridor staying green over dry annual grass. Run again with the button's 300 s budget,
signatures finished 95.1 s after the click (93.3 s in the worker); October's NDVI (band 20)
mapped, and a view saved on it reopened on band 20 in a fresh browser. Hydrography answered in
2.9 s (17 ms cached) with 55 flowlines in 56 clipped pieces: 21 intermittent (10 of them
Figueroa Creek), 34 ephemeral and 1 artificial path; the generalised query answered in 1.5 s.
Known limit: land rasters are Cesium imagery layers, which are not drawn while Google
Photorealistic 3D Tiles are on; the maps above were checked in the open world mode.

## Implemented increment: the Fog of Knowing (fog-v1)

A saved land can show how fine and how fresh the knowledge this workspace has _stored_ about
each patch of it is: soft fog over the map that is thick where nothing is stored, or only old or
coarse records, and thin where there is recent, fine evidence. It is computed, never measured,
and it lifts only when new evidence is stored; computing it again changes nothing.

Policy `fog-v1` (`apps/api/app/analysis/knowledge.py`). Cells are lon/lat-aligned, 20 m by
default (10 to 100 m, grown until the grid is at most 256 x 256), and a cell is inside when its
centre is (holes stay out).

    density = 1 - max over records r covering the cell of Q(class r) x 0.5^(age r / H(class r))

| Class                              | Q    | Half-life | Example                                         |
| ---------------------------------- | ---- | --------- | ----------------------------------------------- |
| Centimetre capture (5 cm or finer) | 1.00 | 180 d     | a drone or phone capture                        |
| Field inspection                   | 0.85 | 120 d     | a person's dated inspection of a mapped feature |
| Aerial imagery (5 cm to 1 m)       | 0.70 | 365 d     | an aligned air photo or historical map          |
| Satellite (1 to 20 m)              | 0.45 | 90 d      | a cloud-masked Sentinel-2 scene                 |
| Regional or unlocated              | 0.20 | 5 y       | a soil map, a 30 m elevation model, a document  |

- An undated record counts at Q x 0.25, as if two half-lives old. The age is from the
  observation (or publication) date, never the retrieval date.
- A record with no published location (all research evidence and field surveys today) counts
  for the whole land with Q capped at the regional 0.20; its half-life stays its class's.
- A record placed less precisely than a cell (a capture's georeference uncertainty, an aligned
  image's leave-one-out RMS) drops one class.
- An inspected point reaches 15 m, a line 5 m; a footprint covers the cells it touches.
- The qualities and half-lives are judgement, not measurement. Any change is a new policy
  version, and the version is shown beside every number.

Counted: public scan sites whose assets meet the land (class from the recorded ground
resolution; a scan with none is assumed centimetre and says so), inspections of the land's
inventory features (on the geometry as inspected), the land's newest 20 rasters (each dated
Sentinel-2 scene on the cells it actually saw, so cloud-masked cells earn nothing; the
`sentinel-2-signature` dataset counts the same way from its observations; ESA WorldCover as
2021; Copernicus and USGS 3DEP elevation as regional and undated), archive images aligned to the
ground, research evidence and field surveys. Not counted, and listed in the legend: the Google
photorealistic basemap (no capture date), public NAIP air photos and Sentinel-2 scenes that are
not stored here (the Time Lantern shows them; they count once stored), and mission plans, the
simulated fleet and its tracks, Living Survey motion and inferred fill. GBIF records withhold
their coordinates and are never drawn as points. Derived analyses (solar, scenarios) and name
lookups are not observations.

API, read-only and workspace-scoped (viewers may ask): `GET /land/{id}/knowledge?cellM&asOf`
returns the grid as two `LonLatGrid`s (the shared wire format in `app/schemas/grids.py`, row 0 at
the north: density u8, 0..250 = density x 250, 255 outside; best record u16, 65535 none), the
records, a summary and the policy. `GET /land/{id}/knowledge/cell` lists every record covering
one cell with its credit arithmetic. `POST /land/{id}/knowledge/scout-areas` returns the foggiest
patches within a budget, nearest a focus point when the fog is even. Each kind of record is read
in one statement, so the query count does not grow with the evidence.

Console: the strip's "Fog of knowing" pill shows the fog over the selected land, one ground
drape (`cesium/GroundGridLayer.ts`) whose noise breaks it into fog rather than a heatmap. While
it shows, a plain click inside the land opens the cell's card, on a mission zone's fill as well
(outside the land, or before the grid is in, the zone is selected as usual; the map menu has
"What do we know here" too); otherwise a click on the ground still does nothing. Saving an
inspection, editing the inventory, aligning an archive image, saving a field survey, finishing a
research run, registering a capture or revising the boundary refreshes everything computed from
the land's records (`invalidateLandDerived`); the grid is also keyed by the boundary revision, so
a revised boundary never shows the old grid. Where fog went it dissolves over 1.5 s. The drape's
fragment shader reads two texels and two 3-octave noise fields (the land's edge comes pre-softened
in the texture), since it covers the screen on the photorealistic tiles. "Fetch latest satellite
look" stores one Sentinel-2 scene from the last 30 days.

"Lift the fog" asks for the foggiest ground within one battery (about 64 acres at the assumed
drone's flight) near the view and drafts a survey over it through a fresh plan card (one left
open, a revision included, is replaced, as "Plan on this land" does): the patches
are private area zones carrying their scout capture (`PlanArea.capture`), the map draws their
real flight lines dashed with "Would lift fog · forecast", and the planner flies them at those
numbers. The draft or the approved plan exports a DJI WPML route (KMZ: `wpmz/template.kml` and
`wpmz/waylines.wpml`) for a Mavic 3 Enterprise class aircraft in DJI Pilot 2 or FlightHub 2:
one waypoint at each end of every drawn lane, areas in the plan's step order as the map routes
them and each with its own capture's lanes, photos every photo spacing along lanes and none on
transit legs, heights relative to the take-off point. Areas set to different heights or overlaps
are not exported as one route. Changing the height restates the detail in the goal and in the
planner's detail answer. Caveats, said in the console as well: the route is a draft that has not
been imported into DJI Pilot 2 or FlightHub 2 or flown; it does not follow terrain (the card
samples the 3D surface under the patches once per outline, at most 150 points reused across
heights, and gives the resulting detail range and lowest clearance); the zip is store-only; the
battery time is assumed; check airspace
(B4UFLY or LAANC) and fly under Part 107. The drone class in `machine_classes.json` is now that
aircraft's wide camera (73.7 degrees across), so lines at 60 m are 27 m apart.

## Implemented increment: the Descent and the Time Lantern

The welcome card asks "Where is your land?" (with the Land flag off, "Where do you want to
go?"). An address or place is geocoded once, on submit, never per keystroke (Nominatim, the
fallback geocoder, forbids autocomplete); coordinates ("34.698, -120.04", with or without ° and
N/S/E/W) skip the geocoder. One place flies there at once, several are offered as choices, none
says so in place. The flight is one glide (`features/descent/descent.ts`, CameraController.glide)
from wherever the camera is to a 700 m square around the point, re-pointed on the way by the
terrain sampled at full detail and by the OpenStreetMap area containing the point, which is drawn
as "Mapped area from OpenStreetMap, not a legal parcel". A geocoder box more than 8 km across is
a town or a region: a plain flight to its box, no lookup. Overpass is never waited for; its
failure is a line on the arrival card. The arrival card offers "Outline this land", which goes
through `landEntry` (`proposeOutline` for the mapped area, a new `drawOutline` otherwise) so a
selected land is never revised by it, and "Look back in time", which opens the Time Lantern over
the place.

The Time Lantern (Y, or Look back in the strip; ADR 0010) shows dated public imagery under the
cursor through the photorealistic world: each year NAIP was flown (Planetary Computer mosaics,
U.S. public domain), the clearest June to September Sentinel-2 scene of each year since 2016
(off by default: 10 m), and each edition of the USGS topographic sheets covering the place, which
this API tiles from USGS's own GeoTIFFs (`/api/v1/imagery/usgs-topo/tiles`, through the raster
allowlist, the collar cut at the neatline). The wheel over the map steps through real
acquisitions only, Alt peeks, Esc closes (a sheet or popover open over the map takes Esc first).
The rim names the source, its date (the record's under the cursor), its licence and its
positional accuracy on whose word: the sentence the year's NAIP records state where they have
one ("+/- 6 meters to ground specification"; one record is read per year and state, and the
four quarter quads over the ranch say the same in every year that has records), "not stated"
where they have none, "could not be read" when the record did not answer, the 1947 National Map
Accuracy Standards for sheets made after them, "predates the standards" before; for a sheet,
also the stated accuracy of its NAD27 to WGS 84 shift ("datum shift ±7 m": the NADCON grids that
would do better are not installed on the server); and every source says "not checked against
today's 3D world". Under the rail the strip says what the date means ("Catalog publication
year; surveying, revision and scanning may have different dates") and every caveat in the
catalog's words (the datum shift, a sheet edge that could not be matched to the map grid, a
scene's cloud), with a link to the record under the lens (or the source's only record). When the
cursor is outside the area listed, it says where the list is for. A source that could not be
reached is counted apart from a source with nothing here ("2 sources could not be reached, 1 has
nothing here"), an empty rail with one of them says the list may be incomplete rather than that
nothing exists, and "Try again" asks again.

`GET /api/v1/imagery/dated` is a public, rate-limited read (each side at most 0.25°; each
provider's answer cached 6 h per area rounded to 0.01°, and a cached answer spends no token). A
catalog missing a provider that could not be reached is held 30 s and marked `no-store`; the next
one asks that provider alone. A saved land's bounds are rounded outward to 0.05° before they
reach it, and a land selected after a descent replaces the descent's area as the place looked
back on. The topo tile route is an `async def`: a tile waits for one of four sheet readers on the
event loop, holding none of the threads every other route runs on, and the reads run on threads
of their own. Past twelve waiting, or four seconds, a tile is answered 503 with Retry-After,
which the lantern's layers wait out and ask again (`cesium/busyTiles.ts`). The bytes a sheet is
read for (its size and ETag, its header and IFDs, the image blocks neighbouring tiles share) are
kept in process, so a tile costs only the ranges no tile has read yet. The three sources are registered with the
research sources (`/research/sources`) and list what exists without storing evidence; the
overview does not run them. Not done: Esri Wayback and the Esri-hosted topo service (not openly
licensed), image registrations and NDVI rasters in the lens (they need workspace auth on the
tiles), and any check of alignment against the Google mesh.
