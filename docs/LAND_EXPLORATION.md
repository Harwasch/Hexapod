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
