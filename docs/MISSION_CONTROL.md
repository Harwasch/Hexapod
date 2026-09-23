# Mission control

The web app is the operator console for autonomous land-management robots. The map is the
hero: one CesiumJS world from Earth down to the machines working a site, with the mission
layer (zones, machines, plans, the agent's activity) drawn over it. This document describes
that layer and the seam where real fleet data plugs in.

## Layout

```text
┌ Project badge ──── Search ───────────────── Map · Plan · Fleet ┐
│ (site picker)                                Imagery Vegetation Zones Tracks
│ Tool rail        (machine markers, zone chips over the world)   Inspector / Dev
│ ┌ panels ┐                                                      ┌ Plans / Fleet window ┐
│ Feeds · Selection card                                          Agent activity stream
└ Command bar ── "select TR-07" · "show zones" · "fly to …" ── readouts ┘
```

- **Project badge** (`features/mission/ProjectCard`) names the active site's project; its menu
  lists catalog sites and opens Add Data.
- **Simulated motion badge** (`features/living/SimulatedBadge`) sits directly under it whenever
  the Living Survey is animating, in the same corner and the same voice the project badge uses
  for a simulated fleet. It has to be ambient: Gaussian splats never write depth and are
  invisible to picking, so a swaying tree cannot be clicked and the label cannot hang off a
  selection. It claims only that the _motion_ is modelled and that the geometry under it is
  never written to; whether that geometry is a measured capture, and at what resolution, is
  stated per site in the Inspector ("Measured and simulated").
- **View tabs** (`ViewTabs`) switch `map` / `plan` / `fleet`. Keys `1` `2` `3`.
- **Layer pills** (`LayerPills`) toggle imagery, vegetation, zones and tracks.
- **Markers and chips** (`MissionOverlays`) are DOM elements positioned every frame from
  `MissionManager` anchors, so they stay crisp and accessible (real buttons, keyboard focus).
- **Selection card** shows a machine (battery, acres, shift, Pause / Camera) or a zone
  (progress, Open plan / Reassign).
- **Plans** and **Fleet** windows list plans with detail + "Show on map", and machines with a
  treatment log. "+ New plan", "Plan a mission" (Fleet) and "Plan here" (a zone without a
  plan) open the plan composer described below.
- **Agent stream** (`AgentStream`, key `a`) is the running log of agent actions and command
  replies. It has no card while the agent is idle: it appears when something runs (a plan
  being drafted, a demo thread), when a command was answered, or when opened with `a`.
- **Command bar** (`CommandBar`) accepts short intents; `lib/intents.ts` parses them locally
  (machine and zone ids, views, layers, measure, camera, site fly-to, then geocoding).

## Planning with the agent

A plan is a mission for the fleet, and the agent drafts it. The flow is goal → draft → review
→ approve:

1. **Goal.** The operator writes what the fleet should achieve ("clear the star thistle from
   Z-14 and Z-21 with two mowers before seed set") in the plan composer, or in the command bar
   as `plan: …` / "draft a plan to …". Zones and machines can be pre-selected with chips; a
   selected zone or machine seeds them.
2. **Draft.** The console sends the goal plus the project's zones, machines and existing plans
   to `POST /api/v1/agent/plan-draft`. The API drafts with Claude (official SDK, structured
   output, model from `ANTHROPIC_MODEL`, default `claude-opus-5`) when `ANTHROPIC_API_KEY`
   is set; otherwise a rule-based planner drafts and every draft says so (`source`, `note`).
   The key never reaches the browser. While drafting, the agent stream shows the running
   thread.
3. **Review.** The draft shows title, objective, zones, machines, cadence and dates, the
   estimate (acres, machine-hours, days), ordered steps, risks and the agent's questions.
   Title and objective are editable; "Redraft" sends an answer or adjustment ("use three
   machines", "skip Z-21") together with the previous draft; "Show on map" flies to the zones.
4. **Approve.** The plan joins the Plans list as _Scheduled_ with its steps and provenance
   (Claude + model, or rule-based). "Revise with agent" reopens the composer with the plan's
   goal and replaces it on approval.

**Anywhere.** Plans do not need a site or a fleet. With no site visited the project is
"Anywhere" (`missions/anywhere.ts`); on any project the composer's WHERE row makes zones on
the spot: "Use current view" takes half the viewport around the view centre (60 m to 5 km each
way), "Draw area" borrows the measurement tool's polygon drawing and turns the finished polygon
into a zone (`missions/areas.ts`, ids `A-01…`). Drawn areas are kept in the browser and stored
on the plans that cover them (`areas` on the plan record), so a plan opened on another device
brings its ground with it. The world redraws zones whenever the store's project changes.

**Planning from the bar.** The bar at the bottom is the one place to talk. A sentence of work
("3D scan this field into a splat", "mow the orchard by Friday with two mowers", or the older
"plan: …") is a plan request (`lib/intents.ts`, `isWorkRequest`). The flow
(`features/mission/planFlow.ts`) finds the ground first: zones named in the goal; else, when
the goal names a kind of ground ("the lake") and is not pointing ("this field"), the mapped
feature of that kind nearest the view centre (OpenStreetMap); else, on a site with its own
zones, the planner picks; else the agent asks for one click on the map. The click resolves
(`missions/ground.ts`) in order: the smallest OpenStreetMap area containing the point
(`is_in`), then, when the API has a key, a picture of the view with the click marked goes to
`POST /api/v1/agent/outline` and Claude traces the field, pond or lot around it (normalized
image coordinates, projected back onto the ground pixel by pixel), and last a rough 14-acre
square labelled as the guess it is. The agent then drafts with defaults and the card
(`PlanCard.tsx`) shows the result: the ground (its corners on the map, dragging redrafts), the
numbers, what it assumed as chips (each clarification's default; tapping one shows the
alternatives and redrafts), the schedule and steps, risks, and Approve. Plain words in the
bar while a draft is open are a change to it ("two drones", "finish by Friday"). The camera
frames the ground at an angle when the draft lands.

**Task families.** The rules planner reads the goal as a treatment (mow, clear, spray…), a
survey (inspect, map, photograph) or a 3D scan (scan, photogrammetry, splat, mesh, lidar) and
writes steps that fit: a treatment gets a survey pass, treatment per zone and a verification
pass; a survey gets a survey plan, a survey per zone and a review; a scan gets a capture plan
(height, overlap, obliques), optional ground control, a capture per zone at a rate set by the
detail level, reconstruction (compute, not machine time) and registration and QA. The Claude
prompt carries the same rule. `apps/api/evals/planner_cases.json` has a case per family.

**Shaping the ground.** Every area (`A-01`…) is editable on the map while its card is open:
white corner handles drag, translucent midpoint handles pull a new corner out of an edge, an
amber centre handle moves the whole area, right-click removes a corner, Esc ends editing;
camera inputs pause while a handle is held (`cesium/AreaEditor.ts`). "Draw" traces a new
outline corner by corner; "Pick again" waits for another click. Outlines from OpenStreetMap
carry © OpenStreetMap contributors (ODbL) on the zone. Areas stored with the plans join the
project when the plans load, but an area shaped in this browser is never replaced by the
stored copy, and new area ids skip the ids the stored plans already use.

**When the card says "Rule-based planner".** The API did not see `ANTHROPIC_API_KEY`. It
reads the root `.env` and the process environment. On Windows, a variable set with `setx` or
System settings reaches only terminals opened afterwards: close the terminal running
`pnpm dev:api`, open a new one, start it again, then check
`curl http://localhost:8000/api/v1/agent/status` says `"provider":"claude"`.

**Persistence.** Approving writes the plan to the API (`POST /api/v1/plans`; revising is
`PUT` and keeps every approved revision in history; lifecycle is `PATCH …/status`). Plans are
keyed by the mission project id and optionally linked to a catalog site. The console shows
the stored record; when the API is offline the plan is kept in the browser
(`twin.mission.v1`) and says so, and persisted plans replace stale console plans on the next
fetch. The `MissionProvider` seam is unchanged.

**Conflicts, rates, progress.** Before approval the review lists conflicts with other
active plans (a machine booked on overlapping days, a zone another plan still covers; pure
functions in `missions/conflicts.ts`). The planner request carries the same booked windows
plus rates learned from the work log per task family (`missions/rates.ts`), so the rules
planner prefers free machines and estimates from what the fleet actually did, saying so in
the assumptions. A dispatched plan's detail shows progress from the work log against where
the schedule expects it today (`missions/progress.ts`) and offers "Replan from here", which
reopens the composer with the drift written into the goal.

**Quality.** `apps/web/src/missions/planDraft.ts` converts between the API draft and the
console's `Plan`; `apps/api/app/services/planner.py` holds both planners behind one `Planner`
protocol (`tests/test_agent.py`). `apps/api/evals/planner_cases.json` is the planner
evaluation set: `uv run python -m app.scripts.eval_planner` runs it against the rules planner
(a test) and against Claude when a key is set. The dev panel's "Planning" row reports the
session's time-to-approve, redrafts and unedited approvals. `docs/PLANNING.md` is the
product record.

## Data flow

```text
catalog site ──▶ SceneBridge ──▶ MissionProvider.projectForSite() ──▶ useMission (Zustand)
                                                   │
                                                   └──▶ scene.mission (MissionManager)
                                                        zones as clamped polygons
                                                        machine tracks as polylines
                                                        per-frame anchors → overlays
```

`apps/web/src/missions/types.ts` defines `Project`, `Machine`, `Zone`, `Plan`, `AgentAction`,
`WorkLogRow` and `Feed`. `MissionProvider` is the only seam: it returns a `Project` for a site
or `null`. A site without mission data still gets a project (`missions/siteProject.ts`: the
site boundary as its one zone, no machines, plans from the API) so planning works for every
catalog site. The project follows the last site visited: the engaged site goes null when the
camera leaves it, the project does not.

## Demo data is simulated

`missions/demo.ts` ships one project, **Blackrock Mesa**, attached to the public Cesium splat
site. Every record carries `simulated: true`, the project badge says so, and nothing in the
demo is presented as a live feed (camera feeds show "no stream"). Replace
`DemoMissionProvider` with a provider backed by your fleet API (REST/WebSocket/MCAP) to get
live positions; the overlays, cards and command bar do not change.

## Design language

The look is the "Robot Land Management UI" design: near-black olive base, dark glass panels
(`packages/ui/src/styles/tokens.css`), teal for working/done, blue for the agent, amber for
attention, Instrument Sans for text and Azeret Mono for readouts. Fonts load from Google
Fonts with system fallbacks; the light glass theme is opt-in in Settings.

## Tests

- `src/__tests__/intents.test.ts` — command parsing.
- `e2e/app.spec.ts` "mission control" — tabs, plan → show on map, fleet → selection,
  command bar replies, layer pills.
