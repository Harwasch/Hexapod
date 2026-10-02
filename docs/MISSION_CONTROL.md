# Mission control

The web app is the operator console for autonomous land-management robots. The map is the
hero: one CesiumJS world from Earth down to the machines working a site, with the mission
layer (zones, machines, plans, the agent's activity) drawn over it. This document describes
that layer and the seam where real fleet data plugs in.

## Layout

```text
┌ Site switcher ── [ ⌘K  Search, run an action or ask the agent ] ── Map · Plan · Fleet ┐
│ Layers    ┌ tool panel ┐                          Toasts      ┌ Plan / Fleet drawer ┐ │
│ Measure   │ (one at a  │   markers and zone       Selection   │ full height; the map│ │
│ Add       │  time)     │   chips; one site pin    Inspector   │ beside it stays live│ │
│ Settings  └────────────┘   above 50 km                        │                     │ │
│              [ Blackrock Mesa  Splat · Mesh · Points ] [ Load ]└─────────────────────┘ │
└ Fleet: 4 working · 2 need attention │ Agent: <task> +4 tasks ⌃          ◎ ⌂ │ Data ┘
```

On a phone (≤ 640 px) the same parts stack: the site switcher with a search button at the
top, one bottom sheet (a tool panel, Plan or Fleet, or what is selected), the data credits
in a thin strip of their own at the right edge, the status line in one row with the compass,
and a tab bar — Map, Plan, Fleet, More — at the bottom. More holds the four tools; the search
button opens the command box full screen. Between 641 and 899 px the rail is a row of the four
labelled tools above the bar.

- **Site switcher** (`features/mission/ProjectCard`) names the site — one name per site,
  its project's when it has one (`features/sites/siteNames.ts`, `docs/GLOSSARY.md`) — and
  leaves room beside the name for a status chip. Its menu is where you change where you are:
  every catalog site (fly there; the one you are at is marked), the current site's saved
  views (open, save the current camera, delete; `features/bookmarks/savedViews.ts`), "Add a
  site", and links to the scan gallery and the data console. `s` opens it on its sites, `b`
  on its saved views; both are command-box actions ("Switch site", "Saved views").
- **Simulated motion badge** (`features/living/SimulatedBadge`) sits directly under it whenever
  the Living Survey is animating, in the same corner and the same voice the project badge uses
  for a simulated fleet. It has to be ambient: Gaussian splats never write depth and are
  invisible to picking, so a swaying tree cannot be clicked and the label cannot hang off a
  selection. It claims only that the _motion_ is modelled and that the geometry under it is
  never written to; whether that geometry is a measured capture, and at what resolution, is
  stated per site in the Inspector ("Measured and simulated").
- **View tabs** (`ViewTabs`) switch `map` / `plan` / `fleet`. Keys `1` `2` `3`.
- **Tool rail** (`features/shell/ToolRail`): four labelled tools — **Layers** (`L`), **Measure**
  (`M`), **Add** (`U`) and **Settings** (`,`). Layers starts with the four favourites
  (Imagery, Vegetation, Zones, Tracks; `features/layers/LayerFavourites`) and has two modes,
  All layers and **Compare** (`C`; the swipe divider outlives the panel,
  `features/compare/Compare`). **Add** has two tabs: **Upload a capture** (drop zone, the
  phone handoff and the captures list) and **Link a source** (ion, 3D Tiles, GeoJSON, imagery,
  STAC, or a site; `features/add-data`). The developer console is a switch in Settings ›
  Advanced (and `D`). Everything that left the rail is still a command-box action.
- **Markers and chips** (`MissionOverlays`) are DOM elements positioned every frame from
  `MissionManager` anchors, so they stay crisp and accessible (real buttons, keyboard focus).
  Above 50 km of altitude a project's markers and chips collapse into one **site pin** with
  the site's name and how many things it stands for; it flies to the site
  (`missions/sitePin.ts`).
- **Selection card** shows a machine (battery, acres, shift, Pause / Camera) or a zone
  (progress, Open plan / Reassign).
- **Plans** and **Fleet** open as a drawer at the right edge, full height (a bottom sheet on a
  phone): plans with detail + "Show on map", and machines with a treatment log. The map
  beside it stays live — it takes clicks, and a Fleet row flies to its machine and opens its
  card next to the drawer, which stays open. A tool panel and the drawer replace each other
  (`state/layout.ts`). "+ New plan", "Plan a mission" (Fleet) and "Plan here" (a zone
  without a plan) open the plan composer described below.
- **Command box** (`features/command-palette/CommandBox`, `⌘K` / `Ctrl+K` or `/`) is the one
  text box: it replaced the search pill, the ⌘K palette and the agent bar. Results are grouped
  — Sites, Zones and Plans (from the project), Layers, Actions with their keys, then Places
  (the geocoder) — and the last row is always “Ask the agent: …”, which runs the words through
  `lib/intents.ts` (machine and zone ids, views, layers, measure, camera, site fly-to,
  geocoding, and a sentence of work to the planner; `useAgentCommand`). Enter runs the
  highlighted row, the arrows move it, Escape closes. Which row is highlighted follows the
  words (`commandResults.ts`): an instruction or a sentence ("where is TR-07", "mow Z-21 by
  Friday") highlights the agent; a name highlights its best match — a site, zone, plan, layer
  or action before a geocoded place, because places arrive a beat later and Enter must not
  change meaning under a fast typist; the local groups are listed above Places, so the
  highlighted best match is the top row. `?` opens the shortcut sheet, printed from the same
  registry (`app/hotkeys.ts`) the box shows each action's key from.
- **Status line** (`StatusLine`, bottom left) is one pill: the fleet in view ("Fleet: 4
  working · 2 need attention", counted as the Fleet window's KPIs count it), the agent's
  line ("Agent: <current task> +N tasks"; a reply holds it for 15 s, so the answer to what
  was just asked is where the operator is looking), and the connection only when it is
  degraded ("Catalog API offline", "Map key rejected", "Graphics paused"). At globe scale
  with no site it sums up the catalog (sites · machines · alerts). It is a polite live
  region. Its chevron (or `a`) opens the **activity log** above it (`AgentActivityLog`):
  every running, queued and done action and the conversation, led by the full sentence of any
  connection fault. Flows that want the operator to read the agent (`streamOpen`) light the
  agent's line up instead of opening thirteen lines over the map.
- **Bottom right**: compass and Earth share one pill with the data credits. Zoom, top-down and
  explore mode have no on-screen buttons; they are keys (wheel / `+` `-`, `T`, `G`) and
  command-box actions. On a phone the credits (the Cesium ion logo, Google's logo when
  Photorealistic 3D Tiles are on, "Data attribution") leave the pill for a strip above the
  status line: the providers' terms want them on screen, so they never fold behind a button
  (`MapCorner` moves CesiumJS's own credit container between the two places).
- **Developer readouts** (altitude, scale band, metres per pixel, the world, the splat
  renderer) and the deployer's "Shared demo map key" note are behind Settings › Advanced ›
  Show developer readouts, off by default. The splat renderer choice (PlayCanvas / Spark /
  Cesium) is there too; the strip at a site keeps only Splat / Mesh / Points, and beside it
  the model's load: "Loading 3D model 42%", or "Couldn't load the 3D model · Retry"
  (`features/sites/SiteLoadStatus`, from the site's load record `SiteManager` writes). The
  active site's failed model is said there only — no toast repeats it with the asset's file
  name; a failure no pill speaks for (a site loading in the background) still raises one.
- **Other pages**: the scan gallery (`view.html`) and the data console (`admin.html`) share a
  slim header — Globe · Scans · Data console (`src/shared/product.ts`) — on the same tokens
  and fonts. The phone's capture page (`upload.html`) stays on its own.

## Planning with the agent

A plan is a mission for the fleet, and the agent drafts it. The flow is goal → draft → review
→ approve:

1. **Goal.** The operator writes what the fleet should achieve ("clear the star thistle from
   Z-14 and Z-21 with two mowers before seed set") in the plan composer, or in the command box
   as `plan: …` / "draft a plan to …". Zones and machines can be pre-selected with chips; a
   selected zone or machine seeds them.
2. **Draft.** The console sends the goal plus the project's zones, machines and existing plans
   to `POST /api/v1/agent/plan-draft`. The API drafts with Claude (official SDK, structured
   output, model from `ANTHROPIC_MODEL`, default `claude-opus-5`) when `ANTHROPIC_API_KEY`
   is set; otherwise a rule-based planner drafts and every draft says so (`source`, `note`).
   The key never reaches the browser. While drafting, the status line says so and the
   activity log shows the running thread.
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

**Planning from the command box.** The command box is the one place to talk. A sentence of work
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
box while a draft is open are a change to it ("two drones", "finish by Friday"). The camera
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
live positions; the overlays, cards, command box and status line do not change.

## Design language

The look is the "Robot Land Management UI" design: near-black olive base, dark glass panels
(`packages/ui/src/styles/tokens.css`), teal for working/done, blue for the agent, amber for
attention, Instrument Sans for text and Azeret Mono for readouts. Fonts load from Google
Fonts with system fallbacks; the light glass theme is opt-in in Settings.

## Tests

- `src/__tests__/intents.test.ts` — command parsing.
- `src/__tests__/commandBox.test.tsx` — result groups (local before places), which row Enter
  runs, the combobox keyboard, the hotkey registry and the shortcut sheet.
- `src/__tests__/layoutTools.test.tsx` — the four tools, every moved entry still a command,
  Layers' favourites and Compare, the site switcher, the drawer beside a selection, the phone
  tab bar, the site pin, the console's labels and the shared header.
- `src/__tests__/siteLoadToast.test.ts` — a failed model is said by the pill, not a toast.
- `src/__tests__/statusLine.test.tsx` — fleet counts and lines, the agent's line, the
  activity log, globe-scale summary, model load feedback.
- `e2e/app.spec.ts` "mission control" — tabs, plan → show on map, a fleet row → its card
  beside the drawer, the agent's replies on the status line, the layer favourites, the site
  pin; "interaction" — the command box, `?`, Settings › Advanced; the site switcher.
- `e2e/layout.spec.ts` — no surfaces overlap at desktop, laptop and phone sizes, and the
  phone layout (tab bar, More, full-screen search, one-row status, the credits strip with and
  without Google's logo).
