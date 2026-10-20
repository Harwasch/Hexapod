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
│              [ Cesium Gaussian splat demo  Splat · Mesh ]      └─────────────────────┘ │
└ Fleet: 4 working · 2 need attention │ Agent: <task> +4 tasks ⌃          ◎ ⌂ │ Data ┘
```

On a phone (≤ 640 px) the same parts stack: the site switcher with a search button at the
top, one bottom sheet (a tool panel, Plan or Fleet, or what is selected), the data credits
in a thin strip of their own at the right edge, the status line in one row with the compass
("4 working · 2 need attention [simulated] │ ◌ +4 ⌃": the labels and the agent's sentence
are out of view there but still read by screen readers, and in the activity log), and a tab
bar — Map, Plan, Fleet, More — at the bottom. More holds the four tools; the search button
opens the command box full screen. Between 641 and 899 px the rail is a row of the four
labelled tools above the bar.

- **Site switcher** (`features/mission/ProjectCard`) names the site — one name per site,
  its project's when it has one (`features/sites/siteNames.ts`, `docs/GLOSSARY.md`) — and
  leaves room beside the name for a status chip. Its menu is where you change where you are:
  every catalog site (fly there; the one you are at is marked) but developer fixtures — the
  synthetic Living Survey tree and audit or probe records, known by their slugs
  (`synthetic-tree`, `audit-…`, `probe-…`) and never by words in a name, listed with
  Settings › Advanced › Show developer readouts on, never a dated capture, always the site you
  are at
  (`features/sites/siteFixtures.ts`; the command box and the status line count the same
  list) — the current site's saved
  views (open, save the current camera, delete; `features/bookmarks/savedViews.ts`), "Add a
  site", and links to the scan gallery and the data console. `s` opens it on its sites, `v`
  on its saved views; both are command-box actions ("Switch site", "Saved views"). Views saved
  in this browser before any site was visited are listed under "Other saved views". Escape
  closes it from anywhere inside, the view-name field `v` focuses included, and hands the
  keyboard back to the badge.
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
- **Selection card** (`SelectionCard`) is the one card for what is selected: a machine
  (battery, acres, shift, Pause / Camera), a zone (progress, Open plan / Reassign), or an
  object of a scan picked in the scene (`features/sites/ObjectCard`: a click, the brush or the
  objects panel; docs/SCENE_OBJECTS.md "Selecting in the scene"). For an object it shows its
  name and category, the candidates the click offered (◀ 2 of 4 ▶), Hide, Show only, Fly to,
  the brush (`B`) with what the painted area matched and "Use painted area", and Delete for an
  object painted in this browser. One selection at a time: a new one replaces the old, so
  picking an object clears a machine or zone and closes the inspector, and picking a machine
  or zone clears the object and puts the brush away (`state/oneSelection.ts`); the outgoing
  card leaves before the next comes in. On a phone it is the bottom sheet, above the strip,
  the credits' line, the status line and the tab bar. A touch screen has no Shift, Alt, wheel
  or Tab, so the brush offers New / Add / Remove and a brush size there instead, and the
  keyboard hints are left out.
- **Clicks on the map** (`cesium/SelectionManager`). A left-click selects a thing on the map
  — a zone, a mapped feature, a 3D tile's feature, an object of a scan (the scene's own
  selection, asked first) — and on empty ground, the globe or a site's surface it does nothing
  at all: no marker, no card, and what is open stays open. Asking about a place is the **map
  menu**: right-click (Ctrl+click on a one-button Mac, a 550 ms long press on touch, our own
  timer since iOS has no `contextmenu`) opens it at the point (`features/map/MapContextMenu`).
  **What's here** opens the inspector for the point, as a left-click used to (on the active
  site's splat scan, where the scan's solids answer, the site's card, so its Placement stays
  reachable from its surface); **Measure from here** starts a distance with its first point
  there (on a scan's surface too: measuring asks the scan's solids before the depth buffer,
  which splats never write); **Plan here** opens the plan composer with the ground at the
  point outlined, and drafts once the bar has a goal; **Fly here** flies halfway to it. It is a
  `role="menu"` (arrows, Home, End; Escape or Tab closes it) and closes on a press elsewhere or
  when the camera moves. It does not open while measuring, exploring, waiting for "the ground
  you mean", painting with the brush or editing an area's corners, where a right-click removes
  a corner. A right- or Ctrl+drag still orbits: Cesium only calls a press that moved less than
  5 px a click. Double-click flies halfway to the point and selects nothing.
- **Objects** beside the representation switcher (`features/sites/InstanceSearch`) opens the
  scan's objects panel as a popover over the regions (`data-hud-popover`, like the site
  switcher and the command box's results).
- **Inferred** beside it, on a scan with inferred fill (`features/sites/InferredStyle`):
  **Show · Highlight · Hide** for what an image model generated where no camera saw, with a
  one-line legend under the strip while it is drawn ("Inferred: generated where no camera
  saw. Not measured."). Highlight turns it purple and hatched and leaves the measured splats
  as they are; the choice is kept on the device (`inferredStyle`), hidden until chosen.
- **Methods**, on a scan that declares other methods' outputs: below.
- **Plans** and **Fleet** open as a drawer at the right edge, full height (a bottom sheet on a
  phone): plans with detail + "Show on map", and machines with a treatment log. The map
  beside it stays live — it takes clicks, and a Fleet row flies to its machine and opens its
  card next to the drawer, which stays open. A tool panel and the drawer replace each other
  (`state/layout.ts`). "+ New plan", "Plan a mission" (Fleet) and "Plan here" (a zone
  without a plan) open the plan composer described below.
- **Command box** (`features/command-palette/CommandBox`, `⌘K` / `Ctrl+K` or `/`) is the one
  text box: it replaced the search pill, the ⌘K palette and the agent bar. Results are grouped
  — Sites, Saved land (by name, with the Land feature on), Zones and Plans (from the project),
  Layers, Actions with their keys (with Land on, "Open Land" and "Outline land" among them),
  then Places (the geocoder) — and the last row is always “Ask the agent: …”, which runs the words through
  `lib/intents.ts` (machine and zone ids, views, layers, measure, camera, site fly-to,
  geocoding, and a sentence of work to the planner; `useAgentCommand`). Enter runs the
  highlighted row, the arrows move it, Escape closes. Which row is highlighted follows the
  words (`commandResults.ts`): an instruction or a sentence ("where is TR-07", "mow Z-21 by
  Friday") highlights the agent; a name highlights its best match — a site, zone, plan, layer
  or action before a geocoded place, because places arrive a beat later and Enter must not
  change meaning under a fast typist; the local groups are listed above Places, so the
  highlighted best match is the top row. Places found for earlier words are neither listed nor
  run while the new words' search is pending ("Searching places…"); Enter then asks the agent,
  which geocodes the words as typed. Words asked while the agent is still answering (a change
  typed while a plan is being drafted) are logged at once and answered in turn. `?` opens the
  shortcut sheet, printed from the same registry (`app/hotkeys.ts`) the box shows each
  action's key from.
- **Keys** come from that one registry; `GlobalHotkeys` (`features/shell/AppShell`) binds
  them. `B` is the brush that paints a scan's objects to select them; saved views are `V`.
  `[` `]` cycle a selected object's candidates, and Tab / Shift+Tab do too while the map or the
  object's card has focus (a click on an object gives the map the keyboard); from anywhere
  else, the page's body included, Tab moves focus as it always does. Escape steps back one
  thing per press (`features/shell/stepBack.ts`): what floats over the HUD (the map menu, the
  shortcut sheet, More, the activity log, the write-token prompt), then the brush, measuring, the site
  switcher, the selected object, a machine's feeds, the plan composer, the machine or zone,
  the Plan / Fleet drawer, the inspector and last the tool panel. A dialog, popover or tooltip
  that closes itself on Escape marks the key as handled, and the app's keys leave a handled
  key alone (`useHotkey`), so one press never does two things.
- **Undo and redo** (`Ctrl+Z` / `⌘Z`; `Ctrl+Shift+Z` / `⌘⇧Z` or `Ctrl+Y`) take back and do
  again what changed what the map shows — see "Undo and redo" below. In a text field, a text
  area or editable content they are the browser's own; from anywhere else, a button or a switch
  included, they are the app's.
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
  Show developer readouts, off by default. The splat renderer choice (CesiumJS, the default /
  PlayCanvas / Spark) is there too; the strip at a site keeps only Splat / Mesh / Points, and beside it
  the model's load: "Loading 3D model 42%", or "Couldn't load the 3D model · Retry"
  (`features/sites/SiteLoadStatus`, from the site's load record `SiteManager` writes). The
  active site's failed model is said there only — no toast repeats it with the asset's file
  name; a failure no pill speaks for (a site loading in the background) still raises one.
- **Other pages**: the scan gallery (`view.html`) and the data console (`admin.html`) share a
  slim header — Globe · Scans · Data console (`src/shared/product.ts`) — on the same tokens
  and fonts. The phone's capture page (`upload.html`) stays on its own.

## Undo and redo

One linear history (`state/history.ts`), as every editor keeps it: `Ctrl+Z` / `⌘Z` takes back
the last step, `Ctrl+Shift+Z` / `⌘⇧Z` or `Ctrl+Y` does it again, and anything new done after
an undo empties what there was to redo. It keeps the last 50 steps. Each press is said in the
toasts' place, one line replacing the last: "Undid: Hide Pumpkin 3" with **Redo** beside it,
"Redid: …" with **Undo**, or "Nothing to undo". The keys are bound by `GlobalHotkeys`
(`features/shell/undoHotkeys.ts`) from the registry and printed on the `?` sheet.

What is a step, and how it is taken back:

| Change                                                                                                                                                  | Where it is recorded                                     | Undo                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Hide, Show, Show only on an object (the selection card), a category's or an object's eye in the objects panel, the search's Hide all / Show only, Reset | `state/instances.ts`, every action that changes `hidden` | the scan's hidden set exactly as it was (a category partly hidden stays partly hidden)                                  |
| A painted object made (**Use painted area**) or deleted                                                                                                 | `state/sceneSelect.ts` `addCustom` / `removeCustom`      | the scan's painted objects as they were, stored again; a deleted one returns at its place, so later ones keep their ids |
| A layer switched on or off (Layers, the layer pills, the command box)                                                                                   | `features/layers/layerVisibility.ts`                     | the layer back; a basemap that a new one switched off comes back on                                                     |
| Zones or tracks switched (the layer pills)                                                                                                              | `features/mission/quickLayers.ts`                        | switched back                                                                                                           |
| Swipe comparison on or off                                                                                                                              | `features/compare/Compare.tsx`                           | switched back (the two layers it showed stay on)                                                                        |
| A measurement removed, or Clear all                                                                                                                     | `features/measure/MeasurePanel.tsx`                      | the same entities back on the map (`MeasurementManager.remove` returns their restore) and in the list in the order made |
| An area's corners moved, added or removed on the map                                                                                                    | `state/mission.ts` `reshapeArea` (from `SceneBridge`)    | the area exactly as it was; the undo is announced as an edit, so an open plan redrafts for it                           |

Not recorded: camera moves and flights, selections and highlights (a click, the cycle keys,
the brush's strokes, the panel's row highlight), the objects search's words, settings (the
photorealistic world is one), and writes to the server (renames, a scan's scale, saved views,
uploads) — those are confirmed where they are made and are not the map's state to flip back.

- **One gesture, one step.** What a press changes in one task is one step: the card's Show
  only (show everything, then hide the rest) is one, named from what it did.
- **A toggle and its toggle back fold together.** The same thing changed again within 0.8 s
  (an eye flicked off and on) joins the step before it, which disappears when the change comes
  to nothing. Nothing folds into a step that was undone or redone.
- **A site's steps stay at the site.** Steps about the active site's things — its scans'
  objects, painted objects, areas — are dropped when the active site changes, so `Ctrl+Z` at
  site B never reaches into site A (which may still be loaded, off screen), and coming back to
  A starts afresh: its scans reload and the old ids mean nothing. Layers, the comparison and
  measurements are the globe's and stay. A step whose scan was loaded again since (a new
  table) is dropped when reached, as is a step that fails, and the next one is undone.
- **Opting in.** A new change becomes undoable with `recordAction(label, apply, revert)` (or
  `record` once applied, with `scope: "site"` and an `alive` check when it belongs to the
  site); while a step is replayed nothing is recorded, so `revert` may call recording actions.

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
brings its ground with it. Areas derived from a saved Land (fog patches, like-here patches,
workable ground) are private and numbered apart, `A-L01…`, so a public plan's area, numbered in
a browser that never saw them, never takes the same id on a shared project; a private mission
stored with an id a public plan's different outline also uses gets an `A-L…` id when the lists
are merged (`separatePrivateAreas`). The world redraws zones whenever the store's project changes.

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
frames the ground at an angle when the draft lands. The map menu's **Plan here** starts from
the other end: the composer opens with the ground at the point resolved the same way
(`outlineAt` in `planFlow.ts`), and the sentence typed next is drafted on it.

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
the assumptions. A simulated project sends no rates: its log was written at the assumed class
rates, so nothing in it was measured. Every zone sends its outline, the simulated demo's too
(its acres are its footprints' measured acres), and a zone with an outline is sized from path
length under the assumed machine classes (`apps/api/app/services/machine_classes.json`), each
labelled "(assumed)" in the draft. The rules planner takes a zone or machine by the rest of a
name that leads with its id ("the west grassland" for "Z-14 West grassland"), and without a
count asked for its crew is the machines whose model suits the work (the mowers to mow, the
seed drill to seed). The crew also settles how a survey is sized: with a drone it is flown,
and with only survey scouts it is driven at the survey rover's assumptions. Flight lines never
go to a machine whose model says it works on the ground, so a scan in a fleet with no drone
has no crew and says the mapping drone is not in this fleet. On the demo, "Mow the star
thistle on the west grassland" drafts two mowers over 414 acres, about 174 machine-hours at
2.4 acres an hour, and "Survey the Figueroa Creek draw" drafts the two scouts over its 44
acres, about 3.7 machine-hours at the survey rover's 10 m swath and 1.5 m/s. A dispatched
plan's detail shows progress from the work log against where
the schedule expects it today (`missions/progress.ts`) and offers "Replan from here", which
reopens the composer with the drift written into the goal.

**Dress rehearsal.** "Rehearse" (beside Approve) plays the draft before anyone approves it:
ghost machines labelled "SIM" drive the lanes the map draws for each zone, at 300x or 600x, and
a hand goes up at a lane's worst point when it breaks a limit its machine class is assumed to
keep: grade along the lane or cross-slope across it over `maxSlopePct` (read every 5 m), or a
lane inside `creekBufferM` of a mapped creek (measured exactly, lane piece against creek piece,
so a lane that crosses a creek between samples reads "Crosses"). Both limits are in
`machine_classes.json` and labelled assumed. Each hand names the value, the limit and the
source; a click on it, on the map or in the card, flies there and offers "Leave this lane out"
(not on a followed line such as a fence), which takes that lane out of the draft's preview
only: the approved plan still drives it, so its hands still count on the approve button, which
reads "Approve despite N hands". The end summary counts lanes per rule out of the lanes that
rule could read ("2 of 26 checked lanes"), says the lanes are the preview's and how far apart,
and names a rule clear only where it was read on every lane; lanes outside the elevation grid,
beyond where creeks were fetched (100 m around a Land, or the demo ranch outline), or an NHD
answer cut at its record limit are said as gaps, never as clear. Machines come from the zone's
steps; their class from the model ("Ranger 2 · mower"), else the plan's class for the zone; a
plan with no machines (a Land's own project) gets one assumed generic machine per class, each
in its own colour after the project's; a drone's lanes are flown but not checked. Slope comes
from the Land's USGS 3DEP raster (labelled with the cell size it was read at), else its
Copernicus GLO-30 (a surface model with the canopy in it), else the globe's terrain sampled over
the lanes; creeks from the Land's USGS NHD flowlines (ephemeral ones on request), or the demo
ranch's supplied Figueroa Creek without a Land. A click on a ghost of a project machine selects
that machine; lanes and creeks are looked through to the zone or place beneath. The clock is
the time to drive the drawn lanes, not the plan's machine-hours; the ground between drawn lanes,
turns and transits are not checked, and the card says so. The planner, ground reader and panel
load on the first "Rehearse". Code: `missions/rehearsal.ts` (pure), `missions/rehearsalClock.ts`,
`features/mission/rehearsalGround.ts`, `dressRehearsal.ts`, `Rehearsal.tsx`,
`cesium/RehearsalManager.ts`.

On steep ground a rehearsal raises a lot of hands (Oak Bench Ranch's west grassland, mowed:
147, one per lane and rule). The map gathers them (`missions/rehearsalClusters.ts`): hands of
one kind within 80 m of each other or on neighbouring lanes of a zone share one marker, at
their first hand, with a badge counting the hands raised so far; the radius doubles until at
most 15 markers stand. A click on a marker opens its first hand in the card. The card lists
hands by rule with their counts ("Too steep across · 65 lanes", "Too steep up or downhill ·
55", "Inside a creek buffer · 27"), each group open to every lane; nothing is dropped from a
count.

**Work only the ground it can.** When hands go up on a Land's ground (the Land's own project,
or the demo ranch with the seeded Oak Bench Ranch Land selected over it), the card asks the API
what the machines could work instead (`POST /land/{id}/workable-ground`, docs/LAND_EXPLORATION.md)
and offers it as one action with what it cuts: "−137 ac too steep, −39 ac steep edge, −13 ac
creek buffer, −1.1 ac too small to work → 225 ac". Each zone with slope or creek hands is cut to
the strictest class driving it, on the raster the rehearsal read slope from; the steep edge is
the half cell (5 m) beside too-steep cells, where slope read between cells takes in the steep
one. Hands on lanes left out of the preview count too, since the approved plan still drives
them. The answer is cached per zone outline (a hash of all of it, so a dragged corner asks
again) and dropped with the Land's other derived answers when its records change. Taking it
makes the answer a private area derived from the Land ("West grassland,
workable", attributed "Workable ground for a robotic brush mower: slope ≤ 25 % (assumed), 5 m
clear of steeper cells, on USGS 3DEP 1/3″ DEM, published …, 15 m creek buffer (assumed) on USGS
NHD, from West grassland (Z-14)"), puts it in the source zone's place in the plan, keeps the
goal, machines and answers, and redrafts (`features/mission/workableGround.ts`); the card's
ground reads "GROUND · DERIVED FROM YOUR LAND". The source zone is set aside for that card
(`PlanComposerState.setAside`, followed through a second cut of the workable area), so a goal
that names it ("the west grassland") does not bring all of it back. A zone with nothing left
within the limits is not reworked: it stays whole, and the offer and the note say so. The ground
left out is hatched on the map while the card is open (red too steep, lighter red its edge, blue
a creek buffer, labelled "Left out: too steep · derived"; slivers are counted, not drawn), and
the new draft is rehearsed again straight to its verdict. On Oak Bench Ranch, 414 ac and 174
machine-hours become 223 ac and 101 h, and 147 hands become none, so the button reads "Approve
plan"; where hands remain on the workable area, the card says why (slope hands: lanes run up to
left-out ground between cells; creek hands: the cut had no creeks to buffer, NHD not read
then), counts them against the reworked zone's hands before, gives hands on other ground (a
kept fence) their own reason, and says what Approve now counts against before. A
followed line (a fence) has no ground to leave out and keeps its hands; without a Land the card
says "Save this ground as a Land (or open the Land it is on) to read its terrain", and without
the Land's elevation it asks for the bare-earth analysis, each with "Open in Land". If NHD does
not answer, the cut and the redrafted plan's ground note say the creek buffer was not applied,
and the next rehearsal can still raise creek hands.

**Quality.** `apps/web/src/missions/planDraft.ts` converts between the API draft and the
console's `Plan`; `apps/api/app/services/planner.py` holds both planners behind one `Planner`
protocol (`tests/test_agent.py`). `apps/api/evals/planner_cases.json` is the planner
evaluation set: `uv run python -m app.scripts.eval_planner` runs it against the rules planner
(a test) and against Claude when a key is set. The dev panel's "Planning" row reports the
session's time-to-approve, redrafts and unedited approvals. `docs/PLANNING.md` is the
product record.

## Comparing methods (bake-offs)

A bake-off is judged by eye, in the app, on the same scan: object segmentation, inferred fill
and motion skins each have candidate methods, published beside the scan's tiles and declared on
its tileset (`extras.variants`, docs/SCENE_OBJECTS.md "Variants"). When the scan shown declares
any, **Methods** appears in the strip beside Splat · Mesh · Points
(`features/sites/CompareMethods`). It opens a panel with one row per system the scan offers:

- **Objects**, **Fill**, **Motion**: **Today** (what the scan publishes now, the default) and
  each method by its label -- labels are shown, it is not blind -- and under the row what the
  picked method does, in a sentence.
- A pick swaps in place: no reload, the camera where it is, under PlayCanvas, Spark or
  CesiumJS. Objects reload the objects panel and clear the selection (painted objects stay);
  Fill swaps the inferred layer (picking one while Inferred is on Hide shows it); Motion swaps
  the skin the wind moves.
- The pick is kept per scan for the session, so moving between scans and back keeps it; a new
  session starts from Today.
- A row says "Loading" while a pick's files arrive and "Did not load: …" if they do not, and
  then draws nothing for that system rather than the method before.
- Under that, **What to look for**: where the difference between methods shows. Objects: click
  things (the first click should pick the whole object, the next its part; ground types are
  under Ground & soil in the Objects panel). Fill: set Inferred to Highlight and orbit. Motion:
  turn up Settings › Simulated wind and watch trees and shrubs; rigid things like the spool
  should not bend. A method can say its own (`look` in its entry).
- Each row is one radio group (arrow keys move, Space or Enter picks): segments side by side
  while the labels fit one line, otherwise one row a method with the picked one lit, as on the
  spool, whose motion bake-off has five. It fits a 400 px phone, and the panel scrolls on a
  short screen.

Generated content is easy to tell apart while comparing fills: **Highlight** (beside the
switcher) turns every inferred splat purple and hatched and changes no measured pixel, and
**Hide** takes it away.

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

`missions/demo.ts` ships one project, **Oak Bench Ranch**, attached to the demo ranch: a real
place on California's Central Coast (oak savanna along Figueroa Creek, near Santa Ynez) whose
outline is a demo area, not a property line. The name, machines, zones, plans and work log are
simulated, with no affiliation with the landowner. Every record carries `simulated: true`, the
project badge and the onboarding card say so, and nothing in the demo is presented as a live
feed (camera feeds show "no stream"). `DemoMissionProvider` attaches it to the seeded
`demo-ranch` site and its offline copy `builtin-demo-ranch` and to nothing else. Replace it with
a provider backed by your fleet API (REST/WebSocket/MCAP) to get live positions; the overlays,
cards, command box and status line do not change.

**One file holds the place.** `apps/api/app/seed/demo_ranch.json` has the outline, Figueroa
Creek (USGS NHD), the WorldCover grassland and oak patches, the simulated fence, the two machine
bases, the camera framing and where each layer came from (credited on the site record:
WorldCover's CC BY 4.0 requires it). Its `derived` block (zone footprints, geodesic areas, chip
anchors, creek and fence lengths) is written by `python -m app.seed.demo_ranch --write` and never
by hand; `tests/test_demo_ranch.py` fails when it is stale. The API seed and the web app
(`missions/demoRanchData.ts` imports the JSON) read only this file, so every acreage is computed:
a zone's acres are its footprint's, a plan's acres are its zones', each working machine stands on
the coverage passes the plan preview draws, and the badge's area is the catalog site's, formatted
as the switcher formats it. To move the ranch: edit the source half, run the build, run
`src/__tests__/demoRanch.test.ts` (it prints a camera block to paste when the Overview no longer
frames every zone and machine at 1440×900 and 1280×720), then `python -m app.seed` (and
`python -m app.seed.publish --catalog-only` for the offline catalog). The seed is not part of a
deploy: run both after one.

**The arrival** is the site's authored Overview bookmark, looking north up the valley. A window
it would crop (a phone held upright) gets a fit for its own shape instead (`demoRanchArrival`,
`missions/framing.ts`), and the site pin's "whole site" flight (`MissionManager.flyToProject`)
always fits the live window.

The **Redmond** site (`cesium-splat-demo`, where Cesium's Gaussian-splat sample sits) is the
3D-scan showcase, with no project of its own. Its splat (ion 4547222) is withdrawn in this
deployment; `VITE_DEFAULT_SPLAT_ASSET_ID` or adding the asset from the ion Asset Depot brings it
back. Until then the onboarding card's "or see a 3D scan" goes to the first site that has a
splat.

## Design language

The look is the "Robot Land Management UI" design: near-black olive base, dark glass panels
(`packages/ui/src/styles/tokens.css`), teal for working/done, blue for the agent, amber for
attention, Instrument Sans for text and Azeret Mono for readouts. Fonts load from Google
Fonts with system fallbacks; the light glass theme is opt-in in Settings.

## Tests

- `src/__tests__/intents.test.ts` — command parsing.
- `src/__tests__/commandBox.test.tsx` — result groups (local before places), which row Enter
  runs, the combobox keyboard, the hotkey registry and the shortcut sheet.
- `src/__tests__/selectionCard.test.tsx` — `B` the brush and `V` saved views, `useHotkey`
  leaving a handled key alone, the object card (names, cycling, actions, the touch screen's
  brush), one selection at a time, Escape's steps, Tab only from the map or the card.
- `src/__tests__/layoutTools.test.tsx` — the four tools, every moved entry still a command,
  Layers' favourites and Compare, the site switcher, the drawer beside a selection, the phone
  tab bar, the site pin, the console's labels and the shared header.
- `src/__tests__/siteLoadToast.test.ts` — a failed model is said by the pill, not a toast.
- `src/__tests__/history.test.ts` — the history (order, redo emptied by a new step, the 50
  kept, dead and failing steps skipped, nothing recorded while replaying, the site's steps
  dropped on a site change) and each recorded change undone and redone exactly: objects
  (a category partly hidden included, Show only as one step, quick toggles folded, a reloaded
  scan), painted objects, measurements, layers in an exclusive group, an area reshaped.
- `src/__tests__/undoHotkeys.test.tsx` — the keys and their line, a text field, a text area
  and editable content left alone, the registry and the `?` sheet, the selection card's Hide and
  Show only taken back and done again, a measurement removed and Clear all.
- `src/__tests__/statusLine.test.tsx` — fleet counts and lines, the agent's line, the
  activity log, globe-scale summary, model load feedback.
- `e2e/app.spec.ts` "mission control" — tabs, plan → show on map, a fleet row → its card
  beside the drawer, the agent's replies on the status line, the layer favourites, the site
  pin; "interaction" — the command box, `?`, Settings › Advanced; the site switcher; "a scan
  object in the selection card" — one press of `B`, then of `V`; one Escape, one step; Tab
  not trapped on the body; an object replacing a machine and back; the touch screen's brush.
- `e2e/undo.spec.ts` — `Ctrl+Z` / `Ctrl+Shift+Z` / `Ctrl+Y` in the app, the line and its
  button, the command box keeping its own undo, a layer switched back, an area's corner put
  back and taken off again, both keys on the sheet;
  `e2e/undoYard.spec.ts` (a scene spec) — on the synthetic yard, the card's Hide and Show
  only taken back with two presses and done again with two.
- `e2e/layout.spec.ts` — no surfaces overlap at desktop, laptop and phone sizes (the
  selection card for a machine and for an object with the brush out among them), and the
  phone layout (tab bar, More, full-screen search, one-row status, the credits strip with and
  without Google's logo).
- `src/__tests__/compareMethods.test.tsx` — the methods panel: a row per system, Today first,
  labels shown, picks from the keyboard, long names one a row in a vertical radio group (no
  native select), what to look for (the system's, or the method's own), "Did not load"; and
  Show · Highlight · Hide with its legend. `src/__tests__/variants.test.ts` — `extras.variants`
  read defensively (`look` included), the session's picks, the objects and skin swaps.
- `e2e/variants.spec.ts` — on the synthetic yard under PlayCanvas, Spark and CesiumJS: an
  objects pick changes what the objects panel lists, a fill pick draws its layer and not the
  other's, Highlight changes only inferred pixels, Hide removes them, a motion pick swaps the
  skin; the panel at 400 px and from the keyboard, with the yard's names and the spool's long
  ones.
